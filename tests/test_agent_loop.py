import json

import pytest

import audit
import tools
from agent import run_turn
from providers.base import Call, ProviderResponse
from providers.fake import FakeProvider


@pytest.fixture(autouse=True)
def use_tmp_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_log_path", tmp_path / "audit.jsonl")
    audit.reset_session()


def _assistant(text=None, calls=None):
    raw = {"role": "assistant", "content": text, "tool_calls": calls}
    function_calls = []
    for c in calls or []:
        args_raw = c["function"]["arguments"]
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        function_calls.append(Call(id=c["id"], name=c["function"]["name"], args=args))
    return ProviderResponse(
        text=text,
        function_calls=function_calls,
        raw_assistant_content=raw,
    )


def _tool_call(name, args, id_="1"):
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_single_text_response_returns_immediately():
    fp = FakeProvider(script=[_assistant(text="hello")])
    resp = run_turn([], "hi", provider=fp, max_steps=8, system_prompt="sys")
    assert resp.text == "hello"
    assert resp.trace == []
    assert resp.refusal_category is None


def test_one_tool_call_then_final_answer():
    fp = FakeProvider(
        script=[
            _assistant(calls=[_tool_call("list_cancers", {})]),
            _assistant(text="There are 10 cancers."),
        ]
    )
    resp = run_turn([], "what cancers", provider=fp, max_steps=8, system_prompt="sys")
    assert resp.text == "There are 10 cancers."
    assert len(resp.trace) == 1
    assert resp.trace[0]["name"] == "list_cancers"
    assert resp.trace[0]["result"] == tools.list_cancers()


def test_step_limit_returns_message():
    looping = [
        _assistant(calls=[_tool_call("list_cancers", {}, id_=str(i))]) for i in range(20)
    ]
    fp = FakeProvider(script=looping)
    resp = run_turn([], "loop", provider=fp, max_steps=3, system_prompt="sys")
    assert "step" in resp.text.lower() or "ran out" in resp.text.lower()
    assert len(resp.trace) == 3


def test_pii_refusal_skips_provider():
    fp = FakeProvider(script=[])  # empty script — any provider call raises
    resp = run_turn(
        [], "contact jane@example.com", provider=fp, max_steps=8, system_prompt="sys"
    )
    assert resp.refusal_category == "pii"
    assert "patient" in resp.text.lower()
    assert fp.calls == []


def test_clinical_refusal_skips_provider():
    fp = FakeProvider(script=[])
    resp = run_turn(
        [],
        "Which patients are high risk?",
        provider=fp,
        max_steps=8,
        system_prompt="sys",
    )
    assert resp.refusal_category == "risk_stratification"
    assert fp.calls == []


def test_parallel_tool_calls_in_one_step():
    fp = FakeProvider(
        script=[
            _assistant(
                calls=[
                    _tool_call("get_targets", {"cancer_name": "lung"}, id_="a"),
                    _tool_call("get_targets", {"cancer_name": "breast"}, id_="b"),
                ]
            ),
            _assistant(text="done"),
        ]
    )
    resp = run_turn([], "compare", provider=fp, max_steps=8, system_prompt="sys")
    assert len(resp.trace) == 2
    assert resp.trace[0]["name"] == "get_targets"
    assert resp.trace[1]["name"] == "get_targets"


def test_plot_expressions_yields_figure():
    fp = FakeProvider(
        script=[
            _assistant(
                calls=[
                    _tool_call(
                        "plot_expressions",
                        {"expressions": {"X": 0.5}, "title": "t"},
                    )
                ]
            ),
            _assistant(text="here is the chart"),
        ]
    )
    resp = run_turn([], "plot", provider=fp, max_steps=8, system_prompt="sys")
    assert len(resp.figures) == 1


def test_audit_log_written_on_normal_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_log_path", tmp_path / "a.jsonl")
    audit.reset_session()
    fp = FakeProvider(script=[_assistant(text="ok")])
    run_turn([], "hi", provider=fp, max_steps=8, system_prompt="sys")
    line = (tmp_path / "a.jsonl").read_text().strip()
    row = json.loads(line)
    assert row["refusal_category"] is None
    assert row["final_response"] == "ok"


def test_audit_log_redacts_pii_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_log_path", tmp_path / "a.jsonl")
    audit.reset_session()
    fp = FakeProvider(script=[])
    run_turn(
        [], "contact jane@example.com", provider=fp, max_steps=8, system_prompt="sys"
    )
    row = json.loads((tmp_path / "a.jsonl").read_text().strip())
    assert row["refusal_category"] == "pii"
    assert row["user_message"] == "<redacted>"
    assert "email" in row["redacted_patterns"]
