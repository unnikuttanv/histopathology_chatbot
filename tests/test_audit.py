import json

import pytest

import audit


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "test.jsonl"
    monkeypatch.setattr(audit, "_log_path", path)
    audit.reset_session()
    return path


def test_log_turn_writes_jsonl_with_all_fields(audit_path):
    audit.log_turn(
        user_message="top 5 in lung",
        provider="ollama",
        model="llama3.1:8b",
        tool_calls=[
            {"name": "top_genes", "args": {"cancer_name": "lung", "n": 5}, "result": {}}
        ],
        final_response="Here are the top 5 ...",
        step_count=2,
        step_limit_hit=False,
    )
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    for key in (
        "ts", "turn", "user_message", "provider", "model", "actual_model",
        "tool_calls", "final_response", "step_count",
        "step_limit_hit", "refusal_category", "redacted_patterns",
    ):
        assert key in row
    assert row["refusal_category"] is None
    assert row["redacted_patterns"] is None
    assert row["turn"] == 1


def test_log_refusal_pii_redacts_user_message(audit_path):
    audit.log_refusal(
        category="pii",
        user_message="contact jane@example.com",
        patterns=["email"],
        redacted=True,
        final_response="refusal text",
    )
    row = json.loads(audit_path.read_text().strip())
    assert row["refusal_category"] == "pii"
    assert row["user_message"] == "<redacted>"
    assert row["redacted_patterns"] == ["email"]


def test_log_refusal_clinical_keeps_user_message(audit_path):
    audit.log_refusal(
        category="diagnosis",
        user_message="does this patient have X",
        patterns=None,
        redacted=False,
        final_response="refusal text",
    )
    row = json.loads(audit_path.read_text().strip())
    assert row["refusal_category"] == "diagnosis"
    assert row["user_message"] == "does this patient have X"
    assert row["redacted_patterns"] is None


def test_turn_counter_increments(audit_path):
    audit.log_turn(
        user_message="a", provider="ollama", model="m",
        tool_calls=[], final_response="x", step_count=0, step_limit_hit=False,
    )
    audit.log_turn(
        user_message="b", provider="ollama", model="m",
        tool_calls=[], final_response="y", step_count=0, step_limit_hit=False,
    )
    rows = [json.loads(line) for line in audit_path.read_text().strip().splitlines()]
    assert [r["turn"] for r in rows] == [1, 2]
