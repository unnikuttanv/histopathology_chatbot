import pytest

from providers.base import Call, ProviderResponse
from providers.fake import FakeProvider


def test_call_dataclass():
    c = Call(id="x", name="get_targets", args={"cancer_name": "lung"})
    assert c.id == "x"
    assert c.name == "get_targets"
    assert c.args == {"cancer_name": "lung"}


def test_provider_response_with_text():
    r = ProviderResponse(text="hello", function_calls=[], raw_assistant_content={})
    assert r.text == "hello"
    assert r.function_calls == []


def test_fake_provider_emits_scripted_responses_in_order():
    script = [
        ProviderResponse(
            text=None,
            function_calls=[Call(id="1", name="list_cancers", args={})],
            raw_assistant_content={"role": "assistant", "tool_calls": []},
        ),
        ProviderResponse(
            text="done",
            function_calls=[],
            raw_assistant_content={"role": "assistant", "content": "done"},
        ),
    ]
    fp = FakeProvider(script=script)
    r1 = fp.chat([], tools=[])
    r2 = fp.chat([], tools=[])
    assert r1.function_calls[0].name == "list_cancers"
    assert r2.text == "done"


def test_fake_provider_raises_when_script_exhausted():
    fp = FakeProvider(script=[])
    with pytest.raises(RuntimeError):
        fp.chat([], tools=[])
