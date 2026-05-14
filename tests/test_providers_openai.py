from types import SimpleNamespace
from unittest.mock import MagicMock

import providers.openai as openai_provider
from providers.base import Call


def _fake_completion(*, content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _fake_tool_call(name, args_json, id_="call_1"):
    fn = SimpleNamespace(name=name, arguments=args_json)
    return SimpleNamespace(id=id_, type="function", function=fn)


def test_openai_chat_returns_text(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(content="hello")
    monkeypatch.setattr(openai_provider, "_make_client", lambda key: fake_client)

    p = openai_provider.OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    resp = p.chat(history=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.text == "hello"
    assert resp.function_calls == []


def test_openai_chat_returns_tool_calls(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(
        tool_calls=[_fake_tool_call("get_targets", '{"cancer_name": "lung"}')]
    )
    monkeypatch.setattr(openai_provider, "_make_client", lambda key: fake_client)

    p = openai_provider.OpenAIProvider(api_key="sk-test", model="m")
    resp = p.chat(history=[], tools=[])
    assert len(resp.function_calls) == 1
    assert isinstance(resp.function_calls[0], Call)
    assert resp.function_calls[0].name == "get_targets"
    assert resp.function_calls[0].args == {"cancer_name": "lung"}


def test_openai_chat_handles_malformed_arguments(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(
        tool_calls=[_fake_tool_call("get_targets", "not-json")]
    )
    monkeypatch.setattr(openai_provider, "_make_client", lambda key: fake_client)

    p = openai_provider.OpenAIProvider(api_key="sk-test", model="m")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls[0].args == {}


def test_openai_chat_handles_provider_exception(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("boom")
    monkeypatch.setattr(openai_provider, "_make_client", lambda key: fake_client)

    p = openai_provider.OpenAIProvider(api_key="sk-test", model="m")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls == []
    assert "provider error" in (resp.text or "")
