"""Tests for the LiteLLM-backed unified provider.

We mock `litellm.completion` so the tests stay offline. The provider only
cares about the OpenAI-shape response, which LiteLLM normalises across all
backends.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from providers.litellm_provider import LiteLLMProvider, _friendly_error


def _text_response(content: str, model: str = "gemini/gemini-2.5-flash"):
    """Build a fake litellm ModelResponse with a plain text message."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], model=model)


def _tool_call_response(name: str, args: dict, call_id: str = "call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    tc = SimpleNamespace(id=call_id, function=fn, type="function")
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def test_chat_returns_text(monkeypatch):
    monkeypatch.setattr(
        "litellm.completion", lambda **kw: _text_response("hello")
    )
    p = LiteLLMProvider(model="gemini/gemini-2.5-flash", name="gemini", api_key="k")
    out = p.chat([{"role": "user", "content": "hi"}], [])
    assert out.text == "hello"
    assert out.function_calls == []
    assert out.raw_assistant_content["content"] == "hello"
    assert out.raw_assistant_content["tool_calls"] is None


def test_chat_returns_tool_calls(monkeypatch):
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kw: _tool_call_response("list_cancers", {}),
    )
    p = LiteLLMProvider(model="openai/gpt-4o-mini", name="openai", api_key="k")
    out = p.chat([{"role": "user", "content": "cancers?"}], [{"type": "function"}])
    assert out.text is None
    assert len(out.function_calls) == 1
    assert out.function_calls[0].name == "list_cancers"
    assert out.function_calls[0].args == {}
    # Canonical assistant content carries tool_calls and no text content.
    assert out.raw_assistant_content["content"] is None
    assert out.raw_assistant_content["tool_calls"][0]["function"]["name"] == "list_cancers"


def test_chat_passes_fallbacks_and_retries(monkeypatch):
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return _text_response("ok")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(
        model="gemini/gemini-2.5-flash",
        name="gemini",
        api_key="k",
        fallbacks=["ollama_chat/llama3.1:8b"],
        num_retries=3,
    )
    p.chat([{"role": "user", "content": "hi"}], [])
    assert captured["fallbacks"] == ["ollama_chat/llama3.1:8b"]
    assert captured["num_retries"] == 3
    assert captured["model"] == "gemini/gemini-2.5-flash"
    assert captured["api_key"] == "k"


def test_chat_passes_tools_with_tool_choice(monkeypatch):
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return _text_response("")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(model="openai/gpt-4o-mini", name="openai", api_key="k")
    tools = [{"type": "function", "function": {"name": "x"}}]
    p.chat([{"role": "user", "content": "hi"}], tools)
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"


def test_chat_omits_tool_choice_when_no_tools(monkeypatch):
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return _text_response("")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(model="openai/gpt-4o-mini", name="openai", api_key="k")
    p.chat([{"role": "user", "content": "hi"}], [])
    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_chat_passes_api_base_for_ollama(monkeypatch):
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return _text_response("ok")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(
        model="ollama_chat/llama3.1:8b",
        name="ollama",
        api_base="http://localhost:11434",
    )
    p.chat([{"role": "user", "content": "hi"}], [])
    assert captured["api_base"] == "http://localhost:11434"
    assert "api_key" not in captured  # Ollama needs no key


def test_chat_returns_friendly_text_on_exception(monkeypatch):
    def fake(**kw):
        raise RuntimeError("503 UNAVAILABLE — model overloaded")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(model="gemini/x", name="gemini", api_key="k")
    out = p.chat([{"role": "user", "content": "hi"}], [])
    assert "overloaded" in out.text.lower()
    assert out.function_calls == []


def test_chat_translates_string_arguments_to_dict(monkeypatch):
    """LiteLLM emits arguments as JSON-encoded strings (OpenAI style)."""
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kw: _tool_call_response("get_targets", {"cancer_name": "lung"}),
    )
    p = LiteLLMProvider(model="openai/x", name="openai", api_key="k")
    out = p.chat([{"role": "user", "content": "lung"}], [{"type": "function"}])
    assert out.function_calls[0].args == {"cancer_name": "lung"}


def test_chat_handles_malformed_json_arguments(monkeypatch):
    fn = SimpleNamespace(name="get_targets", arguments="{not valid json")
    tc = SimpleNamespace(id="c1", function=fn, type="function")
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    monkeypatch.setattr("litellm.completion", lambda **kw: resp)
    p = LiteLLMProvider(model="openai/x", name="openai", api_key="k")
    out = p.chat([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert out.function_calls[0].args == {}


def test_chat_cleans_assistant_history_for_openai_format(monkeypatch):
    """Assistant messages with tool_calls=None should have content-only shape."""
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return _text_response("ok")

    monkeypatch.setattr("litellm.completion", fake)
    p = LiteLLMProvider(model="openai/x", name="openai", api_key="k")
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": None},
    ]
    p.chat(history, [])
    sent = captured["messages"]
    # Cleaned assistant message should not carry tool_calls=None
    assert "tool_calls" not in sent[1]
    assert sent[1]["content"] == "hello"


@pytest.mark.parametrize(
    "msg,expected_substring",
    [
        ("503 UNAVAILABLE", "overloaded"),
        ("429 RESOURCE_EXHAUSTED rate limit", "quota"),
        ("404 NOT_FOUND model x", "Model not found"),
        ("401 UNAUTHENTICATED bad key", "authentication"),
        ("some random crash", "Provider error"),
    ],
)
def test_friendly_error_messages(msg, expected_substring):
    text = _friendly_error(RuntimeError(msg))
    assert expected_substring.lower() in text.lower()


def test_friendly_error_for_connection_failure_by_type_name():
    class APIConnectionError(RuntimeError):
        pass

    text = _friendly_error(APIConnectionError("Cannot connect to host localhost:11434"))
    assert "couldn't reach" in text.lower()
    assert "host.docker.internal" in text


def test_friendly_error_for_all_fallback_attempts_failed():
    text = _friendly_error(RuntimeError("All fallback attempts failed. Check logs."))
    assert "both the cloud llm and the local ollama fallback failed" in text.lower()


def test_friendly_error_for_ollama_runner_oom():
    """Real LiteLLM message when Ollama can't allocate memory for a model."""
    msg = (
        "litellm.APIConnectionError: Ollama_chatException - "
        '{"error":"llama runner process has terminated: '
        "llama_init_from_model: failed to initialize the context: "
        'failed to allocate compute pp buffers"}'
    )
    text = _friendly_error(RuntimeError(msg))
    assert "out of memory" in text.lower()
    assert "llama3.2:1b" in text  # actionable suggestion
    # Must NOT misclassify as a connection issue.
    assert "host.docker.internal" not in text


def test_friendly_error_for_rate_limit_wrapped_in_connection_error():
    """Gemini 429s sometimes surface as APIConnectionError after fallback wrapping."""

    class APIConnectionError(RuntimeError):
        pass

    msg = (
        "litellm.RateLimitError: geminiException - "
        '{"error":{"code":429,"message":"You exceeded your current quota",'
        '"status":"RESOURCE_EXHAUSTED"}}'
    )
    text = _friendly_error(APIConnectionError(msg))
    assert "rate limit" in text.lower() or "quota" in text.lower()


def test_friendly_error_does_not_leak_full_traceback():
    """`str(exc)` from LiteLLM can be a multi-line traceback dump.
    The user-facing message must collapse it to one short line.
    """
    huge = "Provider error\n" + ("Traceback line\n" * 200)
    text = _friendly_error(RuntimeError(huge))
    assert "\n" not in text
    assert len(text) < 500


def test_provider_exposes_name_and_model():
    p = LiteLLMProvider(model="gemini/gemini-2.5-flash", name="gemini", api_key="k")
    assert p.name == "gemini"
    assert p.model == "gemini/gemini-2.5-flash"


def test_chat_captures_actual_model_from_response(monkeypatch):
    """When a fallback runs, litellm's response.model differs from the configured one."""
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kw: _text_response("ok", model="ollama_chat/llama3.2:1b"),
    )
    p = LiteLLMProvider(
        model="gemini/gemini-2.5-flash", name="gemini", api_key="k"
    )
    out = p.chat([{"role": "user", "content": "hi"}], [])
    assert out.actual_model == "ollama_chat/llama3.2:1b"
