from unittest.mock import MagicMock

import providers.ollama as ollama_provider
from providers.base import Call


def test_ollama_chat_returns_tool_calls_when_present(monkeypatch):
    fake_resp = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "list_cancers",
                        "arguments": {},
                    }
                }
            ],
        }
    }
    fake_client = MagicMock()
    fake_client.chat.return_value = fake_resp
    monkeypatch.setattr(ollama_provider, "_make_client", lambda host: fake_client)

    p = ollama_provider.OllamaProvider(model="llama3.1:8b", host="http://localhost:11434")
    resp = p.chat(history=[{"role": "user", "content": "hi"}], tools=[])

    assert resp.text is None or resp.text == ""
    assert len(resp.function_calls) == 1
    assert isinstance(resp.function_calls[0], Call)
    assert resp.function_calls[0].name == "list_cancers"
    assert resp.function_calls[0].args == {}


def test_ollama_chat_returns_text_when_no_tool_calls(monkeypatch):
    fake_resp = {"message": {"role": "assistant", "content": "Hello there"}}
    fake_client = MagicMock()
    fake_client.chat.return_value = fake_resp
    monkeypatch.setattr(ollama_provider, "_make_client", lambda host: fake_client)

    p = ollama_provider.OllamaProvider(model="m", host="http://x")
    resp = p.chat(history=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.text == "Hello there"
    assert resp.function_calls == []


def test_ollama_chat_handles_string_arguments(monkeypatch):
    fake_resp = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_targets",
                        "arguments": '{"cancer_name": "lung"}',
                    }
                }
            ],
        }
    }
    fake_client = MagicMock()
    fake_client.chat.return_value = fake_resp
    monkeypatch.setattr(ollama_provider, "_make_client", lambda host: fake_client)

    p = ollama_provider.OllamaProvider(model="m", host="http://x")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls[0].args == {"cancer_name": "lung"}


def test_ollama_chat_handles_malformed_arguments(monkeypatch):
    fake_resp = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_targets", "arguments": "not-json"}}
            ],
        }
    }
    fake_client = MagicMock()
    fake_client.chat.return_value = fake_resp
    monkeypatch.setattr(ollama_provider, "_make_client", lambda host: fake_client)

    p = ollama_provider.OllamaProvider(model="m", host="http://x")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls[0].args == {}


def test_ollama_chat_handles_provider_exception(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("network down")
    monkeypatch.setattr(ollama_provider, "_make_client", lambda host: fake_client)

    p = ollama_provider.OllamaProvider(model="m", host="http://x")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls == []
    assert "provider error" in (resp.text or "")
