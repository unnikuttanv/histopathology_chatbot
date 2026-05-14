from unittest.mock import MagicMock
from types import SimpleNamespace

import providers.gemini as gemini_provider


def _make_fake_response(*, text=None, function_calls=None):
    """Build a fake genai response object with the shape gemini.py expects."""
    fc = function_calls or []

    parts = []
    for name, args in fc:
        part = SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)
        parts.append(part)
    if text is not None:
        text_part = SimpleNamespace(function_call=None, text=text)
        parts.append(text_part)

    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[candidate])


def _patch_client(monkeypatch, fake_client):
    monkeypatch.setattr(gemini_provider, "_make_client", lambda key: fake_client)


def test_gemini_chat_returns_text(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_fake_response(text="hello")
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="gemini-2.5-flash")
    resp = p.chat(history=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.text == "hello"
    assert resp.function_calls == []


def test_gemini_chat_returns_function_call(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_fake_response(
        function_calls=[("list_cancers", {})]
    )
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="m")
    resp = p.chat(history=[{"role": "user", "content": "hi"}], tools=[])
    assert len(resp.function_calls) == 1
    assert resp.function_calls[0].name == "list_cancers"
    assert resp.function_calls[0].args == {}


def test_gemini_chat_handles_provider_exception(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("boom")
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="m")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls == []
    # Non-transient errors fall through to the friendly-message helper.
    assert resp.text and "boom" in resp.text


def test_gemini_chat_retries_on_503_and_succeeds(monkeypatch):
    # Make sleep instant so the test stays fast.
    monkeypatch.setattr(gemini_provider, "_BACKOFF_BASE", 0)
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        RuntimeError("503 UNAVAILABLE: overloaded"),
        _make_fake_response(text="recovered"),
    ]
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="m")
    resp = p.chat(history=[], tools=[])
    assert resp.text == "recovered"
    assert fake_client.models.generate_content.call_count == 2


def test_gemini_chat_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(gemini_provider, "_BACKOFF_BASE", 0)
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError(
        "503 UNAVAILABLE: still overloaded"
    )
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="m")
    resp = p.chat(history=[], tools=[])
    assert resp.function_calls == []
    assert "overloaded" in resp.text.lower()
    assert fake_client.models.generate_content.call_count == gemini_provider._MAX_ATTEMPTS


def test_gemini_chat_does_not_retry_on_404(monkeypatch):
    monkeypatch.setattr(gemini_provider, "_BACKOFF_BASE", 0)
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError(
        "404 NOT_FOUND: model missing"
    )
    _patch_client(monkeypatch, fake_client)

    p = gemini_provider.GeminiProvider(api_key="x", model="m")
    resp = p.chat(history=[], tools=[])
    assert fake_client.models.generate_content.call_count == 1
    assert "model" in resp.text.lower()
