"""Unified LLM provider built on LiteLLM.

Replaces per-SDK providers with a single class that handles Gemini, OpenAI,
and Ollama (and anything else LiteLLM supports) through one OpenAI-compatible
API. Retries and runtime fallback are delegated to LiteLLM.
"""
from __future__ import annotations

import json
import uuid

import litellm

from providers.base import Call, ProviderResponse


def _friendly_error(exc: Exception) -> str:
    """Map a LiteLLM exception to a single-sentence, user-facing message.

    LiteLLM's `str(exc)` can include the full nested traceback — never echo
    that into the chat. Match on the exception type and the first line only.

    Order matters: more specific patterns (Ollama runner crashed, rate limit,
    auth) must run before more generic ones (connection error, "all fallbacks
    failed"), because the inner cause is often more actionable than the outer
    wrapper LiteLLM puts on top of it.
    """
    type_name = type(exc).__name__
    # FULL text — searched for diagnostic signals. Litellm wraps inner errors
    # so the first line is just the wrapper; the real diagnostics are deeper.
    full = str(exc) if str(exc) else ""
    # First line, truncated — used only for the catch-all echo, never to
    # decide a category. Defends against multi-screen tracebacks leaking out.
    first_line = full.splitlines()[0][:500]

    # Ollama runner OOM / model load failure — Ollama panics when it can't
    # allocate buffers for the model. Most actionable signal we can give.
    if "llama runner process has terminated" in full or "failed to allocate" in full:
        return (
            "The local Ollama model couldn't load — usually out of memory. "
            "Try a smaller model: `ollama pull llama3.2:1b` then set "
            "`OLLAMA_MODEL=llama3.2:1b` in `.env`."
        )

    # Rate limit / quota — Gemini's 429 surfaces as RESOURCE_EXHAUSTED in the
    # JSON body, often wrapped in APIConnectionError after fallback chains.
    if (
        "RESOURCE_EXHAUSTED" in full
        or "RateLimit" in type_name
        or "quota" in full.lower()
        or " 429" in full  # leading space avoids matching e.g. "1429"
    ):
        return (
            "LLM quota/rate limit hit. Wait a minute and retry, or switch "
            "providers in `.env`."
        )

    if (
        "PERMISSION_DENIED" in full
        or "UNAUTHENTICATED" in full
        or "Authentication" in type_name
        or " 401" in full
    ):
        return "LLM authentication failed — check the API key in `.env`."

    if "UNAVAILABLE" in full or "ServiceUnavailable" in type_name or " 503" in full:
        return (
            "The primary LLM service is temporarily overloaded. "
            "Try again in a moment."
        )

    # Both primary and fallback exhausted — check BEFORE the generic 404 branch
    # because litellm's fallback chain often surfaces a misleading "404 page
    # not found" wrapper that has nothing to do with the actual model name.
    if "All fallback attempts failed" in full:
        return (
            "Both the cloud LLM and the local Ollama fallback failed. "
            "Most often this means the cloud provider hit a rate limit "
            "and the local model can't handle the request (e.g. small "
            "models often fail at tool calling). Try `LLM_PROVIDER=ollama` "
            "with a tool-capable model like `llama3.1:8b` in `.env`."
        )

    if "NOT_FOUND" in full or "NotFound" in type_name or " 404" in full:
        return (
            "Model not found. Check the model name in `.env` — for Gemini try "
            "`gemini-2.5-flash` or `gemini-2.0-flash`."
        )

    # Pure connection failures (Ollama not running, network down, wrong host).
    if "ConnectionError" in type_name or "Cannot connect" in full or "Connection refused" in full:
        return (
            "Couldn't reach the LLM service. If you're using the local Ollama "
            "fallback, make sure Ollama is running (`ollama serve`) and "
            "`OLLAMA_HOST` in `.env` points to it. In Docker, use "
            "`http://host.docker.internal:11434`."
        )

    return f"Provider error: {type_name}: {first_line}"


class LiteLLMProvider:
    """Single provider that wraps `litellm.completion`.

    `model` is a LiteLLM-style model string (`gemini/gemini-2.5-flash`,
    `openai/gpt-4o-mini`, `ollama_chat/llama3.1:8b`). `fallbacks` is a list of
    fallback model strings tried on transient errors.
    """

    def __init__(
        self,
        *,
        model: str,
        name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        fallbacks: list[str] | None = None,
        temperature: float = 0.0,
        num_retries: int = 3,
    ):
        self.model = model
        self.name = name
        self.api_key = api_key
        self.api_base = api_base
        self.fallbacks = fallbacks or []
        self.temperature = temperature
        self.num_retries = num_retries

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse:
        kwargs = {
            "model": self.model,
            "messages": _clean_history(history),
            "temperature": self.temperature,
            "num_retries": self.num_retries,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.fallbacks:
            kwargs["fallbacks"] = self.fallbacks

        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001 — LiteLLM raises broadly
            return ProviderResponse(
                text=_friendly_error(e),
                function_calls=[],
                raw_assistant_content={"role": "assistant", "content": ""},
            )

        return _translate(resp)


def _translate(resp) -> ProviderResponse:
    """Translate a LiteLLM ModelResponse (OpenAI-shape) to ProviderResponse."""
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or None
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    actual_model = getattr(resp, "model", None)

    calls: list[Call] = []
    for tc in tool_calls_raw:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", "") or ""
        args_raw = getattr(fn, "arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        calls.append(
            Call(id=getattr(tc, "id", None) or str(uuid.uuid4()), name=name, args=args)
        )

    canonical_assistant = {
        "role": "assistant",
        "content": content if not calls else None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.args)},
            }
            for c in calls
        ]
        if calls
        else None,
    }

    return ProviderResponse(
        text=content if not calls else None,
        function_calls=calls,
        raw_assistant_content=canonical_assistant,
        actual_model=actual_model,
    )


def _clean_history(history: list[dict]) -> list[dict]:
    """Strip None-valued tool_calls/content from assistant messages.

    LiteLLM forwards to OpenAI, which rejects `tool_calls: null` next to
    a string content. Either send content+no-tool_calls, or tool_calls+content="".
    """
    out = []
    for m in history:
        if m.get("role") == "assistant":
            cleaned = {"role": "assistant"}
            if m.get("tool_calls"):
                cleaned["tool_calls"] = m["tool_calls"]
                cleaned["content"] = m.get("content") or ""
            else:
                cleaned["content"] = m.get("content") or ""
            out.append(cleaned)
        else:
            out.append(m)
    return out
