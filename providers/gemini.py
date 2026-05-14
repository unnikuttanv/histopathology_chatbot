"""Gemini provider via the google-genai SDK.

Handles translation between our canonical OpenAI-style history /
tool spec and Gemini's native shape.
"""
from __future__ import annotations

import json
import time
import uuid

from google import genai
from google.genai import types as genai_types

from providers.base import Call, ProviderResponse

# Transient Gemini errors worth retrying. 503 (overloaded), 429 (rate limit),
# 500 (server error). 404 / 401 / 400 are config errors — never retry.
_RETRYABLE_STATUS = {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE_EXCEEDED"}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.0  # seconds; doubled per attempt


def _is_transient(exc: Exception) -> bool:
    msg = str(exc)
    if any(code in msg for code in ("503", "429", "500", "504")):
        return True
    return any(s in msg for s in _RETRYABLE_STATUS)


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "503" in msg or "UNAVAILABLE" in msg:
        return (
            "The Gemini service is temporarily overloaded. Try again in a moment, "
            "or set `LLM_PROVIDER=ollama` in `.env` for the local fallback."
        )
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return (
            "Gemini quota/rate limit hit. Wait a minute and retry, or switch "
            "providers in `.env`."
        )
    if "404" in msg or "NOT_FOUND" in msg:
        return (
            f"Gemini model not found: {msg}. "
            "Check `GEMINI_MODEL` in `.env` — try `gemini-2.5-flash` or `gemini-2.0-flash`."
        )
    if "401" in msg or "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg:
        return "Gemini authentication failed — check `GEMINI_API_KEY` in `.env`."
    return f"Provider error: {type(exc).__name__}: {exc}"


def _make_client(api_key: str):
    """Factory split out so tests can monkeypatch."""
    return genai.Client(api_key=api_key)


def _to_gemini_tool(tool_spec: list[dict]) -> "genai_types.Tool":
    """Convert OpenAI-style tool spec to one Gemini Tool object."""
    decls = []
    for t in tool_spec:
        fn = t["function"]
        decls.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return genai_types.Tool(function_declarations=decls)


def _history_to_gemini(history: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate canonical OpenAI-style history into Gemini contents.

    Returns (system_instruction, contents_list).
    """
    system_instruction: str | None = None
    out: list[dict] = []
    for m in history:
        role = m.get("role")
        if role == "system":
            # Last system message wins.
            system_instruction = m.get("content") or system_instruction
            continue
        if role == "user":
            out.append({"role": "user", "parts": [{"text": m.get("content") or ""}]})
        elif role == "assistant":
            if m.get("tool_calls"):
                parts = []
                for tc in m["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(
                        {
                            "function_call": {
                                "name": tc["function"]["name"],
                                "args": args,
                            }
                        }
                    )
                out.append({"role": "model", "parts": parts})
            else:
                out.append(
                    {"role": "model", "parts": [{"text": m.get("content") or ""}]}
                )
        elif role == "tool":
            out.append(
                {
                    "role": "user",  # Gemini expects function responses inside a user turn
                    "parts": [
                        {
                            "function_response": {
                                "name": m["name"],
                                "response": {"result": _maybe_json(m.get("content"))},
                            }
                        }
                    ],
                }
            )
    return system_instruction, out


def _maybe_json(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._client = _make_client(api_key)

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse:
        system_from_history, contents = _history_to_gemini(history)
        system_instruction = system_from_history or self.system_prompt or None

        config_kwargs = {"temperature": self.temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [_to_gemini_tool(tools)]

        resp = None
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                )
                break
            except Exception as e:  # noqa: BLE001 — provider SDK raises broadly
                last_exc = e
                if attempt + 1 < _MAX_ATTEMPTS and _is_transient(e):
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                return ProviderResponse(
                    text=_friendly_error(e),
                    function_calls=[],
                    raw_assistant_content={"role": "assistant", "content": ""},
                )
        if resp is None:
            return ProviderResponse(
                text=_friendly_error(last_exc) if last_exc else "Unknown provider error",
                function_calls=[],
                raw_assistant_content={"role": "assistant", "content": ""},
            )

        function_calls: list[Call] = []
        text_chunks: list[str] = []

        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    args_obj = getattr(fc, "args", {}) or {}
                    args = dict(args_obj)
                    function_calls.append(
                        Call(id=str(uuid.uuid4()), name=fc.name, args=args)
                    )
                else:
                    txt = getattr(part, "text", None)
                    if txt:
                        text_chunks.append(txt)

        text = "".join(text_chunks) if text_chunks else None

        canonical_assistant = {
            "role": "assistant",
            "content": text if not function_calls else None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for c in function_calls
            ]
            if function_calls
            else None,
        }

        return ProviderResponse(
            text=text,
            function_calls=function_calls,
            raw_assistant_content=canonical_assistant,
        )
