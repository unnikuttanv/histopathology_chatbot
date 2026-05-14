"""OpenAI provider — uses the openai Python client.

Our canonical history format is already OpenAI-shaped, so this provider
passes `history` and `tools` through with minimal translation.
"""
from __future__ import annotations

import json
import uuid

from openai import OpenAI

from providers.base import Call, ProviderResponse


def _make_client(api_key: str) -> OpenAI:
    """Factory split out so tests can monkeypatch."""
    return OpenAI(api_key=api_key)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = _make_client(api_key)

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse:
        # OpenAI expects assistant tool_calls without a `tool_calls: null` key when empty.
        cleaned_history = _clean_history_for_openai(history)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=cleaned_history,
                tools=tools or None,
                temperature=self.temperature,
            )
        except Exception as e:
            return ProviderResponse(
                text=f"<provider error: {type(e).__name__}: {e}>",
                function_calls=[],
                raw_assistant_content={"role": "assistant", "content": ""},
            )

        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or None
        tool_calls_raw = getattr(msg, "tool_calls", None) or []

        calls: list[Call] = []
        for tc in tool_calls_raw:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", "")
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
                Call(id=getattr(tc, "id", str(uuid.uuid4())), name=name, args=args)
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
        )


def _clean_history_for_openai(history: list[dict]) -> list[dict]:
    """Strip None-valued tool_calls and content from assistant messages.

    OpenAI's API rejects an assistant message with `tool_calls: null` alongside
    a string content; we either send content+no-tool_calls, or tool_calls+no-content.
    """
    out = []
    for m in history:
        if m.get("role") == "assistant":
            cleaned = {"role": "assistant"}
            if m.get("tool_calls"):
                cleaned["tool_calls"] = m["tool_calls"]
                # OpenAI requires content to be present even if empty string
                cleaned["content"] = m.get("content") or ""
            else:
                cleaned["content"] = m.get("content") or ""
            out.append(cleaned)
        else:
            out.append(m)
    return out
