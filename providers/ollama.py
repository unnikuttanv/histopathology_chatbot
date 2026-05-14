"""Ollama provider — talks to a local Ollama server."""
from __future__ import annotations

import json
import uuid
from typing import Any

import ollama

from providers.base import Call, ProviderResponse


def _make_client(host: str):
    """Factory split out so tests can monkeypatch."""
    return ollama.Client(host=host)


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, temperature: float = 0.0):
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = _make_client(host)

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse:
        try:
            raw = self._client.chat(
                model=self.model,
                messages=history,
                tools=tools or None,
                options={"temperature": self.temperature},
            )
        except Exception as e:
            return ProviderResponse(
                text=f"<provider error: {type(e).__name__}: {e}>",
                function_calls=[],
                raw_assistant_content={"role": "assistant", "content": ""},
            )

        # Some ollama client versions return a Pydantic-like model object; coerce to dict-ish.
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()

        msg = (raw or {}).get("message") or {}
        content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls") or []

        if not isinstance(msg, dict) or (not tool_calls_raw and content == "" and not raw):
            return ProviderResponse(
                text="<unparseable response>",
                function_calls=[],
                raw_assistant_content={"role": "assistant", "content": ""},
            )

        calls: list[Call] = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw: Any = fn.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            calls.append(Call(id=str(uuid.uuid4()), name=name, args=args))

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
