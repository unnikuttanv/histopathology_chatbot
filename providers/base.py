"""Provider abstraction.

Each LLM provider implements `chat(history, tools) -> ProviderResponse`.
The agent loop is provider-agnostic and works in terms of these types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class ProviderResponse:
    text: str | None
    function_calls: list[Call]
    raw_assistant_content: Any  # dict-shaped, in canonical OpenAI form
    actual_model: str | None = None  # The model that actually served the response
    # (may differ from the configured one when a fallback ran).


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse: ...
