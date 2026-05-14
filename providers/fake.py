"""FakeProvider for agent-loop tests. Emits a scripted sequence of responses."""
from __future__ import annotations

from typing import Iterable

from providers.base import ProviderResponse


class FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, script: Iterable[ProviderResponse]):
        self._script = list(script)
        self._i = 0
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def chat(self, history: list[dict], tools: list[dict]) -> ProviderResponse:
        if self._i >= len(self._script):
            raise RuntimeError("FakeProvider script exhausted")
        self.calls.append((list(history), list(tools)))
        resp = self._script[self._i]
        self._i += 1
        return resp
