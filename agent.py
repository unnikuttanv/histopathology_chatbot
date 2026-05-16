"""Provider-agnostic agent loop with safety pre-filters and audit logging."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import audit
import safety
import tools as toolkit
from providers.base import LLMProvider, ProviderResponse


@dataclass
class AgentResponse:
    text: str
    figures: list[Any] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    refusal_category: str | None = None
    actual_model: str | None = None  # Model that produced the final answer.


_STEP_LIMIT_MSG = (
    "The agent reached its step limit before producing a final answer. "
    "Try a more specific question, or break it into smaller steps."
)


def run_turn(
    history: list[dict],
    user_message: str,
    *,
    provider: LLMProvider,
    max_steps: int,
    system_prompt: str,
) -> AgentResponse:
    # --- Safety pre-filters run BEFORE any LLM call. ---
    pii = safety.detect_pii(user_message)
    if pii.matched:
        audit.log_refusal(
            category="pii",
            user_message=user_message,
            patterns=pii.patterns,
            redacted=True,
            final_response=safety.PII_REFUSAL,
        )
        return AgentResponse(text=safety.PII_REFUSAL, refusal_category="pii")

    clinical = safety.detect_clinical_question(user_message)
    if clinical.matched:
        refusal_text = safety.refusal_for(clinical.category)
        audit.log_refusal(
            category=clinical.category,
            user_message=user_message,
            patterns=None,
            redacted=False,
            final_response=refusal_text,
        )
        return AgentResponse(text=refusal_text, refusal_category=clinical.category)

    # --- Main agent loop. ---
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": user_message})

    figures: list[Any] = []
    trace: list[dict] = []
    final_text: str | None = None
    actual_model: str | None = None
    step = 0
    step_limit_hit = False

    for step in range(1, max_steps + 1):
        resp: ProviderResponse = provider.chat(history, toolkit.TOOL_SPEC)
        # Track the actual model from the most recent response — this is the
        # one that produced the final answer (or the last tool call).
        actual_model = resp.actual_model or actual_model

        if not resp.function_calls:
            history.append(resp.raw_assistant_content)
            final_text = resp.text or ""
            break

        history.append(resp.raw_assistant_content)
        for call in resp.function_calls:
            result, fig = toolkit.dispatch(call.name, call.args)
            if fig is not None:
                figures.append(fig)
            trace.append(
                {"name": call.name, "args": call.args, "result": _safe_json(result)}
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(_safe_json(result)),
                }
            )
    else:
        final_text = _STEP_LIMIT_MSG
        step_limit_hit = True

    audit.log_turn(
        user_message=user_message,
        provider=getattr(provider, "name", "?"),
        model=getattr(provider, "model", "?"),
        actual_model=actual_model,
        tool_calls=trace,
        final_response=final_text or "",
        step_count=step,
        step_limit_hit=step_limit_hit,
    )

    return AgentResponse(
        text=final_text or "",
        figures=figures,
        trace=trace,
        actual_model=actual_model,
    )


def _safe_json(obj):
    """Best-effort coerce a tool result to a JSON-serialisable structure."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"_repr": repr(obj)}
