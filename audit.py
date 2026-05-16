"""Append-only JSONL audit log per session.

One file per session, set at app start via init_session(). Records
every turn — normal or refusal — with a single unified schema.
Refusals set refusal_category; PII refusals additionally redact
user_message.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_log_path: Path | None = None
_turn_counter: int = 0


def init_session(log_dir: Path | str = "logs") -> Path:
    """Open a new session log file under `log_dir`. Returns the path."""
    global _log_path, _turn_counter
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    _log_path = log_dir / f"{stamp}.jsonl"
    _turn_counter = 0
    return _log_path


def reset_session() -> None:
    """Reset the turn counter without changing the log path. For tests."""
    global _turn_counter
    _turn_counter = 0


def _write(record: dict) -> None:
    if _log_path is None:
        return  # logging not initialised; silently no-op
    with _log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _next_turn() -> int:
    global _turn_counter
    _turn_counter += 1
    return _turn_counter


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def log_turn(
    *,
    user_message: str,
    provider: str,
    model: str,
    tool_calls: list[dict],
    final_response: str,
    step_count: int,
    step_limit_hit: bool,
    actual_model: str | None = None,
) -> None:
    _write(
        {
            "ts": _now_iso(),
            "turn": _next_turn(),
            "user_message": user_message,
            "provider": provider,
            "model": model,
            "actual_model": actual_model,  # null if call failed before a response.
            "tool_calls": tool_calls,
            "final_response": final_response,
            "step_count": step_count,
            "step_limit_hit": step_limit_hit,
            "refusal_category": None,
            "redacted_patterns": None,
        }
    )


def log_refusal(
    *,
    category: str,
    user_message: str,
    patterns: list[str] | None,
    redacted: bool,
    final_response: str,
) -> None:
    _write(
        {
            "ts": _now_iso(),
            "turn": _next_turn(),
            "user_message": "<redacted>" if redacted else user_message,
            "provider": None,
            "model": None,
            "actual_model": None,
            "tool_calls": [],
            "final_response": final_response,
            "step_count": 0,
            "step_limit_hit": False,
            "refusal_category": category,
            "redacted_patterns": patterns,
        }
    )
