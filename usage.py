"""Thread-safe token-usage accumulator (per model) — modelled on eXercise.

Every successful AI call records its prompt/completion/reasoning token counts
here. ``output`` (completion_tokens) already includes the thinking portion;
``thinking`` is tracked separately and is informational only.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_usage: dict[str, dict[str, int]] = {}  # model → {"input","output","thinking","calls"}


def record(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> None:
    with _lock:
        e = _usage.setdefault(model, {"input": 0, "output": 0, "thinking": 0, "calls": 0})
        e["input"] += int(input_tokens or 0)
        e["output"] += int(output_tokens or 0)
        e["thinking"] += int(thinking_tokens or 0)
        e["calls"] += 1


def record_response_usage(model: str, usage_obj) -> None:
    """Extract token counts from an OpenAI-compat ``usage`` object and record them."""
    if not usage_obj:
        return
    inp = getattr(usage_obj, "prompt_tokens", 0) or 0
    out = getattr(usage_obj, "completion_tokens", 0) or 0
    details = getattr(usage_obj, "completion_tokens_details", None)
    think = (getattr(details, "reasoning_tokens", 0) if details else 0) or \
            getattr(usage_obj, "reasoning_tokens", 0) or 0
    record(model, inp, out, think)


def snapshot() -> dict[str, dict[str, int]]:
    with _lock:
        return {m: dict(v) for m, v in _usage.items()}


def reset() -> None:
    with _lock:
        _usage.clear()
