"""Compact OpenAI-compatible client factory — trimmed from eXercise/ai_client.py.

Provider inferred from the model-name prefix:
  qwen*   → DashScope  (base https://dashscope.aliyuncs.com/compatible-mode/v1, DASHSCOPE_API_KEY)
  gemini* → Gemini OpenAI-compat (GEMINI_API_KEY / GOOGLE_API_KEY)

Keeps only what this tool needs: routing, model-spec parsing, thinking on/off
kwargs, and JSON-fence stripping. No usage tracking / caching / observers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _ProviderDef:
    name: str
    base_url: str
    api_key_env: str
    prefixes: tuple


_PROVIDERS = [
    _ProviderDef(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        ("gemini",),
    ),
    _ProviderDef(
        "qwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        ("qwen",),
    ),
]
_DEFAULT_MODEL = "qwen3.7-plus"


def provider_for_model(model: str) -> str:
    m = model.lower()
    for p in _PROVIDERS:
        if any(m.startswith(pre) for pre in p.prefixes):
            return p.name
    return "qwen"


_LEGACY_EFFORT = {"off": 0, "low": 1024, "high": 8192}


def parse_model_spec(value: str) -> tuple[str, int | None, int | None]:
    """Parse ``"<model>[, <thinking_tokens>][, <max_output_tokens>]"``."""
    parts = [p.strip() for p in value.split(",")]
    model = parts[0]
    nums: list[int] = []
    for p in parts[1:]:
        if not p:
            continue
        low = p.lower()
        if low in _LEGACY_EFFORT:
            nums.append(_LEGACY_EFFORT[low])
        elif p.lstrip("-").isdigit():
            nums.append(int(p))
    thinking = nums[0] if len(nums) >= 1 else None
    max_tokens = nums[1] if len(nums) >= 2 else None
    return model, thinking, max_tokens


def build_thinking_kwargs(provider: str, thinking_tokens: int | None) -> tuple[bool, dict]:
    """Return ``(use_stream, extra_kwargs)``. thinking off → non-streaming."""
    if provider == "gemini":
        if thinking_tokens is None:
            return True, {}
        if thinking_tokens == 0:
            return False, {"reasoning_effort": "none"}
        return True, {"reasoning_effort": "low" if thinking_tokens <= 1024 else "high"}
    if provider == "qwen":
        if not thinking_tokens:
            return False, {"extra_body": {"enable_thinking": False}}
        # Pass the actual thinking budget (token count), not just the on flag.
        return True, {"extra_body": {"enable_thinking": True, "thinking_budget": int(thinking_tokens)}}
    return False, {}


def make_ai_client(model_spec: str):
    """Return ``(client, model, provider, thinking_tokens, max_tokens)`` or None.

    None when the OpenAI SDK is missing or the provider's API key is unset.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    model, thinking, max_tokens = parse_model_spec(model_spec)
    provider = provider_for_model(model)
    pdef = next((p for p in _PROVIDERS if p.name == provider), _PROVIDERS[0])
    api_key = os.environ.get(pdef.api_key_env, "").strip()
    if not api_key and pdef.name == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key, base_url=pdef.base_url)
    except Exception:
        return None
    return client, model, provider, thinking, max_tokens


def strip_json_fences(raw: str) -> str:
    """Strip ```json fences; else return the first balanced { … } block."""
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s)
    if fence:
        return fence.group(1).strip()
    start = s.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(s[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return s
