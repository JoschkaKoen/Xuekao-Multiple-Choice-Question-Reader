"""Per-field extraction with Qwen-VL — one call per field (name, class, answers).

Each returns a plain dict (cached verbatim). Name is guarded against the roster
(exact set membership); class is normalized to the EMP-preferred label.
"""

from __future__ import annotations

import json
import logging

import config
import prompts
from ai_client import strip_json_fences
from ai_helpers import ai_image_call

log = logging.getLogger("xuekao")

NAME_SENTINELS = {"NONAME", "UNREADABLE", "NOMATCH"}
CLASS_SENTINELS = {"NONE", "UNREADABLE", "NOMATCH"}


def parse_json(raw: str):
    if not raw:
        return None
    try:
        return json.loads(strip_json_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return None


def norm_options(v) -> list[str]:
    """Coerce A/B/C/D options from str|list|None → ordered de-duped list."""
    items: list[str] = []
    if isinstance(v, str):
        items = [c for c in v.upper() if c in "ABCD"]
    elif isinstance(v, (list, tuple)):
        for x in v:
            items += [c for c in str(x).upper() if c in "ABCD"]
    out: list[str] = []
    for c in items:
        if c not in out:
            out.append(c)
    return out


def _conf(d: dict, default: int = 0) -> int:
    try:
        return max(0, min(5, int(d.get("confidence", default))))
    except (TypeError, ValueError):
        return default


def _problem(d: dict) -> str:
    return str(d.get("problem", "")).strip()


def read_name(client, model, provider, b64, roster_names, thinking=0, max_tokens=None) -> dict:
    raw = ai_image_call(client, b64, prompts.build_name_prompt(roster_names),
                        model_id=model, provider=provider,
                        max_tokens=max_tokens or 512, thinking_tokens=thinking)
    d = parse_json(raw)
    if d is None:
        return {"matched_name": "UNREADABLE", "raw_name": "", "confidence": 0,
                "problem": "no JSON returned"}
    name = str(d.get("matched_name", "")).strip()
    if name not in NAME_SENTINELS and name not in set(roster_names):
        name = "NOMATCH"  # exact set-membership guard (not fuzzy)
    return {"matched_name": name, "raw_name": str(d.get("raw_name", "")).strip(),
            "confidence": _conf(d), "problem": _problem(d)}


def read_class(client, model, provider, b64, thinking=0, max_tokens=None) -> dict:
    raw = ai_image_call(client, b64, prompts.build_class_prompt(config.CLASS_ALLOWED),
                        model_id=model, provider=provider,
                        max_tokens=max_tokens or 256, thinking_tokens=thinking)
    d = parse_json(raw)
    if d is None:
        return {"class_norm": "UNREADABLE", "class_raw": "", "confidence": 0,
                "problem": "no JSON returned"}
    cls = str(d.get("class", "")).strip().upper()
    if cls in CLASS_SENTINELS:
        norm = cls
    elif cls in config.CLASS_ALLOWED:
        norm = config.CLASS_ALIASES.get(cls, cls)
    else:
        norm = "NOMATCH"
    return {"class_norm": norm, "class_raw": str(d.get("class_raw", "")).strip(),
            "confidence": _conf(d), "problem": _problem(d)}


def read_answers(client, model, provider, b64, thinking=0, max_tokens=None,
                 n_questions=None, double_qs=None) -> dict:
    raw = ai_image_call(client, b64, prompts.build_answers_prompt(n_questions, double_qs),
                        model_id=model, provider=provider,
                        max_tokens=max_tokens or 2000, thinking_tokens=thinking)
    d = parse_json(raw)
    if d is None:
        return {"answers": {}, "confidence": 0, "problem": "no JSON returned"}
    answers: dict[str, list[str]] = {}
    raw_ans = d.get("answers", {})
    if isinstance(raw_ans, dict):
        for k, v in raw_ans.items():
            key = str(k).strip()
            if key.isdigit():
                answers[key] = norm_options(v)
    return {"answers": answers, "confidence": _conf(d),
            "problem": str(d.get("problem", "")).strip()}
