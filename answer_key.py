"""Official answer-key extraction (one Qwen call) + validation.

The key drives the per-folder question count and per-question weights, so a bad
read mis-grades everyone — we validate it (contiguous ids, valid options,
positive points) before grading.
"""

from __future__ import annotations

import logging

import config
import orient
import prompts
from ai_helpers import ai_image_call
from extract import norm_options, parse_json
from pdf_render import page_count, render_color, to_jpeg_b64

log = logging.getLogger("xuekao")


def render_key_b64s(pdf_path) -> list[str]:
    """Render every key page, orient upright, downscale → JPEG base64."""
    out: list[str] = []
    for i in range(page_count(pdf_path)):
        rgb = render_color(pdf_path, i, config.DPI)
        rgb, _rot, _m = orient.correct_orientation(rgb, config.DPI)
        out.append(to_jpeg_b64(rgb, config.KEY_LONG_EDGE))
    return out


def extract_key(client, model, provider, thinking, max_tokens, pdf_path) -> dict:
    b64s = render_key_b64s(pdf_path)
    raw = ai_image_call(client, b64s, prompts.build_key_prompt(),
                        model_id=model, provider=provider,
                        max_tokens=max_tokens or 2048, thinking_tokens=thinking)
    return normalize_key(parse_json(raw))


def normalize_key(d) -> dict:
    if not d or not isinstance(d.get("questions"), list):
        return {"subject": "", "questions": [], "total_questions": 0, "total_points": 0,
                "valid": False, "problems": ["no questions parsed from key"]}
    qs = []
    for it in d["questions"]:
        if not isinstance(it, dict):
            continue
        try:
            q = int(it.get("q"))
        except (TypeError, ValueError):
            continue
        try:
            pts = float(it.get("points", 0) or 0)
        except (TypeError, ValueError):
            pts = 0.0
        qs.append({"q": q, "correct": norm_options(it.get("correct")), "points": pts})
    seen: set[int] = set()
    uniq = []
    for x in sorted(qs, key=lambda z: z["q"]):
        if x["q"] in seen:
            continue
        seen.add(x["q"])
        uniq.append(x)
    key = {"subject": str(d.get("subject", "")).strip(), "questions": uniq,
           "total_questions": len(uniq),
           "total_points": round(sum(x["points"] for x in uniq), 2)}
    key["problems"] = validate_key(uniq)
    key["valid"] = bool(uniq) and not key["problems"]
    return key


def validate_key(qs) -> list[str]:
    if not qs:
        return ["empty key"]
    problems: list[str] = []
    nums = [x["q"] for x in qs]
    if nums != list(range(1, len(qs) + 1)):
        problems.append(f"question ids not contiguous 1..N (got {nums})")
    for x in qs:
        if not x["correct"]:
            problems.append(f"Q{x['q']}: no correct option")
        elif any(c not in "ABCD" for c in x["correct"]):
            problems.append(f"Q{x['q']}: invalid option(s) {x['correct']}")
        if x["points"] <= 0:
            problems.append(f"Q{x['q']}: non-positive points ({x['points']})")
    return problems


def double_questions(key) -> list[int]:
    return [x["q"] for x in key.get("questions", []) if len(x["correct"]) >= 2]
