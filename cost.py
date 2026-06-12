"""Token-cost estimation + report — modelled on eXercise's cost_report.

Rates are RMB per 1M tokens, ``(input, output)``. ``output`` already includes
the thinking portion (Qwen/Gemini bill thinking as output), so it is multiplied
by the output rate; the Thinking column is informational only.

Default rates are copied from eXercise's "AI API costs.xlsx" (so the numbers
match). To refresh, drop an "AI API costs.xlsx" (Provider | Model | Input Cost |
Output Cost) into the project dir — it overrides the table below.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import config
import usage as usage_mod

log = logging.getLogger("course_reader")

# model → (input RMB / 1M, output RMB / 1M). Source: eXercise "AI API costs.xlsx".
PRICING: dict[str, tuple[float, float]] = {
    "qwen3.7-plus": (2.0, 8.0),
    "qwen3.6-plus": (2.0, 12.0),
    "qwen3.7-max": (12.0, 36.0),
    "qwen3-max": (2.5, 10.0),
    "qwen-plus": (0.8, 2.0),
    "qwen3.6-flash": (1.2, 7.2),
    "qwen3.5-flash": (0.2, 2.0),
    "qwen-flash": (0.15, 1.5),
    "qwen-vl-max": (3.0, 9.0),
    "qwen-vl-plus": (1.5, 4.5),
    "qwen3-vl-plus": (1.0, 10.0),
    "gemini-2.5-flash": (2.38, 10.19),
    "gemini-2.5-flash-lite": (0.68, 2.72),
}


def _load_pricing() -> dict[str, tuple[float, float]]:
    """Embedded table, overlaid by a local 'AI API costs.xlsx' if present."""
    rates = dict(PRICING)
    xlsx = config.PROJECT_DIR / "AI API costs.xlsx"
    if not xlsx.exists():
        return rates
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active
        rows = iter(ws.rows)
        headers = [str(c.value).strip().lower() if c.value else "" for c in next(rows)]
        mc = next((i for i, h in enumerate(headers) if "model" in h), None)
        ic = next((i for i, h in enumerate(headers) if "input" in h), None)
        oc = next((i for i, h in enumerate(headers) if "output" in h), None)
        if None not in (mc, ic, oc):
            for row in rows:
                model = str(row[mc].value or "").strip()
                if not model:
                    continue
                try:
                    rates[model] = (float(row[ic].value or 0), float(row[oc].value or 0))
                except (TypeError, ValueError):
                    pass
        wb.close()
    except Exception:  # noqa: BLE001 — pricing is best-effort
        pass
    return rates


def compute_cost(usage: dict[str, dict[str, int]]) -> tuple[float, dict[str, dict]]:
    """Return (total_rmb, per_model_breakdown)."""
    rates = _load_pricing()
    breakdown: dict[str, dict] = {}
    total = 0.0
    for model, c in usage.items():
        ir, orr = rates.get(model, (0.0, 0.0))
        cost = c.get("input", 0) / 1_000_000 * ir + c.get("output", 0) / 1_000_000 * orr
        total += cost
        breakdown[model] = {
            "input_tokens": c.get("input", 0),
            "output_tokens": c.get("output", 0),
            "thinking_tokens": c.get("thinking", 0),
            "calls": c.get("calls", 0),
            "rate_in": ir,
            "rate_out": orr,
            "cost_rmb": round(cost, 6),
        }
    return round(total, 6), breakdown


def log_summary() -> None:
    u = usage_mod.snapshot()
    if not u:
        return
    total, bd = compute_cost(u)
    log.info("── token cost (RMB) ──────────────────────────")
    for model, d in bd.items():
        note = "" if d["rate_in"] or d["rate_out"] else "  (no rate listed)"
        log.info("  %s: in=%s out=%s think=%s calls=%d  ¥%.4f%s",
                 model, f"{d['input_tokens']:,}", f"{d['output_tokens']:,}",
                 f"{d['thinking_tokens']:,}", d["calls"], d["cost_rmb"], note)
    log.info("  TOTAL: ¥%.4f", total)


def write_report(out_dir: Path) -> Path | None:
    u = usage_mod.snapshot()
    if not u:
        return None
    total, bd = compute_cost(u)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_usage": bd,
        "total_input_tokens": sum(v["input_tokens"] for v in bd.values()),
        "total_output_tokens": sum(v["output_tokens"] for v in bd.values()),
        "total_thinking_tokens": sum(v["thinking_tokens"] for v in bd.values()),
        "total_calls": sum(v["calls"] for v in bd.values()),
        "total_cost_rmb": total,
    }
    (out_dir / "cost.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# AI token cost (RMB)", "",
        "| Model | Input | Output | Thinking | Calls | Cost (¥) |",
        "|-------|-------|--------|----------|-------|----------|",
    ]
    for model, d in bd.items():
        lines.append(
            f"| {model} | {d['input_tokens']:,} | {d['output_tokens']:,} "
            f"| {d['thinking_tokens']:,} | {d['calls']} | {d['cost_rmb']:.4f} |"
        )
    lines.append(
        f"| **Total** | **{payload['total_input_tokens']:,}** "
        f"| **{payload['total_output_tokens']:,}** "
        f"| **{payload['total_thinking_tokens']:,}** "
        f"| **{payload['total_calls']}** | **{total:.4f}** |"
    )
    lines.append("")
    lines.append("_Output tokens include thinking; the Thinking column is informational. "
                 "Rates: RMB/1M tokens (eXercise 'AI API costs.xlsx')._")
    (out_dir / "cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "cost.json"
