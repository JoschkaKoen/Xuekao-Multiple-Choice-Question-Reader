"""Results workbook — four sheets:
  Answers      — per-student answers vs the official key + marks calculation
  Marks        — the marks-per-student table (sorted)
  Per-question — item analysis across all students (% correct)
  Review       — absentees, no-match / low-confidence / fallback / mark anomalies
"""

from __future__ import annotations

import logging

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

import config

log = logging.getLogger("xuekao")

GREEN = PatternFill("solid", fgColor="C6EFCE")
RED = PatternFill("solid", fgColor="FFC7CE")
YELLOW = PatternFill("solid", fgColor="FFEB9C")
GREY = PatternFill("solid", fgColor="E7E6E6")
HEADF = Font(color="FFFFFF", bold=True)
HEAD = PatternFill("solid", fgColor="305496")
BOLD = Font(bold=True)
CENTER = Alignment(horizontal="center", vertical="center")

_TAG_FILL = {"correct": GREEN, "wrong": RED, "over": RED, "under": RED, "blank": YELLOW}


def _name_disp(rec: dict) -> str:
    return rec["name"]["matched_name"] or "?"


def _class_disp(rec: dict) -> str:
    return rec["cls"]["class_norm"] or "?"


def _serial(name: str, roster: list[str]) -> str:
    try:
        return str(roster.index(name) + 1)
    except ValueError:
        return ""


def _header(ws, values, row=1):
    for j, v in enumerate(values, 1):
        c = ws.cell(row=row, column=j, value=v)
        c.fill = HEAD
        c.font = HEADF
        c.alignment = CENTER


def _answers_sheet(ws, records, key, roster):
    ws.title = "Answers"
    qs = [it["q"] for it in key["questions"]]
    total = key["total_points"]
    head = ["序号", "Sheet", "Name", "Class"] + [f"Q{q}" for q in qs] + ["#Correct", f"Score/{total:g}"]
    _header(ws, head)
    # official-answer reference row
    ws.cell(row=2, column=3, value="official answers →").font = BOLD
    for j, it in enumerate(key["questions"], 5):
        c = ws.cell(row=2, column=j, value="".join(it["correct"]))
        c.font = BOLD
        c.fill = GREY
        c.alignment = CENTER
    ws.cell(row=2, column=5 + len(qs) + 1, value=f"pts {total:g}").font = BOLD

    r = 3
    for rec in sorted(records, key=lambda x: x["page"]):
        score = rec["score"]
        name = _name_disp(rec)
        ws.cell(row=r, column=1, value=_serial(name, roster))
        ws.cell(row=r, column=2, value=rec["page"])
        ws.cell(row=r, column=3, value=name)
        ws.cell(row=r, column=4, value=_class_disp(rec))
        for j, q in enumerate(qs, 5):
            marked, tag, _pts = score.per_q.get(q, (frozenset(), "blank", 0.0))
            c = ws.cell(row=r, column=j, value="".join(sorted(marked)) or "·")
            c.alignment = CENTER
            c.fill = _TAG_FILL.get(tag, YELLOW)
        ws.cell(row=r, column=5 + len(qs), value=score.n_correct).alignment = CENTER
        ws.cell(row=r, column=5 + len(qs) + 1, value=score.total).alignment = CENTER
        r += 1

    ws.freeze_panes = "E3"
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 8


def _marks_sheet(ws, records, roster):
    _header(ws, ["Rank", "序号", "Name", "Class", "Score", "#Correct", "Flags"])
    rows = []
    for rec in records:
        rows.append((rec["score"].total, rec["score"].n_correct, _name_disp(rec),
                     _class_disp(rec), rec))
    rows.sort(key=lambda t: (-t[0], -t[1], t[2]))
    for i, (total, ncorr, name, cls, rec) in enumerate(rows, 1):
        flags = "; ".join(_review_flags(rec))
        ws.append([i, _serial(name, roster), name, cls, total, ncorr, flags])
    ws.freeze_panes = "A2"
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["G"].width = 40


def _perq_sheet(ws, stats, key):
    _header(ws, ["Q", "Correct", "Points", "#Answered", "#Correct", "% Correct", "#Blank", "#Wrong"])
    for s in stats:
        ws.append([s["q"], s["correct"], s["points"], s["answered"],
                   s["n_correct"], s["pct"], s["blank"], s["wrong"]])
        c = ws.cell(row=ws.max_row, column=6)
        c.fill = GREEN if s["pct"] >= 60 else (YELLOW if s["pct"] >= 35 else RED)
        c.alignment = CENTER
    ws.append([])
    ws.append(["TOTAL", "", key["total_points"], "", "", "", "", ""])
    ws.cell(row=ws.max_row, column=1).font = BOLD
    ws.freeze_panes = "A2"


def _review_flags(rec: dict) -> list[str]:
    flags = []
    nm, cl, an = rec["name"], rec["cls"], rec["ans"]
    if nm["matched_name"] in ("NONAME", "UNREADABLE", "NOMATCH"):
        flags.append(f"name={nm['matched_name']}(raw={nm.get('raw_name', '')!r})")
    elif nm["confidence"] <= config.LOW_CONFIDENCE:
        flags.append(f"name low-conf={nm['confidence']}")
    if nm.get("problem"):
        flags.append(f"name note: {nm['problem']}")
    if cl["class_norm"] in ("NONE", "UNREADABLE", "NOMATCH"):
        flags.append(f"class={cl['class_norm']}(raw={cl.get('class_raw', '')!r})")
    elif cl["confidence"] <= config.LOW_CONFIDENCE:
        flags.append(f"class low-conf={cl['confidence']}")
    if cl.get("problem"):
        flags.append(f"class note: {cl['problem']}")
    if an["confidence"] <= config.LOW_CONFIDENCE:
        flags.append(f"answers low-conf={an['confidence']}")
    if an.get("problem"):
        flags.append(f"answers note: {an['problem']}")
    if rec.get("geom") and rec["geom"].get("frame_source") not in (None, "marks"):
        flags.append(f"anchor fallback={rec['geom']['frame_source']}")
    score = rec.get("score")
    if score:
        flags.extend(score.problems)
        anomalies = [t for (_m, t, _p) in score.per_q.values() if t in ("over", "under")]
        if anomalies:
            flags.append(f"{len(anomalies)} over/under marks")
    return flags


def _review_sheet(ws, records, key, roster, low_conf):
    _header(ws, ["Type", "序号", "Sheet", "Name", "Class", "N.conf", "C.conf", "A.conf", "Detail"])
    # key problems first
    if not key.get("valid", False):
        ws.append(["KEY", "", "", "", "", "", "", key.get("confidence", ""),
                   "; ".join(key.get("problems", [])) or "invalid key"])
    elif key.get("problem"):
        ws.append(["KEY", "", "", "", "", "", "", key.get("confidence", ""),
                   f"key note: {key['problem']}"])
    # absentees: roster names with no matched sheet
    matched = {_name_disp(r) for r in records
               if r["name"]["matched_name"] not in ("NONAME", "UNREADABLE", "NOMATCH")}
    for nm in roster:
        if nm not in matched:
            ws.append(["ABSENT", _serial(nm, roster), "", nm, "", "", "", "",
                       "no sheet matched this roster name"])
    # duplicate matches (same roster name on >1 sheet)
    from collections import Counter
    cnt = Counter(_name_disp(r) for r in records
                  if r["name"]["matched_name"] not in ("NONAME", "UNREADABLE", "NOMATCH"))
    dups = {n for n, c in cnt.items() if c > 1}
    # per-record flags
    for rec in sorted(records, key=lambda x: x["page"]):
        flags = _review_flags(rec)
        if _name_disp(rec) in dups:
            flags.insert(0, "DUPLICATE name match")
        if flags:
            ws.append(["FLAG", _serial(_name_disp(rec), roster), rec["page"],
                       _name_disp(rec), _class_disp(rec), rec["name"]["confidence"],
                       rec["cls"]["confidence"], rec["ans"]["confidence"], "; ".join(flags)])
    ws.freeze_panes = "A2"
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["I"].width = 70


def write_workbook(records, key, stats, roster, out_path, low_conf=None):
    if low_conf is None:
        low_conf = config.LOW_CONFIDENCE
    wb = openpyxl.Workbook()
    _answers_sheet(wb.active, records, key, roster)
    _marks_sheet(wb.create_sheet("Marks"), records, roster)
    _perq_sheet(wb.create_sheet("Per-question"), stats, key)
    _review_sheet(wb.create_sheet("Review"), records, key, roster, low_conf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))
    log.info("wrote %s (%d students)", out_path.name, len(records))
