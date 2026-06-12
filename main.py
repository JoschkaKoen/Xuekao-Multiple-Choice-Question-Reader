"""Xuekao MCQ grader — CLI orchestration (one input folder per run).

  python main.py <folder> [options]

Two stages: (A) CPU geometry — render→orient(portrait)→deskew→anchor→color crops,
cached to disk; (B) ALL Qwen calls (1 key + name/class/answers per page) in one
parallel pool, each result cached. Resume is the DEFAULT (successful steps are
not rerun); --fresh forces recompute, --redo KIND re-runs one kind.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import answer_key
import config
import cost
import extract
import grade
import jsonstore
import orient
import roster as roster_mod
import usage
import writer
from ai_client import make_ai_client
from geometry import page_geometry
from logging_setup import Progress, setup_logging
from pdf_render import jpeg_b64, page_count, save_image, save_jpeg
import anchors

log = logging.getLogger("xuekao")

FIELDS = ("class", "name", "answers")


def parse_pages(spec: str | None, total: int) -> list[int]:
    """1-based page spec ("1", "1-3", "1,4,9") → sorted unique 0-based indices."""
    if not spec:
        return list(range(total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return sorted({p for p in out if 0 <= p < total})


def _clients():
    out = {}
    for kind, spec in (("identity", config.IDENTITY_MODEL),
                       ("answers", config.ANSWERS_MODEL),
                       ("key", config.KEY_MODEL)):
        c = make_ai_client(spec)
        if c is None:
            raise SystemExit(f"No API client for {kind} ({spec}); set DASHSCOPE_API_KEY in .env")
        out[kind] = c  # (client, model, provider, thinking, max_tokens)
    return out


# --- Stage A: geometry ------------------------------------------------------
def run_geometry(cfg, page_idx, boxes, global_rot, fresh, redo, debug_crops) -> dict:
    geom_path = cfg.cache_dir / f"geom_p{page_idx + 1}.json"
    crop_paths = {f: cfg.crops_dir / f"p{page_idx + 1}_{f}.jpg" for f in boxes}
    if (not fresh and "geometry" not in redo and jsonstore.valid(geom_path)
            and all(p.exists() for p in crop_paths.values())):
        log.info("geom p%d: cache hit", page_idx + 1)
        return jsonstore.read_json(geom_path)
    g = page_geometry(cfg.first_page, page_idx, config.DPI, global_rot)
    for field, (_box, long_edge) in boxes.items():
        save_jpeg(g.crops[field], crop_paths[field], long_edge)
    if debug_crops:
        fx0, fy0, fx1, fy1 = g.frame
        fw, fh = fx1 - fx0, fy1 - fy0
        cpx = {f: (int(fx0 + b[0] * fw), int(fy0 + b[1] * fh),
                   int(fx0 + b[2] * fw), int(fy0 + b[3] * fh)) for f, (b, _le) in boxes.items()}
        save_image(anchors.debug_overlay(g.rgb, g.marks, g.frame, cpx),
                   cfg.debug_dir / f"overlay_p{page_idx + 1}.png")
    geomd = {"rotation_cw": g.rotation_cw, "orient_method": g.orient_method,
             "skew_deg": round(g.skew_deg, 3), "frame": list(g.frame),
             "frame_source": g.frame_source, "n_marks": g.n_marks}
    jsonstore.write_json(geom_path, geomd)
    log.info("geom p%d: rot=%d(%s) skew=%.2f marks=%d src=%s", page_idx + 1, g.rotation_cw,
             g.orient_method, g.skew_deg, g.n_marks, g.frame_source)
    return geomd


# --- Stage B: AI ------------------------------------------------------------
def run_pipeline(cfg, pages, args):
    usage.reset()
    names = roster_mod.load_roster(cfg.roster)
    if not names:
        raise SystemExit("Empty roster — cannot match names.")
    clients = _clients()
    boxes = config.crop_boxes_for_path(cfg.first_page)
    redo = set(args.redo or [])

    # Stage A — geometry (parallel, CPU-bound)
    total = page_count(cfg.first_page)
    log.info("%s: %d pages; processing %d", cfg.first_page.name, total, len(pages))
    global_rot = orient.global_rotation(cfg.first_page, total, config.ORIENT_DPI)
    log.info("global orientation vote: rot=%d", global_rot)
    gworkers = max(1, min(config.GEOM_WORKERS, len(pages)))
    with ThreadPoolExecutor(max_workers=gworkers, thread_name_prefix="geom") as ex:
        list(ex.map(lambda p: run_geometry(cfg, p, boxes, global_rot, args.fresh, redo,
                                           args.debug_crops), pages))
    if args.geometry_only:
        log.info("geometry-only: crops + overlays in %s", cfg.out_dir)
        return

    # Key extraction first (1 call) → drives N + weights; validate before grading.
    key = _get_key(cfg, clients["key"], args, redo)
    if not key.get("valid", False):
        log.warning("KEY problems: %s", "; ".join(key.get("problems", [])) or "invalid")
    n_q = key.get("total_questions") or None
    dbl = answer_key.double_questions(key)
    log.info("key: subject=%s N=%s points=%s double=%s", key.get("subject"),
             key.get("total_questions"), key.get("total_points"), dbl or "-")

    # Stage B — all field calls in ONE parallel pool
    tasks = [(kind, p) for p in pages for kind in FIELDS]
    progress = Progress(len(tasks))
    results: dict[tuple, dict] = {}
    workers = max(1, min(config.MAX_WORKERS, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ai") as ex:
        futs = {ex.submit(_field_task, cfg, kind, p, clients, names, n_q, dbl, args, redo, progress): (kind, p)
                for (kind, p) in tasks}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    # Assemble + grade + write
    records = []
    for p in pages:
        nm = results[("name", p)]
        cl = results[("class", p)]
        an = results[("answers", p)]
        geomd = jsonstore.read_json(cfg.cache_dir / f"geom_p{p + 1}.json")
        score = grade.grade_student(an["answers"], key)
        records.append({"page": p + 1, "name": nm, "cls": cl, "ans": an,
                        "geom": geomd, "score": score})
    stats = grade.item_stats([r["score"] for r in records], key)
    writer.write_workbook(records, key, stats, names, cfg.results_xlsx)
    cost.write_report(cfg.out_dir)
    cost.log_summary()
    log.info("DONE → %s", cfg.results_xlsx)


def _get_key(cfg, key_client, args, redo) -> dict:
    path = cfg.cache_dir / "key.json"
    if not args.fresh and "key" not in redo and jsonstore.valid(path):
        log.info("key: cache hit")
        return jsonstore.read_json(path)
    log.info("key: extracting from %s", cfg.answer_key.name)
    c, m, pv, th, mx = key_client
    key = answer_key.extract_key(c, m, pv, th, mx, cfg.answer_key)
    jsonstore.write_json(path, key)
    return key


def _field_task(cfg, kind, page, clients, names, n_q, dbl, args, redo, progress) -> dict:
    path = cfg.cache_dir / f"{kind}_p{page + 1}.json"
    if not args.fresh and kind not in redo and jsonstore.valid(path):
        log.info("%s p%d: cache hit", kind, page + 1)
        return jsonstore.read_json(path)
    b64 = jpeg_b64(cfg.crops_dir / f"p{page + 1}_{kind}.jpg")
    t0 = time.monotonic()
    log.info("%s p%d: start", kind, page + 1)
    if kind == "name":
        c, m, pv, th, mx = clients["identity"]
        res = extract.read_name(c, m, pv, b64, names, th, mx)
    elif kind == "class":
        c, m, pv, th, mx = clients["identity"]
        res = extract.read_class(c, m, pv, b64, th, mx)
    else:  # answers
        c, m, pv, th, mx = clients["answers"]
        res = extract.read_answers(c, m, pv, b64, th, mx, n_questions=n_q, double_qs=dbl)
    jsonstore.write_json(path, res)
    log.info("%s p%d: done conf=%s %.1fs", kind, page + 1, res.get("confidence", "-"),
             time.monotonic() - t0)
    progress.tick(f"{kind} p{page + 1}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade Xuekao MCQ answer sheets (one folder per run).")
    ap.add_argument("folder", help="input folder, e.g. input_files_1 or input_files_2")
    ap.add_argument("--pages", help="1-based page spec: '1', '1-3', '1,4,9' (default: all)")
    ap.add_argument("--limit", type=int, help="process at most N pages")
    ap.add_argument("--debug-crops", action="store_true", help="also save anchor/crop overlays")
    ap.add_argument("--geometry-only", action="store_true", help="Stage A only, no AI")
    ap.add_argument("--fresh", action="store_true", help="ignore caches, recompute everything")
    ap.add_argument("--redo", action="append",
                    choices=["geometry", "key", "name", "class", "answers"],
                    help="recompute one kind (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    folder = args.folder.rstrip("/")
    if folder not in config.VALID_FOLDERS:
        raise SystemExit(f"folder must be one of {config.VALID_FOLDERS}, got {folder!r}")
    cfg = config.for_folder(folder)
    setup_logging(cfg.log_path, args.verbose)
    for p in (cfg.first_page, cfg.answer_key, cfg.roster):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    total = page_count(cfg.first_page)
    pages = parse_pages(args.pages, total)
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit("no pages selected")
    run_pipeline(cfg, pages, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
