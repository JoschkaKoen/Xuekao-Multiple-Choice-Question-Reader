"""Configuration.

Loads ``default.env`` (committed defaults) then ``.env`` (secrets + overrides,
wins) — mirroring Course Selection Reader / eXercise env layering. Paths are
folder-parametric: the program processes ONE input folder per run, resolved via
:func:`for_folder`. Everything else (models, DPI, class set, anchor params) is
shared.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv may be absent before the venv is installed
    def load_dotenv(*_a, **_k):  # type: ignore
        return False

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / "default.env", override=False)
load_dotenv(PROJECT_DIR / ".env", override=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- Models (spec: "<model>, <thinking_tokens>[, <max_output_tokens>]") -------
IDENTITY_MODEL = os.getenv("IDENTITY_MODEL", "qwen3.7-plus, 2048, 512")
ANSWERS_MODEL = os.getenv("ANSWERS_MODEL", "qwen3.7-plus, 2048, 3000")
KEY_MODEL = os.getenv("KEY_MODEL", "qwen3.7-plus, 1024, 2048")

# --- Rendering --------------------------------------------------------------
DPI = _int("DPI", 300)
ORIENT_DPI = _int("ORIENT_DPI", 150)
IDENTITY_LONG_EDGE = _int("IDENTITY_LONG_EDGE", 2200)
MCQ_LONG_EDGE = _int("MCQ_LONG_EDGE", 3000)
KEY_LONG_EDGE = _int("KEY_LONG_EDGE", 3000)
JPEG_QUALITY = _int("JPEG_QUALITY", 92)

# --- Behaviour --------------------------------------------------------------
MAX_WORKERS = _int("MAX_WORKERS", 64)
GEOM_WORKERS = _int("GEOM_WORKERS", 6)
LOW_CONFIDENCE = _int("LOW_CONFIDENCE", 2)

# --- Class normalization (set: EMP1–EMP3 / A2801–A2804, prefer EMP) ----------
# The AI matches the class field to one CLASS_ALLOWED label; we also store a
# normalized EMP-preferred value via CLASS_ALIASES. The A2804↔EMP question
# (counts differ 3 vs 4) is confirmed on the test sheets; default keeps A2804.
CLASS_ALLOWED = ["EMP1", "EMP2", "EMP3", "A2801", "A2802", "A2803", "A2804"]
CLASS_ALIASES = {"A2801": "EMP1", "A2802": "EMP2", "A2803": "EMP3", "A2804": "A2804"}

# --- Black-mark anchor detection (tuned in calibration Step 1) ---------------
# A pixel is "black" if all of R,G,B < MARK_BLACK_MAX (timing squares + ink;
# the red printing and most pen ink do NOT pass this).
MARK_BLACK_MAX = _int("MARK_BLACK_MAX", 95)
# Timing-square side at 300 dpi is ~25–55 px; scale by dpi/300. Used to filter
# connected components to the fiducial marks.
MARK_MIN_SIDE_300 = _float("MARK_MIN_SIDE_300", 14.0)
MARK_MAX_SIDE_300 = _float("MARK_MAX_SIDE_300", 95.0)
MARK_MARGIN_FRAC = _float("MARK_MARGIN_FRAC", 0.07)  # marks sit within this fraction of an edge

# Crop boxes as fractions of the detected content frame [x0,y0,x1,y1].
# One crop + one AI call PER FIELD: name, class, answers (the grid together).
# Calibrated visually in Step 1 (see _calibrate.py).
# Portrait (track-on-top) frame fractions, calibrated from Skim point boxes on
# upright pages. Source measurements were in portrait PDF points with a
# bottom-left origin; these fractions are relative to the detected mark frame.
CLASS_CROP = (0.374, 0.101, 0.656, 0.150)      # 班级 field (EMP1 / A28xx)
NAME_CROP = (0.652, 0.030, 0.958, 0.111)       # 姓名 field (handwritten name)
ANSWERS_CROP = (0.020, 0.240, 0.998, 0.500)    # default 选择题 I+II grid

ANSWER_CROPS_BY_FOLDER = {
    "input_files_1": (0.020, 0.240, 0.998, 0.500),
    "input_files_2": (0.020, 0.240, 0.998, 0.460),
}

# Per-field crop registry: name → (frame-fraction box, downscale long-edge).
CROP_BOXES = {
    "class": (CLASS_CROP, IDENTITY_LONG_EDGE),
    "name": (NAME_CROP, IDENTITY_LONG_EDGE),
    "answers": (ANSWERS_CROP, MCQ_LONG_EDGE),
}


def crop_boxes_for_folder(folder: str) -> dict:
    boxes = dict(CROP_BOXES)
    if folder in ANSWER_CROPS_BY_FOLDER:
        boxes["answers"] = (ANSWER_CROPS_BY_FOLDER[folder], MCQ_LONG_EDGE)
    return boxes


def crop_boxes_for_path(path: Path | str) -> dict:
    p = Path(path)
    for folder in ANSWER_CROPS_BY_FOLDER:
        if folder in p.parts:
            return crop_boxes_for_folder(folder)
    return CROP_BOXES

VALID_FOLDERS = ["input_files_1", "input_files_2"]


@dataclass
class FolderCfg:
    name: str
    first_page: Path
    answer_key: Path
    roster: Path
    out_dir: Path
    cache_dir: Path
    crops_dir: Path
    debug_dir: Path
    log_path: Path
    results_xlsx: Path


def for_folder(name: str) -> FolderCfg:
    """Resolve all paths for input folder *name* (e.g. 'input_files_1')."""
    base = PROJECT_DIR / name
    out = PROJECT_DIR / "out" / name
    return FolderCfg(
        name=name,
        first_page=base / "first_page.pdf",
        answer_key=base / "answer_sheet.pdf",
        roster=base / "name_list.xlsx",
        out_dir=out,
        cache_dir=out / "cache",
        crops_dir=out / "crops",
        debug_dir=out / "debug",
        log_path=out / "run.log",
        results_xlsx=out / "results.xlsx",
    )
