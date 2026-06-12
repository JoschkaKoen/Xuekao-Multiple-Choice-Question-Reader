"""Page orientation — reliable black-mark detection, with OSD backstop.

The sheets render landscape (fitz honors /Rotate) but the true reading
orientation is PORTRAIT with the long black timing track on the top margin.
Where the marks sit IS the orientation, so the mark detector determines the
rotation directly — far more reliable than Tesseract OSD, which is confused by
this mixed-orientation form (it abstains with low confidence). OSD / a global
vote are only backstops for the rare page whose marks can't be read.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

import anchors
import config
from pdf_render import render_color, rotate_cw

log = logging.getLogger("xuekao")

_OSD_CONF_MIN = 2.0


def rotation_from_marks(rgb: np.ndarray, dpi: int = config.DPI) -> int | None:
    """CW degrees to canonical portrait from the black marks, or None if too few."""
    marks = anchors.find_marks(rgb, dpi)
    if len(marks) < 6:
        return None
    return anchors.rotation_cw_from_marks(marks, rgb.shape[1], rgb.shape[0])


def rotation_from_osd(rgb: np.ndarray) -> int | None:
    """CW degrees to upright per Tesseract OSD, or None if unavailable/low-conf."""
    try:
        import pytesseract  # noqa: PLC0415
        osd = pytesseract.image_to_osd(Image.fromarray(rgb), output_type=pytesseract.Output.DICT)
        if float(osd.get("orientation_conf", 0)) < _OSD_CONF_MIN:
            return None
        return int(osd.get("rotate", 0)) % 360
    except Exception:  # noqa: BLE001
        return None


def global_rotation(pdf_path, total_pages: int, dpi: int = config.ORIENT_DPI) -> int:
    """Majority mark-based rotation over a spread sample of pages (the global vote)."""
    from collections import Counter
    if total_pages <= 0:
        return 0
    n = min(9, total_pages)
    idxs = sorted({round(i * (total_pages - 1) / max(1, n - 1)) for i in range(n)})
    votes: list[int] = []
    for p in idxs:
        try:
            r = rotation_from_marks(render_color(pdf_path, p, dpi), dpi)
        except Exception:  # noqa: BLE001
            r = None
        if r is not None:
            votes.append(r)
    if not votes:
        return 0
    return Counter(votes).most_common(1)[0][0]


def correct_orientation(rgb: np.ndarray, dpi: int = config.DPI, global_rot: int | None = None):
    """Rotate *rgb* to canonical portrait. Returns (rotated_rgb, rotation_cw, method)."""
    r = rotation_from_marks(rgb, dpi)
    method = "marks"
    if r is None:
        r = rotation_from_osd(rgb)
        method = "osd"
    if r is None:
        r = global_rot if global_rot is not None else 0
        method = "global"
    return rotate_cw(rgb, r), int(r), method
