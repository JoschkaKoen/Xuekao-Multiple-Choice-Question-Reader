"""Per-page geometry: render → orient (portrait) → deskew → anchor frame → crops.

Produces the color field crops (name / class / answers) for one student page,
plus geometry metadata for resume + Review. CPU-only; no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import anchors
import config
import orient
from deskew import deskew_color
from pdf_render import crop_frac, render_color, to_gray

log = logging.getLogger("xuekao")


@dataclass
class Geometry:
    page_idx: int
    rotation_cw: int
    orient_method: str
    skew_deg: float
    frame: tuple
    frame_source: str
    n_marks: int
    crops: dict = field(default_factory=dict)  # name -> color np.ndarray

    @property
    def fallback(self) -> bool:
        return self.frame_source != "marks"


def page_geometry(pdf_path, page_idx: int, dpi: int = config.DPI,
                  global_rot: int | None = None) -> Geometry:
    rgb = render_color(pdf_path, page_idx, dpi)
    rgb, rot, method = orient.correct_orientation(rgb, dpi, global_rot)
    rgb, skew = deskew_color(rgb, to_gray(rgb))
    marks = anchors.find_marks(rgb, dpi)
    frame, source = anchors.resolve_frame(rgb, marks, dpi)
    crops = {name: crop_frac(rgb, frame, box) for name, (box, _le) in config.CROP_BOXES.items()}
    return Geometry(page_idx, rot, method, skew, frame, source, len(marks), crops)
