"""Black-mark (OMR fiducial) detection → page coordinate frame → crop boxes.

The answer sheets carry printed solid-black registration marks made for machine
recognition: a vertical timing track of evenly-spaced black squares down the
left margin plus corner fiducials. They are high-contrast and content-
independent, so they give a reliable frame for (a) per-page orientation and
(b) anchoring the identity / MCQ crops. The red printed frame is the documented
fallback.

Detection runs on the COLOR render (RGB); crops are cut from color elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class Mark:
    cx: int
    cy: int
    x: int
    y: int
    w: int
    h: int
    area: int


def black_mask(rgb: np.ndarray, max_val: int | None = None) -> np.ndarray:
    """Boolean mask of near-black pixels (all of R,G,B < max_val)."""
    m = config.MARK_BLACK_MAX if max_val is None else max_val
    return np.all(rgb < m, axis=2)


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the red printed rulings (R high, clearly above G and B)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 100) & (r - g > 35) & (r - b > 35)


def find_marks(rgb: np.ndarray, dpi: int = config.DPI) -> list[Mark]:
    """Detect solid-black fiducial marks near the page margins."""
    H, W = rgb.shape[:2]
    scale = dpi / 300.0
    min_side = config.MARK_MIN_SIDE_300 * scale
    max_side = config.MARK_MAX_SIDE_300 * scale
    margin = config.MARK_MARGIN_FRAC
    mask = black_mask(rgb).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    marks: list[Mark] = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if area < 0.5 * min_side * min_side:
            continue
        ar = w / float(h)
        if ar < 0.35 or ar > 2.8:            # roughly square-ish
            continue
        if area / float(w * h) < 0.55:        # solid fill
            continue
        cx, cy = int(cent[k][0]), int(cent[k][1])
        near_edge = (
            cx < margin * W or cx > (1 - margin) * W
            or cy < margin * H or cy > (1 - margin) * H
        )
        if not near_edge:
            continue
        marks.append(Mark(cx, cy, x, y, w, h, area))
    return marks


def _edge_groups(marks: list[Mark], W: int, H: int, margin: float):
    left = [m for m in marks if m.cx < margin * W]
    right = [m for m in marks if m.cx > (1 - margin) * W]
    top = [m for m in marks if m.cy < margin * H]
    bottom = [m for m in marks if m.cy > (1 - margin) * H]
    return left, right, top, bottom


def track_side(marks: list[Mark], W: int, H: int) -> str:
    """Which margin holds the dominant timing track ('L'/'R'/'T'/'B')."""
    left, right, top, bottom = _edge_groups(marks, W, H, config.MARK_MARGIN_FRAC)
    counts = {"L": len(left), "R": len(right), "T": len(top), "B": len(bottom)}
    return max(counts, key=counts.get)


# Canonical orientation is PORTRAIT with the long timing track on the TOP margin
# (fitz renders these sheets landscape with the track on the LEFT, so the common
# case is a 90° CW turn). Map: current dominant-track side → degrees CW to bring
# that track to the top.
_SIDE_TO_ROT_CW = {"T": 0, "L": 90, "R": 270, "B": 180}


def rotation_cw_from_marks(marks: list[Mark], W: int, H: int) -> int:
    """Degrees CW to rotate the render into canonical portrait (track on top)."""
    return _SIDE_TO_ROT_CW.get(track_side(marks, W, H), 0)


def detect_frame(marks: list[Mark], W: int, H: int, dpi: int = config.DPI):
    """Content frame (x0,y0,x1,y1) = bounding box of the black marks.

    The timing track spans one full edge and there are marks on the opposite
    edge too, so the mark bbox brackets the printed content in both axes,
    independent of orientation. Returns (frame|None, info) — None (degenerate)
    when marks don't bracket both axes, so the caller falls back to the red frame.
    """
    info = {"n": len(marks), "side": track_side(marks, W, H) if marks else "?"}
    if len(marks) < 6:
        return None, info
    xs = [m.cx for m in marks]
    ys = [m.cy for m in marks]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    info["bbox"] = (x0, y0, x1, y1)
    if (x1 - x0) < 0.45 * W or (y1 - y0) < 0.45 * H:
        return None, info
    return (x0, y0, x1, y1), info


def red_frame_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the red printed frame (fallback content rectangle)."""
    mask = red_mask(rgb).astype(np.uint8)
    if mask.sum() < 1000:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resolve_frame(rgb: np.ndarray, marks: list[Mark], dpi: int = config.DPI):
    """Best content frame: black marks bbox → red printed frame → whole page.

    Returns (frame, source) with source ∈ {"marks","red","page"}. Always returns
    a frame (never None) so a page is never dropped.
    """
    H, W = rgb.shape[:2]
    frame, _info = detect_frame(marks, W, H, dpi)
    if frame is not None:
        return frame, "marks"
    red = red_frame_bbox(rgb)
    if red is not None:
        return red, "red"
    m = 0.03
    return (int(m * W), int(m * H), int((1 - m) * W), int((1 - m) * H)), "page"


def debug_overlay(rgb: np.ndarray, marks: list[Mark], frame, crops_px: dict) -> np.ndarray:
    """Draw detected marks (green), the frame (blue) and crop rects (orange/red)."""
    img = np.ascontiguousarray(rgb.copy())
    for m in marks:
        cv2.rectangle(img, (m.x, m.y), (m.x + m.w, m.y + m.h), (0, 200, 0), 3)
    if frame is not None:
        x0, y0, x1, y1 = frame
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 4)
    colors = {"class": (255, 140, 0), "name": (160, 0, 200), "answers": (220, 0, 0)}
    for name, (x0, y0, x1, y1) in crops_px.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), colors.get(name, (255, 0, 255)), 5)
    return img
