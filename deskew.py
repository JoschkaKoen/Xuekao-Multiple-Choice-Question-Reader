"""Small-angle deskew — ported from the eXercise xscore pipeline (via Course
Selection Reader). Writing-line detection (preferred) + projection-variance
fallback, then a bicubic warp. The angle is measured on a GRAY copy and applied
to the COLOR image (white border fill).

Source: eXercise xscore/preprocessing/deskew/{angle,warp,types}.py
"""

from __future__ import annotations

import cv2
import numpy as np

# --- tuning constants (inlined from eXercise deskew/types.py) ----------------
_SWEEP_MIN = -3.0
_SWEEP_MAX = 3.0
_SWEEP_STEP = 0.01
_SWEEP_COARSE_STEP = 0.1
_SWEEP_FINE_HALF = 0.15
_MIN_APPLY_DEG = 0.01      # skip warp below this (matches fine-sweep resolution)
_MAX_APPLY_DEG = 1.0       # zero out detections beyond this (likely bad signal)
_REFINE_MAX_ITERS = 5

_WL_CLOSE_PX = 15
_WL_OPEN_FRAC = 0.08
_WL_OPEN_MIN_PX = 200
_WL_MIN_WIDTH_FRAC = 0.30
_WL_MAX_HEIGHT_PX = 30
_WL_MIN_PIXELS = 50
_WL_MIN_LINES = 3


def _warp(gray: np.ndarray, angle: float) -> np.ndarray:
    """Bicubic rotation with white (255) border fill — gray, unconditional."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _warp_color(img: np.ndarray, angle: float) -> np.ndarray:
    """Bicubic rotation with white border fill for a 3-channel image."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def _best_angle_projection_variance(
    thresh: np.ndarray, angle_min: float, angle_max: float, angle_step: float
) -> float:
    h, w = thresh.shape[:2]
    cx, cy = w // 2, h // 2
    best_angle, best_var = 0.0, -1.0
    for angle in np.arange(angle_min, angle_max + angle_step / 2, angle_step):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            thresh, M, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        proj = np.sum(rotated, axis=0, dtype=np.float64)
        v = float(np.var(proj))
        if v > best_var:
            best_var, best_angle = v, float(angle)
    return best_angle


def get_deskew_angle(gray: np.ndarray) -> float:
    """Skew angle via vertical-projection variance (coarse then fine sweep)."""
    _, thresh_full = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coarse_best = _best_angle_projection_variance(
        thresh_full, _SWEEP_MIN, _SWEEP_MAX, _SWEEP_COARSE_STEP
    )
    fine_lo = max(_SWEEP_MIN, coarse_best - _SWEEP_FINE_HALF)
    fine_hi = min(_SWEEP_MAX, coarse_best + _SWEEP_FINE_HALF)
    return _best_angle_projection_variance(thresh_full, fine_lo, fine_hi, _SWEEP_STEP)


def detect_writing_line_angle(gray: np.ndarray) -> tuple[float | None, int]:
    """Median angle of dotted/printed horizontal lines; (None, 0) if none found."""
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (_WL_CLOSE_PX, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k)

    open_w = max(_WL_OPEN_MIN_PX, int(_WL_OPEN_FRAC * w))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, 1))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    angles: list[float] = []
    for k in range(1, n_lab):
        _x, _y, ww, hh, _area = stats[k]
        if ww < _WL_MIN_WIDTH_FRAC * w or hh > _WL_MAX_HEIGHT_PX:
            continue
        ys, xs = np.where(labels == k)
        if len(xs) < _WL_MIN_PIXELS:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, _x0, _y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        ang = float(np.degrees(np.arctan2(vy, vx)))
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        angles.append(ang)
    if not angles:
        return None, 0
    return float(np.median(angles)), len(angles)


def iterative_deskew_angle(gray: np.ndarray) -> tuple[float, int, str]:
    """Refine skew angle, preferring writing-line detection; (angle, iters, method)."""
    _, n_first = detect_writing_line_angle(gray)
    if n_first >= _WL_MIN_LINES:
        total = 0.0
        cur = gray
        for it in range(_REFINE_MAX_ITERS):
            a, n = detect_writing_line_angle(cur)
            if a is None or n < _WL_MIN_LINES:
                return total, it, "wl"
            if abs(a) < _MIN_APPLY_DEG:
                return total, it, "wl"
            total += a
            cur = _warp(gray, total)
        return total, _REFINE_MAX_ITERS, "wl"

    total = 0.0
    cur = gray
    for it in range(_REFINE_MAX_ITERS):
        a = get_deskew_angle(cur)
        if abs(a) < _MIN_APPLY_DEG:
            return total, it, "proj"
        total += a
        cur = _warp(gray, total)
    return total, _REFINE_MAX_ITERS, "proj"


def gate_angle(angle: float) -> float:
    """Zero out implausibly large detected angles (likely a bad signal)."""
    return 0.0 if abs(angle) > _MAX_APPLY_DEG else angle


def deskew_color(color: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure skew on *gray*, apply to *color*. Returns (deskewed_color, angle)."""
    angle, _iters, _method = iterative_deskew_angle(gray)
    angle = gate_angle(angle)
    if abs(angle) < _MIN_APPLY_DEG:
        return color, 0.0
    return _warp_color(color, angle), angle
