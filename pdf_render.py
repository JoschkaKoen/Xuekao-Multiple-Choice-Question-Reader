"""PDF rasterization (COLOR), rotation, crop, downscale → JPEG/base64.

fitz `get_pixmap` honors the page's /Rotate, so renders come out upright
(landscape for these answer sheets). We render in COLOR (csRGB) — crops keep
their color — and derive a gray copy only for orientation/deskew/anchor maths.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

import config
from ai_helpers import page_to_jpeg_b64


def page_count(pdf_path: Path) -> int:
    with fitz.open(str(pdf_path)) as d:
        return d.page_count


def render_color(pdf_path: Path, page_idx: int, dpi: int = config.DPI) -> np.ndarray:
    """Render one PDF page to an upright RGB array (H, W, 3) uint8 (fitz honors /Rotate)."""
    with fitz.open(str(pdf_path)) as d:
        pix = d[page_idx].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # defensive: drop alpha if present
        img = img[:, :, :3]
    return np.ascontiguousarray(img[:, :, :3])


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rotate_cw(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 using exact np.rot90 (no interpolation)."""
    k = (int(degrees) // 90) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=-k))  # np.rot90 is CCW; -k → CW


def crop_frac(img: np.ndarray, frame: tuple[int, int, int, int],
              box: tuple[float, float, float, float], pad_frac: float = 0.0) -> np.ndarray:
    """Crop *img* to *box* (x0,y0,x1,y1 fractions of *frame* = (fx0,fy0,fx1,fy1))."""
    fx0, fy0, fx1, fy1 = frame
    fw, fh = (fx1 - fx0), (fy1 - fy0)
    x0 = fx0 + box[0] * fw
    y0 = fy0 + box[1] * fh
    x1 = fx0 + box[2] * fw
    y1 = fy0 + box[3] * fh
    px, py = pad_frac * fw, pad_frac * fh
    H, W = img.shape[:2]
    x0 = max(0, int(round(x0 - px)));  y0 = max(0, int(round(y0 - py)))
    x1 = min(W, int(round(x1 + px)));  y1 = min(H, int(round(y1 + py)))
    return img[y0:y1, x0:x1]


def downscale_pil(arr: np.ndarray, long_edge: int) -> Image.Image:
    """PIL image downscaled so its longest edge ≤ long_edge (no upscaling)."""
    pil = Image.fromarray(arr)
    w, h = pil.size
    scale = long_edge / float(max(w, h))
    if scale < 1.0:
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return pil


def to_jpeg_b64(arr: np.ndarray, long_edge: int, quality: int = config.JPEG_QUALITY) -> str:
    return page_to_jpeg_b64(downscale_pil(arr, long_edge), quality=quality)


def save_image(arr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def save_jpeg(arr: np.ndarray, path: Path, long_edge: int, quality: int = config.JPEG_QUALITY) -> Path:
    """Downscale a crop to *long_edge* and save as JPEG (what the AI receives)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    downscale_pil(arr, long_edge).convert("RGB").save(str(path), "JPEG", quality=quality)
    return path


def jpeg_b64(path: Path) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
