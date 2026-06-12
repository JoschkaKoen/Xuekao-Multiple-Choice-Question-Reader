"""Dev tool: full geometry (render→orient→deskew→frame), then dump a grid
overlay + the field crops on the CORRECTED (portrait) page for visual
calibration. Usage: python _calibrate.py <folder> <pages> [grid]
"""
import sys

import cv2
import numpy as np
from PIL import Image

import anchors
import config
import orient
import pdf_render
from deskew import deskew_color
from pdf_render import crop_frac, render_color, to_gray


def ds(arr, long_edge=1900):
    p = Image.fromarray(arr)
    w, h = p.size
    s = long_edge / float(max(w, h))
    return np.asarray(p.resize((int(w * s), int(h * s)), Image.LANCZOS)) if s < 1 else arr


def draw_grid(rgb, frame):
    img = np.ascontiguousarray(rgb.copy())
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0
    for i in range(0, 21):
        fx = x0 + int(i / 20 * fw)
        cv2.line(img, (fx, y0), (fx, y1), (0, 0, 255), 2 if i % 2 == 0 else 1)
        if i % 2 == 0:
            cv2.putText(img, f"{i/20:.1f}", (fx + 2, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    for j in range(0, 21):
        fy = y0 + int(j / 20 * fh)
        cv2.line(img, (x0, fy), (x1, fy), (255, 0, 0), 2 if j % 2 == 0 else 1)
        if j % 2 == 0:
            cv2.putText(img, f"{j/20:.1f}", (x0 + 2, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    return img


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "input_files_1"
    pages = [int(x) - 1 for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1"])]
    grid = "grid" in sys.argv
    cfg = config.for_folder(folder)
    crop_boxes = config.crop_boxes_for_folder(folder)
    outdir = cfg.debug_dir
    outdir.mkdir(parents=True, exist_ok=True)
    for p in pages:
        rgb = render_color(cfg.first_page, p, config.DPI)
        rgb, rot, method = orient.correct_orientation(rgb, config.DPI)
        rgb, skew = deskew_color(rgb, to_gray(rgb))
        marks = anchors.find_marks(rgb, config.DPI)
        frame, source = anchors.resolve_frame(rgb, marks, config.DPI)
        fx0, fy0, fx1, fy1 = frame
        fw, fh = fx1 - fx0, fy1 - fy0
        crops_px = {name: (int(fx0 + box[0] * fw), int(fy0 + box[1] * fh),
                           int(fx0 + box[2] * fw), int(fy0 + box[3] * fh))
                    for name, (box, _le) in crop_boxes.items()}
        ov = anchors.debug_overlay(rgb, marks, frame, crops_px)
        pdf_render.save_image(ds(ov), outdir / f"overlay_p{p+1}.png")
        if grid:
            pdf_render.save_image(ds(draw_grid(rgb, frame)), outdir / f"grid_p{p+1}.png")
        for name, (box, _le) in crop_boxes.items():
            pdf_render.save_image(crop_frac(rgb, frame, box), outdir / f"{name}_p{p+1}.png")
        print(f"p{p+1}: {rgb.shape[1]}x{rgb.shape[0]} rot={rot}({method}) skew={skew:.2f} "
              f"marks={len(marks)} frame={frame} src={source}")


if __name__ == "__main__":
    main()
