#!/usr/bin/env python3
"""Render axis-label crops from corpus KM figures for the VLM-calibration pilot.

For each corpus PDF: locate the top KM figure, render it, detect plot box(es),
and for every box where classical CV finds >=2 tick positions on BOTH axes
(the precondition for VLM calibration -- positions are CV's job), crop the x/y
label bands to PNG. Emits a manifest the read-workflow consumes. No OCR, no
network -- pure CV, so it is fast.

Usage: python render_corpus_crops.py [--out artifacts/vlm_crops] [--dpi 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import figure_locator as FL
from raster_km import (PlotBox, detect_plot_boxes, detect_plot_box,
                       detect_tick_positions, render_page)
from vlm_calibrate import axis_label_crop

CORPUS_DIRS = ["corpus", "corpus_unpaywall"]


def _iter_pdfs():
    for d in CORPUS_DIRS:
        for p in sorted(Path(d).glob("*.pdf")):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vlm_crops")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for pdf in _iter_pdfs():
        stem = pdf.stem.replace(".", "_")
        try:
            cands = FL.locate_km_figures(str(pdf))
        except Exception as exc:
            print(f"[skip] {pdf.name}: locate failed ({str(exc)[:50]})")
            continue
        if not cands:
            print(f"[no-km] {pdf.name}")
            continue
        c = cands[0]
        try:
            g = render_page(str(pdf), c.page_index, dpi=args.dpi)
        except Exception as exc:
            print(f"[skip] {pdf.name}: render failed ({str(exc)[:50]})")
            continue
        if c.bbox:
            sc = args.dpi / 72.0
            x0, t, x1, b = c.bbox
            g = g[int(t * sc):int(b * sc), int(x0 * sc):int(x1 * sc)]

        boxes = detect_plot_boxes(g)
        if not boxes:
            single = detect_plot_box(g)
            boxes = [single] if single is not None else []
        if not boxes:
            print(f"[no-box] {pdf.name}")
            continue

        for k, box in enumerate(boxes):
            xpos = sorted(detect_tick_positions(g, box, "x"))
            ypos = sorted(detect_tick_positions(g, box, "y"))
            if len(xpos) < 2 or len(ypos) < 2:
                continue  # VLM calibration needs CV positions on both axes
            x_crop = axis_label_crop(g, box, "x")
            y_crop = axis_label_crop(g, box, "y")
            if x_crop.size == 0 or y_crop.size == 0:
                continue
            ddir = out_root / stem / f"box{k}"
            ddir.mkdir(parents=True, exist_ok=True)
            xp = ddir / "x_axis.png"
            yp = ddir / "y_axis.png"
            Image.fromarray(x_crop).save(xp)
            Image.fromarray(y_crop).save(yp)
            manifest.append({
                "id": f"{stem}/box{k}",
                "pdf": str(pdf),
                "page_index": c.page_index,
                "kind": c.kind,
                "box": [box.x0, box.y0, box.x1, box.y1],
                "n_xticks": len(xpos),
                "n_yticks": len(ypos),
                "x_png": str(xp.resolve()),
                "y_png": str(yp.resolve()),
            })
            print(f"[crop] {pdf.name} box{k}: x={len(xpos)} y={len(ypos)} ticks")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} crop-entries -> {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()
