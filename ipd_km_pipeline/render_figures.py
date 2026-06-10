#!/usr/bin/env python3
"""Render full KM figures (for VLM structured auto-labelling, lever 2).

For each corpus PDF (+ the ADVANCE ground-truth paper), locate the top KM figure
and render its region to a PNG, emitting a manifest the label workflow consumes.
Unlike render_corpus_crops.py (axis bands), this saves the WHOLE figure so the
VLM can count panels/arms and read both axes. Pure CV/render -- no OCR, no net.

Usage: python render_figures.py [--out artifacts/vlm_figures] [--dpi 150]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import figure_locator as FL
from raster_km import render_page

# corpus dirs + the ADVANCE ground-truth PDF (papers_to_process)
CORPUS_DIRS = ["corpus", "corpus_unpaywall", "corpus_pmc", "papers_to_process"]


def _iter_pdfs():
    for d in CORPUS_DIRS:
        for p in sorted(Path(d).glob("*.pdf")):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vlm_figures")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for pdf in _iter_pdfs():
        stem = pdf.stem.replace(".", "_")
        try:
            cands = FL.locate_km_figures(str(pdf))
        except Exception as exc:
            print(f"[skip] {pdf.name}: {str(exc)[:50]}")
            continue
        if not cands:
            print(f"[no-km] {pdf.name}")
            continue
        c = cands[0]
        try:
            g = render_page(str(pdf), c.page_index, dpi=args.dpi)
        except Exception as exc:
            print(f"[skip] {pdf.name}: render {str(exc)[:40]}")
            continue
        if c.bbox:
            sc = args.dpi / 72.0
            x0, t, x1, b = c.bbox
            g = g[int(t * sc):int(b * sc), int(x0 * sc):int(x1 * sc)]
        if g.size == 0 or min(g.shape) < 40:
            print(f"[tiny] {pdf.name}")
            continue
        png = out_root / f"{stem}.png"
        Image.fromarray(g).save(png)
        manifest.append({
            "id": stem, "pdf": str(pdf), "page_index": c.page_index,
            "kind": c.kind, "png": str(png.resolve()),
        })
        print(f"[fig] {pdf.name} p{c.page_index} {g.shape}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} figures -> {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()
