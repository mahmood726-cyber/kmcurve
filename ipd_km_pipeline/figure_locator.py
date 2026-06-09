#!/usr/bin/env python3
"""
Locate the Kaplan-Meier figure in a PDF (page + region + vector/raster kind).
=============================================================================

The corpus scan showed that "biggest image on a page" conflates KM curves with
flowcharts, baseline tables, forest plots and covariate dot-plots. To run the
pipeline on real PDFs we must find the *survival* figure specifically. We do
that by combining two signals:

  1. CAPTION text -- a figure caption mentioning Kaplan-Meier / overall or
     progression/disease/recurrence/event-free survival / cumulative incidence
     / time-to-event. (Vector text, reliable even when the figure is raster.)
  2. CONTENT near the caption -- a plottable region: either vector curve paths
     (vector figure) or a large embedded image with numeric tick labels around
     it (raster figure).

Returns ranked candidates so the caller can render/extract the most likely KM
figure first. Pure pdfplumber; no OCR needed for location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber

# caption keywords (lowercased match)
_KM_RE = re.compile(
    r"kaplan|meier|"
    r"(?:overall|progression[\s-]*free|disease[\s-]*free|recurrence[\s-]*free|"
    r"event[\s-]*free|relapse[\s-]*free|metastasis[\s-]*free)\s+survival|"
    r"survival\s+(?:curve|probability|analysis|function|rate)|"
    r"cumulative\s+incidence|time[\s-]*to[\s-]*event",
    re.IGNORECASE,
)
_FIG_RE = re.compile(r"\b(fig(?:ure)?\.?\s*\d+)", re.IGNORECASE)
_NUM_RE = re.compile(r"\d{1,3}(?:\.\d+)?%?")


@dataclass
class KMFigure:
    page_index: int
    kind: str  # "vector" | "raster"
    caption: str
    score: float
    bbox: Optional[tuple] = None  # (x0, top, x1, bottom) of the figure region
    n_vector_curves: int = 0
    n_numeric_words: int = 0


def _big_images(page):
    pw, ph = page.width or 1, page.height or 1
    out = []
    for im in page.images:
        w = abs(im.get("x1", 0) - im.get("x0", 0))
        h = abs(im.get("bottom", 0) - im.get("top", 0))
        if w * h > 0.05 * pw * ph:
            out.append((im["x0"], im["top"], im["x1"], im["bottom"]))
    return out


def locate_km_figures(pdf_path: str, max_candidates: int = 5) -> List[KMFigure]:
    """Return ranked KM-figure candidates for a PDF."""
    cands: List[KMFigure] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not _KM_RE.search(text):
                continue  # no survival-figure language on this page
            # caption snippet: the line mentioning the KM keyword
            caption = ""
            for line in text.splitlines():
                if _KM_RE.search(line):
                    caption = line.strip()[:160]
                    break
            words = page.extract_words()
            n_num = sum(1 for w in words if _NUM_RE.fullmatch(w["text"]))
            n_curves = len(page.curves)
            big = _big_images(page)

            # decide kind + score
            has_caption_fig = bool(_FIG_RE.search(caption))
            if n_curves >= 200 and n_num >= 6:
                kind, bbox = "vector", None
                score = 3.0 + (1.0 if has_caption_fig else 0.0)
            elif big and n_num >= 6:
                kind, bbox = "raster", max(big, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                score = 2.0 + (1.0 if has_caption_fig else 0.0)
            elif n_curves >= 200:
                kind, bbox, score = "vector", None, 1.5
            elif big:
                kind, bbox, score = "raster", max(big, key=lambda b: (b[2]-b[0])*(b[3]-b[1])), 1.0
            else:
                continue  # caption but no plottable content on this page
            cands.append(KMFigure(page_index=i, kind=kind, caption=caption,
                                  score=score, bbox=bbox,
                                  n_vector_curves=n_curves, n_numeric_words=n_num))
    cands.sort(key=lambda c: -c.score)
    return cands[:max_candidates]


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    paths = sys.argv[1:]
    if not paths:
        base = Path(__file__).resolve().parent
        paths = [str(p) for d in ("corpus", "corpus_unpaywall")
                 for p in sorted((base / d).glob("*.pdf"))]
    print(f"Locating KM figures in {len(paths)} PDFs\n" + "=" * 70)
    found = 0
    for p in paths:
        try:
            cands = locate_km_figures(p)
        except Exception as exc:
            print(f"{Path(p).name}: ERROR {type(exc).__name__}")
            continue
        if cands:
            found += 1
            c = cands[0]
            print(f"{Path(p).name[:34]:<34} p{c.page_index} {c.kind:<6} "
                  f"score={c.score} :: {c.caption[:60]}")
        else:
            print(f"{Path(p).name[:34]:<34} -- no KM figure caption found")
    print("=" * 70)
    print(f"located a KM figure in {found}/{len(paths)} PDFs")
