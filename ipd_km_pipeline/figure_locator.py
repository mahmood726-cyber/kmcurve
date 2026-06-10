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
    caption_anchored: bool = False  # KM keyword sits in a real figure caption


def _big_images(page):
    pw, ph = page.width or 1, page.height or 1
    out = []
    for im in page.images:
        w = abs(im.get("x1", 0) - im.get("x0", 0))
        h = abs(im.get("bottom", 0) - im.get("top", 0))
        if w * h > 0.05 * pw * ph:
            out.append((im["x0"], im["top"], im["x1"], im["bottom"]))
    return out


def _caption_blocks(text: str) -> List[str]:
    """Figure-caption blocks on a page: a line whose 'Figure N' sits at the START
    (a caption) -- not mid-sentence (an inline cross-reference like '... in Fig.
    3 ...'). Each block is the caption line plus up to two continuation lines.
    """
    lines = text.splitlines()
    out: List[str] = []
    for idx, line in enumerate(lines):
        m = _FIG_RE.search(line)
        if not m:
            continue
        pre = line[: m.start()].strip()
        # leading text before "Figure N" => inline reference, not a caption;
        # a "(" prefix is a parenthetical cross-ref "(Figure 3C). ..."
        if len(pre) > 3 or pre.endswith("("):
            continue
        # a subpanel ref ("Figure 3C)", "Fig 2A,") has a letter glued to the
        # number then punctuation/space -- a real caption has ". ", " <Title>", ":"
        after = line[m.end(): m.end() + 2]
        if re.match(r"^[A-Za-z][\)\s,.;]", after):
            continue
        out.append(" ".join(lines[idx: idx + 3]).strip()[:240])
    return out


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def locate_km_figures(pdf_path: str, max_candidates: int = 5,
                      require_caption: bool = False) -> List[KMFigure]:
    """Return ranked KM-figure candidates for a PDF.

    A page is a strong candidate only when the survival-figure keyword appears in
    a real FIGURE CAPTION ("Figure N. ... Kaplan-Meier ..."), not merely in body
    text (a Methods sentence + an unrelated image was the dominant false-positive
    -- 7/15 in the lever-2 corpus). Caption-anchored pages always outrank
    body-text-only pages; ``require_caption=True`` drops body-only candidates
    entirely (use for high-precision corpus building).
    """
    cands: List[KMFigure] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not _KM_RE.search(text):
                continue  # no survival-figure language anywhere on the page

            blocks = _caption_blocks(text)
            km_caption = next((b for b in blocks if _KM_RE.search(b)), "")

            words = page.extract_words()
            n_num = sum(1 for w in words if _NUM_RE.fullmatch(w["text"]))
            n_curves = len(page.curves)
            big = _big_images(page)

            # plottable-content base score
            if n_curves >= 200 and n_num >= 6:
                kind, bbox, base = "vector", None, 3.0
            elif big and n_num >= 6:
                kind, bbox, base = "raster", max(big, key=_area), 2.0
            elif n_curves >= 200:
                kind, bbox, base = "vector", None, 1.5
            elif big:
                kind, bbox, base = "raster", max(big, key=_area), 1.0
            else:
                continue  # KM language but no plottable content

            if km_caption:
                score, caption = base + 2.0, km_caption          # real KM caption
            else:
                if require_caption:
                    continue
                # body-text mention only -> a Methods/Results page with an
                # unrelated figure; keep only as a weak fallback far below any
                # caption-anchored page.
                snippet = blocks[0] if blocks else next(
                    (ln.strip() for ln in text.splitlines() if _KM_RE.search(ln)), "")
                score, caption = base * 0.25, snippet
            cands.append(KMFigure(
                page_index=i, kind=kind, caption=caption[:200], score=round(score, 3),
                bbox=bbox, n_vector_curves=n_curves, n_numeric_words=n_num,
                caption_anchored=bool(km_caption)))
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
