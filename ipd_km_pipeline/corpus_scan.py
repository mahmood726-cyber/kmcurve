#!/usr/bin/env python3
"""
Extraction-robustness scan over a corpus of real OA PDFs.
=========================================================

Measures WHERE the extractor breaks across the messy variety of real figures,
without needing verified ground-truth HRs (so it scales freely). For each PDF
it auto-locates candidate KM-figure pages and buckets the best outcome into a
failure taxonomy that drives feature prioritisation:

  extracted .......... >=1 panel calibrated with >=2 separated arms (success)
  vector_no_arms ..... calibrated panel but <2 arms separated
  vector_calib_failed  axis frame found but tick calibration R^2 < 0.999
  vector_no_panel .... vector curve content but no axis frame detected
  raster_figure ...... figure is a flattened raster image (no vector curves)
                       -> needs the raster fallback path
  no_figure_found .... no plot-like page at all

Run:  python ipd_km_pipeline/corpus_scan.py [corpus_dir] [--json out.json]
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

import vector_km as V
from manual_calibration import figure_is_vector

_NUM = re.compile(r"\d{1,3}(?:\.\d+)?%?")

# ordered best -> worst so we can keep the best per PDF
RANK = ["extracted", "vector_no_arms", "vector_calib_failed",
        "vector_no_panel", "raster_figure", "no_figure_found"]


@dataclass
class PdfScan:
    pdf: str
    n_pages: int = 0
    status: str = "no_figure_found"
    km_page: Optional[int] = None
    n_panels: int = 0
    max_calib_r2: float = 0.0
    max_arms: int = 0
    best_separation_conf: Optional[float] = None
    at_risk_rows: int = 0
    has_raster_figure: bool = False
    detail: str = ""


def _is_plot_page(page) -> bool:
    nvec = len(page.curves) + len(page.lines)
    nnum = sum(1 for w in page.extract_words() if _NUM.fullmatch(w["text"]))
    return nvec >= 40 and nnum >= 5


def _page_has_big_image(page) -> bool:
    pw, ph = page.width or 1, page.height or 1
    for im in page.images:
        w = abs(im.get("x1", 0) - im.get("x0", 0))
        h = abs(im.get("bottom", 0) - im.get("top", 0))
        if w * h > 0.10 * pw * ph:  # image covering >10% of the page
            return True
    return False


def scan_pdf(path: Path) -> PdfScan:
    res = PdfScan(pdf=path.name)
    try:
        with pdfplumber.open(str(path)) as pdf:
            res.n_pages = len(pdf.pages)
            candidates, raster_pages = [], []
            for i, page in enumerate(pdf.pages):
                try:
                    if _is_plot_page(page):
                        candidates.append(i)
                    elif _page_has_big_image(page) and len(page.curves) < 20:
                        raster_pages.append(i)
                except Exception:
                    continue
            res.has_raster_figure = bool(raster_pages)
    except Exception as exc:
        res.status = "no_figure_found"
        res.detail = f"open failed: {type(exc).__name__}: {exc}"
        return res

    best_rank = len(RANK)  # worst
    for i in candidates:
        try:
            panels = V.extract_km_from_pdf(str(path), i, monotone="auto")
        except Exception as exc:
            res.detail = f"p{i}: {type(exc).__name__}"
            continue
        if not panels:
            continue
        for pr in panels:
            res.n_panels = max(res.n_panels, len(panels))
            r2 = min(pr.x_fit.r2, pr.y_fit.r2)
            res.max_calib_r2 = max(res.max_calib_r2, r2)
            n_arms = len([a for a in pr.arms if a.n_points() > 0])
            res.max_arms = max(res.max_arms, n_arms)
            if r2 >= 0.999 and n_arms >= 2:
                status = "extracted"
            elif r2 >= 0.999 and n_arms >= 1:
                status = "vector_no_arms"
            elif r2 < 0.999:
                status = "vector_calib_failed"
            else:
                status = "vector_no_panel"
            rk = RANK.index(status)
            if rk < best_rank:
                best_rank = rk
                res.status = status
                res.km_page = i
                res.best_separation_conf = pr.separation_confidence
                if status == "extracted":
                    try:
                        with pdfplumber.open(str(path)) as pdf2:
                            rows = V.extract_at_risk(pdf2.pages[i], pr.panel)
                        res.at_risk_rows = len(rows)
                    except Exception:
                        pass

    if best_rank == len(RANK):
        # no usable vector panel; classify by what we did see
        if res.has_raster_figure:
            res.status = "raster_figure"
        elif candidates:
            res.status = "vector_no_panel"
        else:
            res.status = "no_figure_found"
    return res


def run(corpus_dir: str) -> dict:
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    scans = [scan_pdf(p) for p in pdfs]
    tax: dict = {}
    for s in scans:
        tax[s.status] = tax.get(s.status, 0) + 1
    n = len(scans)
    extracted = tax.get("extracted", 0)
    report = {
        "n_pdfs": n,
        "taxonomy": tax,
        "pct_extracted": (100.0 * extracted / n) if n else 0.0,
        "scans": [asdict(s) for s in scans],
    }

    print("=" * 84)
    print(f"Corpus extraction-robustness scan  --  {n} PDFs from {corpus_dir}")
    print("=" * 84)
    print(f"{'pdf':<18}{'pages':>5}{'kmpg':>5}{'panels':>7}{'R2':>8}{'arms':>5}{'conf':>6}{'NAR':>4}  status")
    print("-" * 84)
    for s in scans:
        r2 = f"{s.max_calib_r2:.3f}" if s.max_calib_r2 else "  -"
        cf = f"{s.best_separation_conf:.2f}" if s.best_separation_conf is not None else " -"
        kp = s.km_page if s.km_page is not None else "-"
        print(f"{s.pdf:<18}{s.n_pages:>5}{str(kp):>5}{s.n_panels:>7}{r2:>8}{s.max_arms:>5}{cf:>6}{s.at_risk_rows:>4}  {s.status}")
    print("-" * 84)
    print(f"taxonomy: {tax}")
    print(f"fully extracted: {extracted}/{n} ({report['pct_extracted']:.0f}%)")
    print("=" * 84)
    return report


if __name__ == "__main__":
    import sys

    default = str(Path(__file__).resolve().parent / "corpus")
    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
    # positional corpus dir = first arg that is neither a flag nor the --json value
    positionals = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--json":
            skip = True
            continue
        if a.startswith("--"):
            continue
        positionals.append(a)
    corpus = positionals[0] if positionals else default
    rep = run(corpus)
    if out:
        Path(out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"wrote {out}")
