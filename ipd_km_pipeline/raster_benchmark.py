#!/usr/bin/env python3
"""
Raster pipeline benchmark over a corpus of real OA PDFs.
========================================================

Runs the full raster path (figure_locator -> render -> auto-calibrate ->
curve extraction -> arm separation -> Guyot) on every PDF in the corpus and
reports robustness + accuracy. Accuracy uses FIGURE-INTERNAL ground truth:
where the numbers-at-risk table reports cumulative events per arm in
parentheses ("0 (12)"), we compare the RECONSTRUCTED event count to it -- no
external HR matching needed.

Per PDF outcome is bucketed (success / no_km_located / calib_fallback /
extract_fail / error). PDFs are gitignored, so this is a local-run tool; the
script is committed and results are recorded in CORPUS_FINDINGS.md.

Run: python ipd_km_pipeline/raster_benchmark.py [dir ...]
Needs RapidOCR + Tesseract.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

import raster_km as R
import figure_locator as FL


@dataclass
class RasterEval:
    pdf: str
    status: str = ""
    page: Optional[int] = None
    kind: str = ""
    n_arms: int = 0
    recon_events: List[int] = field(default_factory=list)
    reported_events: List[Optional[int]] = field(default_factory=list)
    event_abs_err: Optional[float] = None  # mean |recon - reported| where both known
    detail: str = ""


def evaluate(pdf: str) -> RasterEval:
    ev = RasterEval(pdf=Path(pdf).name)
    try:
        cands = FL.locate_km_figures(pdf)
    except Exception as exc:
        ev.status = "error"; ev.detail = f"locate {type(exc).__name__}"; return ev
    if not cands:
        ev.status = "no_km_located"; return ev
    ev.page, ev.kind = cands[0].page_index, cands[0].kind
    try:
        res = R.pdf_raster_to_ipd(pdf)
    except ValueError as exc:
        ev.status = "calib_fallback" if "calibrat" in str(exc) else "extract_fail"
        ev.detail = str(exc)[:60]; return ev
    except Exception as exc:
        ev.status = "error"; ev.detail = f"{type(exc).__name__}: {exc}"[:60]; return ev

    arms = res["arms"]
    ev.n_arms = len(arms)
    if len(arms) < 2:
        ev.status = "extract_fail"; ev.detail = f"{len(arms)} arms"; return ev
    ev.recon_events = [a["n_events"] for a in arms]

    # figure-internal ground truth: reported events from the at-risk table
    try:
        SC = 200 / 72
        g = R.render_page(pdf, res["page"], dpi=200)
        c = cands[0]
        if c.bbox:
            x0, t, x1, b = c.bbox
            g = g[int(t * SC):int(b * SC), int(x0 * SC):int(x1 * SC)]
        box = R.detect_plot_box(g)
        ev.reported_events = R.at_risk_reported_events(g, box) if box else []
    except Exception:
        ev.reported_events = []

    pairs = [(r, rep) for r, rep in zip(ev.recon_events, ev.reported_events)
             if rep is not None]
    if pairs:
        ev.event_abs_err = float(np.mean([abs(r - rep) for r, rep in pairs]))
    # a KM survival figure with ZERO reconstructed events across BOTH arms is
    # almost always a silent extraction failure (flat/missed curves), not a
    # real result -- flag it rather than count it as success.
    if sum(ev.recon_events) == 0:
        ev.status = "extract_suspect"; ev.detail = "0 total events"
    else:
        ev.status = "success"
    return ev


def run(dirs: List[str]) -> dict:
    pdfs = [str(p) for d in dirs for p in sorted(Path(d).glob("*.pdf"))]
    evals = [evaluate(p) for p in pdfs]
    tax: dict = {}
    for e in evals:
        tax[e.status] = tax.get(e.status, 0) + 1
    errs = [e.event_abs_err for e in evals if e.event_abs_err is not None]

    print("=" * 86)
    print(f"Raster pipeline benchmark -- {len(pdfs)} PDFs")
    print("=" * 86)
    print(f"{'pdf':<34}{'pg':>3}{'kind':>7}{'arms':>5}  {'recon_ev':<12}{'reported_ev':<12} status")
    print("-" * 86)
    for e in evals:
        print(f"{e.pdf[:33]:<34}{str(e.page) if e.page is not None else '-':>3}{e.kind:>7}"
              f"{e.n_arms:>5}  {str(e.recon_events):<12}{str(e.reported_events):<12} {e.status}")
    print("-" * 86)
    n_succ = tax.get("success", 0)
    print(f"taxonomy: {tax}")
    print(f"success: {n_succ}/{len(pdfs)}")
    if errs:
        print(f"event-count accuracy (vs at-risk-table reported events) on {len(errs)} arms-sets: "
              f"mean |abs err| = {np.mean(errs):.1f} events; median = {np.median(errs):.1f}")
    print("=" * 86)
    return {"n_pdfs": len(pdfs), "taxonomy": tax, "success": n_succ,
            "event_abs_errs": errs, "evals": [asdict(e) for e in evals]}


if __name__ == "__main__":  # pragma: no cover
    base = Path(__file__).resolve().parent
    dirs = sys.argv[1:] or [str(base / "corpus"), str(base / "corpus_unpaywall")]
    run(dirs)
