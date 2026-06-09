#!/usr/bin/env python3
"""
Competitor benchmark: vector calibration vs the manual-raster-digitizer class
=============================================================================

Every widely used KM-digitization tool -- WebPlotDigitizer (Rohatgi),
IPDfromKM (Liu 2021), SurvdigitizeR (Liu 2023), Engauge, DigitizeIt -- shares
one mechanism this extractor replaces: the user (or an OCR step) fixes the axis
calibration from a RASTER image, either by clicking reference points or by
automated tick OCR. KMcurve instead reads the exact tick coordinates from the
PDF vector layer.

What this script does (honest scope):
- It does NOT run those external tools (browser/manual or R, raster input;
  not runnable headless here). Instead it isolates the variable that differs:
  the CALIBRATION method. Curve-pixel extraction is held IDENTICAL (KMcurve's
  exact pixels) for both arms of the comparison, so any HR difference is due
  to calibration alone -- the fairest possible comparison.
- Vector path: exact tick coordinates -> zero calibration error, deterministic.
- Manual-raster class: 2 reference clicks per axis on the rasterised figure at
  a typical DPI, with Gaussian pixel jitter modelling human/cursor precision.
  Monte-Carlo over the jitter gives the HR distribution those tools incur.

External cross-validation (cited, not re-run): the sibling `wasserstein`
raster+OCR pipeline reports 90% concordance (36/40 trials within published 95%
CI) and median relative HR error ~1.2-2.6% vs ~10.4-36% for R IPDfromKM on
gold-standard AF trials (figures vary across that project's own docs; quoted
as a range). KMcurve's vector path removes the OCR/colour dependency those
pipelines still carry.
"""

from __future__ import annotations

import numpy as np

import vector_km as V
from guyot import logrank_hr, reconstruct_arm
from manual_calibration import two_point_axis


def _panelA_bundle(pdf_path: str, page_index: int = 7):
    """Return per-arm (time, value, nar_times, nar_values) + exact fits."""
    panels = V.extract_km_from_pdf(pdf_path, page_index, monotone="increasing")
    pa = panels[0]
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        rows = V.extract_at_risk(page, pa.panel)
    nar = {}
    for r in rows:
        key = "standard" if "standard" in r.label.lower() else (
            "intensive" if "intensive" in r.label.lower() else r.label.lower())
        nar[key] = (r.times, r.counts)
    arms = []
    for a in pa.arms:
        key = "standard" if "standard" in a.identity.lower() else "intensive"
        nt, nv = nar.get(key, (None, None))
        arms.append({"id": a.identity or a.label, "key": key,
                     "time": a.time, "value": a.value, "nt": nt, "nv": nv})
    return pa, arms


def _hr_from_arms(arm_tv, value_scale=100.0):
    """arm_tv: list of dicts with time, value(CI%), nt, nv. Returns intensive-vs-standard HR."""
    ipd = {}
    for a in arm_tv:
        surv = np.clip(1.0 - np.asarray(a["value"]) / value_scale, 0.0, 1.0)
        total_n = int(a["nv"][0]) if a["nv"] is not None else None
        t, e = reconstruct_arm(a["time"], surv, nar_times=a["nt"], nar_values=a["nv"],
                               total_n=total_n)
        ipd[a["key"]] = (t, e)
    inten, std = ipd["intensive"], ipd["standard"]
    return logrank_hr(inten[0], inten[1], std[0], std[1])["hr"]


def run(pdf_path: str, page_index: int = 7, n_draws: int = 200, seed: int = 0):
    PUB_HR, PUB_CI = 0.90, (0.82, 0.98)
    pa, arms = _panelA_bundle(pdf_path, page_index)
    xf, yf = pa.x_fit, pa.y_fit

    # ---- Vector path: exact, deterministic ----------------------------------
    hr_vec = _hr_from_arms(arms)

    # invert exact calibration to recover the shared curve PIXELS
    pix = []
    for a in arms:
        px = (np.asarray(a["time"]) - xf.intercept) / xf.slope
        py = (np.asarray(a["value"]) - yf.intercept) / yf.slope
        pix.append({**a, "px": px, "py": py})

    # true reference-tick pixel positions (what a user would click)
    xref_vals = (0.0, 60.0)
    yref_vals = (0.0, 25.0)
    xref_px = [(v - xf.intercept) / xf.slope for v in xref_vals]
    yref_px = [(v - yf.intercept) / yf.slope for v in yref_vals]

    rng = np.random.default_rng(seed)
    print("=" * 72)
    print("Competitor benchmark -- calibration method, ADVANCE Panel A")
    print("=" * 72)
    print(f"\nVECTOR PATH (KMcurve): HR = {hr_vec:.3f}  (exact, deterministic, 0 manual clicks)")
    print(f"Published:             HR = {PUB_HR}  (95% CI {PUB_CI[0]}-{PUB_CI[1]})\n")

    print(f"MANUAL-RASTER CLASS (WebPlotDigitizer / IPDfromKM / Engauge-equivalent)")
    print(f"  Monte-Carlo n={n_draws}; 2 ref clicks/axis on rasterised figure; Gaussian jitter.\n")
    print(f"  {'DPI':>4} {'jitter':>7} | {'HR mean':>8} {'HR sd':>7} {'HR range':>15} "
          f"{'calib RMSE':>11} {'% in CI':>8}")
    print("  " + "-" * 66)

    results = {"hr_vector": float(hr_vec), "manual": []}
    for dpi in (150, 300):
        for sigma_px in (1.0, 2.0):
            sigma_pt = sigma_px * 72.0 / dpi
            hrs, calib_err = [], []
            for _ in range(n_draws):
                xpx_j = [p + rng.normal(0, sigma_pt) for p in xref_px]
                ypx_j = [p + rng.normal(0, sigma_pt) for p in yref_px]
                xfit_j = two_point_axis(xpx_j[0], xref_vals[0], xpx_j[1], xref_vals[1])
                yfit_j = two_point_axis(ypx_j[0], yref_vals[0], ypx_j[1], yref_vals[1])
                jarms = []
                for a in pix:
                    t2 = xfit_j.slope * a["px"] + xfit_j.intercept
                    v2 = yfit_j.slope * a["py"] + yfit_j.intercept
                    v2 = np.maximum.accumulate(np.clip(v2, 0, 100))
                    jarms.append({**a, "time": t2, "value": v2})
                hrs.append(_hr_from_arms(jarms))
                # calibration RMSE on the curve (value units) vs exact
                ve = np.concatenate([a["value"] for a in arms])
                vj = np.concatenate([a["value"] for a in jarms])
                calib_err.append(np.sqrt(np.mean((ve - vj) ** 2)))
            hrs = np.array(hrs)
            in_ci = np.mean((hrs >= PUB_CI[0]) & (hrs <= PUB_CI[1])) * 100
            row = {"dpi": dpi, "jitter_px": sigma_px, "hr_mean": float(hrs.mean()),
                   "hr_sd": float(hrs.std()), "hr_min": float(hrs.min()),
                   "hr_max": float(hrs.max()), "calib_rmse": float(np.mean(calib_err)),
                   "pct_in_ci": float(in_ci)}
            results["manual"].append(row)
            print(f"  {dpi:>4} {sigma_px:>5.0f}px | {hrs.mean():>8.3f} {hrs.std():>7.3f} "
                  f"[{hrs.min():.3f},{hrs.max():.3f}] {np.mean(calib_err):>10.3f}% {in_ci:>7.0f}%")

    # ---- Capability matrix (grounded in documented tool behaviour) ----------
    print("\n" + "=" * 72)
    print("CAPABILITY COMPARISON (documented behaviour; not a runtime measurement)")
    print("=" * 72)
    rows = [
        ("Tool", "Input", "Axis calib", "Manual clicks", "At-risk", "Crossing"),
        ("KMcurve (vector)", "PDF vector", "exact (R2=1.0)", "0", "auto", "continuity"),
        ("WebPlotDigitizer", "raster img", "2-4 manual clicks", ">=4 + trace", "manual", "manual"),
        ("IPDfromKM (R)", "raster pts", "manual pre-digitize", ">=4 (external)", "manual entry", "n/a"),
        ("SurvdigitizeR (R)", "raster img", "OCR/auto", "few", "OCR", "colour"),
        ("Engauge/DigitizeIt", "raster img", "2-4 manual clicks", ">=4 + trace", "n/a", "manual"),
        ("wasserstein (sibling)", "raster+OCR", "OCR", "0 (auto)", "OCR", "colour"),
    ]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for ri, r in enumerate(rows):
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
        if ri == 0:
            print("  " + "  ".join("-" * w[i] for i in range(len(w))))
    print("\nNotes: KMcurve's vector path needs zero manual clicks and has no OCR/")
    print("colour dependency. Raster tools that calibrate from manual clicks incur")
    print("the HR variance quantified above; the legacy automated-OCR approach in")
    print("this same repo scored 0% on axis calibration (OCR_INVESTIGATION_RESULTS.md).")
    print("=" * 72)
    return results


if __name__ == "__main__":  # pragma: no cover
    import sys

    try:
        from project_paths import sample_pdf_path

        default_pdf = str(sample_pdf_path())
    except Exception:
        default_pdf = None
    pdf = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    if not pdf:
        print("Usage: python benchmark_competitors.py <pdf> [page_index]")
        sys.exit(1)
    run(pdf, page)
