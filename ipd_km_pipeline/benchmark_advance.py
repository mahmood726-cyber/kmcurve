#!/usr/bin/env python3
"""
Benchmark: vector KM extractor + Guyot IPD vs published ADVANCE ground truth
===========================================================================

Quantifies the full chain on the bundled fixture (NEJMoa0802987, Figure 3) and
compares against values reported in the ADVANCE paper (NEJM 2008;358:2560-72):

* Panel A "Combined major macrovascular and microvascular events":
  - total events ~ 2125
  - HR (intensive vs standard) ~ 0.90 (95% CI 0.82-0.98), log-rank P = 0.01
* Numbers at risk (from the figure's own table) used as the recovery check.

Run:  python ipd_km_pipeline/benchmark_advance.py
"""

from __future__ import annotations

import numpy as np

import vector_km as V
from guyot import km_from_ipd, logrank_hr

# Ground truth transcribed from ADVANCE Figure 3 / text.
ADVANCE = {
    "at_risk_panelA": {
        "Intensive": [5570, 5457, 5369, 5256, 5100, 4957, 4867, 4756, 4599, 4044, 1883, 447],
        "Standard": [5569, 5448, 5342, 5240, 5065, 4903, 4808, 4703, 4545, 3992, 1921, 470],
    },
    "panelA_total_events": 2125,
    "panelA_hr": 0.90,
    "panelA_hr_ci": (0.82, 0.98),
}


def _fmt(x, n=2):
    return f"{x:.{n}f}"


def run(pdf_path: str, page_index: int = 7) -> dict:
    panels = V.extract_km_from_pdf(pdf_path, page_index, monotone="increasing")
    ipd_panels = V.pdf_to_ipd(pdf_path, page_index)
    report: dict = {"panels": []}

    print("=" * 72)
    print("KMcurve vector extractor -- ADVANCE benchmark")
    print("=" * 72)

    # ---- 1. Calibration accuracy --------------------------------------------
    print("\n[1] Axis calibration (exact, OCR-free)")
    cal_ok = True
    for pr in panels:
        xr, yr = pr.x_fit.r2, pr.y_fit.r2
        cal_ok &= xr > 0.999 and yr > 0.999
        print(f"  Panel {pr.panel.index}: x R2={_fmt(xr,6)} ({pr.x_fit.n_ticks} ticks)"
              f"  y R2={_fmt(yr,6)} ({pr.y_fit.n_ticks} ticks)")
    report["calibration_all_exact"] = bool(cal_ok)

    # ---- 2. At-risk recovery vs the figure's table --------------------------
    print("\n[2] Numbers-at-risk recovery (Panel A) vs figure")
    pa_ipd = ipd_panels[0]
    nar_by_label = {}
    for r in pa_ipd.at_risk:
        key = "Standard" if "standard" in r.label.lower() else (
            "Intensive" if "intensive" in r.label.lower() else r.label)
        nar_by_label[key] = [int(v) for v in r.counts]
    nar_exact = True
    for arm, truth in ADVANCE["at_risk_panelA"].items():
        got = nar_by_label.get(arm, [])
        match = got == truth
        nar_exact &= match
        print(f"  {arm:10}: {'EXACT MATCH' if match else 'MISMATCH'}  ({len(got)}/{len(truth)} cols)")
        if not match:
            print(f"      got  = {got}")
            print(f"      true = {truth}")
    report["at_risk_exact"] = bool(nar_exact)

    # ---- 3. Curve fidelity: reconstructed KM vs digitised curve -------------
    print("\n[3] Curve fidelity (reconstructed-IPD KM vs digitised CI), Panel A")
    km_arms = {(a.identity or a.label): a for a in panels[0].arms}
    drifts, rmses = [], []
    for a in pa_ipd.arms:
        et, es = km_from_ipd(a.time, a.event)
        recon_ci = (1 - es[-1]) * 100 if es.size else 0.0
        in_arm = km_arms[a.label]
        input_ci = in_arm.value[-1]
        drift = recon_ci - input_ci
        drifts.append(abs(drift))
        # full-curve RMSE: reconstructed CI sampled at every digitised time
        recon_ci_at = (1 - np.interp(in_arm.time, et, es, left=1.0)) * 100
        rmse = float(np.sqrt(np.mean((recon_ci_at - in_arm.value) ** 2)))
        rmses.append(rmse)
        print(f"  {a.label:18}: digitised={_fmt(input_ci)}%  reconstructed={_fmt(recon_ci)}%"
              f"  endpoint drift={drift:+.2f}%  full-curve RMSE={_fmt(rmse)}%")
    report["max_ci_drift"] = float(max(drifts)) if drifts else None
    report["max_curve_rmse"] = float(max(rmses)) if rmses else None

    # ---- 4. Event counts vs published ---------------------------------------
    print("\n[4] Event counts (Panel A) vs published")
    by = {}
    for a in pa_ipd.arms:
        key = "Standard" if "standard" in a.label.lower() else "Intensive"
        by[key] = a
    total_events = sum(a.n_events for a in pa_ipd.arms)
    err = 100.0 * (total_events - ADVANCE["panelA_total_events"]) / ADVANCE["panelA_total_events"]
    print(f"  Standard  events = {by['Standard'].n_events}  (N={by['Standard'].total_n})")
    print(f"  Intensive events = {by['Intensive'].n_events}  (N={by['Intensive'].total_n})")
    print(f"  total = {total_events}  vs published {ADVANCE['panelA_total_events']}  ({err:+.1f}%)")
    report["total_events"] = int(total_events)
    report["total_events_pct_err"] = float(err)

    # ---- 5. Hazard ratio vs published ---------------------------------------
    print("\n[5] Hazard ratio (intensive vs standard), Panel A")
    inten, std = by["Intensive"], by["Standard"]
    hr = logrank_hr(inten.time, inten.event, std.time, std.event)
    lo, hi = ADVANCE["panelA_hr_ci"]
    in_ci = lo <= hr["hr"] <= hi
    print(f"  reconstructed HR = {_fmt(hr['hr'])}  (log-rank chi2={_fmt(hr['logrank_chi2'],1)})")
    print(f"  published HR     = {ADVANCE['panelA_hr']}  (95% CI {lo}-{hi})")
    print(f"  -> reconstructed HR {'WITHIN' if in_ci else 'OUTSIDE'} published 95% CI")
    report["hr"] = float(hr["hr"])
    report["hr_in_published_ci"] = bool(in_ci)

    # ---- 6. Separation confidence -------------------------------------------
    print("\n[6] Arm-separation confidence (low => near-overlapping, unreliable)")
    for pr in panels:
        print(f"  Panel {pr.panel.index}: {_fmt(pr.separation_confidence)}"
              f"  arms={[a.identity or a.label for a in pr.arms]}")
    report["separation_confidence"] = [pr.separation_confidence for pr in panels]

    print("\n" + "=" * 72)
    print("SUMMARY:",
          f"calib_exact={report['calibration_all_exact']}",
          f"at_risk_exact={report['at_risk_exact']}",
          f"events={report['total_events']}({report['total_events_pct_err']:+.1f}%)",
          f"HR={_fmt(report['hr'])}(in_CI={report['hr_in_published_ci']})",
          f"max_drift={_fmt(report['max_ci_drift'])}%")
    print("=" * 72)
    return report


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
        print("Usage: python benchmark_advance.py <pdf> [page_index]")
        sys.exit(1)
    run(pdf, page)
