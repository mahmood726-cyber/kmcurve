#!/usr/bin/env python3
"""
End-to-end NAR fusion on the first endpoint-matched real pair: PALOMA-3 (OS).
=============================================================================

`fusion_crossmatch.py` surfaced NCT01942135 / PMC9662922 as a fusion candidate,
but its first HR (0.42) was the trial's PFS primary glued onto an OS curve. With
endpoint-matched linkage (fusion_crossmatch._endpoint_key reading the measure
description) the curve resolves to OVERALL SURVIVAL and its true held-out ground
truth is the posted **OS HR 0.814 (0.644-1.029)** -- not the PFS 0.42.

This drives the same two-way fusion as fusion_real_trial.py (RADIANT-4), now on a
pair discovered end-to-end by the corpus cross-match, against the CORRECT OS HR.
Every input is DERIVED, not hardcoded:

  - OS curve anchors + N : registry-ipd's harvested cohort/NCT01942135.json
    (year 1/2/3 KM-estimate survival probabilities -- the ctgov-posted OS curve).
  - OS event counts      : ctgov's 'Overall Survival (OS)-Number of Participants
    Who Died' measure (the censoring info AACT/the curve lacks; kmcurve OCRs the
    identical figure at-risk "N (events)" in production).
  - ground truth OS HR   : ctgov OS hazard-ratio analysis, ENDPOINT-MATCHED to the
    OS curve via fusion_crossmatch._hr_for_endpoint (the bug fix this validates).

registry-only Guyot(anchors, N) has no censoring; FUSION QP(anchors, N, events)
adds the figure's event count. Honest scope: 3 sparse landmark anchors (yr 1/2/3)
+ posted death counts -- this validates the endpoint-matched MECHANISM on a real
discovered pair, not a dense-curve OCR (PMC9662922's figure OCR is the separate
raster_km step). OS HR 0.814 is near 1 (non-significant OS in PALOMA-3), a harder
target than a strongly-separated curve.

Run:  python fusion_paloma3.py [--cohort C:\\Projects\\registry-ipd\\cohort\\NCT01942135.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import fusion_crossmatch as X
from fusion_real_trial import run_trial, _print, _arm_curve
from guyot import reconstruct_ipd_guyot, ipd_to_arrays, logrank_hr

NCT = "NCT01942135"


def _fetch_os(nct: str) -> Tuple[Dict[str, int], Optional[dict]]:
    """(per-group OS deaths, endpoint-matched OS HR) from ctgov for `nct`."""
    import urllib.request
    req = urllib.request.Request(
        f"https://clinicaltrials.gov/api/v2/studies/{nct}",
        headers={"User-Agent": X._UA})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    oms = (d.get("resultsSection", {}).get("outcomeMeasuresModule", {})
           .get("outcomeMeasures", []))

    deaths: Dict[str, int] = {}
    for om in oms:
        if "died" in (om.get("title") or "").lower():
            groups = {g["id"]: g["title"] for g in om.get("groups", [])}
            for cl in om.get("classes", []):
                for cat in cl.get("categories", []):
                    for m in cat.get("measurements", []):
                        deaths[groups.get(m["groupId"], m["groupId"])] = int(m["value"])
            break

    # endpoint-matched OS HR: reuse the crossmatch's fixed linkage so this script
    # validates the SAME code path the cross-match now uses (not a parallel guess).
    oms_lite = [{"title": om.get("title", ""), "description": om.get("description", ""),
                 "timeFrame": om.get("timeFrame", ""),
                 "analyses": [{"paramType": a.get("paramType"), "paramValue": a.get("paramValue"),
                               "ciLowerLimit": a.get("ciLowerLimit"), "ciUpperLimit": a.get("ciUpperLimit")}
                              for a in (om.get("analyses") or [])]}
                for om in oms]
    hr = X._hr_for_endpoint(oms_lite, "os")
    return deaths, hr


def build_trial(cohort_path: Path) -> dict:
    """Corrected trial dict: registry OS anchors + ctgov OS deaths + OS HR."""
    cohort = json.loads(cohort_path.read_text())
    deaths, hr = _fetch_os(cohort["nct_id"])
    if not deaths:
        raise SystemExit("could not read OS death counts from ctgov")
    if not hr or hr.get("value") is None:
        raise SystemExit("no endpoint-matched OS HR found on ctgov")

    arms = []
    for a in cohort["arms"]:
        label = a["label"]
        if label not in deaths:
            raise SystemExit(f"no OS death count for arm {label!r}; have {list(deaths)}")
        arms.append({**a, "total_events": deaths[label]})
    return {
        "nct_id": cohort["nct_id"],
        "condition": cohort.get("condition"),
        "time_unit": cohort.get("time_unit", "months"),
        "arms": arms,
        "hr": {"value": float(hr["value"]),
               "ci_low": float(hr["ci"][0]) if hr["ci"][0] else None,
               "ci_high": float(hr["ci"][1]) if hr["ci"][1] else None,
               "favors_arm_id": "OG000"},
    }


def ocr_nar_from_figure(pdf_path: str, arm_ns: Dict[str, int]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """OCR the at-risk table off the ACTUAL figure -> per-arm (times, N-at-risk).

    Locates the caption-anchored KM figure, calibrates the x-axis (RANSAC dual-OCR,
    fail-closed), reads the numbers-at-risk band, and maps each row to an arm by
    matching its baseline count to the registry N. This is the figure-derived
    censoring information the registry curve lacks -- the last simulated link in
    the PALOMA-3 fusion (vs the ctgov death total used by build_trial)."""
    import figure_locator as FL
    from raster_km import (render_page, detect_plot_boxes, detect_plot_box,
                           auto_calibrate_axes, extract_at_risk_raster)

    cands = FL.locate_km_figures(pdf_path, require_caption=True)
    if not cands:
        raise SystemExit(f"no caption-anchored KM figure in {pdf_path}")
    c = cands[0]
    g = render_page(pdf_path, c.page_index, dpi=200)
    if c.bbox:
        sc = 200 / 72.0
        x0, t, x1, b = c.bbox
        g = g[int(t * sc):int(b * sc), int(x0 * sc):int(x1 * sc)]
    boxes = detect_plot_boxes(g) or ([detect_plot_box(g)] if detect_plot_box(g) else [])
    if not boxes:
        raise SystemExit("no plot box detected in the figure")
    box = boxes[0]
    xfit, _ = auto_calibrate_axes(g, box)          # fail-closed if unreliable
    rows = extract_at_risk_raster(g, box)
    if not rows:
        raise SystemExit("no numbers-at-risk table read from the figure")

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    used = set()
    for row in rows:
        times = np.array([xfit.value(x) for x, _ in row], float)
        vals = np.array([n for _, n in row], float)
        # map this row to the registry arm whose N is closest to the baseline count
        arm = min((a for a in arm_ns if a not in used),
                  key=lambda a: abs(arm_ns[a] - vals[0]), default=None)
        if arm is None:
            continue
        used.add(arm)
        out[arm] = (times, vals)
    return out


def run_ocr_fusion(cohort_path: Path, pdf_path: str) -> dict:
    """Fusion using the FIGURE's OCR'd number-at-risk table (Guyot+NAR), vs
    registry-only Guyot (no censoring), against the posted OS HR."""
    cohort = json.loads(cohort_path.read_text())
    _, hr = _fetch_os(cohort["nct_id"])
    posted_hr = float(hr["value"]) if hr and hr.get("value") else None
    arms = [a for a in cohort["arms"] if a.get("km_points") and a.get("N")]
    if len(arms) != 2:
        raise SystemExit(f"need 2 registry arms with km_points+N, got {len(arms)}")
    arm_ns = {a["label"]: int(a["N"]) for a in arms}
    nar = ocr_nar_from_figure(pdf_path, arm_ns)

    def _recon(arm: dict, with_nar: bool, idx: int):
        t, s = _arm_curve(arm)                       # registry exact anchors -> (time, S)
        nt = nv = None
        if with_nar and arm["label"] in nar:
            nt, nv = nar[arm["label"]]
        ipd = reconstruct_ipd_guyot(t, s, n_risk_times=nt, n_risk_values=nv,
                                    total_n=int(arm["N"]), arm=idx)
        return ipd_to_arrays(ipd)

    exp, ctl = arms[0], arms[1]
    et_r, ee_r = _recon(exp, False, 0); ct_r, ce_r = _recon(ctl, False, 1)
    hr_reg = logrank_hr(et_r, ee_r, ct_r, ce_r)["hr"]
    et_f, ee_f = _recon(exp, True, 0); ct_f, ce_f = _recon(ctl, True, 1)
    hr_fus = logrank_hr(et_f, ee_f, ct_f, ce_f)["hr"]

    def fold(h):
        return round(max(h, posted_hr) / min(h, posted_hr), 3) if (posted_hr and h > 0) else None

    return {
        "trial": cohort["nct_id"], "pdf": Path(pdf_path).name,
        "posted_hr": posted_hr, "posted_ci": [hr["ci"][0], hr["ci"][1]] if hr else [None, None],
        "nar_arms": {k: {"times": [round(x, 1) for x in v[0]],
                         "n_at_risk": [int(n) for n in v[1]]} for k, v in nar.items()},
        "registry_only_hr": round(hr_reg, 3), "registry_only_fold": fold(hr_reg),
        "ocr_nar_fusion_hr": round(hr_fus, 3), "ocr_nar_fusion_fold": fold(hr_fus),
    }


def _print_ocr(r: dict) -> None:
    print(f"\n=== figure-OCR NAR fusion: {r['trial']} ({r['pdf']}) ===")
    for arm, d in r["nar_arms"].items():
        print(f"  OCR'd at-risk [{arm}]: N0={d['n_at_risk'][0]} "
              f"at t={d['times'][0]}..{d['times'][-1]} ({len(d['n_at_risk'])} cols)")
    lo, hi = r["posted_ci"]
    print(f"\n  posted OS HR (ground truth) : {r['posted_hr']}  (95% CI {lo}-{hi})")
    print(f"  registry-only (anchors, NO censoring) : HR {r['registry_only_hr']}"
          f"  (fold {r['registry_only_fold']})")
    print(f"  FUSION (anchors + FIGURE-OCR'd NAR)   : HR {r['ocr_nar_fusion_hr']}"
          f"  (fold {r['ocr_nar_fusion_fold']})")
    inside = lo is not None and float(lo) <= r["ocr_nar_fusion_hr"] <= float(hi)
    reg_inside = lo is not None and float(lo) <= r["registry_only_hr"] <= float(hi)
    print(f"\n  fusion HR inside posted 95% CI? {inside}   |   registry-only inside? {reg_inside}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=rf"C:\Projects\registry-ipd\cohort\{NCT}.json")
    ap.add_argument("--out", default="paloma3_os_fusion.json")
    ap.add_argument("--ocr-pdf", default="corpus_pmc/PMC9662922.pdf",
                    help="OCR the at-risk table off this figure (the figure-derived "
                         "censoring); set '' to use only the ctgov-posted death total")
    args = ap.parse_args()

    trial = build_trial(Path(args.cohort))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(trial, tf)
        tmp = tf.name
    res = run_trial(Path(tmp))
    Path(tmp).unlink(missing_ok=True)

    print("\n*** endpoint-matched pair: PALOMA-3 OVERALL SURVIVAL (ground truth OS HR,"
          " NOT the PFS 0.42) ***")
    print("\n[A] posted-event fusion (QP backend, ctgov death total as the event count)")
    _print(res)

    out = {"posted_event_fusion": res}
    if args.ocr_pdf:
        ocr = run_ocr_fusion(Path(args.cohort), args.ocr_pdf)
        print("\n[B] figure-OCR fusion (no ctgov events -- the at-risk table is read off"
              " the actual figure)")
        _print_ocr(ocr)
        out["figure_ocr_fusion"] = ocr

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
