#!/usr/bin/env python3
"""
NAR fusion on a REAL matched trial: AACT curve anchors + figure event counts.
=============================================================================

The gold-standard fusion experiment (realipd_benchmark.py --fusion) showed, on
42 synthetic-render datasets, that registry-exact curve anchors + a figure's
at-risk/event information beats either source alone. This drives the SAME fusion
on a REAL trial present in BOTH worlds, validated against the trial's PUBLISHED
hazard ratio:

  NCT01524783 -- RADIANT-4 (everolimus vs placebo, advanced NET), Lancet 2016.

- registry side: registry-ipd's harvested AACT record (NCT01524783.json) gives
  the EXACT posted KM-estimate anchors per arm + N. AACT posts NO number-at-risk.
- figure side: a published KM figure prints "N (events)" -- the total events per
  arm -- which kmcurve OCRs (raster_km.at_risk_reported_events). AACT lacks this.
- ground truth: the registry-posted Cox HR 0.48 (95% CI 0.35-0.67), held out.

We reconstruct pseudo-IPD two ways from the SAME exact anchors and compare the
log-rank HR to the posted 0.48:

  registry-only : Guyot(anchors, N)            -- no censoring info  (AACT today)
  FUSION        : QP(anchors, N, total_events) -- + the figure's event count

registry-ipd's own VALIDATION.md documents this trial as the worked example of
the identifiability trap: curve-only HR 0.68 (outside the registry CI) vs
censoring-informed 0.47 vs posted 0.48. This reproduces it as a fusion of the
two projects' real data.

Honest scope: the event counts (107/77) here are taken from the AACT-harvested
JSON (registry-ipd derived them from participant-flow drop_withdrawals). In full
production kmcurve OCRs the identical "N (events)" totals from the figure's
at-risk table; we did not OCR the RADIANT-4 figure itself (the Lancet primary is
not open-access here). So this validates the fusion MECHANISM and the gain on a
real trial with a real posted HR; the figure-OCR link is exercised separately by
the corpus benchmark and test_realipd_benchmark.py.

Run:  python fusion_real_trial.py [--trial C:\\Projects\\registry-ipd\\NCT01524783.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from guyot import reconstruct_arm, logrank_hr
from qp_reconstruct import reconstruct_arm_qp


def _arm_curve(arm: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Per-arm exact KM anchors -> (time, survival) with a (0, 1) origin."""
    km = sorted((p for p in (arm.get("km_points") or []) if p.get("t") is not None),
                key=lambda p: p["t"])
    t = [0.0] + [float(p["t"]) for p in km]
    s = [1.0] + [float(p["S"]) for p in km]
    s = np.minimum.accumulate(np.clip(s, 0.0, 1.0))   # enforce non-increasing
    return np.asarray(t, float), np.asarray(s, float)


def _dedup_arms(trial: dict) -> List[dict]:
    """The harvested JSON repeats arms; keep the first occurrence per label that
    carries km_points + N."""
    out, seen = [], set()
    for a in trial["arms"]:
        lbl = a.get("label")
        if lbl in seen:
            continue
        if a.get("km_points") and a.get("N"):
            out.append(a); seen.add(lbl)
    return out


def run_trial(trial_path: Path) -> dict:
    trial = json.loads(trial_path.read_text())
    arms = _dedup_arms(trial)
    if len(arms) != 2:
        raise ValueError(f"expected 2 arms with km_points+N, got {len(arms)}: "
                         f"{[a.get('label') for a in arms]}")
    posted = trial.get("hr") or {}
    posted_hr = posted.get("value")
    # favors_arm_id OG000 == the experimental arm; order (exp, ctl) so HR<1 = favors exp.
    exp, ctl = arms[0], arms[1]   # everolimus first in this record
    et, es = _arm_curve(exp)
    ct, cs = _arm_curve(ctl)
    n_exp, n_ctl = int(exp["N"]), int(ctl["N"])
    e_exp, e_ctl = int(exp["total_events"]), int(ctl["total_events"])
    fu = max(et[-1], ct[-1])

    # registry-only: Guyot from exact anchors, NO censoring information.
    rt_e, re_e = reconstruct_arm(et, es, total_n=n_exp)
    rt_c, re_c = reconstruct_arm(ct, cs, total_n=n_ctl)
    hr_reg = logrank_hr(rt_e, re_e, rt_c, re_c)["hr"]

    # FUSION: QP from the SAME exact anchors + the figure's event counts.
    qt_e, qe_e = reconstruct_arm_qp(et, es, total_n=n_exp, total_events=e_exp, follow_up_max=fu)
    qt_c, qe_c = reconstruct_arm_qp(ct, cs, total_n=n_ctl, total_events=e_ctl, follow_up_max=fu)
    hr_fuse = logrank_hr(qt_e, qe_e, qt_c, qe_c)["hr"]

    def fold(h):
        return round(max(h, posted_hr) / min(h, posted_hr), 3) if (posted_hr and h > 0) else None

    return {
        "trial": trial["nct_id"],
        "condition": trial.get("condition"),
        "exp": exp["label"], "ctl": ctl["label"],
        "n_exp": n_exp, "n_ctl": n_ctl, "events_exp": e_exp, "events_ctl": e_ctl,
        "posted_hr": posted_hr, "posted_ci": [posted.get("ci_low"), posted.get("ci_high")],
        "registry_only_hr": round(hr_reg, 3), "registry_only_fold": fold(hr_reg),
        "fusion_hr": round(hr_fuse, 3), "fusion_fold": fold(hr_fuse),
    }


def _print(r: dict) -> None:
    print(f"\n=== NAR fusion on a real trial: {r['trial']} ===")
    print(f"  {r['condition']}")
    print(f"  exp: {r['exp']}  (N={r['n_exp']}, events={r['events_exp']})")
    print(f"  ctl: {r['ctl']}  (N={r['n_ctl']}, events={r['events_ctl']})")
    lo, hi = r["posted_ci"]
    print(f"\n  posted Cox HR (ground truth) : {r['posted_hr']}  (95% CI {lo}-{hi})")
    print(f"  registry-only (anchors, NO censoring) : HR {r['registry_only_hr']}"
          f"  (fold vs posted {r['registry_only_fold']})")
    print(f"  FUSION (anchors + figure events)      : HR {r['fusion_hr']}"
          f"  (fold vs posted {r['fusion_fold']})")
    inside = (r["posted_ci"][0] is not None
              and r["posted_ci"][0] <= r["fusion_hr"] <= r["posted_ci"][1])
    reg_inside = (r["posted_ci"][0] is not None
                  and r["posted_ci"][0] <= r["registry_only_hr"] <= r["posted_ci"][1])
    print(f"\n  fusion HR inside the posted 95% CI? {inside}"
          f"   |   registry-only inside? {reg_inside}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default=r"C:\Projects\registry-ipd\NCT01524783.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run_trial(Path(args.trial))
    _print(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")
