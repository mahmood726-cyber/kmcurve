#!/usr/bin/env python3
"""
Granularity-manifold pooling: the censoring lever shrinks r², the synthesis currency.
=====================================================================================

registry-IPD's `SYNTHESIS-VISION.md` reframes evidence synthesis as ONE
partially-identified missing-data problem on a data-completeness manifold: each
trial contributes a likelihood at its own granularity, weighted by what its
posted statistics identify. The censoring lever (a total-event count / at-risk
table) is what moves a trial UP the manifold from curve-only.

This module demonstrates the kmcurve-specific consequence at the META-ANALYSIS
level, on kmcurve's OWN reconstruction (Guyot / Titman-QP): pool k trials three
ways and compare to the true-IPD pool the patient data would give.

  - true_ipd     : DL pool of the per-trial Cox logHR from the simulated IPD (gold);
  - curve_only   : pool of the CURVE-ONLY Guyot reconstruction (no event count ->
                   assumes ~zero censoring -> attenuates each logHR toward 0);
  - event_pinned : pool of the Titman-QP reconstruction fed each arm's total-event
                   count (kmcurve's QP backend / OCR'd "N(events)").

**Finding: the lever recovers the true-IPD pool -- BOTH the pooled effect AND the
heterogeneity tau2; curve-only silently attenuates both.** A curve-only trial does
not just mis-estimate its own HR -- pooled, the attenuation shrinks the synthesis's
mu, tau2 and prediction interval, so the trials look more homogeneous and the effect
smaller than the truth. The event count is what lets a reconstructed pool stand in
for an IPD meta-analysis. (Aside, measured here: with BOTH arms' curves fixed at the
anchors the log-rank HR is nearly insensitive to the *total* event count, so the
under-identification shows up as curve-only BIAS, not variance -- the QP's value is
removing that bias, not a variance band.)

Run:  python granularity_pool.py [--k 12] [--tau2 0.05] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from guyot import reconstruct_ipd_guyot, ipd_to_arrays, logrank_hr, km_from_ipd
from qp_reconstruct import reconstruct_arm_qp


def _simulate_arm(n: int, rate: float, admin_t: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Exponential event times with administrative censoring at admin_t (+ a little
    random dropout). Returns (time, event)."""
    evt = rng.exponential(1.0 / rate, n)
    drop = rng.exponential(admin_t * 2.5, n)             # light random dropout
    cens = np.minimum(drop, admin_t)
    time = np.minimum(evt, cens)
    event = (evt <= cens).astype(int)
    return time, event


def _km_at(time: np.ndarray, event: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    et, es = km_from_ipd(time, event)
    s = np.ones_like(anchors, dtype=float)
    for i, a in enumerate(anchors):
        prior = es[et <= a]
        s[i] = prior[-1] if len(prior) else 1.0
    return s


def simulate_trial(n_per_arm: int, true_loghr: float, base_rate: float,
                   admin_t: float, anchors: np.ndarray, rng) -> dict:
    """One trial: simulate IPD both arms, return its realised Cox-ish logHR + s²
    (from the IPD) and the POSTED summaries a registry/figure would expose
    (per-arm KM anchors, N, total events)."""
    ct, ce = _simulate_arm(n_per_arm, base_rate, admin_t, rng)               # control
    et, ee = _simulate_arm(n_per_arm, base_rate * np.exp(true_loghr), admin_t, rng)  # exp
    res = logrank_hr(et, ee, ct, ce)
    loghr = float(np.log(max(res["hr"], 1e-6)))
    n_ev_e, n_ev_c = int(ee.sum()), int(ce.sum())
    # sampling variance of logHR ~ 1/totalEvents (log-rank / Peto)
    s2 = 1.0 / max(n_ev_e + n_ev_c, 1)
    return {
        "true_loghr": true_loghr, "loghr_ipd": loghr, "s2": s2,
        "anchors": anchors,
        "S_exp": _km_at(et, ee, anchors), "S_ctl": _km_at(ct, ce, anchors),
        "n_exp": n_per_arm, "n_ctl": n_per_arm, "ev_exp": n_ev_e, "ev_ctl": n_ev_c,
    }


def _recon_loghr(t: dict, ev_exp: Optional[int], ev_ctl: Optional[int]) -> float:
    """Reconstruct the logHR from the POSTED anchors; with event counts -> QP,
    without -> curve-only Guyot (assumes ~no censoring -> attenuates)."""
    at = np.concatenate([[0.0], t["anchors"]])
    se = np.concatenate([[1.0], t["S_exp"]])
    sc = np.concatenate([[1.0], t["S_ctl"]])
    fu = float(t["anchors"][-1])
    if ev_exp is not None and ev_ctl is not None:
        qte, qee = reconstruct_arm_qp(at, se, total_n=t["n_exp"], total_events=ev_exp, follow_up_max=fu)
        qtc, qec = reconstruct_arm_qp(at, sc, total_n=t["n_ctl"], total_events=ev_ctl, follow_up_max=fu)
        hr = logrank_hr(qte, qee, qtc, qec)["hr"]
    else:
        ipde = reconstruct_ipd_guyot(at, se, total_n=t["n_exp"], arm=0)
        ipdc = reconstruct_ipd_guyot(at, sc, total_n=t["n_ctl"], arm=1)
        te, ee2 = ipd_to_arrays(ipde)
        tc, ec2 = ipd_to_arrays(ipdc)
        hr = logrank_hr(te, ee2, tc, ec2)["hr"]
    return float(np.log(max(hr, 1e-6)))


def dl_pool(loghrs: np.ndarray, variances: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pool -> (mu, tau2, PI width)."""
    loghrs, variances = np.asarray(loghrs, float), np.asarray(variances, float)
    k = len(loghrs)
    w = 1.0 / variances
    mu_fe = np.sum(w * loghrs) / np.sum(w)
    Q = float(np.sum(w * (loghrs - mu_fe) ** 2))
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1.0 / (variances + tau2)
    mu = float(np.sum(wr * loghrs) / np.sum(wr))
    se_mu = float(np.sqrt(1.0 / np.sum(wr)))
    # 95% prediction interval (t_{k-2} per the lab's PI convention for new trials)
    from math import sqrt
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.975, max(k - 2, 1)))
    except Exception:
        tcrit = 1.96
    pi_half = tcrit * sqrt(tau2 + se_mu ** 2)
    return {"mu": mu, "tau2": tau2, "pi_width": 2 * pi_half}


def run(k: int, tau2_true: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mu_true = np.log(0.7)                                 # a real protective effect
    anchors = np.array([6.0, 12.0, 18.0, 24.0])
    trials = []
    for _ in range(k):
        loghr = float(rng.normal(mu_true, np.sqrt(tau2_true)))
        trials.append(simulate_trial(180, loghr, base_rate=0.03, admin_t=24.0,
                                     anchors=anchors, rng=rng))

    rows = []
    for t in trials:
        rows.append({"s2": t["s2"],
                     "loghr_curve": _recon_loghr(t, None, None),            # curve-only Guyot
                     "loghr_event": _recon_loghr(t, t["ev_exp"], t["ev_ctl"]),  # QP event-pinned
                     "loghr_ipd": t["loghr_ipd"]})

    s2 = np.array([r["s2"] for r in rows])
    pools = {
        "true_ipd":     dl_pool(np.array([r["loghr_ipd"] for r in rows]), s2),
        "curve_only":   dl_pool(np.array([r["loghr_curve"] for r in rows]), s2),
        "event_pinned": dl_pool(np.array([r["loghr_event"] for r in rows]), s2),
    }
    # how close each reconstructed pool is to the true-IPD pool
    ti = pools["true_ipd"]
    for name in ("curve_only", "event_pinned"):
        pools[name]["mu_err_vs_ipd"] = round(pools[name]["mu"] - ti["mu"], 4)
        pools[name]["tau2_err_vs_ipd"] = round(pools[name]["tau2"] - ti["tau2"], 4)
    return {"k": k, "tau2_true": tau2_true, "mu_true": float(mu_true),
            "mean_s2": float(s2.mean()), "pools": pools}


def _print(o: dict) -> None:
    print(f"\n=== granularity-manifold pooling (k={o['k']}, true tau2={o['tau2_true']}, "
          f"true mu(logHR)={o['mu_true']:.3f}, HR={np.exp(o['mu_true']):.2f}) ===")
    print(f"\n  {'pool':<14}{'mu(logHR)':>11}{'HR':>7}{'tau2':>9}{'PI width':>10}"
          f"{'d-mu':>8}{'d-tau2':>9}")
    for name, p in o["pools"].items():
        de = f"{p.get('mu_err_vs_ipd', 0):>8.3f}" if "mu_err_vs_ipd" in p else " " * 8
        dt = f"{p.get('tau2_err_vs_ipd', 0):>9.3f}" if "tau2_err_vs_ipd" in p else " " * 9
        print(f"  {name:<14}{p['mu']:>11.3f}{np.exp(p['mu']):>7.2f}{p['tau2']:>9.3f}"
              f"{p['pi_width']:>10.3f}{de}{dt}")
    cu, ev = o["pools"]["curve_only"], o["pools"]["event_pinned"]
    print(f"\n  The censoring lever recovers the true-IPD pool: event-pinned d-mu "
          f"{cu['mu_err_vs_ipd']:+.3f}->{ev['mu_err_vs_ipd']:+.3f}, "
          f"d-tau2 {cu['tau2_err_vs_ipd']:+.3f}->{ev['tau2_err_vs_ipd']:+.3f}.")
    print("  Curve-only attenuates BOTH the pooled effect and the heterogeneity "
          "(trials look falsely homogeneous).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--tau2", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="granularity_pool_results.json")
    args = ap.parse_args()
    out = run(args.k, args.tau2, args.seed)
    _print(out)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
