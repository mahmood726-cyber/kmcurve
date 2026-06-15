"""Truth-recovery harness: feed the known-truth survival DGP into the repo's
own Guyot reconstruction and measure recovery of median, events, and log-HR.

Also runs a Monte-Carlo HR-CI coverage check (the failure mode that killed the
sibling kmdigitizer: events placed at interval ENDPOINTS biased log-HR toward
null and drove HR-CI coverage to ~0.03). kmcurve.guyot places events spread
ACROSS each interval (np.linspace(t_prev+e, t_curr-2e, n_events)), so we expect
coverage near nominal.
"""
from __future__ import annotations

import numpy as np

from dgp_survival import make_two_arm
from engine import reconstruct_arm, km_from_ipd, logrank_hr


def median_from_km(times: np.ndarray, surv: np.ndarray) -> float:
    """First event time at which S(t) <= 0.5."""
    below = np.where(surv <= 0.5)[0]
    if below.size == 0:
        return float("nan")
    return float(times[below[0]])


def recover_once(seed: int, true_hr: float = 0.65, rate0: float = 0.10):
    arm0, arm1, true_log_hr = make_two_arm(
        seed=seed, rate0=rate0, true_hr=true_hr
    )

    t0, e0 = reconstruct_arm(
        arm0.curve_times, arm0.curve_surv,
        nar_times=arm0.nar_times, nar_values=arm0.nar_values,
        total_n=arm0.n, arm=0,
    )
    t1, e1 = reconstruct_arm(
        arm1.curve_times, arm1.curve_surv,
        nar_times=arm1.nar_times, nar_values=arm1.nar_values,
        total_n=arm1.n, arm=1,
    )

    km0_t, km0_s = km_from_ipd(t0, e0)
    km1_t, km1_s = km_from_ipd(t1, e1)
    med0 = median_from_km(km0_t, km0_s)
    med1 = median_from_km(km1_t, km1_s)

    lr = logrank_hr(t1, e1, t0, e0)
    log_hr_hat = float(np.log(lr["hr"])) if lr["hr"] == lr["hr"] and lr["hr"] > 0 else float("nan")
    # log-HR SE from the log-rank variance: Var(log HR) ~ 1/V
    se_log_hr = float(1.0 / np.sqrt(lr["v"])) if lr["v"] > 0 else float("nan")

    return {
        "true_log_hr": true_log_hr,
        "log_hr_hat": log_hr_hat,
        "se_log_hr": se_log_hr,
        "true_med0": arm0.true_median,
        "true_med1": arm1.true_median,
        "rec_med0": med0,
        "rec_med1": med1,
        "true_events0": arm0.n_events,
        "true_events1": arm1.n_events,
        "rec_events0": int(np.sum(e0 == 1)),
        "rec_events1": int(np.sum(e1 == 1)),
    }


def coverage_study(n_rep: int = 300, true_hr: float = 0.65):
    true_log_hr = np.log(true_hr)
    covered = 0
    log_hr_hats = []
    events_rel_err = []
    med_abs_err = []
    valid = 0
    for r in range(n_rep):
        res = recover_once(seed=1000 + r, true_hr=true_hr)
        lh, se = res["log_hr_hat"], res["se_log_hr"]
        if not (np.isfinite(lh) and np.isfinite(se) and se > 0):
            continue
        valid += 1
        log_hr_hats.append(lh)
        lo, hi = lh - 1.96 * se, lh + 1.96 * se
        if lo <= true_log_hr <= hi:
            covered += 1
        # event recovery (both arms pooled)
        te = res["true_events0"] + res["true_events1"]
        re = res["rec_events0"] + res["rec_events1"]
        if te > 0:
            events_rel_err.append(abs(re - te) / te)
        # median recovery (control arm; treatment median often beyond censor)
        if np.isfinite(res["rec_med0"]):
            med_abs_err.append(abs(res["rec_med0"] - res["true_med0"]) / res["true_med0"])
    return {
        "n_valid": valid,
        "coverage": covered / valid if valid else float("nan"),
        "mean_log_hr_hat": float(np.mean(log_hr_hats)) if log_hr_hats else float("nan"),
        "true_log_hr": true_log_hr,
        "bias_log_hr": float(np.mean(log_hr_hats) - true_log_hr) if log_hr_hats else float("nan"),
        "mean_event_rel_err": float(np.mean(events_rel_err)) if events_rel_err else float("nan"),
        "mean_median_rel_err": float(np.mean(med_abs_err)) if med_abs_err else float("nan"),
    }


if __name__ == "__main__":
    one = recover_once(seed=42)
    print("=== Single recovery (seed=42, true HR=0.65) ===")
    for k, v in one.items():
        print(f"  {k:14s} {v}")
    print()
    cov = coverage_study(n_rep=300, true_hr=0.65)
    print("=== Monte-Carlo (300 reps, true HR=0.65) ===")
    for k, v in cov.items():
        print(f"  {k:20s} {v}")
