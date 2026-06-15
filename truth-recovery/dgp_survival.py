"""Standalone seeded known-truth survival DGP for the kmcurve Guyot harness.

We generate two-arm exponential survival with a KNOWN true hazard ratio, then
render each arm down to exactly the information a digitiser would recover from a
published figure:

  * a digitised survival curve sampled on a time grid  (time, S(t))
  * a numbers-at-risk table at a handful of report times

We then feed ONLY that figure-level information into the repo's own Guyot
reconstruction and measure how well it recovers:

  * the true arm medians           (closed form: ln2 / rate)
  * the true number of events
  * the true log hazard ratio       (= log(rate1 / rate0))

Everything is closed-form / deterministic given the seed; no repo code is used
here, so this DGP is an independent oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ArmTruth:
    rate: float          # exponential hazard
    n: int               # patients enrolled
    admin_censor: float  # administrative censoring time
    true_median: float   # ln2 / rate
    # figure-level recoverables:
    curve_times: np.ndarray
    curve_surv: np.ndarray
    nar_times: np.ndarray
    nar_values: np.ndarray
    # full-truth IPD (oracle only):
    ipd_time: np.ndarray
    ipd_event: np.ndarray

    @property
    def n_events(self) -> int:
        return int(np.sum(self.ipd_event == 1))


def _km_curve(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier S(t) evaluated on ``grid`` (right-continuous step)."""
    uniq = np.unique(time[event == 1])
    s = 1.0
    surv_at = {}
    for t in uniq:
        n_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & (event == 1)))
        if n_risk > 0:
            s *= 1.0 - d / n_risk
        surv_at[t] = s
    out = np.ones_like(grid, dtype=float)
    keys = np.array(sorted(surv_at.keys()))
    for i, g in enumerate(grid):
        prior = keys[keys <= g]
        out[i] = surv_at[prior[-1]] if prior.size else 1.0
    return out


def simulate_arm(
    rate: float,
    n: int,
    admin_censor: float,
    grid: np.ndarray,
    nar_times: np.ndarray,
    rng: np.random.Generator,
) -> ArmTruth:
    raw = rng.exponential(scale=1.0 / rate, size=n)
    time = np.minimum(raw, admin_censor)
    event = (raw <= admin_censor).astype(int)

    surv = _km_curve(time, event, grid)
    nar_values = np.array([int(np.sum(time >= t)) for t in nar_times], dtype=int)

    return ArmTruth(
        rate=rate,
        n=n,
        admin_censor=admin_censor,
        true_median=float(np.log(2.0) / rate),
        curve_times=grid.copy(),
        curve_surv=surv,
        nar_times=nar_times.copy(),
        nar_values=nar_values,
        ipd_time=time,
        ipd_event=event,
    )


def make_two_arm(
    seed: int = 20260615,
    n0: int = 300,
    n1: int = 300,
    rate0: float = 0.10,
    true_hr: float = 0.65,
    admin_censor: float = 36.0,
) -> Tuple[ArmTruth, ArmTruth, float]:
    """Two exponential arms with a KNOWN true HR = rate1/rate0.

    Returns (control_arm, treatment_arm, true_log_hr).
    """
    rng = np.random.default_rng(seed)
    rate1 = rate0 * true_hr
    grid = np.linspace(0.0, admin_censor, 73)         # ~0.5-month figure sampling
    nar_times = np.array([0, 6, 12, 18, 24, 30, 36], dtype=float)

    arm0 = simulate_arm(rate0, n0, admin_censor, grid, nar_times, rng)
    arm1 = simulate_arm(rate1, n1, admin_censor, grid, nar_times, rng)
    return arm0, arm1, float(np.log(true_hr))
