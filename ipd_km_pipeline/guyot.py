#!/usr/bin/env python3
"""
Guyot IPD reconstruction (self-contained)
=========================================

Reconstructs individual-patient data (time, event) from a digitised
survival curve plus the numbers-at-risk table, following:

    Guyot P, Ades AE, Ouwens MJNM, Welton NJ. Enhanced secondary analysis of
    survival data: reconstructing the data from published Kaplan-Meier survival
    curves. BMC Medical Research Methodology 2012, 12:9.

This is a clean in-repo port of the working implementation in the sibling
``wasserstein`` project (``improved_hr_estimation.reconstruct_ipd_guyot``),
vendored here so KMcurve has no cross-repo import dependency. The numerically
delicate event/censoring bookkeeping is reused verbatim; only packaging and the
validation helper (:func:`km_from_ipd`) are new.

Inputs are designed to plug directly into ``vector_km``: pass an arm's
calibrated ``(time, survival)`` and the at-risk row ``(nar_times, nar_values)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import interp1d

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a declared dep
    _HAVE_SCIPY = False


@dataclass
class IPDRecord:
    """Individual patient data record."""

    time: float
    event: int  # 1 = event, 0 = censored
    arm: int = 0


def _interpolate_nar(
    nar_times: np.ndarray,
    nar_values: np.ndarray,
    curve_times: np.ndarray,
    total_n: int,
) -> np.ndarray:
    """Interpolate numbers-at-risk onto the curve timepoints (monotone)."""
    nar_times = np.asarray(nar_times, dtype=float)
    nar_values = np.asarray(nar_values, dtype=float)

    for i in range(1, len(nar_values)):
        if nar_values[i] > nar_values[i - 1]:
            nar_values[i] = nar_values[i - 1]

    if _HAVE_SCIPY and len(nar_times) >= 2:
        try:
            f = interp1d(
                nar_times,
                nar_values,
                kind="linear",
                bounds_error=False,
                fill_value=(nar_values[0], nar_values[-1]),
            )
            interp = f(curve_times)
        except Exception:
            interp = np.interp(curve_times, nar_times, nar_values)
    else:
        interp = np.interp(curve_times, nar_times, nar_values)

    interp = np.clip(interp, 1, total_n)
    for i in range(1, len(interp)):
        if interp[i] > interp[i - 1]:
            interp[i] = interp[i - 1]
    return interp.astype(int)


def reconstruct_ipd_guyot(
    times: np.ndarray,
    survivals: np.ndarray,
    n_risk_times: Optional[np.ndarray] = None,
    n_risk_values: Optional[np.ndarray] = None,
    total_n: Optional[int] = None,
    arm: int = 0,
) -> List[IPDRecord]:
    """Reconstruct IPD from a KM curve via the Guyot algorithm.

    Parameters
    ----------
    times, survivals : digitised curve points (survival in [0, 1]).
    n_risk_times, n_risk_values : the at-risk table (times and counts).
    total_n : total patients; defaults to the first at-risk count.
    arm : arm label written onto each record (0/1).
    """
    times = np.asarray(times, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    survivals = survivals[sort_idx]

    # survival must be monotone non-increasing
    for i in range(1, len(survivals)):
        if survivals[i] > survivals[i - 1]:
            survivals[i] = survivals[i - 1]

    if total_n is None:
        if n_risk_values is not None and len(n_risk_values) > 0:
            total_n = int(n_risk_values[0])
        else:
            total_n = 100

    has_nar = (
        n_risk_times is not None
        and n_risk_values is not None
        and len(n_risk_times) >= 2
        and len(n_risk_values) >= 2
    )
    if has_nar:
        n_at_risk = _interpolate_nar(n_risk_times, n_risk_values, times, total_n)
    else:
        n_at_risk = np.maximum((survivals * total_n).astype(int), 1)

    ipd: List[IPDRecord] = []
    event_carry = 0.0  # fractional-event accumulator (error diffusion)
    km_recon = 1.0  # reconstructed survival so far (Guyot recursion anchor)
    for i in range(1, len(times)):
        s_curr = survivals[i]
        n_prev = n_at_risk[i - 1]
        n_curr = n_at_risk[i] if i < len(n_at_risk) else 1
        t_prev, t_curr = times[i - 1], times[i]

        if km_recon <= 0 or s_curr <= 0:
            continue

        # Guyot recursion: anchor events to the RECONSTRUCTED survival so far,
        # not the raw digitised previous value. If km_recon has drifted above
        # the target S, the conditional probability rises and the extra events
        # pull it back down -- so the reconstructed KM self-corrects onto the
        # digitised curve instead of accumulating drift.
        cond_prob = 1.0 - (s_curr / km_recon)
        cond_prob = min(max(cond_prob, 0.0), 1.0)
        event_carry += n_prev * cond_prob
        n_events = int(np.floor(event_carry))
        event_carry -= n_events
        n_events = max(0, min(n_events, n_prev - 1))
        # advance the reconstructed survival by the integer events actually placed
        if n_prev > 0:
            km_recon *= 1.0 - n_events / n_prev

        n_censored = max(0, n_prev - n_curr - n_events)

        if n_events > 0:
            for et in np.linspace(t_prev + 1e-3, t_curr - 2e-3, n_events):
                ipd.append(IPDRecord(time=float(et), event=1, arm=arm))
        if n_censored > 0:
            # place censored AFTER the interval's events (just before t_curr) so
            # they remain at risk for those events (KM/Breslow convention).
            # Interleaving them would shrink the risk set early and inflate the
            # KM drops -> systematic upward drift in the recomputed curve.
            for ct in np.linspace(t_curr - 1e-3, t_curr - 0.5e-3, n_censored):
                ipd.append(IPDRecord(time=float(ct), event=0, arm=arm))

    n_remaining = total_n - len(ipd)
    if n_remaining > 0:
        final_censored = n_remaining if has_nar else min(n_remaining, 10)
        for _ in range(final_censored):
            ipd.append(IPDRecord(time=float(times[-1]), event=0, arm=arm))

    return ipd


def ipd_to_arrays(ipd: List[IPDRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert IPD records to sorted (time, event) arrays."""
    if not ipd:
        return np.empty(0), np.empty(0, dtype=int)
    t = np.array([r.time for r in ipd], dtype=float)
    e = np.array([r.event for r in ipd], dtype=int)
    order = np.argsort(t)
    return t[order], e[order]


def km_from_ipd(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate from IPD; returns (event_times, survival).

    Used to validate that reconstructed IPD reproduces the input curve.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.size == 0:
        return np.empty(0), np.empty(0)
    uniq = np.unique(time[event == 1])
    surv = []
    s = 1.0
    for t in uniq:
        n_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if n_risk > 0:
            s *= 1 - d / n_risk
        surv.append(s)
    return uniq, np.array(surv)


def logrank_hr(
    t1: np.ndarray,
    e1: np.ndarray,
    t0: np.ndarray,
    e0: np.ndarray,
) -> dict:
    """Log-rank-based hazard ratio (arm1 vs arm0) and chi-square test.

    Uses the Mantel-Haenszel O-E/V estimator: HR = exp((O1 - E1) / V). Returns
    {'hr', 'logrank_chi2', 'o1', 'e1', 'v'}. This is the standard summary HR
    from reconstructed IPD; for crossing hazards a single HR is not meaningful.
    """
    t1 = np.asarray(t1, float); e1 = np.asarray(e1, int)
    t0 = np.asarray(t0, float); e0 = np.asarray(e0, int)
    event_times = np.unique(np.concatenate([t1[e1 == 1], t0[e0 == 1]]))
    o1 = e_1 = v = 0.0
    for t in event_times:
        n1 = int(np.sum(t1 >= t)); n0 = int(np.sum(t0 >= t))
        d1 = int(np.sum((t1 == t) & (e1 == 1)))
        d0 = int(np.sum((t0 == t) & (e0 == 1)))
        n = n1 + n0; d = d1 + d0
        if n < 2 or d == 0:
            continue
        o1 += d1
        e_1 += d * n1 / n
        v += d * (n1 * n0) * (n - d) / (n * n * (n - 1))
    hr = float(np.exp((o1 - e_1) / v)) if v > 0 else float("nan")
    chi2 = float((o1 - e_1) ** 2 / v) if v > 0 else float("nan")
    return {"hr": hr, "logrank_chi2": chi2, "o1": o1, "e1": e_1, "v": v}


def reconstruct_arm(
    time: np.ndarray,
    survival: np.ndarray,
    nar_times: Optional[np.ndarray] = None,
    nar_values: Optional[np.ndarray] = None,
    total_n: Optional[int] = None,
    arm: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: reconstruct one arm and return (time, event) arrays."""
    ipd = reconstruct_ipd_guyot(
        np.asarray(time, dtype=float),
        np.asarray(survival, dtype=float),
        n_risk_times=nar_times,
        n_risk_values=nar_values,
        total_n=total_n,
        arm=arm,
    )
    return ipd_to_arrays(ipd)
