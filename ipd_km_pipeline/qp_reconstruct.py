#!/usr/bin/env python3
"""
Titman-2026 quadratic-program KM reconstruction (events-informed backend).
==========================================================================

A Python port of registry-ipd's ``reconstructArmQP`` (src/engine.js). Where
Guyot assumes constant censoring within each interval, the Titman QP uses the
posted **total event count** as a linear constraint and resolves the leftover
censoring degree-of-freedom by the minimum-norm convex program

    min ½‖c‖²   s.t.  E(c) = E,  c ≥ 0

which has a closed form. On registry-ipd's 51-dataset gold standard the QP beats
Guyot (HR fold-error ~1.05 vs ~1.14); this module makes it a drop-in alternative
reconstruction backend for kmcurve, whose OCR'd at-risk table supplies the
"N (events)" the QP needs. Degrades to curve-only (Guyot-like) when no event
count is given.

Mechanism: on the cumulative-hazard scale the curve fixes the per-interval
hazards h_k, so events are d_k = h_k·n_k and the at-risk recursion
n_{k+1} = n_k(1-h_k) - c_k is LINEAR in the unknown censoring counts c_k; the
total-event count E is the linear constraint E = Σ h_k n_k. Closed form:
c_k = max(0, λ A_k),  A_k = ∂E/∂c_k,  λ = (E - E0)/Σ A_k²,  E0 = N(1 - S_K).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _monotone_decreasing(s: np.ndarray) -> np.ndarray:
    """Survival must be non-increasing (cheap PAVA-equivalent for our inputs,
    which arrive already near-monotone from the pipeline's accumulate step)."""
    return np.minimum.accumulate(np.clip(s, 0.0, 1.0))


def reconstruct_arm_qp(
    times: np.ndarray,
    survival: np.ndarray,
    total_n: int,
    total_events: Optional[int] = None,
    follow_up_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct one arm's pseudo-IPD via the Titman QP -> (time, event).

    Parameters mirror a figure's risk-table row: ``total_n`` is the first
    at-risk cell, ``total_events`` the parenthetical event count (optional)."""
    t = np.asarray(times, float)
    s = np.asarray(survival, float)
    keep = np.isfinite(t) & np.isfinite(s)
    t, s = t[keep], s[keep]
    order = np.argsort(t)
    t, s = t[order], s[order]
    if t.size == 0 or abs(t[0]) > 1e-9:        # ensure an origin anchor (0, 1)
        t = np.concatenate([[0.0], t])
        s = np.concatenate([[1.0], s])
    s = _monotone_decreasing(s)
    nt = t.size
    N = int(total_n) if total_n else 1
    if nt < 2:
        return np.array([]), np.array([])

    # discrete hazard per interval ending at t[k]
    h = np.zeros(nt)
    for k in range(1, nt):
        h[k] = min(0.999, max(0.0, 1.0 - s[k] / s[k - 1] if s[k - 1] > 0 else 0.0))

    # curve-only at-risk n0[k] (1-based, n0[1]=N) and event ceiling E0 = N(1-S_K)
    n0 = np.zeros(nt + 1)
    n0[1] = N
    for k in range(1, nt):
        n0[k + 1] = n0[k] * (1.0 - h[k])
    E0 = float(sum(h[k] * n0[k] for k in range(1, nt)))

    # sensitivities A_k = -Σ_{m>k} h_m Π_{l=k+1}^{m-1}(1-h_l)  (≤ 0).
    # The naive triple loop is O(nt^3); with the EXTRACTED curve nt is hundreds
    # of pixel columns (not registry-ipd's ~8 sparse anchors), so it must be the
    # O(nt) backward recurrence  B_k = h_{k+1} + (1-h_{k+1})·B_{k+1},  A_k = -B_k.
    A = np.zeros(nt)
    B_next = 0.0  # B_{nt-1} = 0 (no intervals beyond the last)
    for k in range(nt - 2, 0, -1):
        B_k = h[k + 1] + (1.0 - h[k + 1]) * B_next
        A[k] = -B_k
        B_next = B_k
    sum_a2 = float(np.sum(A * A)) or 1.0

    E = total_events
    e_target = min(E, E0) if (E is not None and E > 0) else E0
    lam = (e_target - E0) / sum_a2                  # ≤ 0
    c = np.maximum(0.0, lam * A)

    # realise pseudo-IPD: d_k events SPREAD across (t[k-1], t[k]] (not piled at
    # the anchor -> correct at-risk sets -> correct HR); c_k censored at t[k].
    # Fractional EVENT/CENSOR CARRY (error diffusion, as Guyot uses) so that a
    # DENSE extracted curve -- many tiny intervals with h_k*n < 0.5 -- does not
    # lose events to per-interval rounding (the literal sparse-anchor port did).
    ipd_t, ipd_e = [], []
    n = N
    e_carry = c_carry = 0.0
    for k in range(1, nt):
        e_carry += h[k] * n
        dk = int(np.floor(e_carry)); e_carry -= dk
        dk = min(max(dk, 0), n)
        t0, t1 = t[k - 1], t[k]
        for i in range(dk):
            ipd_t.append(t0 + (t1 - t0) * (i + 0.5) / max(1, dk)); ipd_e.append(1)
        c_carry += c[k]
        ck = int(np.floor(c_carry)); c_carry -= ck
        ck = min(max(ck, 0), n - dk)
        for _ in range(ck):
            ipd_t.append(t1); ipd_e.append(0)
        n -= dk + ck
    tail_t = follow_up_max if follow_up_max is not None else t[-1]
    for _ in range(max(0, int(round(n)))):
        ipd_t.append(float(tail_t)); ipd_e.append(0)
    return np.asarray(ipd_t, float), np.asarray(ipd_e, int)
