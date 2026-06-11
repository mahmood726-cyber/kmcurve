"""Tests for the granularity-manifold pooling demo (granularity_pool.py).
Seeded, so deterministic; no network."""
import numpy as np

import granularity_pool as G


def test_dl_pool_zero_heterogeneity():
    # identical estimates -> tau2 == 0 and mu == the common value
    p = G.dl_pool(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1]))
    assert abs(p["mu"] - 0.5) < 1e-9 and p["tau2"] == 0.0


def test_dl_pool_recovers_known_heterogeneity():
    # spread-out estimates with small sampling variance -> positive tau2
    p = G.dl_pool(np.array([-0.6, -0.2, 0.0, 0.4]), np.array([0.01, 0.01, 0.01, 0.01]))
    assert p["tau2"] > 0.05


def test_event_lever_recovers_true_ipd_pool():
    """The headline: pooling the event-pinned (QP) reconstruction matches the
    true-IPD pool on BOTH mu and tau2; curve-only attenuates both."""
    o = G.run(k=16, tau2_true=0.06, seed=1)
    cu, ev = o["pools"]["curve_only"], o["pools"]["event_pinned"]
    # event-pinned is much closer to the true-IPD pooled effect than curve-only
    assert abs(ev["mu_err_vs_ipd"]) < 0.05
    assert abs(ev["mu_err_vs_ipd"]) < abs(cu["mu_err_vs_ipd"]) - 0.1
    # ...and recovers the heterogeneity that curve-only collapses
    assert abs(ev["tau2_err_vs_ipd"]) < abs(cu["tau2_err_vs_ipd"]) - 0.05
    # curve-only attenuates the effect TOWARD zero (less protective)
    assert o["pools"]["curve_only"]["mu"] > o["pools"]["true_ipd"]["mu"]


def test_run_is_seed_deterministic():
    a = G.run(k=10, tau2_true=0.05, seed=3)
    b = G.run(k=10, tau2_true=0.05, seed=3)
    assert a["pools"]["event_pinned"]["mu"] == b["pools"]["event_pinned"]["mu"]
