"""Tests for the granularity-manifold pooling demo (granularity_pool.py).
Seeded, so deterministic; no network. The --real tests use the registry-ipd
gold-standard CSVs (deterministic raster pipeline, no RNG) and skip if absent."""
from pathlib import Path

import numpy as np
import pytest

import granularity_pool as G

_REGISTRY = Path(r"C:\Projects\registry-ipd")


def _registry_available() -> bool:
    return (_REGISTRY / "realipd").is_dir()


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


def test_abstract_hr_loghr_is_2dp_rounding():
    # the HR-only rung reports the printed HR (2 dp) -> ~lossless point estimate
    assert abs(np.exp(G.abstract_hr_loghr(np.log(0.713))) - 0.71) < 1e-9
    assert abs(G.abstract_hr_loghr(np.log(0.70)) - np.log(0.70)) < 1e-9


def test_abstract_tier_recovers_mu_in_sim():
    """The HR-only rung pins the pooled mu (2-dp rounding only), unlike curve_only."""
    o = G.run(k=16, tau2_true=0.06, seed=1)
    ab, cu = o["pools"]["abstract_hr_only"], o["pools"]["curve_only"]
    assert abs(ab["mu_err_vs_ipd"]) < 0.02            # HR-only recovers mu
    assert abs(ab["mu_err_vs_ipd"]) < abs(cu["mu_err_vs_ipd"]) - 0.1


# --- real-data mode (registry-ipd gold standard, deterministic raster pipeline) ---

# A small fixed protective panel for speed; the conclusions are panel-robust.
_REAL_TEST_DS = ["aidssi", "diabetic", "cancer", "gbsg", "burn"]


@pytest.mark.skipif(not _registry_available(), reason="registry-ipd not present")
def test_run_real_curveonly_biases_pooled_mu():
    """REAL data, robust half: curve_only attenuates the pooled effect toward the
    null, while abstract-HR-only and event-pinned recover it. (tau2 recovery is
    deliberately NOT asserted -- it does not replicate on real reconstructions.)"""
    o = G.run_real(_REGISTRY, _REAL_TEST_DS)
    assert o["k"] >= 4
    ip = o["pools"]["true_ipd"]
    ab, ev, cu = (o["pools"]["abstract_hr_only"], o["pools"]["event_pinned"],
                  o["pools"]["curve_only"])
    assert ip["mu"] < 0                                  # protective panel
    # HR-only and event-pinned recover the pooled mu; curve_only is far off
    assert abs(ab["mu_err_vs_ipd"]) < 0.05
    assert abs(ev["mu_err_vs_ipd"]) < abs(cu["mu_err_vs_ipd"])
    # curve_only attenuates TOWARD the null (less protective than the truth)
    assert cu["mu"] > ip["mu"]


@pytest.mark.skipif(not _registry_available(), reason="registry-ipd not present")
def test_run_real_per_trial_error_ordering():
    """Panel-independent metric: curve-only per-trial logHR error >> event-pinned
    >> abstract-HR-only (the manifold's identification ordering for the HR)."""
    o = G.run_real(_REGISTRY, _REAL_TEST_DS)
    e = o["mean_abs_loghr_err"]
    assert e["abstract_hr_only"] < e["event_pinned"] < e["curve_only"]
    assert e["curve_only"] > 0.1                          # curve-only is materially biased


@pytest.mark.skipif(not _registry_available(), reason="registry-ipd not present")
def test_run_real_is_deterministic():
    a = G.run_real(_REGISTRY, _REAL_TEST_DS)
    b = G.run_real(_REGISTRY, _REAL_TEST_DS)
    assert a["pools"]["curve_only"]["mu"] == b["pools"]["curve_only"]["mu"]
