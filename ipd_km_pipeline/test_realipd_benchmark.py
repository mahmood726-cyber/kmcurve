"""Tests for the true-IPD external-accuracy benchmark (realipd_benchmark.py).

The render->extract->reconstruct->score chain is exercised on a real
gold-standard dataset from the sibling registry-ipd repo. Tests skip cleanly
when that data (or cv2/matplotlib) is unavailable, so they never hard-fail in a
bare CI checkout.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
import realipd_benchmark as B  # noqa: E402

REGISTRY = Path(r"C:\Projects\registry-ipd")
HAS_DATA = (REGISTRY / "realipd").is_dir()
skip_no_data = pytest.mark.skipif(not HAS_DATA, reason="registry-ipd gold standard not on disk")


def test_km_steps_monotone_starts_at_one():
    t = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    e = np.array([1, 1, 0, 1, 1])
    times, surv = B.km_steps(t, e)
    assert surv[0] == 1.0
    assert np.all(np.diff(surv) <= 1e-12)       # non-increasing
    assert times[0] == 0.0 and np.all(np.diff(times) > 0)
    assert 0.0 <= surv[-1] <= 1.0


def test_km_steps_empty():
    times, surv = B.km_steps(np.array([]), np.array([]))
    assert times.tolist() == [0.0] and surv.tolist() == [1.0]


def test_at_risk_table_decreasing_and_starts_at_n():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    times, counts = B._at_risk_table(t, t_max=6.0, k=4)
    assert counts[0] == t.size                  # first cell == total N
    assert np.all(np.diff(counts) <= 0)         # at-risk only decreases
    assert times[0] == 0.0 and times[-1] == 6.0


def test_render_geometry_refs_exact():
    steps = (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.7, 0.4]))
    img, geom = B.render_km([steps], t_max=2.0, W=400, H=300)
    assert img.shape == (300, 400)
    # x_refs/y_refs must bracket the data range with the box corners
    assert geom.x_refs[0][1] == 0.0 and geom.x_refs[1][1] == 2.0
    assert geom.y_refs[0][1] == 0.0 and geom.y_refs[1][1] == 1.0
    assert geom.plot.x0 < geom.plot.x1 and geom.plot.y0 < geom.plot.y1
    assert (img < 128).any()                    # some dark curve pixels exist


@skip_no_data
def test_load_true_ipd_two_arms():
    cfg = B.CONFIG_BY_DS["veteran"]
    et, ee, ct, ce = B.load_true_ipd(REGISTRY / "realipd" / "veteran.csv", cfg)
    assert et.size > 20 and ct.size > 20        # both arms populated
    assert set(np.unique(ee).tolist()) <= {0, 1}
    assert (et >= 0).all() and (ct >= 0).all()


@skip_no_data
def test_recon_and_score_chain_pbc():
    """End-to-end on pbc: chain runs, reports both modes, NAR is well-calibrated."""
    cfg = B.CONFIG_BY_DS["pbc"]
    r = B.recon_and_score(REGISTRY / "realipd" / "pbc.csv", cfg)
    assert r["status"] == "scored"
    assert r["true_hr"] > 0
    for k in ("recon_hr_curveonly", "recon_hr_nar", "fold_curveonly", "fold_nar"):
        assert k in r
    # pbc is a near-null, well-behaved case: censoring-informed should be close.
    assert r["fold_nar"] is not None and r["fold_nar"] < 1.5


@skip_no_data
def test_censoring_informed_beats_curve_only_in_aggregate():
    """The headline claim: the at-risk table makes reconstruction far better
    than curve-only across the set (not necessarily every single dataset)."""
    out = B.run(REGISTRY, datasets=["gbsg", "pbc", "diabetic", "colon", "cancer"])
    ci = out["summary"]["censoring_informed"]
    co = out["summary"]["curve_only"]
    assert ci["n"] >= 3
    assert ci["median"] < co["median"]          # NAR improves the median fold-err
    assert ci["median"] < 1.25                  # and lands in a useful range
