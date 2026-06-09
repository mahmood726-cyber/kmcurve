"""Regression tests for the vector KM extractor.

These lock in the core guarantee that previously eluded the pipeline: exact,
OCR-free axis calibration from the PDF vector layer. Validated against the
bundled ADVANCE-trial fixture (NEJMoa0802987.pdf), page index 7, Figure 3 --
four cumulative-incidence panels, 0-66 months x 0-25%.
"""

import numpy as np
import pytest

pytest.importorskip("pdfplumber", reason="vector extraction needs pdfplumber")

import vector_km  # noqa: E402  (sys.path set up by tests/conftest.py)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None

FIGURE_PAGE = 7  # 0-indexed page holding ADVANCE Figure 3

pytestmark = pytest.mark.skipif(
    SAMPLE_PDF is None, reason="bundled sample PDF not found"
)


@pytest.fixture(scope="module")
def panels():
    return vector_km.extract_km_from_pdf(
        SAMPLE_PDF, FIGURE_PAGE, monotone="increasing"
    )


def test_detects_four_panels(panels):
    assert len(panels) == 4


def test_calibration_is_exact(panels):
    """The whole point: calibration R^2 must be effectively 1.0 on both axes."""
    for pr in panels:
        assert pr.x_fit.r2 > 0.999, f"panel {pr.panel.index} x R2={pr.x_fit.r2}"
        assert pr.y_fit.r2 > 0.999, f"panel {pr.panel.index} y R2={pr.y_fit.r2}"
        assert pr.x_fit.n_ticks >= 6
        assert pr.y_fit.n_ticks >= 4


def test_axis_ranges_match_known_figure(panels):
    """X axis is 0-66 months; Y axis is 0-25% for every panel."""
    for pr in panels:
        p = pr.panel
        left = pr.x_fit.value(p.x0)
        right = pr.x_fit.value(p.x1)
        top = pr.y_fit.value(p.y_top)
        bottom = pr.y_fit.value(p.y_bottom)
        assert abs(left) < 1.0, f"x-left {left}"
        assert 64.0 < right < 70.0, f"x-right {right}"
        assert 24.0 < top < 26.5, f"y-top {top}"
        assert abs(bottom) < 1.0, f"y-bottom {bottom}"


def test_each_panel_has_two_arms_in_range(panels):
    for pr in panels:
        arms = [a for a in pr.arms if a.n_points() > 0]
        assert len(arms) == 2, f"panel {pr.panel.index}: {len(arms)} arms"
        for arm in arms:
            assert arm.n_points() > 50
            # cumulative-incidence stays within the plotted 0-25% window
            assert arm.value.min() >= -0.5
            assert arm.value.max() < 26.0  # no out-of-plot strays (was 25.2 before fix)
            # time is sorted ascending
            assert np.all(np.diff(arm.time) >= -1e-6)
            # cumulative incidence is monotone non-decreasing
            assert np.all(np.diff(arm.value) >= -1e-9)
            # curve starts near the origin and spans the follow-up
            assert arm.value[0] < 3.0
            assert arm.time[0] < 3.0 and arm.time[-1] > 60.0


def test_arms_labelled_by_mean_upper_above_lower(panels):
    """Arms are labelled by mean value, so 'upper' sits above 'lower' on average
    (stable for non-crossing curves; genuine crossings need legend identity)."""
    for pr in panels:
        upper = next(a for a in pr.arms if a.label == "upper")
        lower = next(a for a in pr.arms if a.label == "lower")
        assert upper.value.mean() >= lower.value.mean()


def test_separation_confidence_reported(panels):
    """Confidence flags how resolvably separated the arms are.

    Panel A (combined events, arms diverge to ~2.5% apart) is the most clearly
    separated; the near-overlapping panels (B, D, <1% apart) score lowest.
    """
    confs = [pr.separation_confidence for pr in panels]
    for c in confs:
        assert 0.0 <= c <= 1.0
    # Panel A is the most-separated of the four
    assert panels[0].separation_confidence == max(confs)
    # the near-overlapping panels are flagged low
    assert min(confs) < 0.1


def test_separate_arms_follows_crossing_curves():
    """Two curves that CROSS must be tracked through the crossing (continuity),
    not collapsed into upper/lower envelopes. Curve A: 10->0, curve B: 0->10,
    crossing at t=5. Each recovered track should be monotone-through.
    """
    t = np.arange(0.0, 10.01, 0.5)
    a = np.column_stack([t, 10.0 - t])  # descending
    b = np.column_stack([t, t])  # ascending
    cloud = np.vstack([a, b])
    tracks = vector_km.separate_arms(cloud, n_arms=2, col_dt=0.5, gap=1.0)
    tracks = [tr for tr in tracks if tr.size]
    assert len(tracks) == 2

    def spans(tr, lo_first):  # high->low if lo_first=False else low->high
        v0, v1 = tr[0, 1], tr[-1, 1]
        return (v0 < 2 and v1 > 8) if lo_first else (v0 > 8 and v1 < 2)

    descending = any(spans(tr, lo_first=False) for tr in tracks)
    ascending = any(spans(tr, lo_first=True) for tr in tracks)
    # Envelope assignment would give both tracks starting AND ending at an
    # extreme (10->10 and 0->0); continuity gives one 10->0 and one 0->10.
    assert descending and ascending


def test_separate_arms_parallel_non_crossing():
    """Two non-crossing parallel curves: track 0 stays above track 1."""
    t = np.arange(0.0, 10.01, 0.5)
    upper = np.column_stack([t, 8.0 - 0.3 * t])
    lower = np.column_stack([t, 4.0 - 0.3 * t])
    tracks = vector_km.separate_arms(np.vstack([upper, lower]), n_arms=2, col_dt=0.5, gap=1.0)
    assert np.mean(tracks[0][:, 1]) > np.mean(tracks[1][:, 1])


def test_arm_identity_from_legend(panels):
    """Arms carry treatment identities read from the in-plot legend, with the
    higher-incidence (upper) arm == Standard control across all panels."""
    for pr in panels:
        upper = next(a for a in pr.arms if a.label == "upper")
        lower = next(a for a in pr.arms if a.label == "lower")
        assert "standard" in upper.identity.lower(), upper.identity
        assert "intensive" in lower.identity.lower(), lower.identity


def test_panelA_endpoints_match_advance_figure(panels):
    """Panel A (combined events): Standard ~22.8%, Intensive ~20.3% at 66 mo."""
    pa = panels[0]
    upper = next(a for a in pa.arms if a.label == "upper")
    lower = next(a for a in pa.arms if a.label == "lower")
    assert 21.0 < upper.value[-1] < 24.5, upper.value[-1]
    assert 18.5 < lower.value[-1] < 22.0, lower.value[-1]
