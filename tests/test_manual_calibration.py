"""Tests for the 2-click manual calibration core and the vector/raster router."""

import numpy as np
import pytest

pytest.importorskip("pdfplumber")

import manual_calibration as mc  # noqa: E402 (sys.path via conftest)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None

FIGURE_PAGE = 7


# --------------------------------------------------------------------------- #
# Pure calibration unit tests (no PDF)
# --------------------------------------------------------------------------- #
def test_two_point_axis_recovers_linear_map():
    fit = mc.two_point_axis(100.0, 0.0, 300.0, 60.0)
    assert fit.value(100.0) == pytest.approx(0.0)
    assert fit.value(300.0) == pytest.approx(60.0)
    assert fit.value(200.0) == pytest.approx(30.0)


def test_two_point_axis_rejects_degenerate_clicks():
    with pytest.raises(ValueError):
        mc.two_point_axis(150.0, 0.0, 150.0, 60.0)


def test_calibrate_points_with_monotone():
    # x: px 100->0, 300->60 ; y: px 200(top)->100%, 400(bottom)->0%
    xfit = mc.two_point_axis(100.0, 0.0, 300.0, 60.0)
    yfit = mc.two_point_axis(200.0, 100.0, 400.0, 0.0)
    pts = np.array([[100, 220], [200, 260], [300, 240]], dtype=float)  # noisy
    t, v = mc.calibrate_points(pts, xfit, yfit, monotone="decreasing")
    assert t == pytest.approx([0.0, 30.0, 60.0])
    # decreasing enforced despite the noisy middle/last point
    assert np.all(np.diff(v) <= 1e-9)


def test_survival_from_value():
    ci = np.array([0.0, 20.0, 50.0])
    surv = mc.survival_from_value(ci, is_cumulative_incidence=True, scale=100.0)
    assert surv.tolist() == pytest.approx([1.0, 0.8, 0.5])


# --------------------------------------------------------------------------- #
# Router on the fixture
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")
def test_router_flags_vector_figure_and_empty_region():
    import pdfplumber

    with pdfplumber.open(SAMPLE_PDF) as pdf:
        page = pdf.pages[FIGURE_PAGE]
        # Panel A region is dense vector graphics -> vector path
        assert mc.figure_is_vector(page, bbox=(109, 77, 293, 202)) is True
        # A blank top-margin strip has no curves -> raster/fallback
        assert mc.figure_is_vector(page, bbox=(0, 0, 567, 40)) is False
