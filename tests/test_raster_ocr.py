"""Reliable raster auto-calibration test (dual-engine OCR + RANSAC).

Gated on the OCR engines being installed (RapidOCR + Tesseract binary). Renders
the bundled ADVANCE figure to a RASTER image -- a real KM figure with real
rendered labels and known ground truth (x 0-66 months, y 25%->0) -- and asserts
that auto_calibrate_axes recovers both axes with NO clicks.
"""

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("pypdfium2")
pytest.importorskip("rapidocr_onnxruntime")
pytest.importorskip("pytesseract")

import raster_km as R  # noqa: E402

if not R._ensure_tesseract():
    pytest.skip("tesseract binary not available", allow_module_level=True)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None

pytestmark = pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")


@pytest.fixture(scope="module")
def panelB_raster():
    SC = 300 / 72
    g = R.render_page(SAMPLE_PDF, 7, dpi=300)
    # Panel B (pts x[346,530] y[77,202]); generous crop incl label margins
    return g[int(70 * SC): int(245 * SC), int(317 * SC): int(545 * SC)]


def test_auto_calibration_recovers_advance_axes(panelB_raster):
    box = R.detect_plot_box(panelB_raster)
    assert box is not None
    xf, yf = R.auto_calibrate_axes(panelB_raster, box)
    # x: 0-66 months, y: 25% (top) -> 0% (bottom), recovered with no clicks
    assert xf.r2 > 0.999
    assert yf.r2 > 0.999
    assert abs(yf.value(box.y0) - 25.0) < 1.5
    assert abs(yf.value(box.y1) - 0.0) < 1.5


def test_auto_calibration_fails_closed_on_blank():
    import numpy as np

    blank = np.full((400, 500), 255, dtype=np.uint8)
    box = R.PlotBox(x0=50, y0=20, x1=450, y1=350)
    with pytest.raises(ValueError):
        R.auto_calibrate_axes(blank, box)  # no ticks -> fall back to 2-click
