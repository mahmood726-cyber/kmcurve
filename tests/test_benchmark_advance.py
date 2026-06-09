"""End-to-end benchmark regression: the full chain vs published ADVANCE values.

Guards the headline result -- reconstructed HR reproduces the published 0.90
(95% CI 0.82-0.98) -- against regressions in any stage of the pipeline.
"""

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("scipy")

import benchmark_advance  # noqa: E402 (sys.path via conftest)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None

pytestmark = pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")


@pytest.fixture(scope="module")
def report():
    return benchmark_advance.run(SAMPLE_PDF, 7)


def test_calibration_exact(report):
    assert report["calibration_all_exact"] is True


def test_at_risk_recovered_exactly(report):
    assert report["at_risk_exact"] is True


def test_event_count_close_to_published(report):
    assert abs(report["total_events_pct_err"]) < 5.0


def test_hazard_ratio_matches_published(report):
    assert report["hr_in_published_ci"] is True
    assert report["hr"] == pytest.approx(0.90, abs=0.05)


def test_curve_drift_bounded(report):
    assert report["max_ci_drift"] < 2.0
