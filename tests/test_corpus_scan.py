"""Positive-control test for the corpus extraction-robustness scanner.

The scanner auto-locates the KM figure page (no hardcoded page index) and
buckets the extraction outcome. We assert it correctly extracts the bundled
ADVANCE publisher PDF -- which proves a `raster_figure`/failure verdict on the
OA corpus is a real signal, not a scanner bug.
"""

from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

import corpus_scan  # noqa: E402 (sys.path via conftest)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = sample_pdf_path()
except Exception:
    SAMPLE_PDF = None

pytestmark = pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")


def test_scanner_extracts_advance_positive_control():
    s = corpus_scan.scan_pdf(Path(str(SAMPLE_PDF)))
    assert s.status == "extracted"
    assert s.km_page == 7          # auto-located, not hardcoded
    assert s.n_panels == 4
    assert s.max_calib_r2 > 0.999
    assert s.max_arms == 2
    assert s.at_risk_rows >= 2


def test_ranking_orders_best_first():
    # the taxonomy ranking must be ordered best -> worst
    assert corpus_scan.RANK[0] == "extracted"
    assert corpus_scan.RANK[-1] == "no_figure_found"
