"""Tests for the benchmark harness (Phase-0 accuracy-over-corpus runner)."""

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("scipy")

import benchmark_harness as BH  # noqa: E402 (sys.path via conftest)

try:
    from project_paths import sample_pdf_path  # noqa: F401

    _HAVE_PDF = sample_pdf_path() is not None
except Exception:
    _HAVE_PDF = False

pytestmark = pytest.mark.skipif(not _HAVE_PDF, reason="bundled sample PDF not found")


@pytest.fixture(scope="module")
def report():
    return BH.run()


def test_all_advance_panels_evaluated(report):
    assert report["n_trials"] == 1
    assert report["n_panels"] == 4
    assert report["n_scored"] == 4  # all four produced an HR


def test_all_panels_within_published_ci(report):
    assert report["pct_within_ci"] == 100.0
    assert report["failure_taxonomy"].get("within_ci") == 4


def test_median_hr_error_reasonable(report):
    # seed corpus: median relative HR error should be small
    assert report["median_rel_hr_err"] < 0.10


def test_confidence_tracks_accuracy(report):
    """The highest-separation-confidence panel should be the most accurate --
    the signal that makes the confidence score useful for triage at scale."""
    panels = report["panels"]
    best_conf = max(panels, key=lambda p: p["separation_confidence"])
    # Panel A (combined events) is both highest-confidence and lowest-error
    assert best_conf["panel_index"] == 0
    assert best_conf["rel_err"] == min(p["rel_err"] for p in panels)
