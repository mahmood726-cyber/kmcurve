"""Tests for the KM figure locator (find the survival figure by caption + content)."""

import pytest

pytest.importorskip("pdfplumber")

import figure_locator as F  # noqa: E402 (sys.path via conftest)

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None


def test_km_caption_regex_matches_survival_language():
    for s in [
        "Figure 3. Cumulative Incidences of Events",
        "Kaplan-Meier overall survival curves",
        "Fig 2. Progression-free survival by arm",
        "Kaplan Meier plot for PFS",
        "time-to-event analysis",
    ]:
        assert F._KM_RE.search(s), s


def test_km_caption_regex_rejects_non_survival():
    for s in [
        "Figure 1. Study flowchart of enrollment",
        "Table 1. Baseline characteristics",
        "Figure 2. Covariate balance before and after IPTW",
        "Forest plot of subgroup odds ratios",
    ]:
        assert not F._KM_RE.search(s), s


@pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")
def test_locate_advance_km_figure():
    cands = F.locate_km_figures(SAMPLE_PDF)
    assert cands, "no KM figure located in ADVANCE"
    top = cands[0]
    assert top.page_index == 7          # ADVANCE Figure 3 page
    assert top.kind == "vector"         # ADVANCE ships vector figures
    assert "cumulative incidence" in top.caption.lower()
