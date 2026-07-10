"""Tests for Guyot IPD reconstruction and the end-to-end PDF->IPD wiring."""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pdfplumber")

import guyot  # noqa: E402 (sys.path via tests/conftest.py)
import vector_km  # noqa: E402

try:
    from project_paths import sample_pdf_path

    SAMPLE_PDF = str(sample_pdf_path())
except Exception:
    SAMPLE_PDF = None

FIGURE_PAGE = 7


# --------------------------------------------------------------------------- #
# Unit: synthetic round-trip (no PDF needed)
# --------------------------------------------------------------------------- #
def test_km_from_ipd_recovers_known_survival():
    # 100 patients, an event at t=1 (10 die) then t=2 (10 die), rest censored late
    time = np.array([1.0] * 10 + [2.0] * 10 + [5.0] * 80)
    event = np.array([1] * 10 + [1] * 10 + [0] * 80)
    et, es = guyot.km_from_ipd(time, event)
    # S(1) = 1 - 10/100 = 0.9 ; S(2) = 0.9 * (1 - 10/90) = 0.8
    assert et.tolist() == [1.0, 2.0]
    assert es[0] == pytest.approx(0.9, abs=1e-9)
    assert es[1] == pytest.approx(0.8, abs=1e-9)


def test_no_nar_explicit_total_n_accounts_all_patients():
    # No at-risk table, but a genuine total_n=300 is supplied (as the raster
    # path does via at_risk[i][0][1]). The end-of-follow-up tail must censor
    # ALL remaining patients, not cap at 10 -- otherwise the reconstructed
    # curve collapses far below the target. Regression for GUYOT-1.
    t = np.linspace(0, 60, 60)
    S = np.linspace(1.0, 0.5, 60)
    tt, ee = guyot.reconstruct_arm(t, S, total_n=300)
    assert tt.size == 300  # every patient accounted for, no phantom truncation
    et, es = guyot.km_from_ipd(tt, ee)
    assert es[-1] == pytest.approx(0.5, abs=0.05)  # lands on the input endpoint


def test_default_total_n_tail_stays_capped():
    # No total_n and no NAR -> total_n is the fabricated default (100). The
    # unknown-count case must NOT invent a large censored tail; the 10-cap
    # still applies so the reconstruction stays well under the phantom 100.
    t = np.linspace(0, 60, 60)
    S = np.linspace(1.0, 0.5, 60)
    tt, _ = guyot.reconstruct_arm(t, S)  # explicit_n is False here
    assert tt.size < 100


def test_interpolate_nar_does_not_mutate_caller_array():
    # Passing a float ndarray of at-risk counts must not be clobbered by the
    # internal monotone clamp (np.asarray aliased; np.array copies). GUYOT-2.
    nar_v = np.array([300.0, 320.0, 150.0])  # 320 > 300 triggers the clamp
    before = nar_v.copy()
    guyot.reconstruct_ipd_guyot(
        np.array([0, 6, 12.0]), np.array([1.0, 0.8, 0.6]),
        n_risk_times=np.array([0, 6, 12.0]), n_risk_values=nar_v, total_n=300,
    )
    assert np.array_equal(nar_v, before)


def test_empty_curve_returns_empty_records():
    # An empty digitised curve must return no records rather than IndexError
    # from the final-censoring tail dereferencing times[-1]. Regression KM-1.
    assert guyot.reconstruct_ipd_guyot(np.array([]), np.array([]), total_n=100) == []
    t, e = guyot.reconstruct_arm(np.array([]), np.array([]), total_n=100)
    assert t.size == 0 and e.size == 0


def test_pdf_to_ipd_monotone_follows_ci_flag(monkeypatch):
    # pdf_to_ipd must request a DECREASING monotone for a survival axis and an
    # INCREASING one for a cumulative-incidence axis -- hardcoding "increasing"
    # flattens survival curves to a constant (zero events). Regression KMC-CAL-1.
    captured = {}

    def spy(pdf_path, page_index, n_arms=2, monotone="auto"):
        captured["monotone"] = monotone
        raise RuntimeError("stop after capturing monotone")

    monkeypatch.setattr(vector_km, "extract_km_from_pdf", spy)

    with pytest.raises(RuntimeError):
        vector_km.pdf_to_ipd("nonexistent.pdf", 0, value_is_cumulative_incidence=False)
    assert captured["monotone"] == "decreasing"

    captured.clear()
    with pytest.raises(RuntimeError):
        vector_km.pdf_to_ipd("nonexistent.pdf", 0, value_is_cumulative_incidence=True)
    assert captured["monotone"] == "increasing"


def test_reconstruct_round_trips_a_known_curve():
    # Known curve: survival drops 1.0 -> 0.6 over 0..12 months, N=500, with NAR.
    times = np.array([0, 3, 6, 9, 12], dtype=float)
    surv = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    nar_t = np.array([0, 6, 12], dtype=float)
    nar_v = np.array([500, 380, 280], dtype=float)
    t, e = guyot.reconstruct_arm(times, surv, nar_times=nar_t, nar_values=nar_v, total_n=500)
    assert t.size == 500  # every patient accounted for
    assert e.sum() > 0
    # reconstructed final survival should land near the input 0.6
    et, es = guyot.km_from_ipd(t, e)
    assert es[-1] == pytest.approx(0.6, abs=0.05)


# --------------------------------------------------------------------------- #
# End-to-end: PDF -> IPD on the ADVANCE fixture
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")
def test_pdf_to_ipd_advance_panelA():
    panels = vector_km.pdf_to_ipd(SAMPLE_PDF, FIGURE_PAGE)
    assert len(panels) == 4
    pa = panels[0]
    # arms are now paired to at-risk rows via the in-plot legend
    assert pa.nar_mapping == "legend_matched"
    # the Standard arm is paired with the "Standard" at-risk row, etc.
    for a in pa.arms:
        assert a.label.split()[0].lower() in a.nar_label.lower()
    # both at-risk rows recovered with the real ADVANCE totals (~5570)
    totals = sorted(int(r.counts[0]) for r in pa.at_risk)
    assert len(pa.at_risk) == 2
    assert all(5500 < n < 5650 for n in totals), totals

    arms = pa.arms
    assert len(arms) == 2
    for a in arms:
        assert 5400 < a.total_n < 5700  # reconstructed N matches at-risk start
        assert 700 < a.n_events < 1400  # plausible event count per arm
    # combined event count is close to the paper's reported 2125
    total_events = sum(a.n_events for a in arms)
    assert 1900 < total_events < 2300, total_events


@pytest.mark.skipif(SAMPLE_PDF is None, reason="bundled sample PDF not found")
def test_reconstructed_ci_tracks_input_curve():
    """Reconstructed final cumulative incidence ~ the digitised curve endpoint."""
    km_panels = vector_km.extract_km_from_pdf(SAMPLE_PDF, FIGURE_PAGE, monotone="increasing")
    ipd_panels = vector_km.pdf_to_ipd(SAMPLE_PDF, FIGURE_PAGE)
    # IPD arms are labelled by legend identity; key the input curves the same way
    arms_in = {(a.identity or a.label): a for a in km_panels[0].arms}
    for a in ipd_panels[0].arms:
        et, es = guyot.km_from_ipd(a.time, a.event)
        recon_ci = (1 - es[-1]) * 100 if es.size else 0.0
        input_ci = arms_in[a.label].value[-1]
        assert abs(recon_ci - input_ci) < 3.0, (a.label, recon_ci, input_ci)
