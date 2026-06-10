#!/usr/bin/env python3
"""Tests for axis_cross_check + the confidence veto."""
import pytest
from raster_km import PlotBox
from vector_km import _fit_axis
from axis_cross_check import cross_check_calibration, caption_max_followup
from confidence import calibration_confidence

BOX = PlotBox(x0=100, y0=50, x1=500, y1=450)
# x: px 100..500 -> 0..24 months ; y: px 50..450 -> 1..0 survival
XFIT = _fit_axis([(100, 0.0), (500, 24.0)])
YFIT = _fit_axis([(50, 1.0), (450, 0.0)])
META = {"value_scale": 1.0, "y_kind": "survival"}


def test_caption_parse():
    assert caption_max_followup("Kaplan-Meier over 24 months of follow-up") == 24.0
    assert caption_max_followup("median follow-up 36 months; max 60 months") == 60.0
    assert caption_max_followup("no horizon here") is None


def test_consistent_when_external_matches():
    r = cross_check_calibration(XFIT, YFIT, META, BOX, external_max_time=24.0)
    assert r["consistent"] is True
    assert r["flags"]["x_matches_followup"] is True
    assert r["flags"]["y_span_full"] is True


def test_external_mismatch_vetoes():
    # calibrated x-max is 24 but the paper reports 20 -> inconsistent
    r = cross_check_calibration(XFIT, YFIT, META, BOX, external_max_time=20.0)
    assert r["consistent"] is False
    assert "x_matches_followup" in r["vetoes"]


def test_at_risk_alignment():
    # columns at px 100,300,500 should map to 0,12,24 months
    good = cross_check_calibration(XFIT, YFIT, META, BOX,
                                   at_risk_col_times=[(100, 0), (300, 12), (500, 24)])
    assert good["consistent"] is True
    bad = cross_check_calibration(XFIT, YFIT, META, BOX,
                                  at_risk_col_times=[(100, 0), (300, 15), (500, 30)])
    assert bad["consistent"] is False


def test_confidence_veto_overrides_high_score():
    good_meta = dict(META, x_kind="time", y_value_min=0.0, y_value_max=1.0,
                     x_value_min=0.0, x_value_max=24.0)
    inconsistent = {"consistent": False, "vetoes": ["x_matches_followup"]}
    r = calibration_confidence(good_meta, 1.0, 1.0, 5, 5, verify_agree=True,
                               cross_check=inconsistent)
    assert r["confidence"] >= 0.95      # internally still looks great
    assert r["cross_check_veto"] is True
    assert r["auto_accept"] is False     # but external contradiction vetoes
