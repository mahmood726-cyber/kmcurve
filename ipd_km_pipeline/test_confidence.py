#!/usr/bin/env python3
"""Unit tests for calibration_confidence (lever 4)."""
import pytest
from confidence import calibration_confidence

# a clean, plausible survival calibration
GOOD_META = {
    "value_scale": 1.0, "y_kind": "survival", "x_kind": "time",
    "y_value_min": 0.0, "y_value_max": 1.0, "x_value_min": 0.0, "x_value_max": 24.0,
}


def test_clean_verified_is_high_and_auto_accept():
    r = calibration_confidence(GOOD_META, 1.0, 1.0, 5, 5, verify_agree=True)
    assert r["confidence"] >= 0.95
    assert r["auto_accept"] is True


def test_verify_disagree_never_auto_accepts():
    r = calibration_confidence(GOOD_META, 1.0, 1.0, 5, 5, verify_agree=False)
    assert r["auto_accept"] is False
    assert r["confidence"] < 0.6


def test_two_ticks_not_auto_accepted_even_if_verified():
    # 2 ticks can't catch a misread -> low n_ticks score, below auto-accept floor
    r = calibration_confidence(GOOD_META, 1.0, 1.0, 2, 2, verify_agree=True)
    assert r["auto_accept"] is False
    assert r["components"]["n_ticks"] < 0.4


def test_implausible_semantics_lowers_confidence():
    bad = dict(GOOD_META, y_kind="other", y_value_min=0.2, y_value_max=0.6)
    good = calibration_confidence(GOOD_META, 1.0, 1.0, 5, 5, verify_agree=True)
    worse = calibration_confidence(bad, 1.0, 1.0, 5, 5, verify_agree=True)
    assert worse["confidence"] < good["confidence"]
    assert worse["components"]["semantics"] < 1.0


def test_single_read_is_neutral_not_full():
    # unknown verify (single read) should not earn full confidence
    r = calibration_confidence(GOOD_META, 1.0, 1.0, 5, 5, verify_agree=None)
    assert r["components"]["verify"] < 1.0
    assert r["confidence"] < calibration_confidence(
        GOOD_META, 1.0, 1.0, 5, 5, verify_agree=True)["confidence"]


def test_low_r2_drags_down():
    r = calibration_confidence(GOOD_META, 0.95, 1.0, 5, 5, verify_agree=True)
    assert r["components"]["r2"] < 0.6
