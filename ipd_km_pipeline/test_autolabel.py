#!/usr/bin/env python3
"""Tests for the lever-2 label comparison / corpus-record logic."""
import copy
from autolabel import compare_labels, corpus_record, LABEL_SCHEMA

BASE = {
    "is_km_figure": True, "n_panels": 1,
    "panels": [{
        "outcome": "Overall survival",
        "y_axis": {"kind": "survival", "unit": "proportion", "min": 0.0, "max": 1.0},
        "x_axis": {"unit": "months", "min": 0.0, "max": 24.0},
        "n_arms": 2, "arm_labels": ["drug", "placebo"],
        "censoring_marks": True, "at_risk_table": True,
    }],
}


def test_identical_reads_full_confidence_no_review():
    r = compare_labels(BASE, copy.deepcopy(BASE))
    assert r["confidence"] == 1.0
    assert r["needs_review"] is False


def test_arm_count_disagreement_flags_review():
    d = copy.deepcopy(BASE); d["panels"][0]["n_arms"] = 3
    r = compare_labels(BASE, d)
    assert r["needs_review"] is True
    assert r["agree"]["n_arms"] is False
    assert r["confidence"] < 1.0


def test_panel_count_disagreement_flags_review():
    d = copy.deepcopy(BASE); d["n_panels"] = 2; d["panels"] = d["panels"] * 2
    r = compare_labels(BASE, d)
    assert r["needs_review"] is True
    assert r["agree"]["n_panels"] is False


def test_range_only_disagreement_tolerated():
    # x-max 24 vs 26 (~8%) within tol -> not flagged; structural fields agree
    d = copy.deepcopy(BASE); d["panels"][0]["x_axis"]["max"] = 26.0
    r = compare_labels(BASE, d)
    assert r["agree"]["x_range"] is True
    assert r["needs_review"] is False
    # but a 2x range error is caught
    d2 = copy.deepcopy(BASE); d2["panels"][0]["x_axis"]["max"] = 48.0
    assert compare_labels(BASE, d2)["agree"]["x_range"] is False


def test_y_kind_disagreement_flags_review():
    d = copy.deepcopy(BASE); d["panels"][0]["y_axis"]["kind"] = "cumulative_incidence"
    r = compare_labels(BASE, d)
    assert r["needs_review"] is True


def test_corpus_record_shape():
    rec = corpus_record({"id": "fig1", "pdf": "x.pdf", "page_index": 3},
                        BASE, copy.deepcopy(BASE))
    assert rec["id"] == "fig1" and rec["confidence"] == 1.0
    assert rec["needs_human_review"] is False
    assert rec["provenance"] == "vlm-double-read-autolabel"


def test_schema_is_wellformed():
    assert LABEL_SCHEMA["properties"]["panels"]["items"]["required"]


def test_agreed_not_km_is_clean_negative():
    neg = {"is_km_figure": False, "n_panels": 0, "panels": []}
    r = compare_labels(neg, dict(neg))
    assert r["needs_review"] is False        # both agree it's not KM -> clean
    assert r["confidence"] == 1.0


def test_panel_reorder_still_agrees():
    import copy
    a = {"is_km_figure": True, "n_panels": 2, "panels": [
        {"outcome": "OS", "y_axis": {"kind": "survival", "unit": "percent", "min": 0, "max": 100},
         "x_axis": {"unit": "months", "min": 0, "max": 30}, "n_arms": 2, "arm_labels": [],
         "censoring_marks": True, "at_risk_table": False},
        {"outcome": "PFS", "y_axis": {"kind": "survival", "unit": "percent", "min": 0, "max": 100},
         "x_axis": {"unit": "months", "min": 0, "max": 30}, "n_arms": 4, "arm_labels": [],
         "censoring_marks": True, "at_risk_table": False}]}
    b = copy.deepcopy(a); b["panels"] = [a["panels"][1], a["panels"][0]]  # reversed order
    r = compare_labels(a, b)
    assert r["needs_review"] is False        # order-invariant match
