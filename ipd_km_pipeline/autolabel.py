#!/usr/bin/env python3
"""
VLM structured auto-labelling of KM figures (roadmap lever 2).
==============================================================

Lever 2 is the gating resource: a labelled corpus to MEASURE gains and train
lever 3. Hand-labelling 200-1000 figures is intractable -- so this bootstraps it
with the same machinery levers 1+4 already proved: a VLM reads the structured
labels (panel count, per-panel axis type/unit/range, arm count, censoring,
at-risk table), TWO independent reads adversarially verify, and a label
CONFIDENCE gates each figure into ``accepted`` (auto) vs ``needs_human_review``
(the minority a human checks). That is the confidence-gated labelling the whole
roadmap is building toward, applied to corpus construction itself.

This module provides the label SCHEMA (shared with the read workflow), the
two-read agreement/confidence logic, and the corpus-record builder. It performs
no network or rendering itself -- the figure reads come from the workflow (or the
Claude vision API), keeping it pure and unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Label schema (also used as the workflow / API structured-output schema)
# --------------------------------------------------------------------------- #
_AXIS_Y = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["survival", "cumulative_incidence", "other"]},
        "unit": {"type": "string", "enum": ["proportion", "percent", "other"]},
        "min": {"type": "number"}, "max": {"type": "number"},
    },
    "required": ["kind", "unit", "min", "max"],
}
_AXIS_X = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "unit": {"type": "string", "enum": ["months", "years", "weeks", "days", "other"]},
        "min": {"type": "number"}, "max": {"type": "number"},
    },
    "required": ["unit", "min", "max"],
}
_PANEL = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "outcome": {"type": "string", "description": "panel title / outcome, '' if none"},
        "y_axis": _AXIS_Y, "x_axis": _AXIS_X,
        "n_arms": {"type": "integer"},
        "arm_labels": {"type": "array", "items": {"type": "string"}},
        "censoring_marks": {"type": "boolean"},
        "at_risk_table": {"type": "boolean"},
    },
    "required": ["outcome", "y_axis", "x_axis", "n_arms", "arm_labels",
                 "censoring_marks", "at_risk_table"],
}
LABEL_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "is_km_figure": {"type": "boolean"},
        "n_panels": {"type": "integer"},
        "panels": {"type": "array", "items": _PANEL},
    },
    "required": ["is_km_figure", "n_panels", "panels"],
}

READ_PROMPT = (
    "Label this Kaplan-Meier figure for a structured corpus. Report:\n"
    " - is_km_figure: true only if this is a survival/KM curve figure.\n"
    " - n_panels: number of separate KM plots (sub-panels) in the figure.\n"
    " - per panel: outcome title (or ''); y_axis {kind survival/cumulative_incidence/"
    "other, unit proportion/percent/other, min, max}; x_axis {unit months/years/"
    "weeks/days/other, min, max}; n_arms (number of curves); arm_labels (legend "
    "text, [] if none); censoring_marks (tick marks on curves?); at_risk_table "
    "(numbers-at-risk row below?).\n"
    "Read carefully; if a value is not legible use a best estimate and it will be "
    "flagged by the second reader. Return ONLY the structured result."
)

# Structural fields whose disagreement means the figure needs human review.
_RANGE_TOL = 0.10


def _num_close(a, b, tol=_RANGE_TOL):
    if a is None or b is None:
        return a == b
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= tol * scale


def compare_labels(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two independent reads -> per-field agreement + a label confidence.

    Load-bearing structural fields (is_km, n_panels, and per-panel n_arms /
    y_kind / x_unit) must agree for a clean auto-label; axis ranges are compared
    with tolerance; soft fields (outcome text, arm_labels) are not gated. Returns
    ``{confidence, needs_review, agree}`` where ``agree`` is the per-check map.
    """
    agree: Dict[str, bool] = {}
    agree["is_km_figure"] = bool(r1.get("is_km_figure")) == bool(r2.get("is_km_figure"))
    agree["n_panels"] = r1.get("n_panels") == r2.get("n_panels")

    # match panels order-invariantly: two reads may list multi-panel figures in
    # different order, so sort each by a structural key before index comparison.
    def _pkey(p):
        return (p.get("n_arms", 0), p["y_axis"].get("kind", ""),
                p["x_axis"].get("unit", ""),
                round(p["y_axis"].get("max", 0) or 0, 1),
                round(p["x_axis"].get("max", 0) or 0, 1))

    p1 = sorted(r1.get("panels", []), key=_pkey)
    p2 = sorted(r2.get("panels", []), key=_pkey)
    panel_fields = ("n_arms", "y_kind", "x_unit", "y_range", "x_range")
    both_not_km = not r1.get("is_km_figure") and not r2.get("is_km_figure")
    both_km = r1.get("is_km_figure") and r2.get("is_km_figure")
    if both_not_km:
        # both readers agree it is NOT a KM figure -> a clean negative label;
        # panel fields are vacuously satisfied (there are no panels to compare).
        for k in panel_fields:
            agree[k] = True
    elif both_km and agree["n_panels"] and len(p1) == len(p2) and p1:
        per = {k: [] for k in panel_fields}
        for a, b in zip(p1, p2):
            per["n_arms"].append(a.get("n_arms") == b.get("n_arms"))
            per["y_kind"].append(a["y_axis"].get("kind") == b["y_axis"].get("kind"))
            per["x_unit"].append(a["x_axis"].get("unit") == b["x_axis"].get("unit"))
            per["y_range"].append(_num_close(a["y_axis"].get("max"), b["y_axis"].get("max"))
                                  and _num_close(a["y_axis"].get("min"), b["y_axis"].get("min")))
            per["x_range"].append(_num_close(a["x_axis"].get("max"), b["x_axis"].get("max")))
        for k, v in per.items():
            agree[k] = all(v) if v else False
    else:
        for k in panel_fields:
            agree[k] = False

    # weight the structural fields; ranges count less (legitimate read variance)
    weights = {"is_km_figure": 0.20, "n_panels": 0.25, "n_arms": 0.20,
               "y_kind": 0.15, "x_unit": 0.10, "y_range": 0.05, "x_range": 0.05}
    confidence = round(sum(w for k, w in weights.items() if agree.get(k)), 4)

    # structural disagreement -> human review; range-only disagreement is tolerated
    structural = ["is_km_figure", "n_panels", "n_arms", "y_kind", "x_unit"]
    needs_review = not all(agree.get(k) for k in structural)
    return {"confidence": confidence, "needs_review": needs_review, "agree": agree}


def corpus_record(fig: Dict[str, Any], r1: Dict[str, Any], r2: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Build one labelled-corpus record from a figure ref + its two reads."""
    cmp = compare_labels(r1, r2)
    return {
        "id": fig["id"],
        "pdf": fig.get("pdf"),
        "page_index": fig.get("page_index"),
        "label": r1,                       # accepted read (read1); identical to r2 where agreed
        "label_alt": r2,
        "confidence": cmp["confidence"],
        "needs_human_review": cmp["needs_review"],
        "agreement": cmp["agree"],
        "provenance": "vlm-double-read-autolabel",
    }
