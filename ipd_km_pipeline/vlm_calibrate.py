#!/usr/bin/env python3
"""
VLM-assisted axis calibration (roadmap lever 1 -- the dominant-failure lever).
=============================================================================

The raster path's #1 failure is axis CALIBRATION: ``raster_km.auto_calibrate_axes``
detects exact tick *positions* with classical CV, but reads tick *values* with
Tesseract/RapidOCR -- and small axis-label OCR is unreliable (benchmarked
~15-60% per-digit), which is why 7/18 corpus figures land in ``calib_fail``.

This module keeps the division of labour the roadmap prescribes:

  * **classical CV owns geometry** -- ``raster_km.detect_tick_positions`` gives
    sub-pixel tick positions (exact, never hallucinated);
  * **a vision-language model owns reading** -- the ordered tick *values* and
    axis *semantics* (time vs survival vs cumulative-incidence, %, log). The VLM
    leapfrogs the OCR crux but is NEVER trusted for precise geometry.

Two execution paths share one ingest core (so both are testable, and the
agent path needs no network):

  * **API path** (unattended): ``calibrate_via_api`` base64-encodes the axis-label
    crops and calls Claude with a FORCED structured tool call (vision), then
    pairs the returned values with CV positions and fits.
  * **Agent/workflow path** (this repo's "emit packet -> Claude Code -> ingest"
    pattern): ``build_packet`` writes the crops + schema to disk; an agent reads
    the crop images and returns the structured answer; ``ingest_vlm_answer``
    pairs + fits. Identical fit logic to the API path.

Like ``auto_calibrate_axes`` it FAILS CLOSED: if too few CV ticks pair with VLM
values, the fit R^2 is low, or the axis is log-scaled (unsupported), it raises so
the caller falls back to the reliable 2-click path. Correct-or-refuse, never a
confidently-wrong calibration.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vector_km import AxisFit, _fit_axis

# Default model for the unattended API path. Opus 4.8 has the strongest vision;
# override with VLM_CALIBRATE_MODEL (e.g. claude-sonnet-4-6 for cheaper runs).
DEFAULT_MODEL = os.environ.get("VLM_CALIBRATE_MODEL", "claude-opus-4-8")

# JSON Schema for the structured answer. Used both as the forced-tool input_schema
# (API path) and as the contract documented to agents (workflow path). The VLM
# returns ONLY values + semantics -- never pixel coordinates (geometry is CV's job).
_AXIS_PROPS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tick_values": {
            "type": "array",
            "items": {"type": "number"},
            "description": (
                "Numeric value at each labelled tick, in reading order "
                "(x-axis: LEFT to RIGHT; y-axis: TOP to BOTTOM). Include only "
                "ticks that carry a printed number. Do NOT invent evenly-spaced "
                "values the figure does not print."
            ),
        },
        "is_log": {
            "type": "boolean",
            "description": "True only if the axis is explicitly log-scaled.",
        },
        "is_percent": {
            "type": "boolean",
            "description": "True if the axis is labelled in percent (0-100), not a 0-1 proportion.",
        },
        "axis_kind": {
            "type": "string",
            "enum": ["time", "survival", "cumulative_incidence", "other"],
            "description": (
                "time = x-axis (months/years/days); survival = y-axis decreasing "
                "from ~1/100; cumulative_incidence = y-axis increasing from 0; "
                "other = anything else."
            ),
        },
    },
    "required": ["tick_values", "is_log", "is_percent", "axis_kind"],
}

CALIBRATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"x_axis": _AXIS_PROPS, "y_axis": _AXIS_PROPS},
    "required": ["x_axis", "y_axis"],
}

_TOOL_NAME = "report_axis_calibration"
_TOOL_DESCRIPTION = (
    "Report the numeric values printed at each labelled axis tick of a "
    "Kaplan-Meier plot, plus each axis's semantics. Report VALUES and SEMANTICS "
    "only -- never pixel positions. x-axis ticks in LEFT-to-RIGHT order; y-axis "
    "ticks in TOP-to-BOTTOM order. Omit ticks with no printed number."
)

_INSTRUCTION = (
    "These two crops are the X-axis label band (below the plot) and the Y-axis "
    "label band (left of the plot) of a Kaplan-Meier survival figure. Read the "
    "printed number at each tick. Return x-axis values LEFT-to-RIGHT and y-axis "
    "values TOP-to-BOTTOM. Do not guess values that are not printed; omit any "
    "tick whose number you cannot read."
)


# --------------------------------------------------------------------------- #
# Cropping the axis-label bands
# --------------------------------------------------------------------------- #
def axis_label_crop(
    gray: np.ndarray,
    plot,
    axis: str = "x",
    label_band: int = 34,
    y_label_width: int = 60,
    pad: int = 4,
) -> np.ndarray:
    """Crop the band that carries one axis's printed tick labels.

    x-axis: a horizontal strip spanning the plot width, just below the axis.
    y-axis: a vertical strip spanning the plot height, just left of the axis.
    Returns a uint8 (H, W) crop (possibly empty if the band is off-image).
    """
    H, W = gray.shape
    if axis == "x":
        y0 = max(plot.y1 - pad, 0)
        y1 = min(plot.y1 + label_band, H)
        x0 = max(plot.x0 - pad, 0)
        x1 = min(plot.x1 + pad, W)
    else:
        y0 = max(plot.y0 - pad, 0)
        y1 = min(plot.y1 + pad, H)
        x0 = max(plot.x0 - y_label_width, 0)
        x1 = min(plot.x0 + pad, W)
    return gray[y0:y1, x0:x1]


def _encode_png(arr: np.ndarray) -> str:
    """Base64-encode a uint8 grayscale array as a PNG (for the vision API)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def build_packet(gray: np.ndarray, plot, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Build a calibration packet: the two axis-label crops + the answer schema.

    If ``out_dir`` is given, the crops are written as ``x_axis.png`` /
    ``y_axis.png`` there (for the agent/workflow path, where an agent reads the
    PNGs). Always returns base64 strings (for the API path) plus the schema and
    instruction text. Geometry (tick positions) is recomputed at ingest time, so
    the packet carries no pixel coordinates the VLM could anchor to.
    """
    from PIL import Image

    x_crop = axis_label_crop(gray, plot, "x")
    y_crop = axis_label_crop(gray, plot, "y")
    packet: Dict[str, Any] = {
        "instruction": _INSTRUCTION,
        "schema": CALIBRATION_SCHEMA,
        "x_axis_b64": _encode_png(x_crop) if x_crop.size else None,
        "y_axis_b64": _encode_png(y_crop) if y_crop.size else None,
    }
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        if x_crop.size:
            Image.fromarray(x_crop).save(d / "x_axis.png")
            packet["x_axis_path"] = str(d / "x_axis.png")
        if y_crop.size:
            Image.fromarray(y_crop).save(d / "y_axis.png")
            packet["y_axis_path"] = str(d / "y_axis.png")
        (d / "packet.json").write_text(
            json.dumps(
                {k: v for k, v in packet.items() if not k.endswith("_b64")},
                indent=2,
            )
        )
    return packet


# --------------------------------------------------------------------------- #
# Pair VLM values with CV positions and fit (the shared ingest core)
# --------------------------------------------------------------------------- #
def _merge_split_peaks(positions: List[float], min_gap_frac: float = 0.4) -> List[float]:
    """Merge tick peaks that sit closer than ``min_gap_frac`` x median spacing.

    A single rendered tick mark sometimes splits into two adjacent CV peaks
    (anti-aliasing + the small min_gap in detect_tick_positions), which inflates
    the tick COUNT and breaks the count-match against the VLM read. Peaks 6px
    apart when the real tick spacing is ~60px are unambiguously one tick, so we
    collapse each run of near-peaks to its centroid. This is geometric and safe
    (it never invents or drops a real tick); genuine under-detection (a missing
    tick) is left alone, so it still fails closed at the count-match gate.
    """
    pos = sorted(positions)
    if len(pos) < 3:
        return pos
    med = float(np.median(np.diff(pos)))
    if med <= 0:
        return pos
    thr = min_gap_frac * med
    groups: List[List[float]] = [[pos[0]]]
    for p in pos[1:]:
        if p - groups[-1][-1] < thr:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [sum(g) / len(g) for g in groups]


def _pair_and_fit(
    positions: List[float], values: List[float], axis: str, min_r2: float = 0.99
) -> AxisFit:
    """Pair sorted CV tick positions with VLM-read values (same spatial order)
    and robust-fit a linear calibration. Fails closed on count mismatch / low R^2.

    Both sequences are in ascending-position order: x left->right, y top->bottom.
    The VLM is asked for values in exactly that order, so index i of each list is
    the same physical tick. RANSAC (reused from raster_km) rejects a single
    misread; we then require >=3 inliers and high R^2 or raise.
    """
    from raster_km import _ransac_axis_fit

    pos = sorted(positions)
    if len(values) < 2:
        raise ValueError(f"VLM {axis}: only {len(values)} tick values read")
    if len(pos) != len(values):
        # Geometry and reading disagree on tick count -- ambiguous alignment.
        # Fail closed rather than guess which positions/values to drop.
        raise ValueError(
            f"VLM {axis}: {len(pos)} CV ticks vs {len(values)} read values "
            "-- count mismatch, fall back to 2-click"
        )
    pairs = list(zip(pos, values))
    span = (max(values) - min(values)) or 1.0
    tol = max(2.0, 0.03 * abs(span))
    fit, n_in = _ransac_axis_fit(pairs, tol)
    need = 3 if len(pairs) >= 3 else len(pairs)
    if n_in < need or fit.r2 < min_r2:
        raise ValueError(
            f"VLM {axis}: {n_in}/{len(pairs)} ticks agree (R2={fit.r2:.4f}); "
            "unreliable -- fall back to 2-click"
        )
    return fit


def _axis_values(answer_axis: Dict[str, Any], axis: str) -> List[float]:
    """Validate one axis sub-answer and return its tick values (raises on log)."""
    if answer_axis.get("is_log"):
        raise ValueError(f"VLM {axis}: log-scaled axis unsupported -- fall back to 2-click")
    vals = answer_axis.get("tick_values") or []
    return [float(v) for v in vals]


def ingest_vlm_answer(
    gray: np.ndarray, plot, answer: Dict[str, Any], min_r2: float = 0.99,
    verify_agree: Optional[bool] = None,
    external_max_time: Optional[float] = None,
    at_risk_col_times: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[AxisFit, AxisFit, Dict[str, Any]]:
    """Pair a structured VLM answer with classical-CV tick positions and fit.

    ``answer`` matches ``CALIBRATION_SCHEMA`` (``x_axis``/``y_axis`` each with
    ``tick_values`` in reading order + semantics). Returns ``(x_fit, y_fit,
    meta)`` where meta carries the detected semantics. Raises (fail closed) if
    either axis can't be fit. This is the single fit core used by BOTH the API
    and agent/workflow paths -- network-free and unit-tested.
    """
    from raster_km import detect_tick_positions

    xa, ya = answer["x_axis"], answer["y_axis"]
    x_vals = _axis_values(xa, "x")
    y_vals = _axis_values(ya, "y")

    # merge split peaks (one tick read as two adjacent peaks) before pairing;
    # genuine missing ticks are NOT invented, so under-detection still fails closed.
    x_pos = _merge_split_peaks(sorted(detect_tick_positions(gray, plot, "x")))
    y_pos = _merge_split_peaks(sorted(detect_tick_positions(gray, plot, "y")))

    x_fit = _pair_and_fit(x_pos, x_vals, "x", min_r2)
    y_fit = _pair_and_fit(y_pos, y_vals, "y", min_r2)

    meta = {
        "y_is_percent": bool(ya.get("is_percent")),
        "y_kind": ya.get("axis_kind"),
        "x_kind": xa.get("axis_kind"),
        "value_is_cumulative_incidence": ya.get("axis_kind") == "cumulative_incidence",
        "value_scale": 100.0 if ya.get("is_percent") else 1.0,
        "x_ticks": len(x_vals),
        "y_ticks": len(y_vals),
        "x_value_min": min(x_vals), "x_value_max": max(x_vals),
        "y_value_min": min(y_vals), "y_value_max": max(y_vals),
    }
    # lever 4: confidence that PREDICTS correctness (not just R^2), for the
    # auto-accept-vs-human-flag gate. verify_agree comes from the double-read.
    # An external value cross-check (caption follow-up / at-risk times) can veto
    # auto-accept -- the only thing that catches a correlated identical misread.
    from confidence import calibration_confidence
    from axis_cross_check import cross_check_calibration
    xcheck = None
    if external_max_time is not None or at_risk_col_times is not None:
        xcheck = cross_check_calibration(
            x_fit, y_fit, meta, plot,
            external_max_time=external_max_time, at_risk_col_times=at_risk_col_times)
        meta["cross_check"] = xcheck
    meta["confidence"] = calibration_confidence(
        meta, x_fit.r2, y_fit.r2, len(x_vals), len(y_vals),
        verify_agree=verify_agree, cross_check=xcheck)
    return x_fit, y_fit, meta


# --------------------------------------------------------------------------- #
# API path (unattended) -- vision + forced structured tool call
# --------------------------------------------------------------------------- #
def calibrate_via_api(
    gray: np.ndarray,
    plot,
    model: str = DEFAULT_MODEL,
    min_r2: float = 0.99,
) -> Tuple[AxisFit, AxisFit, Dict[str, Any]]:
    """Read tick values from the axis-label crops via the Claude vision API
    (forced structured tool call), then pair + fit. Requires ANTHROPIC_API_KEY
    and the ``anthropic`` SDK. Raises (fail closed) on any failure.
    """
    import anthropic

    packet = build_packet(gray, plot)
    if not packet["x_axis_b64"] or not packet["y_axis_b64"]:
        raise ValueError("VLM: axis-label band off-image; cannot crop")

    client = anthropic.Anthropic()
    content = [
        {"type": "text", "text": packet["instruction"]},
        {"type": "text", "text": "X-axis label band:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": packet["x_axis_b64"]}},
        {"type": "text", "text": "Y-axis label band:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": packet["y_axis_b64"]}},
    ]
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[{
            "name": _TOOL_NAME,
            "description": _TOOL_DESCRIPTION,
            "strict": True,
            "input_schema": CALIBRATION_SCHEMA,
        }],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": content}],
    )
    answer = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if answer is None:
        raise ValueError("VLM: model returned no structured calibration")
    return ingest_vlm_answer(gray, plot, answer, min_r2)


def vlm_calibrate_axes(
    gray: np.ndarray,
    plot,
    answer: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    min_r2: float = 0.99,
    verify_agree: Optional[bool] = None,
) -> Tuple[AxisFit, AxisFit]:
    """Drop-in companion to ``raster_km.auto_calibrate_axes`` -> ``(x_fit, y_fit)``.

    If ``answer`` is provided (agent/workflow path or replay), ingest it; else
    call the Claude vision API. Fails closed (raises) -- caller falls back to
    2-click. Discards the semantics meta to match auto_calibrate_axes' signature;
    use ``calibrate_via_api`` / ``ingest_vlm_answer`` when you want the meta.
    """
    if answer is not None:
        x_fit, y_fit, _ = ingest_vlm_answer(gray, plot, answer, min_r2, verify_agree)
    else:
        x_fit, y_fit, _ = calibrate_via_api(gray, plot, model, min_r2)
    return x_fit, y_fit
