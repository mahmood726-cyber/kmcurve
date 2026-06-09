#!/usr/bin/env python3
"""
Manual (2-click) calibration + vector/raster router
===================================================

The honest answer to *flattened raster* KM figures (scanned PDFs, figures
inserted as images) where ``vector_km`` finds no vector text/paths: a human
clicks two known points on each axis, exactly like WebPlotDigitizer, and the
pixel curve (from any raster pixel-extractor) is calibrated against those
reference points. Full automation of raster axis calibration is unreliable; the
2-click step is the accepted standard even in published Guyot-reconstruction
papers.

This module provides:

* :func:`figure_is_vector` -- the router test: does a region carry enough
  vector text/paths to use the exact ``vector_km`` path, or must we fall back?
* :func:`two_point_axis` / :func:`calibrate_points` -- the calibration core,
  reusing ``vector_km.AxisFit`` so downstream code (and Guyot) is identical
  whether the data came from the vector or raster path.

The pixel-curve input is intentionally decoupled: feed it pixels from the
repo's raster extractor, a WebPlotDigitizer export, or the HTML KMDigitizer.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from vector_km import AxisFit, _fit_axis


# --------------------------------------------------------------------------- #
# Router: is this figure/region vector (exact) or raster (needs 2-click)?
# --------------------------------------------------------------------------- #
def figure_is_vector(
    page,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_curve_points: int = 200,
    min_numeric_words: int = 4,
    tick_pad: float = 35.0,
) -> bool:
    """Heuristic: True if ``page`` (optionally within ``bbox``) has enough
    vector curve geometry AND numeric tick text to use exact extraction.

    ``bbox`` is (x0, top, x1, bottom) in PDF points. A flattened raster figure
    has its curve baked into an image, so it carries almost no vector curve
    points and no separate numeric text near the plot region. The numeric-text
    check pads the bbox by ``tick_pad`` because axis tick labels sit just
    *outside* the plotted area.
    """
    import re

    def _in(x, y, pad=0.0):
        if bbox is None:
            return True
        return (
            bbox[0] - pad <= x <= bbox[2] + pad and bbox[1] - pad <= y <= bbox[3] + pad
        )

    curve_pts = 0
    for c in page.curves:
        for px, py in c.get("pts", []):
            if _in(px, py):
                curve_pts += 1
                if curve_pts >= min_curve_points:
                    break
        if curve_pts >= min_curve_points:
            break

    numeric = 0
    num_re = re.compile(r"\d{1,3}(?:\.\d+)?%?")
    for w in page.extract_words():
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        if _in(cx, cy, pad=tick_pad) and num_re.fullmatch(w["text"]):
            numeric += 1

    return curve_pts >= min_curve_points and numeric >= min_numeric_words


# --------------------------------------------------------------------------- #
# 2-click calibration core
# --------------------------------------------------------------------------- #
def two_point_axis(p1: float, v1: float, p2: float, v2: float) -> AxisFit:
    """Build an AxisFit from two (pixel, value) reference clicks on one axis."""
    if p1 == p2:
        raise ValueError("The two reference clicks must differ in pixel position")
    return _fit_axis([(p1, v1), (p2, v2)])


def axis_from_clicks(refs: Sequence[Tuple[float, float]]) -> AxisFit:
    """Build an AxisFit from >=2 (pixel, value) clicks (least-squares if >2)."""
    if len(refs) < 2:
        raise ValueError("Need >=2 reference clicks to calibrate an axis")
    return _fit_axis(list(refs))


def calibrate_points(
    points_px: np.ndarray,
    x_fit: AxisFit,
    y_fit: AxisFit,
    monotone: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """Map an (N, 2) pixel curve to calibrated (time, value) arrays.

    ``points_px`` columns are (px, py). ``monotone`` may be "increasing"
    (cumulative incidence), "decreasing" (survival) or "none".
    """
    pts = np.asarray(points_px, dtype=float)
    if pts.size == 0:
        return np.empty(0), np.empty(0)
    time = x_fit.slope * pts[:, 0] + x_fit.intercept
    value = y_fit.slope * pts[:, 1] + y_fit.intercept
    order = np.argsort(time)
    time, value = time[order], value[order]
    if monotone == "increasing":
        np.maximum.accumulate(value, out=value)
    elif monotone == "decreasing":
        np.minimum.accumulate(value, out=value)
    return time, value


def survival_from_value(
    value: np.ndarray, is_cumulative_incidence: bool = False, scale: float = 1.0
) -> np.ndarray:
    """Convert a calibrated y-series to survival in [0, 1].

    ``scale`` is 100 for percent axes, 1 for 0-1 proportions.
    """
    v = np.asarray(value, dtype=float) / scale
    surv = (1.0 - v) if is_cumulative_incidence else v
    return np.clip(surv, 0.0, 1.0)
