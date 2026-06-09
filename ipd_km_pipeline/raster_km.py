#!/usr/bin/env python3
"""
Raster fallback: extract KM curves from FLATTENED (image) figures.
==================================================================

The corpus scan (CORPUS_FINDINGS.md) showed most Europe-PMC OA PDFs rasterise
their figures -- the tick labels are baked into the image, so the exact vector
path cannot fire and there is no vector text to calibrate from. This module
handles that common case.

Design: the raster path only needs (1) the figure rendered to pixels and (2) a
dark-curve pixel cloud from the plot interior. Everything downstream is REUSED:
- calibration: ``manual_calibration`` 2-click (the accepted semi-automatic
  standard; raster auto-calibration via OCR is the legacy 0% problem),
- arm separation: ``vector_km.separate_arms`` (velocity-aware continuity),
- IPD: ``guyot.reconstruct_arm`` + ``logrank_hr``.

So a raster figure becomes usable with two axis clicks per axis, and produces
the identical IPD/HR output as the vector path.

``render_*`` needs pypdfium2; the pixel tracer is pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def render_page(pdf_path: str, page_index: int, dpi: int = 300) -> np.ndarray:
    """Render a PDF page to a grayscale (H, W) uint8 array."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_index]
        bmp = page.render(scale=dpi / 72.0, grayscale=True)
        img = bmp.to_pil().convert("L")
        return np.asarray(img, dtype=np.uint8)
    finally:
        pdf.close()


@dataclass
class PlotBox:
    """Plot interior in *pixel* coords of a rendered image (axis box)."""

    x0: int
    y0: int  # top
    x1: int
    y1: int  # bottom


def detect_plot_box(gray: np.ndarray, thresh: int = 140, frac: float = 0.4) -> Optional[PlotBox]:
    """Auto-detect the plot interior from axis lines via projection profiles.

    The x-axis is the strongest near-full-width dark horizontal line and the
    y-axis the strongest near-full-height dark vertical line. Pure numpy (no
    OpenCV). Returns None if no clear axis frame is found. Removes the need for
    a manually supplied PlotBox for single-panel raster figures.
    """
    dark = gray < thresh
    H, W = dark.shape
    col_sum = dark.sum(axis=0)  # vertical lines -> tall columns
    row_sum = dark.sum(axis=1)  # horizontal lines -> wide rows
    # y-axis: a column whose dark count covers >= frac of the height; take the
    # LEFTMOST strong one. x-axis: a row covering >= frac of width; BOTTOM-MOST.
    col_cand = np.where(col_sum >= frac * H)[0]
    row_cand = np.where(row_sum >= frac * W)[0]
    if col_cand.size == 0 or row_cand.size == 0:
        return None
    x0 = int(col_cand[0])     # left (y-axis) column
    y1 = int(row_cand[-1])    # bottom (x-axis) row
    # derive the other two edges from the actual axis-line extents
    xaxis_cols = np.where(dark[y1, :])[0]
    yaxis_rows = np.where(dark[:, x0])[0]
    x1 = int(xaxis_cols.max()) if xaxis_cols.size else W - 1
    y0 = int(yaxis_rows.min()) if yaxis_rows.size else 0
    if x1 - x0 < 0.2 * W or y1 - y0 < 0.2 * H:
        return None
    return PlotBox(x0=x0, y0=y0, x1=x1, y1=y1)


def detect_tick_positions(
    gray: np.ndarray, plot: PlotBox, axis: str = "x",
    thresh: int = 140, band: int = 8, min_gap: int = 4,
) -> List[float]:
    """Detect axis tick-mark pixel positions just outside the plot box.

    Ticks are short dark marks perpendicular to the axis, in the margin band
    beyond it. Returns their centre positions (x for the x-axis, y for the
    y-axis) as peaks in the margin projection. Pure numpy.
    """
    dark = gray < thresh
    if axis == "x":  # ticks below the x-axis (rows y1 .. y1+band)
        strip = dark[plot.y1 + 1: plot.y1 + 1 + band, plot.x0: plot.x1 + 1]
        prof = strip.sum(axis=0)
        offset = plot.x0
    else:            # y-axis ticks left of x0 (cols x0-band .. x0)
        strip = dark[plot.y0: plot.y1 + 1, max(plot.x0 - band, 0): plot.x0]
        prof = strip.sum(axis=1)
        offset = plot.y0
    if prof.size == 0 or prof.max() == 0:
        return []
    peak_thresh = max(1, 0.4 * prof.max())
    peaks, i = [], 0
    while i < len(prof):
        if prof[i] >= peak_thresh:
            j = i
            while j < len(prof) and prof[j] >= peak_thresh:
                j += 1
            peaks.append(offset + (i + j - 1) / 2.0)
            i = j + min_gap
        else:
            i += 1
    return peaks


def auto_axis_fit(tick_positions: List[float], values: List[float]):
    """Build an AxisFit from detected tick POSITIONS + supplied VALUES.

    Reduces calibration input to the list of axis values (e.g. [0,6,..,66]);
    OCR can fill these in later for full automation. Requires len match.
    """
    from manual_calibration import axis_from_clicks

    if len(tick_positions) != len(values) or len(values) < 2:
        raise ValueError(f"tick/value count mismatch: {len(tick_positions)} vs {len(values)}")
    return axis_from_clicks(list(zip(sorted(tick_positions), sorted(values))))


def dark_curve_cloud(
    gray: np.ndarray,
    plot: PlotBox,
    thresh: int = 140,
    margin: int = 2,
) -> np.ndarray:
    """Return (N, 2) dark-pixel coords (x_px, y_px) inside the plot interior.

    Dark pixels (below ``thresh``) within the plot box are curve candidates.
    Axis lines/ticks/labels are excluded by restricting to the interior with a
    small ``margin``. For overlapping same-colour curves the downstream
    continuity tracer resolves which pixel belongs to which arm.
    """
    x0, x1 = plot.x0 + margin, plot.x1 - margin
    y0, y1 = plot.y0 + margin, plot.y1 - margin
    sub = gray[y0:y1, x0:x1]
    ys, xs = np.where(sub < thresh)
    if xs.size == 0:
        return np.empty((0, 2))
    return np.column_stack([xs + x0, ys + y0]).astype(float)


def column_curve_points(
    cloud: np.ndarray, n_curves: int = 2, col_px: float = 1.0, gap_px: float = 4.0
) -> np.ndarray:
    """Collapse a dark-pixel cloud to per-column curve points.

    For each x-column, dark pixels form 1..n_curves vertical runs; we take the
    median y of each run (split at gaps > gap_px). Returns an (M, 2) cloud of
    (x_px, y_px) suitable for calibration + continuity separation. This dramat-
    ically thins the cloud and removes vertical-stroke artefacts.
    """
    if cloud.size == 0:
        return cloud
    out: List[Tuple[float, float]] = []
    xcol = np.round(cloud[:, 0] / col_px) * col_px
    for cx in np.unique(xcol):
        ys = np.sort(cloud[xcol == cx, 1])
        if ys.size == 0:
            continue
        # split into runs at large gaps, keep up to n_curves run-medians
        runs, start = [], 0
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] > gap_px:
                runs.append(ys[start:i]); start = i
        runs.append(ys[start:])
        runs.sort(key=lambda r: -len(r))  # densest runs first
        for r in runs[:n_curves]:
            out.append((cx, float(np.median(r))))
    return np.array(out, dtype=float) if out else np.empty((0, 2))


def raster_to_ipd(
    gray: np.ndarray,
    plot: PlotBox,
    x_refs: List[Tuple[float, float]],
    y_refs: List[Tuple[float, float]],
    n_arms: int = 2,
    value_is_cumulative_incidence: bool = False,
    value_scale: float = 1.0,
    nar: Optional[List[dict]] = None,
):
    """Full raster path: dark cloud -> calibrate (2-click) -> arms -> IPD.

    ``x_refs`` / ``y_refs`` are >=2 (pixel, value) reference clicks per axis (in
    the SAME rendered-image pixel space as ``gray``). Returns a list of dicts
    {label, time, event, n_events} (one per arm), reconstructed via Guyot.
    """
    from manual_calibration import axis_from_clicks, calibrate_points
    from vector_km import separate_arms
    from guyot import reconstruct_arm

    xfit = axis_from_clicks(x_refs)
    yfit = axis_from_clicks(y_refs)

    cloud_px = column_curve_points(dark_curve_cloud(gray, plot), n_curves=n_arms)
    if cloud_px.size == 0:
        return []
    t, v = calibrate_points(cloud_px, xfit, yfit, monotone="none")
    cal = np.column_stack([t, v])
    # gap must scale to the value units (survival 0-1 vs CI 0-100), else arms
    # never split (separate_arms' default gap=1.0 is tuned for percent axes).
    gap = max(0.02 * (float(np.ptp(v)) or 1.0), 1e-6)
    arm_clouds = separate_arms(cal, n_arms=n_arms, gap=gap)

    out = []
    for i, ac in enumerate(arm_clouds):
        if ac.size == 0:
            continue
        order = np.argsort(ac[:, 0])
        tt, vv = ac[order, 0], ac[order, 1]
        if value_is_cumulative_incidence:
            surv = np.clip(1.0 - vv / value_scale, 0.0, 1.0)
            surv = np.minimum.accumulate(surv)
        else:
            surv = np.clip(vv / value_scale, 0.0, 1.0)
            surv = np.minimum.accumulate(surv)
        nt = nar[i]["times"] if nar and i < len(nar) else None
        nv = nar[i]["counts"] if nar and i < len(nar) else None
        total_n = int(nv[0]) if nv is not None else None
        et, ev = reconstruct_arm(tt, surv, nar_times=nt, nar_values=nv, total_n=total_n)
        out.append({"label": f"arm{i}", "time": et, "event": ev,
                    "n_events": int(ev.sum())})
    return out
