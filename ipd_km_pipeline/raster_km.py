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
from pathlib import Path
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


def _ensure_tesseract() -> bool:
    """Point pytesseract at the tesseract binary if it isn't on PATH. Returns
    True if OCR is usable."""
    try:
        import pytesseract
    except Exception:
        return False
    import shutil

    if shutil.which("tesseract"):
        return True
    for cand in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\mahmo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ):
        if Path(cand).exists():
            pytesseract.pytesseract.tesseract_cmd = cand
            return True
    return False


def ocr_tick_values(
    gray: np.ndarray, plot: PlotBox, positions: List[float], axis: str = "x",
    label_band: int = 30, y_label_width: int = 52, upscale: int = 4, thresh: int = 140,
) -> List[Optional[float]]:
    """OCR the numeric label beside each detected tick (digits + decimal).

    For each tick position, crop the label region (below the x-axis / left of
    the y-axis), upscale + binarise, and OCR with a digit whitelist. Returns a
    value per position (None where unreadable). This is the piece that makes
    raster calibration fully automatic -- no human clicks.
    """
    import re as _re

    if not _ensure_tesseract():
        raise RuntimeError("tesseract not available; cannot OCR tick values")
    import pytesseract

    from PIL import Image

    H, W = gray.shape
    pos = sorted(positions)
    half = (np.median(np.diff(pos)) / 2.0) if len(pos) > 1 else 12.0
    whitelist = "-c tessedit_char_whitelist=0123456789."
    out: List[Optional[float]] = []
    for p in pos:
        if axis == "x":
            x0 = int(max(p - half, 0)); x1 = int(min(p + half, W))
            y0 = plot.y1 + 2; y1 = min(plot.y1 + 2 + label_band, H)
        else:  # y-axis labels: wide enough for multi-digit (100); clear the
            # tick marks (which sit just left of the axis) from the right edge.
            y0 = int(max(p - half, 0)); y1 = int(min(p + half, H))
            x0 = max(plot.x0 - 8 - y_label_width, 0); x1 = max(plot.x0 - 8, 0)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            out.append(None); continue
        # tight-crop to the dark glyph bounding box (+pad) so OCR sees a centred
        # digit group rather than a small digit in a sea of white (which skews
        # thresholding and wrecks recognition).
        dmask = crop < thresh
        if dmask.any():
            rr = np.where(dmask.any(axis=1))[0]
            cc = np.where(dmask.any(axis=0))[0]
            r0, r1 = max(rr[0] - 2, 0), min(rr[-1] + 3, crop.shape[0])
            c0, c1 = max(cc[0] - 2, 0), min(cc[-1] + 3, crop.shape[1])
            crop = crop[r0:r1, c0:c1]
        if crop.size == 0:
            out.append(None); continue
        # smooth (LANCZOS) upscale BEFORE binarising -> clean glyph edges; nearest
        # upscaling leaves blocky digits that Tesseract misreads.
        big = Image.fromarray(crop).resize(
            (max(crop.shape[1] * upscale, 1), max(crop.shape[0] * upscale, 1)),
            Image.LANCZOS)
        a = np.asarray(big)
        # per-crop adaptive (mean) threshold beats a fixed cut on anti-aliased
        # upscaled glyphs (diagnosed: 7/8 vs 4/8 digit accuracy).
        t = float(a.mean())
        binimg = np.where(a < t, 0, 255).astype(np.uint8)
        binimg = np.pad(binimg, 16, mode="constant", constant_values=255)  # quiet margin
        val = None
        for psm in (7, 8):  # text-line, then single-word
            txt = pytesseract.image_to_string(
                binimg, config=f"--psm {psm} {whitelist}").strip()
            m = _re.search(r"\d+(?:\.\d+)?", txt)
            if m:
                val = float(m.group()); break
        out.append(val)
    return out


def _ransac_axis_fit(pairs, tol: float):
    """Robust line fit over (position, value) pairs, tolerant of OCR errors.

    Tick positions are exact and evenly spaced and values are an arithmetic
    sequence, so the true calibration line passes through MANY pairs while OCR
    misreads scatter off it. We take the pair-defined line with the most
    inliers (residual <= tol) and least-squares refit on those inliers.
    Returns (AxisFit, n_inliers).
    """
    from vector_km import _fit_axis

    best_inliers: List = []
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            (p1, v1), (p2, v2) = pairs[i], pairs[j]
            if p1 == p2:
                continue
            m = (v2 - v1) / (p2 - p1)
            b = v1 - m * p1
            inliers = [(p, v) for p, v in pairs if abs(m * p + b - v) <= tol]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
    # require >=3 consistent ticks (2 always fit a line perfectly -> degenerate);
    # below that, auto-calibration is unreliable and the caller should fall back
    # to 2-click. With <3 ticks total, all must agree.
    need = 3 if len(pairs) >= 3 else len(pairs)
    if len(best_inliers) < need:
        raise ValueError(
            f"robust fit: only {len(best_inliers)} consistent ticks (need {need}); "
            "fall back to 2-click calibration")
    return _fit_axis(best_inliers), len(best_inliers)


def auto_calibrate_axis(gray: np.ndarray, plot: PlotBox, axis: str = "x",
                        tol: Optional[float] = None, min_inlier_frac: float = 0.75):
    """EXPERIMENTAL, fails closed: detect tick positions + OCR values -> AxisFit.

    *** Tesseract OCR of small axis labels is NOT reliable *** (benchmarked at
    ~15-60% per-digit accuracy, erratic across font sizes). It can even return
    a confidently-wrong line from coincidentally-collinear misreads. So this
    fails CLOSED: it requires a strong inlier majority (>= ``min_inlier_frac``
    of OCR'd ticks AND >= 3) before trusting the fit, and otherwise RAISES so
    the caller falls back to the reliable 2-click path (``auto_axis_fit`` with
    detected positions + human/known values). Do not rely on this for
    unattended calibration; verify the result against the curve.
    """
    positions = sorted(detect_tick_positions(gray, plot, axis=axis))
    values = ocr_tick_values(gray, plot, positions, axis=axis)
    pairs = [(p, v) for p, v in zip(positions, values) if v is not None]
    if len(pairs) < 2:
        raise ValueError(f"auto-calibration {axis}: only {len(pairs)} ticks OCR'd")
    if tol is None:
        vals = [v for _, v in pairs]
        tol = max(2.0, 0.03 * (max(vals) - min(vals) or 1.0))
    fit, n_in = _ransac_axis_fit(pairs, tol)
    if n_in < max(3, int(np.ceil(min_inlier_frac * len(pairs)))):
        raise ValueError(
            f"auto-calibration {axis}: only {n_in}/{len(pairs)} OCR ticks agree "
            "-- unreliable; fall back to 2-click")
    return fit


_RAPIDOCR = None


def _ensure_rapidocr() -> bool:
    """Initialise the RapidOCR singleton; return True if usable."""
    global _RAPIDOCR
    if _RAPIDOCR is not None:
        return True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPIDOCR = RapidOCR()
        return True
    except Exception:
        return False


def _rapidocr_numeric(image: np.ndarray):
    """Run RapidOCR once; return numeric detections as (cx, cy, value).

    RapidOCR is a detector+recognizer, so each number comes WITH its location
    -- no per-tick cropping needed. Returns [] if RapidOCR isn't installed.
    """
    import re as _re
    if not _ensure_rapidocr():
        return []
    res, _ = _RAPIDOCR(image)
    out = []
    for it in (res or []):
        box, txt = it[0], str(it[1]).strip()
        # extract the FIRST number embedded in the label ("Month 18" -> 18,
        # "1.00-" -> 1.0). Skip labels with >1 number (e.g. a merged
        # "Month 6Month 9Month 12" or an at-risk "46 (4)") -- ambiguous position.
        nums = _re.findall(r"\d+(?:\.\d+)?", txt)
        if len(nums) != 1:
            continue
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        out.append((cx, cy, float(nums[0])))
    return out


def _rapidocr_all(image: np.ndarray):
    """All RapidOCR detections as (text, cx, cy). [] if unavailable."""
    out = []
    if not _ensure_rapidocr():
        return out
    try:
        res, _ = _RAPIDOCR(image)
    except Exception:
        return out
    for it in (res or []):
        xs = [p[0] for p in it[0]]; ys = [p[1] for p in it[0]]
        out.append((str(it[1]), sum(xs) / 4.0, sum(ys) / 4.0))
    return out


def extract_at_risk_raster(image: np.ndarray, plot: PlotBox, row_gap: float = 22.0):
    """Parse the numbers-at-risk table below a raster KM plot.

    At-risk cells are usually "N" or "N (events)"; we take the LEADING integer
    of each cell in the band below the x-axis, cluster detections into rows by
    y, sort each row by x. Returns a list of rows, each a list of (x, N) sorted
    by x (so row[0] is the baseline N). Empty if RapidOCR unavailable.
    """
    import re as _re

    band_top = plot.y1 + 0.18 * (plot.y1 - plot.y0)
    cells = []
    for txt, cx, cy in _rapidocr_all(image):
        if cy < band_top:
            continue
        m = _re.match(r"\s*(\d{1,4})", txt)  # leading integer = at-risk count
        if m and plot.x0 - 40 <= cx <= plot.x1 + 40:
            cells.append((cx, cy, int(m.group(1))))
    if not cells:
        return []
    cells.sort(key=lambda z: z[1])
    rows, cur, last = [], [], None
    for cx, cy, n in cells:
        if last is None or abs(cy - last) <= row_gap:
            cur.append((cx, n))
        else:
            rows.append(sorted(cur)); cur = [(cx, n)]
        last = cy
    if cur:
        rows.append(sorted(cur))
    return rows


def at_risk_reported_events(image: np.ndarray, plot: PlotBox, row_gap: float = 22.0):
    """Per at-risk row, the max parenthetical events from 'N (events)' cells.

    The at-risk table often reports cumulative events in parentheses, e.g.
    '46 (4)' ... '0 (12)'. The largest parenthetical in a row is the total
    events for that arm -- FIGURE-INTERNAL ground truth for validating the
    reconstruction without any external HR. Returns a list (one per row, by y)
    of ints, or None for rows with no parenthetical events found.
    """
    import re as _re

    band_top = plot.y1 + 0.18 * (plot.y1 - plot.y0)
    cells = []
    for txt, cx, cy in _rapidocr_all(image):
        if cy < band_top or not (plot.x0 - 40 <= cx <= plot.x1 + 40):
            continue
        m = _re.search(r"\((\d{1,4})\)", txt)  # parenthetical = cumulative events
        if m:
            cells.append((cy, int(m.group(1))))
    if not cells:
        return []
    cells.sort()
    rows, cur, last = [], [], None
    for cy, e in cells:
        if last is None or abs(cy - last) <= row_gap:
            cur.append(e)
        else:
            rows.append(max(cur) if cur else None); cur = [e]
        last = cy
    if cur:
        rows.append(max(cur))
    return rows


def auto_calibrate_axes(
    image: np.ndarray, plot: PlotBox, tol: float = 2.0, min_inlier_frac: float = 0.7,
):
    """Reliable raster auto-calibration via DUAL-ENGINE OCR + RANSAC.

    Collects (position, value) tick candidates from BOTH RapidOCR (band-filtered
    by location) and Tesseract (per detected tick position), for each axis, then
    robust-fits. The two engines are complementary (RapidOCR reads dense x-labels
    that Tesseract misses; Tesseract reads y-labels RapidOCR misses), and RANSAC
    rejects mis-reads and stray numbers (e.g. at-risk counts in the label band).

    Returns (x_fit, y_fit). Each axis FAILS CLOSED (raises) if too few ticks
    agree -- caller falls back to 2-click. Validate against the curve.
    """
    rapid = _rapidocr_numeric(image)
    x_cands: List[Tuple[float, float]] = []
    y_cands: List[Tuple[float, float]] = []
    # RapidOCR: classify by location relative to the axes
    band = max(8, int(0.10 * (plot.y1 - plot.y0)))
    for cx, cy, v in rapid:
        if plot.y1 < cy <= plot.y1 + band * 2 and plot.x0 - band <= cx <= plot.x1 + band:
            x_cands.append((cx, v))
        elif plot.x0 - band * 3 <= cx < plot.x0 and plot.y0 - band <= cy <= plot.y1 + band:
            y_cands.append((cy, v))
    # Tesseract per detected tick position (second, complementary source)
    try:
        xpos = sorted(detect_tick_positions(image, plot, "x"))
        for p, v in zip(xpos, ocr_tick_values(image, plot, xpos, "x")):
            if v is not None:
                x_cands.append((p, v))
        ypos = sorted(detect_tick_positions(image, plot, "y"))
        for p, v in zip(ypos, ocr_tick_values(image, plot, ypos, "y")):
            if v is not None:
                y_cands.append((p, v))
    except Exception:
        pass

    fits = {}
    for name, cands in (("x", x_cands), ("y", y_cands)):
        if len(cands) < 3:
            raise ValueError(f"auto-calibrate {name}: only {len(cands)} tick candidates")
        fit, n_in = _ransac_axis_fit(cands, tol)
        # accept on an ABSOLUTE consistent-tick count + high R^2 on the inliers.
        # (A fraction of all candidates is too strict: stray numbers like at-risk
        # counts inflate the denominator. 4 collinear evenly-spaced round-valued
        # ticks is not a coincidence.) Otherwise fail closed to 2-click.
        # >=4 inliers guards against spurious-collinear misreads; R^2>0.97
        # tolerates real-world OCR position noise while still rejecting garbage
        # fits (which score ~0.2-0.4).
        need = 4 if len(cands) >= 5 else 3
        if n_in < need or fit.r2 < 0.97:
            raise ValueError(
                f"auto-calibrate {name}: {n_in} ticks agree (R2={fit.r2:.3f}); "
                "unreliable -- fall back to 2-click")
        fits[name] = fit
    return fits["x"], fits["y"]


def auto_axis_fit(tick_positions: List[float], values: List[float]):
    """Build an AxisFit from detected tick POSITIONS + supplied VALUES.

    Reduces calibration input to the list of axis values (e.g. [0,6,..,66]);
    OCR can fill these in later for full automation. Requires len match.
    """
    from manual_calibration import axis_from_clicks

    if len(tick_positions) != len(values) or len(values) < 2:
        raise ValueError(f"tick/value count mismatch: {len(tick_positions)} vs {len(values)}")
    return axis_from_clicks(list(zip(sorted(tick_positions), sorted(values))))


def _rapidocr_text_boxes(image: np.ndarray, pad: int = 3):
    """All RapidOCR text bounding boxes (x0, top, x1, bottom), padded.

    Used to mask out IN-PLOT TEXT -- legend ("Acoramidis / placebo / Censored"),
    titles, annotations -- which otherwise contaminate the dark-curve cloud
    (the legend sits low in the plot and produces spurious downward spikes).
    """
    rows = []
    try:
        if _ensure_rapidocr():
            res, _ = _RAPIDOCR(image)
            for it in (res or []):
                xs = [p[0] for p in it[0]]
                ys = [p[1] for p in it[0]]
                rows.append((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
    except Exception:
        pass
    return rows


def dark_curve_cloud(
    gray: np.ndarray,
    plot: PlotBox,
    thresh: int = 140,
    margin: int = 2,
    exclude_boxes=None,
) -> np.ndarray:
    """Return (N, 2) dark-pixel coords (x_px, y_px) inside the plot interior.

    Dark pixels (below ``thresh``) within the plot box are curve candidates.
    Axis lines/ticks/labels are excluded by restricting to the interior with a
    small ``margin``. ``exclude_boxes`` (e.g. OCR text regions) are masked out
    so in-plot legend/annotation text doesn't pollute the curve. For
    overlapping same-colour curves the downstream continuity tracer resolves
    which pixel belongs to which arm.
    """
    x0, x1 = plot.x0 + margin, plot.x1 - margin
    y0, y1 = plot.y0 + margin, plot.y1 - margin
    sub = (gray[y0:y1, x0:x1] < thresh)
    for bx0, bt, bx1, bb in (exclude_boxes or []):
        ix0 = int(max(bx0 - x0, 0)); ix1 = int(max(bx1 - x0, 0))
        iy0 = int(max(bt - y0, 0)); iy1 = int(max(bb - y0, 0))
        sub[iy0:iy1, ix0:ix1] = False
    ys, xs = np.where(sub)
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


def pdf_raster_to_ipd(pdf_path: str, dpi: int = 200,
                      value_is_cumulative_incidence: bool = False, value_scale: float = 1.0):
    """FULL raster pipeline on a real PDF: locate -> render -> detect box ->
    dual-engine auto-calibrate -> text-masked curve extraction -> continuity
    arm separation -> monotone survival -> at-risk OCR -> Guyot IPD.

    Returns {page, kind, arms:[{label,time,event,n_events,final_survival,total_n}]}.
    Raises if no KM figure is located or auto-calibration fails closed (then a
    caller can fall back to 2-click via raster_to_ipd). Needs RapidOCR+Tesseract.
    """
    import figure_locator as _FL
    from manual_calibration import calibrate_points
    from vector_km import separate_arms
    from guyot import reconstruct_arm

    cands = _FL.locate_km_figures(pdf_path)
    if not cands:
        raise ValueError("no KM figure located")
    c = cands[0]
    SC = dpi / 72.0
    g = render_page(pdf_path, c.page_index, dpi=dpi)
    if c.bbox:
        x0, t, x1, b = c.bbox
        g = g[int(t * SC):int(b * SC), int(x0 * SC):int(x1 * SC)]
    box = detect_plot_box(g)
    if box is None:
        raise ValueError("no plot box detected")
    x_fit, y_fit = auto_calibrate_axes(g, box)  # fails closed
    boxes = _rapidocr_text_boxes(g)
    cloud = column_curve_points(dark_curve_cloud(g, box, exclude_boxes=boxes), n_curves=2)
    t_, v_ = calibrate_points(cloud, x_fit, y_fit, monotone="none")
    gap = max(0.02 * (float(np.ptp(v_)) or 1.0), 1e-6)
    arms = separate_arms(np.column_stack([t_, v_]), n_arms=2, gap=gap)
    at_risk = extract_at_risk_raster(g, box)

    out = []
    for i, a in enumerate(arms):
        if a.size == 0:
            continue
        a = a[a[:, 0].argsort()]
        if value_is_cumulative_incidence:
            surv = np.clip(1.0 - a[:, 1] / value_scale, 0.0, 1.0)
        else:
            surv = np.clip(a[:, 1] / value_scale, 0.0, 1.0)
        surv = np.minimum.accumulate(surv)  # KM survival is non-increasing
        tn = at_risk[i][0][1] if i < len(at_risk) and at_risk[i] else None
        et, ev = reconstruct_arm(a[:, 0], surv, total_n=tn)
        out.append({"label": f"arm{i}", "time": et, "event": ev,
                    "n_events": int(ev.sum()), "final_survival": float(surv[-1]),
                    "total_n": tn})
    return {"page": c.page_index, "kind": c.kind, "caption": c.caption, "arms": out}


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
