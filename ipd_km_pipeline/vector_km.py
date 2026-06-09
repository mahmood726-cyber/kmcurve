#!/usr/bin/env python3
"""
Vector KM extractor (pdfplumber backend)
========================================

Extracts Kaplan-Meier / cumulative-incidence curves from *vector* PDF figures
with exact, OCR-free axis calibration.

Why this exists
---------------
The earlier ``vector_curve_extractor.py`` had the right idea but three fatal
flaws that left the wider pipeline stuck on the "axis calibration = 0% success"
problem (see ``OCR_INVESTIGATION_RESULTS.md``):

1. It calibrated from the *whole page*, so at-risk-table counts, p-values and
   "95%" tokens polluted the fit.
2. It fit calibration from only the two extreme-position numbers
   (``x_vals[0]`` / ``x_vals[-1]``) -- one stray token breaks it.
3. It had no figure-region detection and no support for multi-panel pages, and
   it imported ``fitz`` (PyMuPDF), whose C-extension fails to load in this
   environment.

This module fixes all three:

* **Per-panel detection.** It finds each plot's axis frame (a long vertical
  y-axis line meeting a long horizontal x-axis line) and calibrates *inside*
  that panel only.
* **Robust calibration.** It least-squares fits *all* tick labels adjacent to
  each axis and reports R^2 so a bad fit is visible, not silent.
* **No OCR, no rasterization.** Tick values and curve geometry both come from
  the PDF vector layer via pdfplumber.

Validated on the bundled ADVANCE-trial fixture (``NEJMoa0802987.pdf``, page
index 7, Figure 3): 4 panels, calibration R^2 = 1.000000 on both axes.

Remaining known limitation: arms of the *same stroke colour* are separated
geometrically (upper vs lower trajectory per x-column). Crossing same-colour
arms are assigned by vertical order at each column, which matches how a reader
disambiguates them but is not infallible. Colour-coded arms separate cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pdfplumber is the only hard dependency for vector extraction
    import pdfplumber
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "vector_km requires pdfplumber. Install with: pip install pdfplumber"
    ) from exc


_NUMERIC = re.compile(r"\d{1,3}(?:\.\d+)?")


def _to_number(text: str) -> Optional[float]:
    """Parse a tick-label token like '12', '0.0', '25%' to a float, else None."""
    t = text.strip().replace("%", "")
    if _NUMERIC.fullmatch(t):
        return float(t)
    return None


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class AxisFit:
    """Linear pixel -> value calibration for one axis (value = slope*px + b)."""

    slope: float
    intercept: float
    r2: float
    n_ticks: int

    def value(self, px: float) -> float:
        return self.slope * px + self.intercept


@dataclass
class Panel:
    """A single plot region defined by its axis frame, in PDF points."""

    index: int
    x0: float  # left edge (y-axis pixel x)
    x1: float  # right edge (plot right)
    y_top: float
    y_bottom: float  # bottom edge (x-axis pixel y)

    @property
    def yaxis_x(self) -> float:
        return self.x0

    @property
    def xaxis_y(self) -> float:
        return self.y_bottom


@dataclass
class Arm:
    """One reconstructed curve trace within a panel."""

    label: str  # "upper" / "lower" (positional; map to treatment via legend)
    time: np.ndarray
    value: np.ndarray  # calibrated y in the panel's native y-units
    identity: str = ""  # treatment name from the in-plot legend, if matched

    def n_points(self) -> int:
        return int(self.time.size)


@dataclass
class PanelResult:
    panel: Panel
    x_fit: AxisFit
    y_fit: AxisFit
    arms: List[Arm] = field(default_factory=list)
    # fraction of timepoints where the arms are resolvably separated; low
    # values flag near-overlapping same-colour curves whose split is unreliable
    separation_confidence: float = 1.0


# --------------------------------------------------------------------------- #
# Panel / axis-frame detection
# --------------------------------------------------------------------------- #
def _long_horizontals(page, min_len: float, tol: float) -> List[dict]:
    out = []
    for l in page.lines:
        if abs(l["y0"] - l["y1"]) < tol and abs(l["x1"] - l["x0"]) >= min_len:
            out.append(l)
    return out


def _long_verticals(page, min_len: float, tol: float) -> List[dict]:
    out = []
    for l in page.lines:
        if abs(l["x0"] - l["x1"]) < tol and abs(l["bottom"] - l["top"]) >= min_len:
            out.append(l)
    return out


def detect_panels(
    page,
    min_axis_len: float = 60.0,
    tol: float = 0.8,
    match_tol: float = 4.0,
) -> List[Panel]:
    """Detect plot panels by pairing y-axis verticals with x-axis horizontals.

    A panel's y-axis is a vertical line whose *bottom* sits on a horizontal
    x-axis line that starts at (approximately) the same x. The plot's right
    edge is taken as the right end of that x-axis line.
    """
    verts = _long_verticals(page, min_axis_len, tol)
    horis = _long_horizontals(page, min_axis_len, tol)

    panels: List[Panel] = []
    for v in verts:
        vx = (v["x0"] + v["x1"]) / 2.0
        v_bottom = v["bottom"]
        # find an x-axis line whose y ~ v_bottom and whose left end ~ vx.
        # NB: use top/bottom (top-origin) consistently -- y0/y1 are bottom-origin
        # in pdfplumber and would never match a top-origin coordinate.
        for h in horis:
            hy = (h["top"] + h["bottom"]) / 2.0
            if abs(hy - v_bottom) > match_tol:
                continue
            if abs(h["x0"] - vx) > match_tol:
                continue
            panels.append(
                Panel(
                    index=0,
                    x0=vx,
                    x1=h["x1"],
                    y_top=v["top"],
                    y_bottom=hy,
                )
            )
            break

    # de-duplicate panels that resolve to nearly the same frame
    unique: List[Panel] = []
    for p in sorted(panels, key=lambda p: (round(p.y_top), round(p.x0))):
        if any(
            abs(p.x0 - q.x0) < match_tol
            and abs(p.y_bottom - q.y_bottom) < match_tol
            for q in unique
        ):
            continue
        unique.append(p)
    for i, p in enumerate(unique):
        p.index = i
    return unique


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _fit_axis(ticks: Sequence[Tuple[float, float]]) -> AxisFit:
    """Least-squares fit value = slope*pixel + intercept over (pixel, value)."""
    if len(ticks) < 2:
        raise ValueError(f"Need >=2 ticks to calibrate, got {len(ticks)}")
    px = np.array([p for p, _ in ticks], dtype=float)
    val = np.array([v for _, v in ticks], dtype=float)
    A = np.vstack([px, np.ones_like(px)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, val, rcond=None)
    pred = slope * px + intercept
    ss_tot = np.sum((val - val.mean()) ** 2)
    r2 = 1.0 - np.sum((val - pred) ** 2) / ss_tot if ss_tot > 0 else 0.0
    return AxisFit(slope=float(slope), intercept=float(intercept), r2=float(r2), n_ticks=len(ticks))


def calibrate_panel(
    page,
    panel: Panel,
    x_band: float = 14.0,
    y_band: float = 28.0,
) -> Tuple[AxisFit, AxisFit]:
    """Calibrate a panel from tick labels adjacent to its axes only.

    X ticks: numeric tokens just *below* the x-axis line, within the plot's
    x-span. Y ticks: numeric tokens just *left* of the y-axis line, within the
    plot's y-span.
    """
    words = page.extract_words()
    x_ticks: List[Tuple[float, float]] = []
    y_ticks: List[Tuple[float, float]] = []
    for w in words:
        v = _to_number(w["text"])
        if v is None:
            continue
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        # x-axis ticks
        if (
            panel.xaxis_y + 1 < cy < panel.xaxis_y + x_band
            and panel.x0 - 8 <= cx <= panel.x1 + 10
        ):
            x_ticks.append((cx, v))
        # y-axis ticks
        if (
            panel.yaxis_x - y_band < cx < panel.yaxis_x - 1
            and panel.y_top - 6 <= cy <= panel.y_bottom + 6
        ):
            y_ticks.append((cy, v))

    x_ticks = sorted(set(x_ticks))
    y_ticks = sorted(set(y_ticks))
    return _fit_axis(x_ticks), _fit_axis(y_ticks)


def panel_x_ticks(page, panel: Panel, x_band: float = 14.0) -> List[Tuple[float, float]]:
    """Return the panel's x-axis ticks as sorted (pixel_center, value) pairs."""
    ticks: List[Tuple[float, float]] = []
    for w in page.extract_words():
        v = _to_number(w["text"])
        if v is None:
            continue
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        if (
            panel.xaxis_y + 1 < cy < panel.xaxis_y + x_band
            and panel.x0 - 8 <= cx <= panel.x1 + 10
        ):
            ticks.append((cx, v))
    return sorted(set(ticks))


@dataclass
class AtRiskRow:
    """One numbers-at-risk row, anchored to x-tick positions."""

    label: str
    times: np.ndarray
    counts: np.ndarray


def extract_at_risk(
    page,
    panel: Panel,
    x_ticks: Optional[Sequence[Tuple[float, float]]] = None,
    y_search: float = 80.0,
    row_gap: float = 4.0,
    min_filled: float = 0.6,
) -> List[AtRiskRow]:
    """Extract numbers-at-risk rows below a panel by anchoring digits to ticks.

    pdfplumber merges the tightly-spaced at-risk integers into a single token,
    so we work at the character level: for each x-tick column we gather the
    digit chars within half a tick-spacing and join them into one integer. Rows
    are detected by clustering digit chars vertically below the x-axis.
    """
    if x_ticks is None:
        x_ticks = panel_x_ticks(page, panel)
    if len(x_ticks) < 2:
        return []
    tick_px = np.array([p for p, _ in x_ticks], dtype=float)
    tick_val = np.array([v for _, v in x_ticks], dtype=float)
    half = float(np.median(np.diff(np.sort(tick_px)))) / 2.0

    y_lo = panel.xaxis_y + 3.0
    y_hi = panel.xaxis_y + y_search
    digits = [
        c
        for c in page.chars
        if c["text"].isdigit()
        and y_lo < (c["top"] + c["bottom"]) / 2.0 < y_hi
        and panel.x0 - 30 <= (c["x0"] + c["x1"]) / 2.0 <= panel.x1 + 12
    ]
    if not digits:
        return []

    # cluster digit chars into rows by y
    digits.sort(key=lambda c: (c["top"] + c["bottom"]) / 2.0)
    rows: List[List[dict]] = []
    cur: List[dict] = []
    last_y = None
    for c in digits:
        cy = (c["top"] + c["bottom"]) / 2.0
        if last_y is None or abs(cy - last_y) <= row_gap:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
        last_y = cy
    if cur:
        rows.append(cur)

    out: List[AtRiskRow] = []
    for row in rows:
        counts: List[Optional[int]] = []
        for px in tick_px:
            ds = sorted(
                [c for c in row if abs((c["x0"] + c["x1"]) / 2.0 - px) < half],
                key=lambda c: c["x0"],
            )
            s = "".join(c["text"] for c in ds)
            counts.append(int(s) if s else None)
        filled = sum(1 for c in counts if c is not None)
        if filled < min_filled * len(tick_px):
            continue  # not a real at-risk row (e.g. stray label digits)
        # keep only ticks that were filled, preserving alignment
        keep = [i for i, c in enumerate(counts) if c is not None]
        times = tick_val[keep]
        vals = np.array([counts[i] for i in keep], dtype=float)
        # semantic filter: at-risk counts start high and are non-increasing.
        # This rejects the x-axis tick-label row (0, 6, 12, ... which INCREASES)
        # that also lives just below the axis.
        if vals[0] < 20 or vals[0] < vals.max() - 1e-9:
            continue
        # row label: words just left of the panel in this y-band
        cy = float(np.median([(c["top"] + c["bottom"]) / 2.0 for c in row]))
        label = ""
        for w in page.extract_words():
            wy = (w["top"] + w["bottom"]) / 2.0
            if (
                abs(wy - cy) < row_gap
                and w["x1"] <= panel.x0 - 1
                and not any(ch.isdigit() for ch in w["text"])
            ):
                label = (label + " " + w["text"]).strip()
        out.append(AtRiskRow(label=label, times=times, counts=vals))
    return out


# --------------------------------------------------------------------------- #
# Curve geometry
# --------------------------------------------------------------------------- #
def _curve_bbox(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _is_reference_line(pts, panel: Panel) -> bool:
    """True if a path is a gridline / reference line, not a data curve.

    Reference lines are tall+narrow verticals (e.g. dashed markers at t=24/48)
    or long+flat horizontals spanning most of the plot width at constant y.
    """
    if len(pts) < 2:
        return True
    xmin, xmax, ymin, ymax = _curve_bbox(pts)
    w = xmax - xmin
    h = ymax - ymin
    plot_w = panel.x1 - panel.x0
    plot_h = panel.y_bottom - panel.y_top
    # near-vertical spanning a large fraction of plot height
    if w < 1.5 and h > 0.4 * plot_h:
        return True
    # near-horizontal spanning a large fraction of plot width
    if h < 1.5 and w > 0.6 * plot_w:
        return True
    return False


def detect_reference_lines(
    page, panel: Panel, edge_tol: float = 3.0, min_frac: float = 0.4
) -> Tuple[List[float], List[float]]:
    """Find interior gridline / reference-line positions inside a panel.

    Dashed reference lines (e.g. vertical markers at t=24/48) are rendered as
    many short dash segments, so a per-object height test misses them -- but
    they also appear as long ``line`` objects. We return the x-positions of
    interior verticals and y-positions of interior horizontals (excluding the
    axes themselves) so their points can be removed from the curve cloud.
    """
    plot_h = panel.y_bottom - panel.y_top
    plot_w = panel.x1 - panel.x0
    x_refs: List[float] = []
    y_refs: List[float] = []
    for v in _long_verticals(page, min_len=min_frac * plot_h, tol=0.8):
        vx = (v["x0"] + v["x1"]) / 2.0
        if panel.x0 + edge_tol < vx < panel.x1 - edge_tol and (
            panel.y_top - edge_tol <= v["top"] and v["bottom"] <= panel.y_bottom + edge_tol
        ):
            x_refs.append(vx)
    for h in _long_horizontals(page, min_len=min_frac * plot_w, tol=0.8):
        hy = (h["top"] + h["bottom"]) / 2.0
        if panel.y_top + edge_tol < hy < panel.y_bottom - edge_tol and (
            panel.x0 - edge_tol <= h["x0"] and h["x1"] <= panel.x1 + edge_tol
        ):
            y_refs.append(hy)
    return x_refs, y_refs


def extract_panel_point_cloud(
    page,
    panel: Panel,
    pad: float = 1.0,
    x_refs: Optional[Sequence[float]] = None,
    y_refs: Optional[Sequence[float]] = None,
    ref_tol: float = 1.6,
) -> np.ndarray:
    """Collect all data-curve points (pixels) inside a panel.

    Returns an (N, 2) array of (px, py). Reference lines are excluded two ways:
    long single-object gridlines via :func:`_is_reference_line`, and dashed
    reference lines via proximity to ``x_refs`` / ``y_refs`` (their detected
    pixel positions, since dashes survive the per-object test).
    """
    x0, x1 = panel.x0 - pad, panel.x1 + pad
    yt, yb = panel.y_top - pad, panel.y_bottom + pad
    x_refs = list(x_refs or [])
    y_refs = list(y_refs or [])

    def near_ref(px: float, py: float) -> bool:
        if any(abs(px - rx) <= ref_tol for rx in x_refs):
            return True
        if any(abs(py - ry) <= ref_tol for ry in y_refs):
            return True
        return False

    cloud: List[Tuple[float, float]] = []
    for c in page.curves:
        pts = c.get("pts", [])
        inside = [
            (px, py)
            for (px, py) in pts
            if x0 <= px <= x1 and yt <= py <= yb and not near_ref(px, py)
        ]
        if len(inside) < 2:
            continue
        if _is_reference_line(inside, panel):
            continue
        cloud.extend(inside)
    return np.array(cloud, dtype=float) if cloud else np.empty((0, 2))


def extract_legend_labels(
    page, panel: Panel, line_gap: float = 11.0
) -> List[Tuple[str, float, float]]:
    """Find in-plot legend labels as (text, x_center, y_center) in pixels.

    Returns non-numeric word blocks inside the plot box, with p-values and the
    rotated axis title (which sits at x > panel.x1) excluded. Vertically and
    horizontally adjacent words are merged into one label (e.g. "Standard" +
    "control" -> "Standard control").
    """
    cands = []
    for w in page.extract_words():
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        if not (panel.x0 - 2 <= cx <= panel.x1 and panel.y_top - 2 <= cy <= panel.y_bottom + 2):
            continue
        txt = w["text"].strip()
        if _to_number(txt) is not None or "=" in txt or len(txt) < 2:
            continue
        if not any(ch.isalpha() for ch in txt):
            continue
        cands.append({"t": txt, "x0": w["x0"], "x1": w["x1"], "cx": cx, "cy": cy})

    # merge adjacent words (close in y, overlapping/!nearby in x) into labels
    cands.sort(key=lambda c: (c["cy"], c["x0"]))
    used = [False] * len(cands)
    labels: List[Tuple[str, float, float]] = []
    for i, c in enumerate(cands):
        if used[i]:
            continue
        block = [c]
        used[i] = True
        for j in range(i + 1, len(cands)):
            if used[j]:
                continue
            d = cands[j]
            if abs(d["cy"] - block[-1]["cy"]) <= line_gap and not (
                d["x1"] < min(b["x0"] for b in block) - 30
                or d["x0"] > max(b["x1"] for b in block) + 30
            ):
                block.append(d)
                used[j] = True
        block.sort(key=lambda b: (round(b["cy"] / line_gap), b["x0"]))
        text = " ".join(b["t"] for b in block)
        cx = float(np.mean([b["cx"] for b in block]))
        cy = float(np.mean([b["cy"] for b in block]))
        labels.append((text, cx, cy))
    return labels


def assign_arm_identity(
    page, panel: Panel, arms: List["Arm"], x_fit: AxisFit, y_fit: AxisFit
) -> str:
    """Match in-plot legend labels to arms by trajectory proximity.

    For each label, the arm whose pixel-position at the label's time is nearest
    the label is paired (optimal assignment). Sets ``arm.identity`` in place.
    Returns the method used: "legend" if labels were matched, else "" (none).
    """
    labels = extract_legend_labels(page, panel)
    live = [a for a in arms if a.n_points() > 0]
    if not labels or not live or y_fit.slope == 0 or x_fit.slope == 0:
        return ""

    def arm_py_at(arm: "Arm", x_px: float) -> float:
        t = x_fit.slope * x_px + x_fit.intercept
        v = float(np.interp(t, arm.time, arm.value))
        return (v - y_fit.intercept) / y_fit.slope  # value -> pixel y

    # cost[label, arm] = vertical pixel distance from label to arm at label's x
    cost = np.zeros((len(labels), len(live)))
    for li, (_, lx, ly) in enumerate(labels):
        for ai, arm in enumerate(live):
            cost[li, ai] = abs(ly - arm_py_at(arm, lx))

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        rows = list(range(min(len(labels), len(live))))
        cols = list(np.argmin(cost, axis=1)[: len(rows)])
    for li, ai in zip(rows, cols):
        live[ai].identity = labels[li][0]
    return "legend"


def _enforce_monotone(time: np.ndarray, value: np.ndarray, increasing: bool) -> np.ndarray:
    """Force a value series to be monotone in time (CI up, survival down)."""
    out = value.astype(float).copy()
    if increasing:
        np.maximum.accumulate(out, out=out)
    else:
        np.minimum.accumulate(out, out=out)
    return out


def _decimate(time: np.ndarray, value: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points whose value barely changes (keep step structure)."""
    if time.size <= 2:
        return time, value
    keep = [0]
    for i in range(1, time.size - 1):
        if abs(value[i] - value[keep[-1]]) >= eps:
            keep.append(i)
    keep.append(time.size - 1)
    return time[keep], value[keep]


def _cluster_column(vals: np.ndarray, gap: float, n_arms: int) -> List[float]:
    """Cluster a column's values into <= n_arms bands; return medians (desc)."""
    vals = np.sort(vals)
    groups: List[np.ndarray] = []
    start = 0
    for i in range(1, len(vals)):
        if vals[i] - vals[i - 1] > gap:
            groups.append(vals[start:i])
            start = i
    groups.append(vals[start:])
    meds = [float(np.median(g)) for g in groups]
    # merge closest adjacent bands until at most n_arms remain
    while len(meds) > n_arms:
        d = [meds[i + 1] - meds[i] for i in range(len(meds) - 1)]
        j = int(np.argmin(d))
        meds[j] = (meds[j] + meds[j + 1]) / 2.0
        del meds[j + 1]
    return sorted(meds, reverse=True)


def separate_arms(
    points: np.ndarray,
    n_arms: int = 2,
    col_dt: float = 0.5,
    gap: float = 1.0,
) -> List[np.ndarray]:
    """Split a calibrated (time, value) cloud into ``n_arms`` traces.

    ``points`` columns are (time, value). Per time-column bin, points are
    clustered into <= n_arms bands. Bands are assigned to tracks by
    VELOCITY-AWARE CONTINUITY: each band goes to the track whose *predicted*
    next position (last value + last step) is nearest, via optimal assignment.
    This follows each curve THROUGH a crossing, rather than producing upper/
    lower envelopes (which is what naive vertical-order assignment yields and
    is wrong whenever curves cross). Where a column has a single band (curves
    coincident, e.g. near t=0) all tracks share it. Returns one (M, 2) array
    per track, sorted by time. Track order is seeded highest-value-first, so
    for non-crossing curves track 0 == "upper".
    """
    if points.size == 0:
        return [np.empty((0, 2)) for _ in range(n_arms)]

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover - scipy is a declared dep
        linear_sum_assignment = None

    order = np.argsort(points[:, 0])
    points = points[order]
    col = np.round(points[:, 0] / col_dt) * col_dt
    cols = np.unique(col)

    tracks: List[List[Tuple[float, float]]] = [[] for _ in range(n_arms)]
    prev: Optional[List[float]] = None
    vel = [0.0] * n_arms

    for c in cols:
        bands = _cluster_column(points[col == c, 1], gap, n_arms)
        if not bands:
            continue

        if prev is None:
            if len(bands) == n_arms:
                prev = list(bands)
                for k in range(n_arms):
                    tracks[k].append((c, bands[k]))
            else:  # not yet separated -> coincident, all tracks share
                vm = float(np.mean(bands))
                for k in range(n_arms):
                    tracks[k].append((c, vm))
            continue

        predicted = [prev[k] + vel[k] for k in range(n_arms)]

        if len(bands) == 1:
            # curves coincident here: every track gets the shared value, but
            # velocity is FROZEN (momentum) so a true crossing is followed
            # through once the bands separate again.
            b = bands[0]
            for k in range(n_arms):
                prev[k] = b
                tracks[k].append((c, b))
            continue

        # assign bands -> tracks minimising total |band - predicted| (optimal
        # transport over positions). Carrying momentum lets this swap sides
        # across a crossing instead of building upper/lower envelopes.
        cost = np.abs(np.subtract.outer(np.array(bands), np.array(predicted)))
        if linear_sum_assignment is not None:
            rows, cidx = linear_sum_assignment(cost)
        else:  # greedy fallback
            rows, cidx, used = [], [], set()
            for ri in range(len(bands)):
                j = int(np.argmin([cost[ri, k] if k not in used else 1e18
                                   for k in range(n_arms)]))
                rows.append(ri); cidx.append(j); used.add(j)
        assigned = {int(ci): bands[int(ri)] for ri, ci in zip(rows, cidx)}
        for j in range(n_arms):
            if j in assigned:
                vel[j] = assigned[j] - prev[j]
                prev[j] = assigned[j]
                tracks[j].append((c, assigned[j]))
            else:  # unmatched track carries its position, keeps momentum
                tracks[j].append((c, prev[j]))

    arrays = [np.array(t, dtype=float) if t else np.empty((0, 2)) for t in tracks]
    # Stable labelling: order tracks by mean value (desc) so track 0 == "upper"
    # for the common non-crossing case. Genuine crossings need legend identity
    # (see assign_arm_identity); this is only the positional default.
    arrays.sort(key=lambda a: -(a[:, 1].mean() if a.size else -np.inf))
    return arrays


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def extract_km_from_pdf(
    pdf_path: str,
    page_index: int,
    n_arms: int = 2,
    monotone: str = "auto",
    decimate_eps: Optional[float] = None,
) -> List[PanelResult]:
    """Extract calibrated KM/cumulative-incidence curves from one PDF page.

    Parameters
    ----------
    pdf_path : path to the (vector) PDF.
    page_index : 0-indexed page holding the figure.
    n_arms : expected curves per panel (default 2).
    monotone : "auto" (infer from data), "increasing" (cumulative incidence),
        "decreasing" (survival), or "none".
    decimate_eps : if set, drop points whose value changes by less than this
        (in y-units) to compress the trace to its step structure.

    Returns one ``PanelResult`` per detected panel, each with exact axis fits
    and cleaned per-arm calibrated (time, value) traces. ``value`` is in the
    panel's native y-units (e.g. cumulative incidence %).
    """
    results: List[PanelResult] = []
    labels = ["upper", "lower", "arm2", "arm3"]
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        panels = detect_panels(page)
        for panel in panels:
            try:
                x_fit, y_fit = calibrate_panel(page, panel)
            except ValueError:
                # not enough tick labels -> not a real calibratable plot
                continue

            # plotted value/time window from the calibrated axis extents
            v_top = y_fit.value(panel.y_top)
            v_bot = y_fit.value(panel.y_bottom)
            v_lo, v_hi = sorted((v_top, v_bot))
            t_hi = max(x_fit.value(panel.x0), x_fit.value(panel.x1))
            margin_v = 0.02 * (v_hi - v_lo)
            margin_t = 0.02 * t_hi

            x_refs, y_refs = detect_reference_lines(page, panel)
            cloud = extract_panel_point_cloud(page, panel, x_refs=x_refs, y_refs=y_refs)

            # calibrate to (time, value) and clip to the plotted window
            if cloud.size:
                t = x_fit.slope * cloud[:, 0] + x_fit.intercept
                v = y_fit.slope * cloud[:, 1] + y_fit.intercept
                keep = (
                    (t >= -margin_t)
                    & (t <= t_hi + margin_t)
                    & (v >= v_lo - margin_v)
                    & (v <= v_hi + margin_v)
                )
                cal = np.column_stack([t[keep], v[keep]])
            else:
                cal = np.empty((0, 2))

            arm_clouds = separate_arms(cal, n_arms=n_arms)

            # separation confidence: fraction of (aligned) columns where the two
            # arms differ by more than ~2% of the value range. Low => arms
            # near-overlapping and the split is unreliable.
            sep_conf = 1.0
            sep_thresh = max(margin_v, 1e-9)
            if (
                len(arm_clouds) >= 2
                and arm_clouds[0].size
                and arm_clouds[1].size
                and len(arm_clouds[0]) == len(arm_clouds[1])
            ):
                sep_conf = float(
                    np.mean(
                        np.abs(arm_clouds[0][:, 1] - arm_clouds[1][:, 1]) > sep_thresh
                    )
                )

            arms: List[Arm] = []
            for i, ac in enumerate(arm_clouds):
                if ac.size == 0:
                    arm = Arm(label=labels[i], time=np.empty(0), value=np.empty(0))
                    arms.append(arm)
                    continue
                order = np.argsort(ac[:, 0])
                tt, vv = ac[order, 0], ac[order, 1]
                if monotone == "auto":
                    inc = vv[-1] >= vv[0]
                    mono = monotone if monotone == "none" else ("inc" if inc else "dec")
                else:
                    mono = {"increasing": "inc", "decreasing": "dec", "none": "none"}.get(
                        monotone, "none"
                    )
                if mono != "none":
                    vv = _enforce_monotone(tt, vv, increasing=(mono == "inc"))
                if decimate_eps is not None:
                    tt, vv = _decimate(tt, vv, decimate_eps)
                arms.append(Arm(label=labels[i], time=tt, value=vv))

            # resolve treatment identity from the in-plot legend (vector text)
            assign_arm_identity(page, panel, arms, x_fit, y_fit)

            results.append(
                PanelResult(
                    panel=panel,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    arms=arms,
                    separation_confidence=sep_conf,
                )
            )
    return results


@dataclass
class ArmIPD:
    label: str
    time: np.ndarray  # event/censoring times
    event: np.ndarray  # 1 = event, 0 = censored
    total_n: int
    n_events: int
    nar_label: str  # which at-risk row was paired (heuristic)


@dataclass
class PanelIPD:
    panel_index: int
    arms: List[ArmIPD]
    at_risk: List["AtRiskRow"]
    nar_mapping: str  # how arms were paired to at-risk rows


def pdf_to_ipd(
    pdf_path: str,
    page_index: int,
    n_arms: int = 2,
    value_is_cumulative_incidence: bool = True,
    value_scale: float = 100.0,
) -> List[PanelIPD]:
    """End-to-end: vector curve + at-risk table -> reconstructed IPD per arm.

    Pairs curve arms to at-risk rows BY ORDER (upper<->row0, ...), which is a
    heuristic -- arm/row identity is genuinely ambiguous without legend OCR, so
    the raw ``at_risk`` rows and the ``nar_mapping`` flag are returned for the
    caller to verify or override.

    Parameters
    ----------
    value_is_cumulative_incidence : if True, survival = 1 - value/value_scale;
        if False, survival = value/value_scale.
    value_scale : 100 when the y-axis is a percentage, 1 for a 0-1 proportion.
    """
    from guyot import reconstruct_arm  # local import: keeps guyot optional

    # extract all panels' cleaned curves once, index by panel position
    km_by_index = {
        pr.panel.index: pr.arms
        for pr in extract_km_from_pdf(
            pdf_path, page_index, n_arms=n_arms, monotone="increasing"
        )
    }

    out: List[PanelIPD] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        panels = detect_panels(page)
        for panel in panels:
            try:
                calibrate_panel(page, panel)
            except ValueError:
                continue
            x_ticks = panel_x_ticks(page, panel)
            at_risk = extract_at_risk(page, panel, x_ticks=x_ticks)

            arms = km_by_index.get(panel.index)
            if not arms:
                continue

            # pair each arm to an at-risk row: prefer legend-identity token
            # overlap (e.g. arm "Standard control" <-> row "Standard"); fall
            # back to positional order when identity is unavailable.
            row_for_arm, used, by_legend = {}, set(), True
            for i, arm in enumerate(arms):
                toks = set(arm.identity.lower().split())
                best, best_ov = None, 0
                for ri, r in enumerate(at_risk):
                    if ri in used:
                        continue
                    ov = len(toks & set(r.label.lower().split()))
                    if ov > best_ov:
                        best, best_ov = ri, ov
                if best is not None and best_ov > 0:
                    row_for_arm[i] = best
                    used.add(best)
                else:
                    by_legend = False
            if not by_legend or len(row_for_arm) < len(arms):
                # incomplete legend match -> positional fallback for everyone
                row_for_arm = {i: i for i in range(len(arms))}
                by_legend = False

            arm_ipds: List[ArmIPD] = []
            for i, arm in enumerate(arms):
                if arm.n_points() == 0:
                    continue
                if value_is_cumulative_incidence:
                    surv = 1.0 - arm.value / value_scale
                else:
                    surv = arm.value / value_scale
                surv = np.clip(surv, 0.0, 1.0)

                ri = row_for_arm.get(i)
                row = at_risk[ri] if ri is not None and ri < len(at_risk) else None
                nar_t = row.times if row is not None else None
                nar_v = row.counts if row is not None else None
                total_n = int(row.counts[0]) if row is not None else None

                t, e = reconstruct_arm(
                    arm.time, surv, nar_times=nar_t, nar_values=nar_v,
                    total_n=total_n, arm=i,
                )
                arm_ipds.append(
                    ArmIPD(
                        label=arm.identity or arm.label,
                        time=t,
                        event=e,
                        total_n=int(t.size),
                        n_events=int(e.sum()),
                        nar_label=row.label if row is not None else "",
                    )
                )
            out.append(
                PanelIPD(
                    panel_index=panel.index,
                    arms=arm_ipds,
                    at_risk=at_risk,
                    nar_mapping="legend_matched" if by_legend else "by_order_heuristic",
                )
            )
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import sys

    try:
        from project_paths import sample_pdf_path

        default_pdf = str(sample_pdf_path())
    except Exception:
        default_pdf = None

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    page_index = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    if not pdf_path:
        print("Usage: python vector_km.py <pdf> [page_index]")
        sys.exit(1)

    panels = extract_km_from_pdf(pdf_path, page_index, monotone="increasing")
    print(f"Detected {len(panels)} panel(s) on page {page_index} of {pdf_path}\n")
    for pr in panels:
        p = pr.panel
        print(f"Panel {p.index}: box x[{p.x0:.0f},{p.x1:.0f}] y[{p.y_top:.0f},{p.y_bottom:.0f}]")
        print(
            f"  X-calib R2={pr.x_fit.r2:.6f} ({pr.x_fit.n_ticks} ticks): "
            f"left->{pr.x_fit.value(p.x0):.1f}  right->{pr.x_fit.value(p.x1):.1f}"
        )
        print(
            f"  Y-calib R2={pr.y_fit.r2:.6f} ({pr.y_fit.n_ticks} ticks): "
            f"top->{pr.y_fit.value(p.y_top):.1f}  bottom->{pr.y_fit.value(p.y_bottom):.1f}"
        )
        for arm in pr.arms:
            if arm.n_points() == 0:
                continue
            print(
                f"    {arm.label:>5}: n={arm.n_points():4d} "
                f"t[{arm.time.min():.1f},{arm.time.max():.1f}] "
                f"value[{arm.value.min():.1f},{arm.value.max():.1f}] end={arm.value[-1]:.1f}"
            )
        print()
