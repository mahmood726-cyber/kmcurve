#!/usr/bin/env python3
"""
Calibrated confidence for raster axis calibration (roadmap lever 4).
====================================================================

"Confidence-gated full-auto" needs a score that PREDICTS whether a calibration
is actually correct -- not just the fit R^2. Lever 1 showed why R^2 is a weak
signal on its own: tick values are arithmetic and tick positions are evenly
spaced, so *any* in-order alignment fits R^2=1 (the linear degeneracy). A wrong
read can therefore still score R^2=1.

The signals that actually predict correctness:

  * **verify-agreement** -- two INDEPENDENT VLM reads returning the same tick
    values. The strongest anti-misread signal; an uncorrelated misread breaks it.
  * **tick count** -- with >=4 arithmetic ticks pinned to evenly-spaced
    positions, a single misread falls off the line and RANSAC drops it (the fit
    self-corrects); with 2-3 ticks a misread can stay collinear and pass
    silently. So confidence rises sharply with tick count.
  * **semantic plausibility** -- a KM y-axis is survival/cumulative-incidence in
    [0,1] or [0,100] spanning most of the axis; the x-axis is time, non-negative,
    starting near 0. Implausible semantics flag a misread/mis-pair.
  * **R^2** -- a necessary floor, not a discriminator.

``calibration_confidence`` combines these into [0,1] via a weighted geometric
mean (a near-zero component pulls the whole score down) and an ``auto_accept``
gate. Weights are hand-set from the lever-1 failure analysis and VALIDATED for
DISCRIMINATION (correct vs corrupted) in ``validate_confidence.py``; mapping the
score to a true probability needs the labelled corpus (lever 2) and is
deliberately deferred -- see that module's report.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Component weights (geometric-mean exponents). Verify-agreement and tick count
# dominate because they are the signals that actually catch a bad read; R^2 is a
# near-necessary floor and so carries little discriminating weight.
_WEIGHTS = {"verify": 0.35, "n_ticks": 0.30, "semantics": 0.25, "r2": 0.10}
_AUTO_ACCEPT_THRESHOLD = 0.80
_MIN_TICKS_FOR_AUTO = 4


def _ticks_score(n: int) -> float:
    """Map the (min over axes) tick count to a [0,1] confidence contribution.

    2 ticks (a line through 2 points) cannot catch a misread -> low; >=5 ticks
    over-constrain the fit so a stray read is rejected -> ~1.
    """
    return {0: 0.0, 1: 0.0, 2: 0.30, 3: 0.62, 4: 0.86}.get(n, 1.0)


def _r2_score(r2_min: float) -> float:
    """R^2 floor: ~1 above 0.995, falling off sharply below (a real misalignment
    that survived count-match still tends to dent R^2 a little)."""
    if r2_min >= 0.9995:
        return 1.0
    if r2_min >= 0.99:
        return 0.85
    if r2_min >= 0.97:
        return 0.5
    return 0.1


def _semantics_score(meta: Dict[str, Any]) -> float:
    """Plausibility of the read against KM-figure expectations, in [0,1].

    Uses value ranges + axis_kind recorded in ``meta`` (set by
    ``vlm_calibrate.ingest_vlm_answer``). Penalises: y not survival/CI, y range
    not covering most of [0,scale], x not time, x not starting near 0, x not
    increasing.
    """
    score = 1.0
    scale = float(meta.get("value_scale", 1.0)) or 1.0

    y_kind = meta.get("y_kind")
    if y_kind not in ("survival", "cumulative_incidence"):
        score *= 0.55
    ymin, ymax = meta.get("y_value_min"), meta.get("y_value_max")
    if ymin is not None and ymax is not None:
        # scale consistency: a percent axis maxes near 100, a proportion near 1.
        # A contradiction (is_percent but max~1, or proportion but max>>1) is a
        # strong wrong-scale signal -- it silently 100x's the mapped survival.
        is_pct = bool(meta.get("y_is_percent"))
        if is_pct and ymax <= 1.5:
            score *= 0.25
        if (not is_pct) and ymax > 1.5:
            score *= 0.25
        # y should sit within [0, scale] (+small tol) and span most of it
        if ymin < -0.05 * scale or ymax > 1.10 * scale:
            score *= 0.4
        span_frac = (ymax - ymin) / scale if scale else 0.0
        if span_frac < 0.6:
            score *= 0.6      # only a partial slice of the axis was read

    x_kind = meta.get("x_kind")
    if x_kind != "time":
        score *= 0.7
    xmin, xmax = meta.get("x_value_min"), meta.get("x_value_max")
    if xmin is not None and xmax is not None:
        if xmin < -1e-6:                       # time should be non-negative
            score *= 0.4
        if xmax <= xmin:                        # must increase
            score *= 0.2
        elif xmin > 0.25 * (xmax - xmin):       # first tick should be near 0
            score *= 0.7
    return max(0.0, min(1.0, score))


def calibration_confidence(
    meta: Dict[str, Any],
    r2_x: float,
    r2_y: float,
    n_ticks_x: int,
    n_ticks_y: int,
    verify_agree: Optional[bool] = None,
    threshold: float = _AUTO_ACCEPT_THRESHOLD,
) -> Dict[str, Any]:
    """Predict whether a raster calibration is correct. Returns
    ``{confidence, auto_accept, components}``.

    ``verify_agree`` is True/False when two independent VLM reads were compared
    (the workflow / API double-read path), or None when only a single read is
    available (then the verify signal is neutral-ish, never full credit).
    """
    if verify_agree is None:
        verify = 0.65          # single read: unknown -> neutral, capped below 1
    else:
        verify = 1.0 if verify_agree else 0.15

    comps = {
        "verify": verify,
        "n_ticks": _ticks_score(min(n_ticks_x, n_ticks_y)),
        "semantics": _semantics_score(meta),
        "r2": _r2_score(min(r2_x, r2_y)),
    }
    # weighted geometric mean: any near-zero component drags the score down,
    # which is the desired "one bad signal => don't auto-accept" behaviour.
    conf = 1.0
    for k, w in _WEIGHTS.items():
        conf *= max(comps[k], 1e-6) ** w
    conf = round(conf, 4)

    auto_accept = bool(
        conf >= threshold
        and min(n_ticks_x, n_ticks_y) >= _MIN_TICKS_FOR_AUTO
        and (verify_agree is not False)        # an explicit disagreement never auto-accepts
    )
    return {"confidence": conf, "auto_accept": auto_accept, "components": comps}
