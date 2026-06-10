#!/usr/bin/env python3
"""
External value cross-check for axis calibration (closes lever 4's blind spot).
==============================================================================

Lever 4 confidence catches uncorrelated misreads, under-constrained fits, and
scale contradictions -- but NOT a *correlated* identical misread that both VLM
readers make and that stays linear and semantically plausible (e.g. every x tick
read 1.2x off). No internal signal can catch that, because the fit is still a
clean line through plausible values. The only thing that can is an INDEPENDENT
value source.

This module cross-checks a calibration against sources already available in the
raster pipeline / paper:

  * **y-span** -- a survival / cumulative-incidence axis must span most of
    [0, scale]; a tiny span means a partial/mis-read axis.
  * **x vs reported follow-up** -- the calibrated max time must match the
    caption's stated follow-up or the last at-risk-table time column. This is
    the independent source that catches the x-scale blind spot.
  * **at-risk column alignment** -- each at-risk time column maps, under the
    calibration, to (close to) its printed time.

When an external reference is available and disagrees, ``auto_accept`` is VETOED
(forced False) regardless of confidence -- an external contradiction outranks an
internally-consistent-but-possibly-wrong fit. Checks that need a reference which
isn't present are simply skipped (``None``), never assumed pass.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def caption_max_followup(caption: Optional[str]) -> Optional[float]:
    """Best-effort parse of a stated follow-up horizon from a figure caption.

    Looks for "<n> months/years/weeks/days" (optionally "up to"/"median ...
    follow-up of"). Returns the value in the SAME unit the number is written in
    (caller compares against the x-axis, which is in its own unit) -- so this is
    only used when the x-axis unit matches the caption unit; returns None if no
    horizon is found. Conservative: returns the max plausible horizon found.
    """
    if not caption:
        return None
    vals: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(month|year|week|day)s?\b", caption, re.I):
        vals.append(float(m.group(1)))
    return max(vals) if vals else None


def cross_check_calibration(
    x_fit,
    y_fit,
    meta: Dict[str, Any],
    box,
    *,
    external_max_time: Optional[float] = None,
    at_risk_col_times: Optional[List[Tuple[float, float]]] = None,
    rel_tol: float = 0.10,
) -> Dict[str, Any]:
    """Cross-check a calibration against external value sources.

    ``external_max_time``     -- reported max follow-up (caption / last at-risk
                                 column), in the x-axis unit.
    ``at_risk_col_times``     -- list of (pixel_x, printed_time) for at-risk
                                 columns, to verify the calibration maps each.
    Returns ``{consistent, vetoes, flags}``: ``flags`` is per-check
    True/False/None (None = not checked); ``vetoes`` lists failed external
    checks; ``consistent`` is False iff any external check failed.
    """
    flags: Dict[str, Optional[bool]] = {}
    scale = float(meta.get("value_scale", 1.0)) or 1.0

    # y-span: a survival/CI axis covers most of [0, scale]
    yspan = abs(y_fit.value(box.y0) - y_fit.value(box.y1))
    flags["y_span_full"] = bool(yspan >= 0.6 * scale)

    # x vs reported follow-up -- the independent source for the scale blind spot
    if external_max_time is not None and external_max_time > 0:
        cal_max = max(x_fit.value(box.x0), x_fit.value(box.x1))
        flags["x_matches_followup"] = bool(
            abs(cal_max - external_max_time) <= rel_tol * external_max_time)
    else:
        flags["x_matches_followup"] = None

    # at-risk columns map to their printed times
    if at_risk_col_times:
        ref = external_max_time or max((t for _, t in at_risk_col_times), default=1.0) or 1.0
        worst = max(abs(x_fit.value(px) - t) for px, t in at_risk_col_times)
        flags["at_risk_aligned"] = bool(worst <= rel_tol * ref)
    else:
        flags["at_risk_aligned"] = None

    vetoes = [k for k, v in flags.items() if v is False]
    return {"consistent": len(vetoes) == 0, "vetoes": vetoes, "flags": flags}
