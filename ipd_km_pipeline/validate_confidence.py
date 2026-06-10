#!/usr/bin/env python3
"""Validate that calibration confidence DISCRIMINATES correct vs corrupted
calibrations (roadmap lever 4).

Without a labelled corpus (lever 2) we cannot calibrate the score to a true
probability, but we CAN validate the property that matters for the auto-accept
gate: does confidence rank correct calibrations above wrong ones, and does
auto-accept avoid the wrong ones?

Method: take each verified pilot calibration (read1, both readers agreed, fits
cleanly) as a CORRECT reference, then inject corruptions and measure
(confidence, auto_accept, reconstruction-error-vs-reference) for each. The
reconstruction error is max |value_variant(px) - value_ref(px)| over the plot
box, normalised by the reference value span (a proxy for IPD reconstruction
error from a mis-calibration).

Corruptions:
  correct      -- unchanged reference (expect high conf, auto-accept, ~0 error)
  wrong_scale  -- toggle y is_percent (100x value error; implausible span)
  y_kind_other -- y semantics set to "other"
  verify_fail  -- correct values but the two reads DISAGREED
  x_scale_1.2  -- x tick values x1.2: a CORRELATED misread both readers make
                  identically. Stays linear + semantically plausible -> the
                  documented BLIND SPOT (high error, confidence cannot catch it).

Usage: python validate_confidence.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import figure_locator as FL
from raster_km import PlotBox, render_page
import vlm_calibrate as VC

DPI = 200
ROOT = Path("artifacts/vlm_crops")


def _err_vs_ref(fit, ref_fit, lo, hi):
    """max |fit(px)-ref(px)| over [lo,hi], normalised by the ref value span."""
    pxs = np.linspace(lo, hi, 50)
    dev = np.max(np.abs((fit.slope * pxs + fit.intercept) - (ref_fit.slope * pxs + ref_fit.intercept)))
    span = abs(ref_fit.value(hi) - ref_fit.value(lo)) or 1.0
    return float(dev / span)


def _corrupt(ans, kind):
    a = copy.deepcopy(ans)
    if kind == "wrong_scale":
        a["y_axis"]["is_percent"] = not a["y_axis"].get("is_percent", False)
    elif kind == "y_kind_other":
        a["y_axis"]["axis_kind"] = "other"
    elif kind == "x_scale_1.2":
        a["x_axis"]["tick_values"] = [v * 1.2 for v in a["x_axis"]["tick_values"]]
    return a


KINDS = ["correct", "wrong_scale", "y_kind_other", "verify_fail", "x_scale_1.2"]


def main():
    manifest = {e["id"]: e for e in json.loads((ROOT / "manifest.json").read_text())}
    results = json.loads((ROOT / "vlm_reads.json").read_text())["results"]
    by_pdf = {}
    for r in results:
        e = manifest.get(r["id"])
        if e and r["agree_x"] and r["agree_y"]:
            by_pdf.setdefault(e["pdf"], []).append((r, e))

    rows = []
    for pdf, items in by_pdf.items():
        try:
            c = FL.locate_km_figures(pdf)[0]
            g = render_page(pdf, c.page_index, dpi=DPI)
            if c.bbox:
                sc = DPI / 72.0
                x0, t, x1, b = c.bbox
                g = g[int(t * sc):int(b * sc), int(x0 * sc):int(x1 * sc)]
        except Exception:
            continue
        for r, e in items:
            box = PlotBox(*e["box"])
            try:
                rx, ry, _ = VC.ingest_vlm_answer(g, box, r["read1"], verify_agree=True)
            except Exception:
                continue  # only references that calibrate cleanly are usable
            for kind in KINDS:
                ans = r["read1"] if kind in ("correct", "verify_fail") else _corrupt(r["read1"], kind)
                va = False if kind == "verify_fail" else True
                try:
                    xf, yf, meta = VC.ingest_vlm_answer(g, box, ans, verify_agree=va)
                except Exception:
                    rows.append({"id": e["id"], "kind": kind, "rejected": True,
                                 "conf": 0.0, "auto": False, "err": None})
                    continue
                err = max(_err_vs_ref(xf, rx, box.x0, box.x1),
                          _err_vs_ref(yf, ry, box.y0, box.y1))
                conf = meta["confidence"]
                rows.append({"id": e["id"], "kind": kind, "rejected": False,
                             "conf": conf["confidence"], "auto": conf["auto_accept"],
                             "err": round(err, 4)})

    # ---- report ----
    print(f"{'kind':<14}{'n':>3}{'mean_conf':>11}{'auto_accept_rate':>18}{'mean_err':>10}")
    by_kind = {}
    for x in rows:
        by_kind.setdefault(x["kind"], []).append(x)
    for kind in KINDS:
        xs = by_kind.get(kind, [])
        if not xs:
            continue
        n = len(xs)
        mc = np.mean([x["conf"] for x in xs])
        ar = np.mean([1.0 if x["auto"] else 0.0 for x in xs])
        errs = [x["err"] for x in xs if x["err"] is not None]
        me = np.mean(errs) if errs else float("nan")
        print(f"{kind:<14}{n:>3}{mc:>11.3f}{ar:>18.2f}{me:>10.3f}")

    # discrimination: do correct calibrations out-score the detectable corruptions?
    detectable = [x for x in rows if x["kind"] in ("wrong_scale", "y_kind_other", "verify_fail")]
    correct = [x for x in rows if x["kind"] == "correct"]
    blind = [x for x in rows if x["kind"] == "x_scale_1.2"]
    if correct and detectable:
        pairs = [(c["conf"], d["conf"]) for c in correct for d in detectable]
        sep = np.mean([1.0 if cc > dd else 0.0 for cc, dd in pairs])
        print(f"\nDISCRIMINATION (correct conf > detectable-corruption conf): {sep:.3f}")
    if correct:
        fa = np.mean([1.0 if x["auto"] else 0.0 for x in detectable])
        tp = np.mean([1.0 if x["auto"] else 0.0 for x in correct])
        print(f"auto-accept: correct={tp:.2f}  detectable-corruptions={fa:.2f} (want 0)")
    if blind:
        ba = np.mean([1.0 if x["auto"] else 0.0 for x in blind])
        be = np.mean([x["err"] for x in blind if x["err"] is not None])
        print(f"\nBLIND SPOT (correlated identical misread, x_scale_1.2): "
              f"auto-accept={ba:.2f} at mean_err={be:.3f} -- confidence CANNOT catch a "
              f"misread both readers make identically + that stays linear & plausible. "
              f"Documented limit; needs an external value cross-check (at-risk table / caption).")

    (ROOT / "confidence_validation.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {ROOT/'confidence_validation.json'}")


if __name__ == "__main__":
    main()
