#!/usr/bin/env python3
"""Evaluate VLM calibration vs the OCR baseline on the SAME corpus boxes.

Consumes the read-verify workflow output (``vlm_reads.json``: a list of
{id, read1, read2, agree_x, agree_y}) joined with the crop manifest. For each
box it re-renders the figure, runs the existing OCR ``auto_calibrate_axes``
(baseline) and the VLM ``ingest_vlm_answer`` (read1, gated on the adversarial
agreement), and reports the head-to-head. The headline number is how many boxes
the VLM calibrates that OCR fails closed on -- the lever-1 rescue rate.

Usage: python eval_vlm_calibration.py [vlm_reads.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import figure_locator as FL
from raster_km import PlotBox, auto_calibrate_axes, render_page
from vlm_calibrate import ingest_vlm_answer

DPI = 200
ROOT = Path("artifacts/vlm_crops")


def main():
    reads_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vlm_reads.json"
    manifest = {e["id"]: e for e in json.loads((ROOT / "manifest.json").read_text())}
    payload = json.loads(reads_path.read_text())
    results = payload["results"] if isinstance(payload, dict) else payload

    by_pdf: dict = {}
    for r in results:
        e = manifest.get(r["id"])
        if e:
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
        except Exception as exc:
            print(f"[skip pdf] {Path(pdf).name}: {str(exc)[:50]}")
            continue
        for r, e in items:
            box = PlotBox(*e["box"])
            try:
                auto_calibrate_axes(g, box)
                ocr = "ok"
            except Exception:
                ocr = "fail"
            if not (r["agree_x"] and r["agree_y"]):
                vlm = "skip(verify-disagree)"
            else:
                try:
                    xf, yf, meta = ingest_vlm_answer(g, box, r["read1"])
                    vlm = f"ok(r2x={xf.r2:.3f},r2y={yf.r2:.3f})"
                except Exception as exc:
                    vlm = f"fail({str(exc)[:34]})"
            rescue = ocr == "fail" and vlm.startswith("ok")
            rows.append({"id": r["id"], "ocr": ocr, "vlm": vlm, "rescue": rescue})
            mark = "  <-- VLM RESCUE" if rescue else ""
            print(f"{r['id']:<42} OCR={ocr:<5} VLM={vlm}{mark}")

    n = len(rows)
    ocr_ok = sum(x["ocr"] == "ok" for x in rows)
    vlm_ok = sum(x["vlm"].startswith("ok") for x in rows)
    rescues = sum(x["rescue"] for x in rows)
    agree = sum(1 for r in results if r.get("agree_x") and r.get("agree_y"))
    summary = {
        "n_boxes": n, "verified_agree_both": agree,
        "ocr_calibrates": ocr_ok, "vlm_calibrates": vlm_ok,
        "vlm_rescues_over_ocr": rescues,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    (ROOT / "eval_summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"-> {ROOT/'eval_summary.json'}")


if __name__ == "__main__":
    main()
