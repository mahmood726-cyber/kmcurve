#!/usr/bin/env python3
"""
End-to-end NAR fusion on the first endpoint-matched real pair: PALOMA-3 (OS).
=============================================================================

`fusion_crossmatch.py` surfaced NCT01942135 / PMC9662922 as a fusion candidate,
but its first HR (0.42) was the trial's PFS primary glued onto an OS curve. With
endpoint-matched linkage (fusion_crossmatch._endpoint_key reading the measure
description) the curve resolves to OVERALL SURVIVAL and its true held-out ground
truth is the posted **OS HR 0.814 (0.644-1.029)** -- not the PFS 0.42.

This drives the same two-way fusion as fusion_real_trial.py (RADIANT-4), now on a
pair discovered end-to-end by the corpus cross-match, against the CORRECT OS HR.
Every input is DERIVED, not hardcoded:

  - OS curve anchors + N : registry-ipd's harvested cohort/NCT01942135.json
    (year 1/2/3 KM-estimate survival probabilities -- the ctgov-posted OS curve).
  - OS event counts      : ctgov's 'Overall Survival (OS)-Number of Participants
    Who Died' measure (the censoring info AACT/the curve lacks; kmcurve OCRs the
    identical figure at-risk "N (events)" in production).
  - ground truth OS HR   : ctgov OS hazard-ratio analysis, ENDPOINT-MATCHED to the
    OS curve via fusion_crossmatch._hr_for_endpoint (the bug fix this validates).

registry-only Guyot(anchors, N) has no censoring; FUSION QP(anchors, N, events)
adds the figure's event count. Honest scope: 3 sparse landmark anchors (yr 1/2/3)
+ posted death counts -- this validates the endpoint-matched MECHANISM on a real
discovered pair, not a dense-curve OCR (PMC9662922's figure OCR is the separate
raster_km step). OS HR 0.814 is near 1 (non-significant OS in PALOMA-3), a harder
target than a strongly-separated curve.

Run:  python fusion_paloma3.py [--cohort C:\\Projects\\registry-ipd\\cohort\\NCT01942135.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import fusion_crossmatch as X
from fusion_real_trial import run_trial, _print

NCT = "NCT01942135"


def _fetch_os(nct: str) -> Tuple[Dict[str, int], Optional[dict]]:
    """(per-group OS deaths, endpoint-matched OS HR) from ctgov for `nct`."""
    import urllib.request
    req = urllib.request.Request(
        f"https://clinicaltrials.gov/api/v2/studies/{nct}",
        headers={"User-Agent": X._UA})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    oms = (d.get("resultsSection", {}).get("outcomeMeasuresModule", {})
           .get("outcomeMeasures", []))

    deaths: Dict[str, int] = {}
    for om in oms:
        if "died" in (om.get("title") or "").lower():
            groups = {g["id"]: g["title"] for g in om.get("groups", [])}
            for cl in om.get("classes", []):
                for cat in cl.get("categories", []):
                    for m in cat.get("measurements", []):
                        deaths[groups.get(m["groupId"], m["groupId"])] = int(m["value"])
            break

    # endpoint-matched OS HR: reuse the crossmatch's fixed linkage so this script
    # validates the SAME code path the cross-match now uses (not a parallel guess).
    oms_lite = [{"title": om.get("title", ""), "description": om.get("description", ""),
                 "timeFrame": om.get("timeFrame", ""),
                 "analyses": [{"paramType": a.get("paramType"), "paramValue": a.get("paramValue"),
                               "ciLowerLimit": a.get("ciLowerLimit"), "ciUpperLimit": a.get("ciUpperLimit")}
                              for a in (om.get("analyses") or [])]}
                for om in oms]
    hr = X._hr_for_endpoint(oms_lite, "os")
    return deaths, hr


def build_trial(cohort_path: Path) -> dict:
    """Corrected trial dict: registry OS anchors + ctgov OS deaths + OS HR."""
    cohort = json.loads(cohort_path.read_text())
    deaths, hr = _fetch_os(cohort["nct_id"])
    if not deaths:
        raise SystemExit("could not read OS death counts from ctgov")
    if not hr or hr.get("value") is None:
        raise SystemExit("no endpoint-matched OS HR found on ctgov")

    arms = []
    for a in cohort["arms"]:
        label = a["label"]
        if label not in deaths:
            raise SystemExit(f"no OS death count for arm {label!r}; have {list(deaths)}")
        arms.append({**a, "total_events": deaths[label]})
    return {
        "nct_id": cohort["nct_id"],
        "condition": cohort.get("condition"),
        "time_unit": cohort.get("time_unit", "months"),
        "arms": arms,
        "hr": {"value": float(hr["value"]),
               "ci_low": float(hr["ci"][0]) if hr["ci"][0] else None,
               "ci_high": float(hr["ci"][1]) if hr["ci"][1] else None,
               "favors_arm_id": "OG000"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=rf"C:\Projects\registry-ipd\cohort\{NCT}.json")
    ap.add_argument("--out", default="paloma3_os_fusion.json")
    args = ap.parse_args()

    trial = build_trial(Path(args.cohort))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(trial, tf)
        tmp = tf.name
    res = run_trial(Path(tmp))
    Path(tmp).unlink(missing_ok=True)

    print("\n*** endpoint-matched pair: PALOMA-3 OVERALL SURVIVAL (ground truth OS HR,"
          " NOT the PFS 0.42) ***")
    _print(res)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
