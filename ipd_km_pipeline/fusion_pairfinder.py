#!/usr/bin/env python3
"""
Targeted pair-finder: start from trials that POST a registry KM curve.
======================================================================

`fusion_crossmatch.py` searches the OA figure corpus for trials that also post a
ctgov curve (figure -> registry). This goes the other way (registry -> figure):
start from the trials that ALREADY post a structured 2-arm KM curve + HR (the
registry side is guaranteed), then look for an open-access PRIMARY publication
whose KM figure kmcurve could OCR. If one exists, it is a fully end-to-end
fusion pair by construction.

Source list: registry-ipd's harvested gallery (`realipd/gallery_results.json`
-> `all_with_hr`) -- real AACT trials with km_points + N + events + a posted HR.

Per trial:
  ctgov references -> the RESULT-type PMID (the primary results paper, the one
  that carries the KM figure + at-risk table) -> is it open-access in PMC?

Finding (2026-06-10): of 30 curve+HR trials, 25 link a publication but **0 have
an open-access PRIMARY in PMC** -- the trials that post structured registry
curves are predominantly industry RCTs whose primaries are paywalled, while the
OA papers that cite them are reviews/secondary analyses without the matching
2-arm figure. Combined with fusion_crossmatch (0/327 OA PDFs post a curve), the
fully-OA end-to-end fusion is bounded by an OPEN-ACCESS GAP, not by capability:
the two data sources are anti-correlated on openness. This is the measured
reason the RADIANT-4 demonstration (fusion_real_trial.py) takes the event count
from the AACT harvest rather than OCR-ing the (paywalled) Lancet figure -- and a
direct, quantified reinforcement of registry-ipd's POLICY.md ask.

No AACT snapshot needed (ctgov + NCBI/Europe-PMC APIs). Cached + incremental.

Run:  python fusion_pairfinder.py [--gallery <registry-ipd>/realipd/gallery_results.json]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

_UA = "Mozilla/5.0 (kmcurve-fusion-pairfinder research)"
CACHE = Path(__file__).resolve().parent / ".fusion_pairfinder_cache.json"


def _get(url: str, timeout: int = 30, retries: int = 4) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def result_pmid(nct: str) -> Optional[str]:
    """The RESULT-type linked PMID (the primary results paper) -- the one likely
    to carry the trial's KM figure + at-risk table. Falls back to any linked PMID."""
    data = _get(f"https://clinicaltrials.gov/api/v2/studies/{nct}?fields=ReferencesModule")
    if data is None:
        return None
    refs = (json.loads(data).get("protocolSection", {})
            .get("referencesModule", {}).get("references", []))
    result = [r["pmid"] for r in refs if r.get("type") == "RESULT" and r.get("pmid")]
    any_pmid = [r["pmid"] for r in refs if r.get("pmid")]
    return result[0] if result else (any_pmid[0] if any_pmid else None)


def pmids_to_pmc(pmids: List[str]) -> Dict[str, Optional[str]]:
    """Batch PMID -> PMCID via the NCBI ID converter (PMCID present => in PMC)."""
    out: Dict[str, Optional[str]] = {p: None for p in pmids}
    for i in range(0, len(pmids), 180):                      # idconv batch cap
        chunk = ",".join(pmids[i:i + 180])
        data = _get(f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={chunk}"
                    f"&format=json&tool=kmcurve&email=research@example.org")
        if data is None:
            continue
        for rec in json.loads(data).get("records", []):
            if rec.get("pmid") and rec.get("pmcid"):
                out[rec["pmid"]] = rec["pmcid"]
        time.sleep(0.34)
    return out


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            pass
    return {"nct_pmid": {}}


def run(gallery_path: Path) -> dict:
    gal = json.loads(gallery_path.read_text())
    trials = gal.get("all_with_hr") or gal.get("trials") or []
    cache = _load_cache()

    # 1) NCT -> primary (RESULT) PMID (cached).
    for t in trials:
        nct = t.get("nct") or t.get("nct_id")
        if nct and nct not in cache["nct_pmid"]:
            cache["nct_pmid"][nct] = result_pmid(nct)
            time.sleep(0.15)
    CACHE.write_text(json.dumps(cache))

    # 2) batch the primary PMIDs -> PMC OA.
    ncts = [(t.get("nct") or t.get("nct_id")) for t in trials]
    pmids = [cache["nct_pmid"].get(n) for n in ncts]
    pmc = pmids_to_pmc([p for p in pmids if p])

    rows = []
    for t in trials:
        nct = t.get("nct") or t.get("nct_id")
        pmid = cache["nct_pmid"].get(nct)
        pmcid = pmc.get(pmid) if pmid else None
        rows.append({"nct": nct, "condition": t.get("condition"),
                     "registry_hr": t.get("registry_HR") or t.get("registry_hr"),
                     "result_pmid": pmid, "oa_pmcid": pmcid})
    ready = [r for r in rows if r["oa_pmcid"]]
    return {
        "n_trials": len(trials),
        "n_with_primary_pmid": sum(1 for r in rows if r["result_pmid"]),
        "n_oa_primary": len(ready),       # fusion-ready end-to-end pairs
        "ready": ready,
        "rows": rows,
    }


def _print(out: dict) -> None:
    print("\n=== targeted fusion pair-finder (registry curve -> OA figure) ===")
    print(f"  curve+HR trials            : {out['n_trials']}")
    print(f"  with a linked primary PMID : {out['n_with_primary_pmid']}")
    print(f"  with an OA PRIMARY in PMC  : {out['n_oa_primary']}  (fusion-ready pairs)")
    if out["ready"]:
        print(f"\n  {'NCT':<14}{'PMCID':<13}{'regHR':>7}  condition")
        for r in out["ready"]:
            print(f"  {r['nct']:<14}{r['oa_pmcid']:<13}{str(r['registry_hr']):>7}  "
                  f"{(r['condition'] or '')[:34]}")
        print("\n  -> fetch the PMC PDF + OCR its at-risk table (raster_km) + the AACT"
              "\n     anchors (cohort/) -> fully real, no-simulated-link NAR fusion.")
    else:
        print("\n  No OA-primary pairs: the trials that post a registry curve have"
              "\n  PAYWALLED primaries (open-access gap). The fusion mechanism is proven"
              "\n  (42 datasets + RADIANT-4); the end-to-end OA reach is the bottleneck.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery",
                    default=r"C:\Projects\registry-ipd\realipd\gallery_results.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = run(Path(args.gallery))
    _print(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
