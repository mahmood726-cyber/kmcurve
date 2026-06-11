#!/usr/bin/env python3
"""
Forward pair-finder: ctgov-search -> strong-HR 2-arm trial -> OA primary -> figure.
==================================================================================

`fusion_crossmatch.py` is BACKWARD (corpus PDF -> NCT): it only sees trials whose
NCT happens to be cited by an OA PDF we already downloaded -- and the strongly-
separated ones turned out to be cited only by subgroup/secondary papers with no
extractable figure (the open-access wall; see CORPUS_FINDINGS.md). `fusion_pairfinder.py`
is FORWARD but bounded to registry-ipd's 30-trial gallery (0 OA primaries).

This goes forward at SCALE: search ClinicalTrials.gov directly for results-posting
RCTs, keep those that post a 2-arm KM curve with a STRONGLY-SEPARATED endpoint-
matched HR (|log HR| >= --min-sep), then for each follow the RESULT-type PMID ->
PMC open-access -> download the rendered PDF -> verify a KM figure actually
EXTRACTS. A hit is a fully end-to-end fusion pair *with a large true effect* --
the high-signal validation PALOMA-3 (flat OS 0.814) cannot provide.

Composes existing, tested parts: fusion_crossmatch.{fetch_study,_km_from_study,
_hr_separation,figure_extractable}, fusion_pairfinder.{result_pmid,pmids_to_pmc},
acquire_corpus.fetch_pdf. ctgov + NCBI/Europe-PMC APIs only; cached + incremental.

Run:  python fusion_forward_finder.py [--query "..."] [--max-ncts 300] [--min-sep 0.4]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

import fusion_crossmatch as X
from fusion_pairfinder import result_pmid, pmids_to_pmc

CACHE = Path(__file__).resolve().parent / ".fusion_forward_cache.json"
_SEARCH = "https://clinicaltrials.gov/api/v2/studies"


def search_ncts(query: str, max_ncts: int, page_size: int = 100,
                agg: str = "results:with,phase:3") -> List[str]:
    """NCT ids of results-posting studies matching `query`, paged.

    NB: do NOT put "hazard ratio" in `query.term` -- it is not in ctgov's
    searchable text index, so it collapses the result count (e.g. 'overall
    survival hazard ratio' -> 93 vs 'overall survival' -> 9308). Search on the
    indexed survival term and let the client-side `_km_from_study` /
    `_hr_separation` filter to a posted 2-arm curve with a strong HR."""
    ncts: List[str] = []
    token = None
    while len(ncts) < max_ncts:
        params = {"query.term": query, "aggFilters": agg,
                  "fields": "NCTId", "pageSize": str(min(page_size, max_ncts - len(ncts)))}
        if token:
            params["pageToken"] = token
        data = X._get(f"{_SEARCH}?{urllib.parse.urlencode(params)}")
        if data is None:
            break
        try:
            d = json.loads(data)
        except Exception:
            break
        for s in d.get("studies", []):
            nct = (s.get("protocolSection", {}).get("identificationModule", {})
                   .get("nctId"))
            if nct:
                ncts.append(nct)
        token = d.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return ncts[:max_ncts]


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            pass
    return {"study": {}}        # nct -> trimmed ctgov study (same shape fetch_study returns)


def run(query: str, max_ncts: int, min_sep: float, corpus_dir: Path,
        verify_figures: bool = True, agg: str = "results:with,phase:3") -> dict:
    cache = _load_cache()
    ncts = search_ncts(query, max_ncts, agg=agg)
    print(f"ctgov search -> {len(ncts)} results-posting studies", flush=True)

    # 1) detect a posted 2-arm curve + STRONGLY-SEPARATED endpoint-matched HR.
    strong = []
    for i, nct in enumerate(ncts, 1):
        if nct not in cache["study"]:
            cache["study"][nct] = X.fetch_study(nct)
            if i % 20 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  scanned {i}/{len(ncts)} studies; {len(strong)} strong so far",
                      flush=True)
        km = X._km_from_study(cache["study"][nct], nct).get("km")
        if not km or not km.get("hr_endpoint_matched") or km.get("n_groups") != 2:
            continue
        sep = X._hr_separation((km.get("hr") or {}).get("value"))
        if sep is not None and sep >= min_sep:
            strong.append({"nct": nct, "endpoint": km["endpoint"],
                           "posted_hr": km["hr"]["value"], "posted_ci": km["hr"]["ci"],
                           "n_timepoints": km["n_timepoints"], "hr_separation": round(sep, 3)})
    CACHE.write_text(json.dumps(cache))
    strong.sort(key=lambda r: r["hr_separation"], reverse=True)
    print(f"-> {len(strong)} strongly-separated 2-arm posted-curve trials "
          f"(|log HR| >= {min_sep})", flush=True)

    # 2) primary RESULT PMID -> PMC open-access.
    for r in strong:
        r["result_pmid"] = result_pmid(r["nct"])
        time.sleep(0.12)
    pmc = pmids_to_pmc([r["result_pmid"] for r in strong if r["result_pmid"]])
    for r in strong:
        r["oa_pmcid"] = pmc.get(r["result_pmid"]) if r.get("result_pmid") else None

    # 3) for OA primaries, download + verify a KM figure actually EXTRACTS.
    from acquire_corpus import fetch_pdf
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for r in strong:
        r["figure_extractable"] = None
        if not r.get("oa_pmcid") or not verify_figures:
            continue
        pmcid = r["oa_pmcid"].replace("PMC", "")
        pdf = corpus_dir / f"PMC{pmcid}.pdf"
        if not pdf.exists():
            try:
                got = fetch_pdf(pmcid, corpus_dir)
            except Exception:
                got = None
            if not got:
                r["figure_extractable"] = False
                continue
        r["figure_extractable"] = X.figure_extractable(pdf)

    ready = [r for r in strong if r.get("oa_pmcid") and r.get("figure_extractable")]
    return {
        "query": query, "n_scanned": len(ncts), "min_sep": min_sep,
        "n_strong_2arm": len(strong),
        "n_strong_oa_primary": sum(1 for r in strong if r.get("oa_pmcid")),
        "n_ready": len(ready),               # strong + OA + extractable figure
        "ready": ready, "strong": strong,
    }


def _print(out: dict) -> None:
    print("\n=== forward fusion finder (ctgov -> strong HR -> OA figure) ===")
    print(f"  studies scanned             : {out['n_scanned']}")
    print(f"  strong 2-arm posted curves  : {out['n_strong_2arm']}  (|log HR| >= {out['min_sep']})")
    print(f"  ...with an OA primary in PMC: {out['n_strong_oa_primary']}")
    print(f"  ...with an EXTRACTABLE figure (fusion-ready): {out['n_ready']}")
    if out["ready"]:
        print(f"\n  {'NCT':<13}{'PMCID':<13}{'ep':<5}{'HR':>6}{'sep':>6}")
        for r in out["ready"]:
            print(f"  {r['nct']:<13}{r['oa_pmcid']:<13}{r['endpoint']:<5}"
                  f"{str(r['posted_hr']):>6}{r['hr_separation']:>6}")
        print("\n  -> harvest the NCT anchors (registry-ipd) + OCR this PDF's at-risk"
              "\n     table (raster_km) for a STRONG-EFFECT end-to-end NAR fusion.")
    else:
        print("\n  No strong + OA + extractable pair (the open-access wall). The strong"
              "\n  trials' OA primaries are absent/paywalled or their figures don't extract.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="overall survival",
                    help="ctgov searchable term (NOT 'hazard ratio' -- unindexed); "
                         "the HR is detected client-side")
    ap.add_argument("--agg", default="results:with,phase:3",
                    help="ctgov aggFilters (e.g. 'results:with' to drop the phase-3 limit)")
    ap.add_argument("--max-ncts", type=int, default=1500)
    ap.add_argument("--min-sep", type=float, default=0.4)
    ap.add_argument("--corpus", default="corpus_pmc")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip PDF download + figure check (registry-side scan only)")
    ap.add_argument("--out", default="forward_candidates.json")
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    corpus = Path(args.corpus) if Path(args.corpus).is_absolute() else base / args.corpus
    out = run(args.query, args.max_ncts, args.min_sep, corpus,
              verify_figures=not args.no_figures, agg=args.agg)
    _print(out)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
