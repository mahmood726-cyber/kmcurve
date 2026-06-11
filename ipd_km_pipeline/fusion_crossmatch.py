#!/usr/bin/env python3
"""
Cross-match the figure corpus against ClinicalTrials.gov posted KM curves.
=========================================================================

The fully end-to-end NAR fusion (fusion_real_trial.py, but with kmcurve OCR-ing
the at-risk table off the ACTUAL figure) needs a single trial that is in BOTH
worlds at once:

  (a) a corpus PDF whose KM figure kmcurve can extract, AND
  (b) a ClinicalTrials.gov record that POSTS a structured KM-estimate curve
      (>= 3 survival timepoints) -- the exact registry anchors -- ideally with a
      posted hazard ratio as held-out ground truth.

This scan finds those pairs. It is cheap, cached, and incremental, so it can be
re-run as the corpus grows toward thousands of PDFs (matches are rare -- the
registry census shows only ~0.4-0.8% of trials post a structured curve):

  1. extract NCT ids from each corpus PDF (pdfplumber text + regex; cached per PDF)
  2. for each unique NCT, query the public ctgov API v2 for a posted KM-estimate
     outcome (>=3 timepoint classes + a survival-type title) + a posted HR (cached)
  3. emit ranked fusion-ready candidates (PDF <-> NCT) to JSON

A hit means: harvest the NCT's anchors (registry-ipd harvest_trial.py) + run
kmcurve on the PDF to OCR its at-risk table -> the first fully real, no-simulated-
link NAR fusion. No AACT snapshot needed (ctgov API only).

Run:  python fusion_crossmatch.py [--corpus corpus_pmc] [--limit N] [--out cands.json]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

_NCT = re.compile(r"NCT0\d{7}")
_UA = "Mozilla/5.0 (kmcurve-fusion-crossmatch research)"
_SURV_KW = ("surviv", "event-free", "event free", "progression-free", "progression free",
            "remaining", "alive", "free of", "kaplan", "recurrence-free", "disease-free",
            "relapse-free", "time to", "freedom from")
CACHE = Path(__file__).resolve().parent / ".fusion_crossmatch_cache.json"


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


def ncts_in_pdf(pdf: Path) -> List[str]:
    """Unique NCT ids cited anywhere in the PDF text. Uses PyMuPDF (fitz) when available -- ~10x faster
    than pdfplumber (~1.3s vs ~15s/PDF), which matters as the corpus scales to thousands of PDFs -- and
    falls back to pdfplumber. NCT extraction is just a regex over the text, so the two agree; the cached
    pdf->NCT map stays valid across the switch."""
    found = set()
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf) as doc:
            for page in doc:
                found.update(_NCT.findall(page.get_text() or ""))
        return sorted(found)
    except ImportError:
        pass
    except Exception:
        return []
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                found.update(_NCT.findall(page.extract_text() or ""))
    except Exception:
        return []
    return sorted(found)


def fetch_study(nct: str, sleep: float = 0.2) -> dict:
    """Fetch + TRIM a ctgov v2 study to the survival outcome measures (title,
    paramType, class titles, group count, HR analyses). Cached so re-detection
    after a detector change never re-fetches. Returns a minimal dict in the same
    shape `_km_from_study` parses."""
    data = _get(f"https://clinicaltrials.gov/api/v2/studies/{nct}")
    time.sleep(sleep)
    if data is None:
        return {"error": "fetch_failed"}
    try:
        d = json.loads(data)
    except Exception:
        return {"error": "bad_json"}
    if not d.get("hasResults"):
        return {"hasResults": False}
    oms = (d.get("resultsSection", {}).get("outcomeMeasuresModule", {})
           .get("outcomeMeasures", []))
    keep = []
    for om in oms:
        title = om.get("title") or ""
        if not any(k in title.lower() for k in _SURV_KW):
            continue  # only survival measures carry the curve + HR; shrinks cache
        keep.append({
            "title": title, "paramType": om.get("paramType", ""),
            "classes": [{"title": c.get("title", "")} for c in om.get("classes", [])],
            "groups": [{} for _ in om.get("groups", [])],
            "analyses": [{"paramType": a.get("paramType"), "paramValue": a.get("paramValue"),
                          "ciLowerLimit": a.get("ciLowerLimit"), "ciUpperLimit": a.get("ciUpperLimit")}
                         for a in (om.get("analyses") or [])],
        })
    return {"hasResults": True,
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": keep}}}


# class titles that are SUBGROUP strata, not KM-curve timepoints (false positives)
_SUBGROUP_KW = ("sex:", "race:", "age", "gender", "region", "ethnic", "subgroup",
                "ecog", "stage:", "histolog", "mutation", "smoking", "weight",
                "prior ", "baseline ", "country", "<", ">", "=", "male", "female")
# paramTypes that are NOT a survival probability (a curve posts a probability/%)
_NON_CURVE_PARAM = ("median", "mean", "geometric", "least squares", "hazard", "count")
_TIME_RE = re.compile(r"(month|year|week|day|\bmo\b|\byr\b|\bwk\b|time\s*point|^\s*[\d.]+\s*$)",
                      re.IGNORECASE)


def _is_timepoint(class_title: str) -> bool:
    """A KM-curve class is a TIME (e.g. 'Month 6', '12 months', '24'), NOT a
    subgroup stratum (e.g. 'Sex: Male', 'Race: Caucasian')."""
    t = (class_title or "").strip().lower()
    if not t:
        return True   # untitled classes in a probability measure are usually timepoints
    if any(k in t for k in _SUBGROUP_KW):
        return False
    return bool(_TIME_RE.search(t))


def _km_from_study(d: dict, nct: str) -> dict:
    """Pure parse: detect a posted KM-estimate CURVE (a survival-probability
    outcome with >=3 TIMEPOINT classes -- not a subgroup/median table) and a
    posted HR (from any survival analysis in the trial)."""
    if not d.get("hasResults"):
        return {"nct": nct, "has_results": False}
    oms = (d.get("resultsSection", {}).get("outcomeMeasuresModule", {})
           .get("outcomeMeasures", []))
    # any survival HR posted anywhere in the trial (the curve measure often
    # differs from the measure carrying the HR analysis)
    trial_hr = None
    for om in oms:
        if any(k in (om.get("title") or "").lower() for k in _SURV_KW):
            for a in (om.get("analyses") or []):
                if "hazard" in (a.get("paramType") or "").lower():
                    trial_hr = {"value": a.get("paramValue"),
                                "ci": [a.get("ciLowerLimit"), a.get("ciUpperLimit")]}
                    break
        if trial_hr:
            break
    best = None
    for om in oms:
        title = (om.get("title") or "")
        param = (om.get("paramType") or "")
        is_surv = any(k in title.lower() for k in _SURV_KW)
        is_curve_param = not any(k in param.lower() for k in _NON_CURVE_PARAM)
        classes = om.get("classes", [])
        n_tp = sum(1 for c in classes if _is_timepoint(c.get("title", "")))
        n_groups = len(om.get("groups", []))
        if is_surv and is_curve_param and n_tp >= 3 and n_groups >= 2:
            cand = {"title": title[:80], "n_timepoints": n_tp, "param": param,
                    "n_groups": n_groups, "hr": trial_hr}
            if best is None or n_tp > best["n_timepoints"]:
                best = cand
    return {"nct": nct, "has_results": True, "km": best}


_CACHE_VERSION = 2  # bump when the cached study shape changes


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            c = json.loads(CACHE.read_text())
            if c.get("version") == _CACHE_VERSION:
                return c
            # stale schema: keep the expensive pdf->NCT map, drop ctgov entries
            return {"version": _CACHE_VERSION,
                    "pdf_ncts": c.get("pdf_ncts", {}), "nct_info": {}}
        except Exception:
            pass
    return {"version": _CACHE_VERSION, "pdf_ncts": {}, "nct_info": {}}


def run(corpus_dir: Path, limit: Optional[int] = None) -> dict:
    cache = _load_cache()
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    # 1) NCTs per PDF (cached per filename).
    new_pdfs = 0
    for pdf in pdfs:
        if pdf.name not in cache["pdf_ncts"]:
            cache["pdf_ncts"][pdf.name] = ncts_in_pdf(pdf)
            new_pdfs += 1
            if new_pdfs % 20 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  scanned {new_pdfs} new PDFs for NCT ids...", flush=True)
    CACHE.write_text(json.dumps(cache))

    # 2) trimmed ctgov study per unique NCT (cached as raw study, not a verdict,
    #    so a detector change re-detects offline without re-fetching).
    pdf_ncts = {k: v for k, v in cache["pdf_ncts"].items() if k in {p.name for p in pdfs}}
    all_ncts = sorted({n for ns in pdf_ncts.values() for n in ns})
    new_ncts = 0
    for nct in all_ncts:
        if nct not in cache["nct_info"]:
            cache["nct_info"][nct] = fetch_study(nct)
            new_ncts += 1
            if new_ncts % 10 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  queried {new_ncts} new NCTs on ctgov...", flush=True)
    CACHE.write_text(json.dumps(cache))

    # 3) detect posted KM curves + rank fusion-ready candidates (offline).
    candidates = []
    n_with_results = 0
    for nct in all_ncts:
        study = cache["nct_info"].get(nct, {})
        if study.get("hasResults"):
            n_with_results += 1
    for pdf_name, ncts in pdf_ncts.items():
        for nct in ncts:
            km = _km_from_study(cache["nct_info"].get(nct, {}), nct).get("km")
            if km:
                candidates.append({
                    "pdf": pdf_name, "nct": nct,
                    "n_timepoints": km["n_timepoints"], "n_groups": km.get("n_groups"),
                    "posted_hr": (km.get("hr") or {}).get("value"),
                    "posted_ci": (km.get("hr") or {}).get("ci"),
                    "outcome": km["title"],
                })
    # best first: has HR, then more timepoints
    candidates.sort(key=lambda c: (c["posted_hr"] is not None, c["n_timepoints"]), reverse=True)
    return {
        "n_pdfs": len(pdfs),
        "n_pdfs_with_nct": sum(1 for ns in pdf_ncts.values() if ns),
        "n_unique_ncts": len(all_ncts),
        "n_ncts_with_results": n_with_results,
        "n_fusion_candidates": len(candidates),
        "candidates": candidates,
    }


def _print(out: dict) -> None:
    print(f"\n=== fusion cross-match ===")
    print(f"  PDFs scanned          : {out['n_pdfs']}")
    print(f"  PDFs citing an NCT     : {out['n_pdfs_with_nct']}")
    print(f"  unique NCTs            : {out['n_unique_ncts']}")
    print(f"  NCTs with posted results: {out['n_ncts_with_results']}")
    print(f"  FUSION-READY candidates : {out['n_fusion_candidates']}")
    if out["candidates"]:
        print(f"\n  {'PDF':<30}{'NCT':<14}{'tpts':>5}{'postedHR':>10}  outcome")
        print("  " + "-" * 84)
        for c in out["candidates"][:25]:
            hr = c["posted_hr"] if c["posted_hr"] is not None else "-"
            print(f"  {c['pdf'][:29]:<30}{c['nct']:<14}{c['n_timepoints']:>5}{str(hr):>10}  "
                  f"{c['outcome'][:34]}")
        print("\n  -> harvest the NCT anchors (registry-ipd harvest_trial.py) + OCR the PDF's"
              "\n     at-risk table (raster_km) for a fully real NAR fusion.")
    else:
        print("\n  No fusion-ready pairs yet. Re-run as the corpus grows (matches are rare:"
              "\n  ~0.4-0.8% of trials post a structured KM curve).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_pmc")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    corpus = Path(args.corpus) if Path(args.corpus).is_absolute() else base / args.corpus
    out = run(corpus, args.limit)
    _print(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
