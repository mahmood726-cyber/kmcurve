#!/usr/bin/env python3
"""
Abstract-event-count fusion: validate a STRONG-effect pair without the figure.
==============================================================================

`fusion_forward_finder.py` proved the open-access wall is TOTAL for strongly-
separated trials: the figures are paywalled. But the QP fusion does not need the
figure -- it needs a per-arm total-EVENT count, and the trial's PubMed ABSTRACT
is free even when the primary is paywalled and routinely prints it:

    "death occurred in 107 of 205 ... versus 77 of 97 ..."       (X-of-N counts)
    "disease recurrence or death ... in 11% ... versus 46% ..."   (percentages)

This is registry-IPD's "abstract event-count lever" (KMCURVE-SYNERGY.md idea 5),
applied to kmcurve's strong forward-finder trials. Two extractors:
  - X-of-N counts: reuse registry-IPD's validated `abstract_events.extract_events`
    (100% precision on 161 abstracts);
  - per-arm event PERCENTAGES (this module's extension -- abstracts post rates far
    more often than "X of N"): events_i = round(rate_i * N_i), matched to a
    registry arm by its CURVE-implied event fraction (1 - S_final), label-free.

For each trial: registry curve anchors (registry-IPD cohort) + abstract events ->
Titman-QP -> log-rank HR vs the posted strong HR. No figure anywhere.

Run:  python fusion_abstract_lever.py [--candidates forward_candidates.json] [--nct NCT...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import fusion_crossmatch as X
from qp_reconstruct import reconstruct_arm_qp
from guyot import logrank_hr

sys.path.insert(0, r"C:\Projects\registry-ipd\harvest")
try:
    import abstract_events as AE          # registry-ipd's validated X-of-N extractor
except Exception:
    AE = None

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# PubMed publication types that are NOT the primary results paper.
_NON_PRIMARY = re.compile(
    r"comment|editorial|letter|reply|errat|retract|correction|news|biography|"
    r"published\s+erratum", re.IGNORECASE)


def _get(url: str, timeout: int = 30, retries: int = 4) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": X._UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def primary_pmid(nct: str) -> Optional[str]:
    """The trial's PRIMARY results paper PMID. ctgov references list replies/
    comments too, so search PubMed by the trial's secondary-source id (`<nct>[si]`)
    and pick the earliest record that is an original article (NOT a comment/reply/
    erratum), preferring RCT/Clinical-Trial publication types."""
    data = _get(f"{_EUTILS}/esearch.fcgi?db=pubmed&term={nct}[si]&retmax=30&retmode=json")
    if data is None:
        return None
    ids = json.loads(data).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None
    # one efetch for all candidates -> (pmid, year, pubtypes, title)
    summ = _get(f"{_EUTILS}/efetch.fcgi?db=pubmed&id={','.join(ids)}&rettype=medline&retmode=text")
    if summ is None:
        return ids[0]
    text = summ.decode("utf-8", "replace")
    recs = []
    for block in re.split(r"\nPMID- ", "\n" + text):
        pm = re.match(r"\s*(\d+)", block)
        if not pm:
            continue
        pmid = pm.group(1)
        ptypes = " ".join(re.findall(r"^PT  - (.+)$", block, re.MULTILINE))
        year = re.search(r"^DP  - (\d{4})", block, re.MULTILINE)
        is_rct = bool(re.search(r"randomized controlled trial|clinical trial, phase", ptypes, re.I))
        primary = not _NON_PRIMARY.search(ptypes)
        recs.append((primary, is_rct, int(year.group(1)) if year else 9999, pmid))
    if not recs:
        return ids[0]
    # primary first, then RCT-typed, then EARLIEST year (the primary precedes letters)
    recs.sort(key=lambda r: (not r[0], not r[1], r[2]))
    return recs[0][3]


def fetch_abstract(pmid: str) -> str:
    data = _get(f"{_EUTILS}/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text")
    return data.decode("utf-8", "replace") if data else ""


# --- percentage -> EVENT-FRACTION extension (abstracts post rates more than counts) ---
_PCT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d)?)\s*(?:%|per\s*cent|percent)", re.IGNORECASE)
# the percentage GOVERNOR (the ~45 chars before it) decides whether the % counts
# SURVIVORS (-> event fraction = 1-p) or EVENTS (-> event fraction = p). A clause
# can contain both words (CheckMate 066: "rate of SURVIVAL was 72.9% ... hazard
# ratio for DEATH") so classify on the LOCAL governor, not the whole clause.
_SURVIVOR_GOV = re.compile(
    r"surviv|alive|free\s+(?:of|from)|without\s+(?:progress|recurren|relaps|event|"
    r"death)|remain|progression[-\s]free|event[-\s]free|disease[-\s]free|"
    r"recurrence[-\s]free|relapse[-\s]free", re.IGNORECASE)
_EVENT_GOV = re.compile(
    r"die|died|death|mortalit|occurr|experienc|had\s+(?:an?\s+)?(?:event|progress|"
    r"recurren|relaps|death)|develop|progress(?:ed|ion)|recurr|relaps|"
    r"event(?:s)?\s+(?:occurr|were|in)", re.IGNORECASE)


def extract_event_fractions(abstract: str, endpoint: Optional[str]) -> List[float]:
    """Per-arm EVENT fractions (in 0-1) parsed from the abstract's percentages.

    Critical: a "%" can report SURVIVORS ("rate of survival was 72.9%") or EVENTS
    ("death occurred in 27%"); these are 1-p apart, so misreading one as the other
    inverts the count. Each % is classified by its LOCAL governor (the words just
    before it), and survivor-% is converted to an event fraction as 1-p. A % whose
    governor is ambiguous (both/neither survivor and event marker) is REJECTED --
    a wrong count silently corrupts the QP, so over-inclusion is the enemy.
    Clause-level guards (endpoint scope, not adverse/enrolment/negated) still apply."""
    if AE is None or not abstract:
        return []
    text = AE._normalise(abstract)
    ep_rx = AE._ENDPOINT_RX.get(endpoint) if endpoint else None
    out, prev_kind, prev_end = [], None, None
    for m in _PCT.finditer(text):
        clause, _ = AE._clause(text, m.start(), m.end())
        if AE._NEG.search(clause) or AE._ENROLL.search(clause) or AE._ADVERSE.search(clause):
            prev_kind = None
            continue
        if ep_rx and not ep_rx.search(clause):
            prev_kind = None
            continue
        gov = text[max(0, m.start() - 45):m.start()]           # the local governor
        is_surv, is_event = bool(_SURVIVOR_GOV.search(gov)), bool(_EVENT_GOV.search(gov))
        if is_surv != is_event:
            kind = "surv" if is_surv else "event"
        elif (prev_kind is not None and prev_end is not None       # comparator inheritance:
              and AE._COMPARATOR.search(text[prev_end:m.start()])  # "72.9% ... versus 42.1%"
              and m.start() - prev_end < 80):                      # the 2nd % inherits the 1st's kind
            kind = prev_kind
        else:
            prev_kind = None
            continue                                            # ambiguous -> reject
        p = float(m.group(1)) / 100.0
        if not 0.0 < p < 1.0:
            prev_kind = None
            continue
        out.append(1.0 - p if kind == "surv" else p)            # survivors -> event fraction
        prev_kind, prev_end = kind, m.end()
    return out


def _arm_curve(arm: dict) -> Tuple[np.ndarray, np.ndarray]:
    km = sorted((p for p in arm["km_points"] if p.get("t") is not None), key=lambda p: p["t"])
    t = np.array([0.0] + [float(p["t"]) for p in km])
    s = np.minimum.accumulate(np.clip(np.array([1.0] + [float(p["S"]) for p in km]), 0, 1))
    return t, s


# strongly-separated trials MUST have clearly different per-arm event fractions; a
# near-equal pair from the percentage path means the wrong numbers were grabbed
# (response/disease-control/AE rates), so reject it. Empirical: NCT00699816 RFS 0.63
# extracted [108,106] (~equal) -> inverted HR 1.45; NCT04737187 [234,234].
_MIN_FRAC_GAP = 0.12


def _events_for_arms(abstract: str, endpoint: Optional[str],
                     arms: List[dict]) -> Tuple[Optional[List[int]], str, str]:
    """Per-arm total-event counts from the abstract, matched to `arms`.

    Returns (events_per_arm or None, source, confidence). The X-of-N path
    (registry-ipd, validated 100% precision) is HIGH confidence. The percentage
    path is LOW confidence and gated: abstracts carry many percentages (response,
    AE, survival-at-t) so it can grab the wrong one and emit a confident-but-wrong
    count (the silent-substitution trap) -- it is flagged needs-verification and
    rejected when the two event fractions are implausibly near-equal for a strong
    trial."""
    ns = [int(a["N"]) for a in arms]
    implied = [1.0 - float(_arm_curve(a)[1][-1]) for a in arms]   # curve-implied event fraction

    # (1) X-of-N counts (registry-ipd, validated) -- HIGH confidence
    if AE is not None:
        ev = AE.extract_events(abstract, endpoint=endpoint) or AE.extract_events(abstract)
        if ev and len(ev.get("events", [])) >= 2:
            counts, enns = ev["events"], ev["ns"]
            matched = [None, None]
            for e, n in zip(counts, enns):
                j = min(range(len(ns)), key=lambda k: abs(ns[k] - n))
                if matched[j] is None and 0 <= e <= ns[j]:
                    matched[j] = e
            if all(m is not None for m in matched):
                return matched, "abstract_x_of_n", "high"

    # (2) event PERCENTAGES -> counts (experimental extension) -- LOW confidence
    fracs = extract_event_fractions(abstract, endpoint)
    if len(fracs) >= 2:
        fracs = fracs[:2]
        if abs(fracs[0] - fracs[1]) < _MIN_FRAC_GAP:             # near-equal -> wrong numbers
            return None, "percentage_rejected_near_equal", "none"
        order = sorted(range(2), key=lambda i: implied[i])
        frac_sorted = sorted(fracs)
        events = [0, 0]
        for rank, arm_i in enumerate(order):
            events[arm_i] = int(round(frac_sorted[rank] * ns[arm_i]))
        if all(0 < e <= n for e, n in zip(events, ns)):
            return events, "abstract_percentage", "low"
    return None, "none", "none"


def run_one(nct: str, posted_hr: float, endpoint: Optional[str],
            cohort_dir: Path) -> Optional[dict]:
    cohort_path = cohort_dir / f"{nct}.json"
    if not cohort_path.exists():
        return {"nct": nct, "skip": "no registry cohort"}
    cohort = json.loads(cohort_path.read_text())
    arms = [a for a in cohort["arms"] if a.get("km_points") and a.get("N")]
    if len(arms) != 2:
        return {"nct": nct, "skip": f"{len(arms)} registry arms with km_points (need 2)"}

    pmid = primary_pmid(nct)
    abstract = fetch_abstract(pmid) if pmid else ""
    events, source, confidence = _events_for_arms(abstract, endpoint, arms)
    if events is None:
        return {"nct": nct, "pmid": pmid, "skip": f"no usable event count ({source})",
                "posted_hr": posted_hr}

    # order (lower-event arm, higher-event arm) so HR<1 favours the better arm
    lo, hi = (0, 1) if events[0] <= events[1] else (1, 0)
    recon = []
    for idx in (lo, hi):
        t, s = _arm_curve(arms[idx])
        qt, qe = reconstruct_arm_qp(t, s, total_n=int(arms[idx]["N"]),
                                    total_events=events[idx], follow_up_max=float(t[-1]))
        recon.append((qt, qe))
    hr = logrank_hr(recon[0][0], recon[0][1], recon[1][0], recon[1][1])["hr"]
    fold = round(max(hr, posted_hr) / min(hr, posted_hr), 3) if posted_hr and hr > 0 else None
    return {
        "nct": nct, "pmid": pmid, "endpoint": endpoint, "events_source": source,
        "confidence": confidence, "needs_verification": confidence != "high",
        "events": events, "arm_Ns": [int(a["N"]) for a in arms],
        "posted_hr": posted_hr, "fusion_hr": round(hr, 3), "fold_vs_posted": fold,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="forward_candidates.json")
    ap.add_argument("--cohort-dir", default=r"C:\Projects\registry-ipd\cohort")
    ap.add_argument("--nct", nargs="*", default=None)
    ap.add_argument("--out", default="abstract_lever_results.json")
    args = ap.parse_args()

    strong = json.loads(Path(args.candidates).read_text())["strong"]
    if args.nct:
        strong = [s for s in strong if s["nct"] in set(args.nct)]

    results = []
    for s in sorted(strong, key=lambda r: -r["hr_separation"]):
        r = run_one(s["nct"], float(s["posted_hr"]), s.get("endpoint"),
                    Path(args.cohort_dir))
        if r:
            r["hr_separation"] = s["hr_separation"]
            results.append(r)
            if "skip" in r:
                print(f"  {r['nct']} (sep {s['hr_separation']:.2f}): SKIP - {r['skip']}", flush=True)
            else:
                tag = "HIGH" if r["confidence"] == "high" else "LOW/verify"
                print(f"  {r['nct']} (sep {s['hr_separation']:.2f}, {r['events_source']}, "
                      f"{tag}): events {r['events']} -> fusion HR {r['fusion_hr']} vs posted "
                      f"{r['posted_hr']} (fold {r['fold_vs_posted']})", flush=True)

    high = [r for r in results if r.get("confidence") == "high"]
    low = [r for r in results if r.get("confidence") == "low"]
    print(f"\n{len(high)}/{len(results)} validated from a HIGH-confidence (X-of-N) abstract "
          f"count; {len(low)} more LOW-confidence (percentage, needs human verification).")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
