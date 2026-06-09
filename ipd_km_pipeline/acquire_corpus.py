#!/usr/bin/env python3
"""
Acquire a corpus of open-access KM-curve PDFs from PubMed Central.
==================================================================

Fetches real, diverse OA PDFs so the extractor can be benchmarked for
robustness across the messy variety of real-world figures (vector vs raster,
log axes, >2 arms, missing at-risk tables, etc.). PDFs land in a gitignored
corpus dir with a manifest recording PMCID / DOI / licence for provenance.

Pipeline per article:
  esearch (NCBI PMC, OA subset) -> PMC IDs -> Europe PMC rendered PDF
  (https://europepmc.org/articles/PMC<id>?pdf=render, returns application/pdf).

NCBI/EuropePMC etiquette: a few requests/sec; we sleep between calls.
Run:  python ipd_km_pipeline/acquire_corpus.py --n 30 --query "..."
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_PDF = "https://europepmc.org/articles/PMC{id}?pdf=render"
DEFAULT_QUERY = '"kaplan-meier"[Body] AND randomized AND open access[filter]'
_UA = "Mozilla/5.0 (KMcurve-corpus research)"


def _get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def esearch_pmc(query: str, n: int, retstart: int = 0) -> List[str]:
    from urllib.parse import quote_plus

    url = (f"{EUTILS}/esearch.fcgi?db=pmc&term={quote_plus(query)}"
           f"&retmax={n}&retstart={retstart}&retmode=json&sort=relevance")
    data = json.loads(_get(url))
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pdf(pmcid: str, dest: Path) -> Optional[Path]:
    """Download a rendered OA PDF from Europe PMC. Verifies the %PDF magic."""
    data = _get(EPMC_PDF.format(id=pmcid))
    if not data[:5].startswith(b"%PDF"):
        return None  # got an HTML interstitial / not actually a PDF
    out = dest / f"PMC{pmcid}.pdf"
    out.write_bytes(data)
    return out


def acquire(query: str, n: int, out_dir: Path, sleep: float = 0.34) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"pdfs": []}
    have = {e["pmcid"] for e in manifest["pdfs"]}

    ids = esearch_pmc(query, n * 4)  # over-fetch; not all render a PDF
    got, skipped = 0, {"no_pdf": 0, "error": 0, "dup": 0}
    for pmcid in ids:
        if got >= n:
            break
        if pmcid in have:
            skipped["dup"] += 1
            continue
        try:
            pdf = fetch_pdf(pmcid, out_dir)
            time.sleep(sleep)
            if pdf is None:
                skipped["no_pdf"] += 1
                continue
            manifest["pdfs"].append({"pmcid": pmcid, "pdf": pdf.name})
            got += 1
            print(f"  [{got}/{n}] PMC{pmcid}  {pdf.stat().st_size // 1024} KB")
        except Exception as exc:
            skipped["error"] += 1
            print(f"  skip PMC{pmcid}: {type(exc).__name__}: {exc}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nacquired {got} new PDFs (total {len(manifest['pdfs'])}); skipped {skipped}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "corpus"))
    args = ap.parse_args()
    print(f"Acquiring up to {args.n} OA KM PDFs -> {args.out}")
    acquire(args.query, args.n, Path(args.out))
