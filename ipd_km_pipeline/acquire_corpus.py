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
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_PDF = "https://europepmc.org/articles/PMC{id}?pdf=render"
DEFAULT_QUERY = '"kaplan-meier"[Body] AND randomized AND open access[filter]'
_UA = "Mozilla/5.0 (KMcurve-corpus research)"


def _get(url: str, timeout: int = 30, retries: int = 4, max_seconds: float = 90.0) -> bytes:
    """Fetch with bounded exponential backoff AND a wall-clock deadline.

    Local DNS resolution fails in bursts under rapid-fire requests (getaddrinfo
    failed / Errno 11001) then recovers; retrying turns those transient bursts
    into successes. A genuine 404/410/403 is NOT retried; only connection/DNS/
    5xx-class errors are.

    The wall-clock deadline (chunked read with a per-fetch ``max_seconds`` budget)
    is essential: urllib's ``timeout`` is PER-SOCKET-OPERATION, so a server that
    trickles bytes slowly can stall a single fetch far beyond ``timeout`` without
    ever tripping it -- observed hanging the whole acquisition on one slow PDF.
    The budget bounds any single fetch and lets the retry/skip logic move on."""
    import urllib.error

    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        try:
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                chunks = []
                while True:
                    if time.monotonic() - start > max_seconds:
                        raise TimeoutError(f"fetch exceeded {max_seconds}s wall-clock")
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            # 404/410/403/500 are permanent for an EPMC ?pdf=render URL (no
            # renderable PDF for this article); 429/502/503 are worth retrying.
            if exc.code in (404, 410, 403, 500) or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except TimeoutError:
            # Socket or wall-clock timeout = this fetch is too slow. Retrying the
            # same slow URL won't help and would burn retries*max_seconds on one
            # bad PDF -- fail fast so the caller skips it and moves on. (DNS bursts
            # raise URLError below, which IS retried.)
            raise
        except Exception:
            # URLError (DNS getaddrinfo bursts, conn reset) -- transient, back off.
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")  # loop always returns or raises


def esearch_pmc(query: str, n: int, retstart: int = 0, retries: int = 4) -> List[str]:
    from urllib.parse import quote_plus

    url = (f"{EUTILS}/esearch.fcgi?db=pmc&term={quote_plus(query)}"
           f"&retmax={n}&retstart={retstart}&retmode=json&sort=relevance")
    # Bounded backoff: a transient blip on the page fetch must not kill a
    # multi-hour resumable acquisition run.
    for attempt in range(retries):
        try:
            data = json.loads(_get(url))
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  esearch failed at retstart={retstart} "
                      f"after {retries} tries: {type(exc).__name__}")
                return []
            time.sleep(2 ** attempt)  # 1, 2, 4 s
    return []


def fetch_pdf(pmcid: str, dest: Path) -> Optional[Path]:
    """Download a rendered OA PDF from Europe PMC. Verifies the %PDF magic."""
    data = _get(EPMC_PDF.format(id=pmcid))
    if not data[:5].startswith(b"%PDF"):
        return None  # got an HTML interstitial / not actually a PDF
    out = dest / f"PMC{pmcid}.pdf"
    out.write_bytes(data)
    return out


EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}?email={email}"


def epmc_search(query: str, n: int) -> List[dict]:
    """Europe PMC REST search -> list of {id, source, doi, pdf_urls}."""
    from urllib.parse import quote

    url = (f"{EPMC_SEARCH}?query={quote(query)}&format=json"
           f"&pageSize={min(n, 100)}&resultType=core")
    data = json.loads(_get(url))
    out = []
    for r in data.get("resultList", {}).get("result", []):
        pdfs = [u.get("url") for u in r.get("fullTextUrlList", {}).get("fullTextUrl", [])
                if u.get("documentStyle") == "pdf"]
        out.append({"id": r.get("id"), "source": r.get("source"),
                    "doi": r.get("doi", ""), "pdf_urls": pdfs})
    return out


def _save_pdf_bytes(data: bytes, out: Path) -> Optional[Path]:
    if not data[:5].startswith(b"%PDF"):
        return None
    out.write_bytes(data)
    return out


def acquire_unpaywall(query: str, n: int, out_dir: Path,
                      email: str = "research@example.org", sleep: float = 0.34) -> dict:
    """Publisher OA PDFs via Unpaywall (DOIs from a Europe PMC MED search).

    Tests whether PUBLISHER PDFs preserve vector figures (unlike PMC-rendered).
    Publisher servers often block bot PDF access, so yield can be low -- that
    is reported honestly, not hidden.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"pdfs": []}
    have = {e.get("doi") for e in manifest["pdfs"]}

    hits = epmc_search(f"{query} AND SRC:MED", n * 6)
    got, skipped = 0, {"no_doi": 0, "not_oa": 0, "blocked": 0, "dup": 0, "error": 0}
    for h in hits:
        if got >= n:
            break
        doi = h["doi"]
        if not doi:
            skipped["no_doi"] += 1
            continue
        if doi in have:
            skipped["dup"] += 1
            continue
        try:
            uw = json.loads(_get(UNPAYWALL.format(doi=doi, email=email)))
            time.sleep(sleep)
            loc = uw.get("best_oa_location") or {}
            pdf_url = loc.get("url_for_pdf")
            if not pdf_url:
                skipped["not_oa"] += 1
                continue
            safe = doi.replace("/", "_")
            pdf = _save_pdf_bytes(_get(pdf_url, timeout=60), out_dir / f"{safe}.pdf")
            time.sleep(sleep)
            if pdf is None:
                skipped["blocked"] += 1
                continue
            manifest["pdfs"].append({"doi": doi, "pdf": pdf.name, "source": "unpaywall"})
            got += 1
            print(f"  [{got}/{n}] {doi}  {pdf.stat().st_size // 1024} KB")
        except Exception as exc:
            skipped["error"] += 1
            print(f"  skip {doi}: {type(exc).__name__}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nunpaywall: acquired {got} (total {len(manifest['pdfs'])}); skipped {skipped}")
    return manifest


def acquire_biorxiv(query: str, n: int, out_dir: Path, sleep: float = 0.34) -> dict:
    """bioRxiv/medRxiv preprint PDFs (DOI prefix 10.1101) via Europe PMC + the
    predictable .full.pdf URL. Preprints usually keep vector figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"pdfs": []}
    have = {e.get("doi") for e in manifest["pdfs"]}

    hits = epmc_search(f"{query} AND SRC:PPR", n * 8)
    got, skipped = 0, {"not_biorxiv": 0, "blocked": 0, "dup": 0, "error": 0}
    for h in hits:
        if got >= n:
            break
        doi = h["doi"]
        if not doi.startswith("10.1101/"):  # bioRxiv/medRxiv prefix
            skipped["not_biorxiv"] += 1
            continue
        if doi in have:
            skipped["dup"] += 1
            continue
        try:
            server = "biorxiv"  # medRxiv shares the URL scheme; try biorxiv host
            url = f"https://www.{server}.org/content/{doi}v1.full.pdf"
            safe = doi.replace("/", "_")
            pdf = _save_pdf_bytes(_get(url, timeout=60), out_dir / f"{safe}.pdf")
            time.sleep(sleep)
            if pdf is None:
                pdf = _save_pdf_bytes(_get(f"https://www.medrxiv.org/content/{doi}v1.full.pdf",
                                           timeout=60), out_dir / f"{safe}.pdf")
                time.sleep(sleep)
            if pdf is None:
                skipped["blocked"] += 1
                continue
            manifest["pdfs"].append({"doi": doi, "pdf": pdf.name, "source": "biorxiv"})
            got += 1
            print(f"  [{got}/{n}] {doi}  {pdf.stat().st_size // 1024} KB")
        except Exception as exc:
            skipped["error"] += 1
            print(f"  skip {doi}: {type(exc).__name__}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nbiorxiv: acquired {got} (total {len(manifest['pdfs'])}); skipped {skipped}")
    return manifest


def acquire(query: str, n: int, out_dir: Path, sleep: float = 0.34,
            page: int = 200, flush_every: int = 5) -> dict:
    """Acquire up to ``n`` NEW PMC OA PDFs, paginating the relevance-sorted
    esearch via ``retstart`` so the corpus can grow to thousands across
    resumable runs (the manifest's ``pmcid`` set dedups, so a re-run walks past
    everything already on disk). The manifest is flushed every ``flush_every``
    new PDFs, so an interrupted multi-hour run keeps what it acquired."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"pdfs": []}
    manifest.setdefault("no_pdf", [])  # PMCIDs with no renderable PDF (permanent)
    # `have` skips both downloaded AND known-dead ids, so a re-run does NOT
    # re-attempt the permanently-failing ids (HTTP 500 / interstitial) scattered
    # through the relevance list -- the cause of restarts stalling on dead PDFs.
    have = {e["pmcid"] for e in manifest["pdfs"]} | set(manifest["no_pdf"])

    # Reconcile: adopt any PMC*.pdf already on disk but missing from the manifest
    # (recovers PDFs from a run killed before its last flush, so they are never
    # re-downloaded). Filenames are written as PMC<id>.pdf by fetch_pdf().
    adopted = 0
    for p in sorted(out_dir.glob("PMC*.pdf")):
        pmcid = p.stem[3:]  # strip "PMC"
        if pmcid and pmcid not in have:
            manifest["pdfs"].append({"pmcid": pmcid, "pdf": p.name})
            have.add(pmcid)
            adopted += 1
    if adopted:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  reconciled {adopted} orphan PDF(s) into manifest")

    got, skipped = 0, {"no_pdf": 0, "error": 0, "dup": 0}
    retstart = 0
    since_flush = 0
    while got < n:
        ids = esearch_pmc(query, page, retstart=retstart)  # next page
        if not ids:
            print(f"  esearch exhausted at retstart={retstart}")
            break
        retstart += len(ids)
        for pmcid in ids:
            if got >= n:
                break
            if pmcid in have:
                skipped["dup"] += 1
                continue
            have.add(pmcid)  # don't re-attempt within this run
            try:
                pdf = fetch_pdf(pmcid, out_dir)
                time.sleep(sleep)
                if pdf is None:
                    skipped["no_pdf"] += 1
                    manifest["no_pdf"].append(pmcid)  # interstitial = permanent
                else:
                    manifest["pdfs"].append({"pmcid": pmcid, "pdf": pdf.name})
                    got += 1
                    print(f"  [{got}/{n}] PMC{pmcid}  {pdf.stat().st_size // 1024} KB")
                since_flush += 1                       # success OR blacklist mutated
            except urllib.error.HTTPError as exc:
                # 404/410/403/500 = no renderable PDF -> blacklist so future runs
                # skip it instead of re-failing the same dead id every restart.
                skipped["no_pdf"] += 1
                manifest["no_pdf"].append(pmcid)
                since_flush += 1
            except Exception as exc:
                # transient (timeout / DNS getaddrinfo) -- do NOT blacklist; a
                # later run retries it.
                skipped["error"] += 1
                print(f"  skip PMC{pmcid}: {type(exc).__name__}: {exc}")
            # Flush every ``flush_every`` manifest mutations (downloads AND
            # blacklist entries) so a failure-heavy run killed before its next
            # success still persists the dead-id blacklist it accumulated.
            if since_flush >= flush_every:
                manifest_path.write_text(json.dumps(manifest, indent=2))
                since_flush = 0
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nacquired {got} new PDFs (total {len(manifest['pdfs'])}); skipped {skipped}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--source", choices=["pmc", "unpaywall", "biorxiv"], default="pmc")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--email", default="research@example.org")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else base / f"corpus_{args.source}"
    print(f"Acquiring up to {args.n} {args.source} KM PDFs -> {out}")
    if args.source == "pmc":
        acquire(args.query, args.n, out)
    elif args.source == "unpaywall":
        acquire_unpaywall(args.query, args.n, out, email=args.email)
    elif args.source == "biorxiv":
        acquire_biorxiv(args.query, args.n, out)
