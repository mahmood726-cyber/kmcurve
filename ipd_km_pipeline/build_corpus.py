#!/usr/bin/env python3
"""Build the confidence-gated labelled corpus from the double-read workflow
output (roadmap lever 2).

Consumes the figure-autolabel workflow result (``vlm_labels.json``: a list of
{id, read1, read2}) and emits ``labels.jsonl`` -- one corpus record per figure
with the merged label, a double-read confidence, and a ``needs_human_review``
flag (structural disagreement). Reports the confidence-gated accept rate: the
fraction auto-accepted vs the minority a human must check.

Usage: python build_corpus.py [vlm_labels.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from autolabel import corpus_record

ROOT = Path("artifacts/vlm_figures")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vlm_labels.json"
    payload = json.loads(src.read_text())
    results = payload["results"] if isinstance(payload, dict) else payload

    records = []
    for r in results:
        rec = corpus_record({"id": r["id"], "pdf": r.get("png"), "page_index": None},
                            r["read1"], r["read2"])
        records.append(rec)

    out = ROOT / "labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n = len(records)
    accepted = [r for r in records if not r["needs_human_review"]]
    review = [r for r in records if r["needs_human_review"]]
    confs = [r["confidence"] for r in records]
    # which structural fields most often drive review
    fail_fields = Counter()
    for r in review:
        for k in ("is_km_figure", "n_panels", "n_arms", "y_kind", "x_unit"):
            if not r["agreement"].get(k):
                fail_fields[k] += 1
    npanels = Counter(r["label"].get("n_panels") for r in records)

    print(f"corpus: {n} figures")
    print(f"  auto-accepted (no review):   {len(accepted)}/{n} "
          f"({100*len(accepted)/n:.0f}%)")
    print(f"  needs human review:          {len(review)}/{n}")
    print(f"  mean double-read confidence: {sum(confs)/n:.3f}")
    print(f"  n_panels distribution:       {dict(sorted(npanels.items(), key=lambda x:(x[0] is None, x[0])))}")
    if fail_fields:
        print(f"  review driven by:            {dict(fail_fields)}")
    print(f"-> {out}")

    (ROOT / "corpus_summary.json").write_text(json.dumps({
        "n": n, "auto_accepted": len(accepted), "needs_review": len(review),
        "mean_confidence": round(sum(confs)/n, 4),
        "review_fields": dict(fail_fields),
    }, indent=2))


if __name__ == "__main__":
    main()
