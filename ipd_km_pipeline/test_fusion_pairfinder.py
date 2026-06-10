"""Network-free test for the targeted fusion pair-finder (fusion_pairfinder.py).
Monkeypatches the ctgov/NCBI lookups so the run() aggregation + OA classification
are tested without hitting the network."""
import json

import fusion_pairfinder as P


def test_run_classifies_oa_primary(tmp_path, monkeypatch):
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps({"all_with_hr": [
        {"nct": "NCT00000001", "condition": "A", "registry_HR": 0.5},   # OA primary
        {"nct": "NCT00000002", "condition": "B", "registry_HR": 1.2},   # paywalled primary
        {"nct": "NCT00000003", "condition": "C", "registry_HR": 0.8},   # no linked pmid
    ]}))
    monkeypatch.setattr(P, "CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(P, "result_pmid",
                        lambda nct: {"NCT00000001": "111", "NCT00000002": "222"}.get(nct))
    # only PMID 111 is open-access in PMC
    monkeypatch.setattr(P, "pmids_to_pmc",
                        lambda pmids: {"111": "PMC111", "222": None})

    out = P.run(gallery)
    assert out["n_trials"] == 3
    assert out["n_with_primary_pmid"] == 2          # 111 + 222
    assert out["n_oa_primary"] == 1                 # only 111 is OA
    assert out["ready"][0]["nct"] == "NCT00000001"
    assert out["ready"][0]["oa_pmcid"] == "PMC111"


def test_run_no_oa_primary(tmp_path, monkeypatch):
    """The real-world case: curve-posting trials with paywalled primaries -> 0."""
    gallery = tmp_path / "g.json"
    gallery.write_text(json.dumps({"all_with_hr": [{"nct": "NCT9", "registry_HR": 0.6}]}))
    monkeypatch.setattr(P, "CACHE", tmp_path / "c.json")
    monkeypatch.setattr(P, "result_pmid", lambda nct: "999")
    monkeypatch.setattr(P, "pmids_to_pmc", lambda pmids: {"999": None})  # not OA
    out = P.run(gallery)
    assert out["n_oa_primary"] == 0 and out["ready"] == []
