"""Network-free tests for the forward fusion finder (fusion_forward_finder.py).
Monkeypatches every network/IO touch so the test pins the selection logic:
keep 2-arm, endpoint-matched, strongly-separated curves; mark a pair READY only
when it is OA AND its figure extracts."""
import json

import fusion_forward_finder as F
import fusion_crossmatch as X


def _study(title, n_groups, n_tp, hr, *, desc="time to death due to any cause"):
    classes = [{"title": f"Month {m}"} for m in range(n_tp)]
    analyses = ([{"paramType": "Hazard Ratio (HR)", "paramValue": hr[0],
                  "ciLowerLimit": hr[1], "ciUpperLimit": hr[2]}] if hr else [])
    return {"hasResults": True, "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
        {"title": title, "paramType": "NUMBER", "description": desc, "timeFrame": "",
         "classes": classes, "groups": [{} for _ in range(n_groups)], "analyses": analyses}]}}}


def test_keeps_only_strong_two_arm_matched(monkeypatch, tmp_path):
    studies = {
        "NCT_STRONG": _study("Overall Survival", 2, 4, ("0.45", "0.35", "0.58")),   # strong, 2-arm -> keep
        "NCT_FLAT":   _study("Overall Survival", 2, 4, ("0.92", "0.80", "1.05")),   # flat -> drop (sep<0.4)
        "NCT_3ARM":   _study("Overall Survival", 3, 4, ("0.40", "0.30", "0.55")),   # strong but 3-arm -> drop
        "NCT_NOHR":   _study("Overall Survival", 2, 4, None),                       # no HR -> drop
    }
    monkeypatch.setattr(F, "search_ncts", lambda q, m, page_size=100: list(studies))
    monkeypatch.setattr(X, "fetch_study", lambda nct, sleep=0.2: studies[nct])
    # OA + figure resolution, deterministic per NCT
    monkeypatch.setattr(F, "result_pmid", lambda nct: "PMID_" + nct)
    monkeypatch.setattr(F, "pmids_to_pmc",
                        lambda pmids: {p: ("PMC999" if p == "PMID_NCT_STRONG" else None) for p in pmids})
    import acquire_corpus
    monkeypatch.setattr(acquire_corpus, "fetch_pdf", lambda pmcid, dest: dest / f"PMC{pmcid}.pdf")
    monkeypatch.setattr(X, "figure_extractable", lambda pdf: True)
    monkeypatch.setattr(F, "CACHE", tmp_path / "c.json")

    out = F.run("q", max_ncts=10, min_sep=0.4, corpus_dir=tmp_path, verify_figures=True)
    assert out["n_strong_2arm"] == 1                       # only NCT_STRONG survives the filter
    assert out["strong"][0]["nct"] == "NCT_STRONG"
    assert out["n_ready"] == 1 and out["ready"][0]["oa_pmcid"] == "PMC999"


def test_strong_but_not_extractable_is_not_ready(monkeypatch, tmp_path):
    studies = {"NCT_STRONG": _study("Overall Survival", 2, 4, ("0.45", "0.35", "0.58"))}
    monkeypatch.setattr(F, "search_ncts", lambda q, m, page_size=100: list(studies))
    monkeypatch.setattr(X, "fetch_study", lambda nct, sleep=0.2: studies[nct])
    monkeypatch.setattr(F, "result_pmid", lambda nct: "PMID1")
    monkeypatch.setattr(F, "pmids_to_pmc", lambda pmids: {"PMID1": "PMC42"})
    import acquire_corpus
    monkeypatch.setattr(acquire_corpus, "fetch_pdf", lambda pmcid, dest: dest / f"PMC{pmcid}.pdf")
    monkeypatch.setattr(X, "figure_extractable", lambda pdf: False)   # OA but no extractable figure
    monkeypatch.setattr(F, "CACHE", tmp_path / "c.json")

    out = F.run("q", max_ncts=10, min_sep=0.4, corpus_dir=tmp_path, verify_figures=True)
    assert out["n_strong_2arm"] == 1 and out["n_strong_oa_primary"] == 1
    assert out["n_ready"] == 0                              # OA but figure didn't extract -> not ready
