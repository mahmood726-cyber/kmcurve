"""Network-free tests for the fusion cross-match scan (fusion_crossmatch.py):
the NCT regex and the ctgov KM-curve detector (parse split from fetch)."""
import fusion_crossmatch as X


def test_nct_regex():
    txt = "Registered at ClinicalTrials.gov NCT01524783 and also NCT00867113; not NCTxyz."
    assert set(X._NCT.findall(txt)) == {"NCT01524783", "NCT00867113"}


def test_km_detector_finds_curve_and_hr():
    """A RADIANT-4-shaped study: a survival outcome with many timepoint classes
    + a hazard-ratio analysis is detected as a posted KM curve with an HR."""
    study = {
        "hasResults": True,
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
            {"title": "Probability of Participants Remaining Event-Free (PFS)",
             "paramType": "NUMBER",
             "classes": [{} for _ in range(11)],
             "analyses": [{"paramType": "Hazard Ratio (HR)", "paramValue": "0.48",
                           "ciLowerLimit": "0.35", "ciUpperLimit": "0.67"}]},
            {"title": "Overall Response Rate", "paramType": "NUMBER", "classes": [{}]},
        ]}},
    }
    info = X._km_from_study(study, "NCT01524783")
    assert info["has_results"] is True
    assert info["km"]["n_timepoints"] == 11
    assert info["km"]["hr"]["value"] == "0.48"


def test_km_detector_rejects_non_survival_and_no_results():
    # many timepoints but not a survival title -> not a KM curve
    study = {"hasResults": True, "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
        {"title": "Change From Baseline in Chromogranin A", "classes": [{} for _ in range(29)]}]}}}
    assert X._km_from_study(study, "NCT1")["km"] is None
    # no posted results
    assert X._km_from_study({"hasResults": False}, "NCT2")["has_results"] is False


def test_km_detector_needs_three_timepoints():
    study = {"hasResults": True, "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
        {"title": "Overall Survival", "classes": [{}, {}]}]}}}  # only 2 timepoints
    assert X._km_from_study(study, "NCT3")["km"] is None
