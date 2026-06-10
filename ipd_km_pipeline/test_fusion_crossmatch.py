"""Network-free tests for the fusion cross-match scan (fusion_crossmatch.py):
the NCT regex and the ctgov KM-curve detector, including the subgroup-table /
median-table / single-arm false positives that the naive >=3-classes heuristic
flagged (NCT00363415 was a real subgroup-median false positive)."""
import fusion_crossmatch as X


def _study(measures):
    return {"hasResults": True,
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": measures}}}


def _km_measure(title, class_titles, n_groups, param="NUMBER", hr=None):
    m = {"title": title, "paramType": param,
         "classes": [{"title": t} for t in class_titles],
         "groups": [{} for _ in range(n_groups)]}
    if hr:
        m["analyses"] = [{"paramType": "Hazard Ratio (HR)", "paramValue": hr[0],
                          "ciLowerLimit": hr[1], "ciUpperLimit": hr[2]}]
    return m


def test_nct_regex():
    txt = "Registered NCT01524783 and also NCT00867113; not NCTxyz or NCT123."
    assert set(X._NCT.findall(txt)) == {"NCT01524783", "NCT00867113"}


def test_detects_real_2arm_km_curve_with_hr():
    study = _study([_km_measure(
        "Progression-free Survival", [f"Month {m}" for m in range(0, 33, 3)],
        n_groups=2, hr=("0.48", "0.35", "0.67"))])
    info = X._km_from_study(study, "NCT1")
    assert info["km"]["n_timepoints"] == 11
    assert info["km"]["n_groups"] == 2
    assert info["km"]["hr"]["value"] == "0.48"


def test_rejects_subgroup_median_table():
    """NCT00363415 regression: 15 classes that are SUBGROUPS with median-OS
    values, not KM timepoints -> must NOT be flagged as a posted curve."""
    subgroups = ["Sex: Male", "Sex: Female", "Race: Caucasian", "Age <65", "Age >=65"]
    study = _study([_km_measure("Overall Survival (Subgroups)", subgroups,
                                n_groups=2, param="MEDIAN")])
    assert X._km_from_study(study, "NCT00363415")["km"] is None


def test_rejects_single_arm_curve():
    """NCT00867113 regression: a real KM curve but a single arm -> no 2-arm HR."""
    study = _study([_km_measure("Recurrence-free Survival", ["Month 12", "Month 24", "Month 36"],
                                n_groups=1)])
    assert X._km_from_study(study, "NCT00867113")["km"] is None


def test_rejects_median_paramtype_even_with_time_titles():
    study = _study([_km_measure("Overall Survival", ["Year 1", "Year 2", "Year 3"],
                                n_groups=2, param="MEDIAN")])
    assert X._km_from_study(study, "NCT2")["km"] is None


def test_needs_three_timepoints_and_results():
    study = _study([_km_measure("Overall Survival", ["Month 6", "Month 12"], n_groups=2)])
    assert X._km_from_study(study, "NCT3")["km"] is None
    assert X._km_from_study({"hasResults": False}, "NCT4")["has_results"] is False


def test_is_timepoint_helper():
    assert X._is_timepoint("Month 6") and X._is_timepoint("24 months") and X._is_timepoint("12")
    assert not X._is_timepoint("Sex: Male") and not X._is_timepoint("Race: Caucasian")
