"""Network-free tests for the fusion cross-match scan (fusion_crossmatch.py):
the NCT regex and the ctgov KM-curve detector, including the subgroup-table /
median-table / single-arm false positives that the naive >=3-classes heuristic
flagged (NCT00363415 was a real subgroup-median false positive)."""
import fusion_crossmatch as X


def _study(measures):
    return {"hasResults": True,
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": measures}}}


def _km_measure(title, class_titles, n_groups, param="NUMBER", hr=None,
                description="", time_frame=""):
    m = {"title": title, "paramType": param,
         "description": description, "timeFrame": time_frame,
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


def test_hr_separation():
    import math
    assert X._hr_separation(None) is None
    assert X._hr_separation("n/a") is None
    assert X._hr_separation("0") is None                       # HR<=0 -> undefined
    assert abs(X._hr_separation("1.0") - 0.0) < 1e-9           # no effect -> flat
    assert abs(X._hr_separation("0.814") - abs(math.log(0.814))) < 1e-9
    # a strongly-separated HR (0.40) out-scores a flat one (0.814)
    assert X._hr_separation("0.40") > X._hr_separation("0.814")
    # symmetric in log space: 0.5 and 2.0 are equally separated
    assert abs(X._hr_separation("0.5") - X._hr_separation("2.0")) < 1e-9


def test_endpoint_key_families():
    assert X._endpoint_key("Overall Survival (OS)") == "os"
    assert X._endpoint_key("Progression-Free Survival (PFS)") == "pfs"
    assert X._endpoint_key("Disease-free Survival") == "dfs"
    # PFS/DFS must win over the bare 'survival' that 'os' would otherwise catch
    assert X._endpoint_key("Progression-free survival (overall)") == "pfs"
    # a generic survival-probability curve title is NOT a known endpoint
    assert X._endpoint_key("Survival Probabilities at Year 1, Year 2, and Year 3") is None


def test_does_not_borrow_mismatched_endpoint_hr():
    """PALOMA-3 / NCT01942135 regression: the curve is an OS 'Survival
    Probabilities' measure (no HR); the only posted HR (0.422) is PFS. The old
    'any survival HR in the trial' logic glued the PFS HR onto the OS curve.
    Now the endpoint must match -> the generic OS curve gets NO HR."""
    study = _study([
        _km_measure("Progression-Free Survival (PFS)", ["Median"], n_groups=2,
                    param="MEDIAN", hr=("0.422", "0.318", "0.560")),
        _km_measure("Survival Probabilities at Year 1, Year 2, and Year 3",
                    ["Year 1", "Year 2", "Year 3"], n_groups=2, param="NUMBER"),
    ])
    km = X._km_from_study(study, "NCT01942135")["km"]
    assert km is not None and km["n_timepoints"] == 3       # the curve is still detected
    assert km["hr"] is None and km["hr_endpoint_matched"] is False   # but no mismatched HR


def test_endpoint_key_resolves_generic_title_via_description():
    """A generic 'Survival Probabilities' curve title carries no endpoint by
    title, but its description (overall survival -> death/alive, no progression)
    resolves it to OS; a description naming a competing endpoint wins."""
    # PALOMA-3 OS landmark-survival description (no 'progression' anywhere)
    assert X._endpoint_key(
        "Survival Probabilities at Year 1, Year 2, and Year 3",
        description="probability of survival after randomization based on the "
                    "Kaplan-Meier estimate; censored to last date known to be alive",
        time_frame="From randomization until death (assessed up to 36 months)") == "os"
    # same generic title but a disease-free description must NOT become OS
    assert X._endpoint_key(
        "Survival Probabilities at Year 1, Year 2, and Year 3",
        description="disease-free survival, time to recurrence") == "dfs"
    # no description -> still fails closed
    assert X._endpoint_key("Survival Probabilities at Year 1, Year 2, and Year 3") is None


def test_paloma3_pair_uses_os_hr_not_pfs_via_description():
    """NCT01942135 end state: with descriptions present, the OS survival-probability
    curve resolves to OS and borrows the OS HR (0.81) -- NOT the PFS HR (0.42)."""
    study = _study([
        _km_measure("Progression-Free Survival (PFS)", ["Median"], n_groups=2,
                    param="MEDIAN", hr=("0.422", "0.318", "0.560"),
                    description="time to first documentation of progression or death"),
        _km_measure("Overall Survival (OS)", ["Median"], n_groups=2, param="MEDIAN",
                    hr=("0.814", "0.644", "1.029"),
                    description="time from randomization to date of death due to any cause"),
        _km_measure("Survival Probabilities at Year 1, Year 2, and Year 3",
                    ["Year 1", "Year 2", "Year 3"], n_groups=2, param="NUMBER",
                    description="probability of survival based on the Kaplan-Meier "
                                "estimate; censored to last date known to be alive",
                    time_frame="From randomization until death"),
    ])
    km = X._km_from_study(study, "NCT01942135")["km"]
    assert km["endpoint"] == "os" and km["hr_endpoint_matched"] is True
    assert km["hr"]["value"] == "0.814"          # the OS HR, not the PFS 0.422


def test_borrows_hr_from_sibling_measure_same_endpoint():
    """The legitimate cross-measure case: an OS curve carries no HR, but a sibling
    OS median measure does -> the SAME-endpoint HR is attached."""
    study = _study([
        _km_measure("Overall Survival (OS)", ["Year 1", "Year 2", "Year 3"],
                    n_groups=2, param="NUMBER"),                     # curve, no HR
        _km_measure("Overall Survival (OS)", ["Median"], n_groups=2,
                    param="MEDIAN", hr=("0.81", "0.64", "1.03")),    # HR sibling
    ])
    km = X._km_from_study(study, "NCTos")["km"]
    assert km["hr"]["value"] == "0.81" and km["hr_endpoint_matched"] is True
    assert km["hr"]["endpoint"] == "os"
