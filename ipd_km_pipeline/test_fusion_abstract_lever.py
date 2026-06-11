"""Network-free tests for the abstract-event-count lever (fusion_abstract_lever.py).
Pins the percentage extraction + arm-matching + skip logic; no PubMed calls."""
import pytest

import fusion_abstract_lever as L


pytestmark = pytest.mark.skipif(L.AE is None,
                                reason="registry-ipd abstract_events not importable")


def test_non_primary_pubtype_regex():
    assert L._NON_PRIMARY.search("Comment")
    assert L._NON_PRIMARY.search("Published Erratum")
    assert L._NON_PRIMARY.search("Letter")
    assert not L._NON_PRIMARY.search("Randomized Controlled Trial")
    assert not L._NON_PRIMARY.search("Journal Article")


def test_event_governor_vs_survivor_governor():
    # EVENT phrasing -> event fraction = p
    ev = "Progression or death occurred in 25% of the combination group versus 62%."
    assert sorted(L.extract_event_fractions(ev, endpoint="PFS")) == [0.25, 0.62]
    # SURVIVOR phrasing -> event fraction = 1 - p (the CheckMate-066 trap); the 2nd
    # % (42.1%) inherits 'survivor' from the 1st via the "compared with" comparator
    surv = ("the overall rate of survival was 72.9% in the nivolumab group, as "
            "compared with 42.1% in the dacarbazine group (hazard ratio for death, 0.42)")
    fr = sorted(round(x, 3) for x in L.extract_event_fractions(surv, endpoint="OS"))
    assert fr == [0.271, 0.579]      # survivors converted to event fractions


def test_extract_rejects_adverse_and_enrolment():
    adverse = "Grade 3 adverse events occurred in 40% versus 35% of patients."
    assert L.extract_event_fractions(adverse, endpoint=None) == []
    enrol = "Among randomized patients, 55% versus 52% were female."
    assert L.extract_event_fractions(enrol, endpoint=None) == []


def test_events_for_arms_percentage_matches_by_curve_fraction():
    abstract = ("Disease recurrence or death occurred in 20% in the experimental group "
                "versus 50% in the control group.")
    # arm0 has the HIGHER survival (fewer events) -> should get the SMALLER percentage
    arms = [
        {"N": 100, "km_points": [{"t": 12, "S": 0.9}, {"t": 24, "S": 0.8}]},   # implied ~0.2
        {"N": 100, "km_points": [{"t": 12, "S": 0.7}, {"t": 24, "S": 0.5}]},   # implied ~0.5
    ]
    events, source, confidence = L._events_for_arms(abstract, "DFS", arms)
    assert source == "abstract_percentage" and confidence == "low"   # percentage = low confidence
    assert events == [20, 50]                         # 20% -> low-event arm0, 50% -> arm1


def test_events_for_arms_rejects_near_equal_fractions():
    # strongly-separated trials can't have ~equal per-arm event fractions; reject
    # (the NCT00699816/[108,106]-inverts-the-HR false positive)
    abstract = ("Disease recurrence or death occurred in 48% in the experimental group "
                "versus 50% in the control group.")
    arms = [{"N": 100, "km_points": [{"t": 12, "S": 0.55}]},
            {"N": 100, "km_points": [{"t": 12, "S": 0.50}]}]
    events, source, confidence = L._events_for_arms(abstract, "DFS", arms)
    assert events is None and source == "percentage_rejected_near_equal"


def test_events_for_arms_none_when_no_fraction():
    abstract = "The hazard ratio for overall survival was 0.62 (95% CI 0.50-0.77)."
    events, source, confidence = L._events_for_arms(abstract, "OS",
                                        [{"N": 100, "km_points": [{"t": 12, "S": 0.8}]},
                                         {"N": 100, "km_points": [{"t": 12, "S": 0.6}]}])
    assert events is None and source == "none"
