"""Network-free regression for the endpoint-matched PALOMA-3 OS fusion
(fusion_paloma3.py). Monkeypatches the ctgov fetch so the test pins the
reconstruction + endpoint-matched-HR assembly, not the live API.

Locks in the headline: against the CORRECT OS HR 0.814 (not the PFS 0.42),
registry-only Guyot falls OUTSIDE the posted CI while events-informed FUSION
lands INSIDE it -- the whole point of fixing the endpoint linkage."""
import json
from pathlib import Path

import pytest

import fusion_paloma3 as P
from fusion_real_trial import run_trial


# PALOMA-3 OS landmark survival (ctgov-posted KM estimate, year 1/2/3), N per arm.
_COHORT = {
    "nct_id": "NCT01942135", "condition": "Metastatic Breast Cancer",
    "time_unit": "months",
    "arms": [
        {"arm_id": "OG000", "label": "Palbociclib + Fulvestrant", "role": "experimental",
         "N": 347, "follow_up_max": 36.0,
         "km_points": [{"t": 12.0, "S": 0.855}, {"t": 24.0, "S": 0.653}, {"t": 36.0, "S": 0.496}]},
        {"arm_id": "OG001", "label": "Placebo + Fulvestrant", "role": "comparator",
         "N": 174, "follow_up_max": 36.0,
         "km_points": [{"t": 12.0, "S": 0.848}, {"t": 24.0, "S": 0.573}, {"t": 36.0, "S": 0.408}]},
    ],
}
# ctgov: OS deaths per arm + the endpoint-matched OS hazard ratio (NOT PFS 0.42).
_DEATHS = {"Palbociclib + Fulvestrant": 201, "Placebo + Fulvestrant": 109}
_OS_HR = {"value": "0.814", "ci": ["0.644", "1.029"], "endpoint": "os"}


def test_paloma3_os_fusion_inside_ci_registry_outside(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_fetch_os", lambda nct: (_DEATHS, _OS_HR))
    cohort = tmp_path / "NCT01942135.json"
    cohort.write_text(json.dumps(_COHORT))

    trial = P.build_trial(cohort)
    assert trial["hr"]["value"] == 0.814                 # OS HR, not the PFS 0.422
    assert trial["arms"][0]["total_events"] == 201       # ctgov OS deaths wired in
    assert trial["arms"][1]["total_events"] == 109

    tmp = tmp_path / "trial.json"
    tmp.write_text(json.dumps(trial))
    res = run_trial(Path(tmp))

    lo, hi = res["posted_ci"]
    fusion_inside = lo <= res["fusion_hr"] <= hi
    registry_inside = lo <= res["registry_only_hr"] <= hi
    assert fusion_inside and not registry_inside          # the headline of the fix
    assert res["fusion_fold"] < 1.15                      # events-informed within ~15% of truth


_PDF = Path(__file__).resolve().parent / "corpus_pmc" / "PMC9662922.pdf"


def _ocr_deps_ok():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import figure_locator  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _PDF.exists(), reason="PMC9662922.pdf not present (gitignored corpus)")
@pytest.mark.skipif(not _ocr_deps_ok(), reason="rapidocr/figure_locator unavailable")
def test_figure_ocr_nar_fusion_inside_ci(monkeypatch, tmp_path):
    """End-to-end: OCR the at-risk table off the REAL PMC9662922 figure (no ctgov
    events) -> Guyot+NAR fusion lands inside the posted OS CI; registry-only does
    not. The HR ground truth is mocked (no network); the OCR + reconstruction are
    real."""
    monkeypatch.setattr(P, "_fetch_os", lambda nct: (_DEATHS, _OS_HR))
    cohort = tmp_path / "NCT01942135.json"
    cohort.write_text(json.dumps(_COHORT))

    r = P.run_ocr_fusion(cohort, str(_PDF))
    # the at-risk baselines OCR'd off the figure match the registry arm sizes
    bases = sorted(d["n_at_risk"][0] for d in r["nar_arms"].values())
    assert bases == [174, 347]
    lo, hi = float(r["posted_ci"][0]), float(r["posted_ci"][1])
    assert lo <= r["ocr_nar_fusion_hr"] <= hi              # figure-only fusion inside CI
    assert not (lo <= r["registry_only_hr"] <= hi)         # registry-only outside
    assert r["ocr_nar_fusion_fold"] < 1.2


def test_build_trial_fails_closed_without_os_hr(monkeypatch, tmp_path):
    """No endpoint-matched OS HR -> refuse to assert a validation pair."""
    monkeypatch.setattr(P, "_fetch_os", lambda nct: (_DEATHS, None))
    cohort = tmp_path / "c.json"
    cohort.write_text(json.dumps(_COHORT))
    try:
        P.build_trial(cohort)
        assert False, "expected SystemExit on missing OS HR"
    except SystemExit:
        pass
