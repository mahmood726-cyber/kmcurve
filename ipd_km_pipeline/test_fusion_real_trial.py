"""Test the real-trial NAR-fusion demo (fusion_real_trial.py) on RADIANT-4.

Validates the fusion MECHANISM on a real trial with a real posted HR:
registry-only (AACT anchors, no censoring) falls outside the posted CI; adding
the figure's event count (fusion) pulls the reconstructed HR inside it.
Skips cleanly when registry-ipd's harvested trial JSON is absent.
"""
from pathlib import Path

import pytest

import fusion_real_trial as F

TRIAL = Path(r"C:\Projects\registry-ipd\NCT01524783.json")
skip_no_trial = pytest.mark.skipif(not TRIAL.exists(),
                                   reason="registry-ipd RADIANT-4 trial JSON not on disk")


@skip_no_trial
def test_radiant4_fusion_beats_registry_only():
    r = F.run_trial(TRIAL)
    assert r["trial"] == "NCT01524783"
    assert r["posted_hr"] == 0.48
    assert r["n_exp"] == 205 and r["n_ctl"] == 97
    assert r["events_exp"] == 107 and r["events_ctl"] == 77
    # both reconstruct an HR favouring the experimental arm (<1)
    assert 0 < r["fusion_hr"] < 1 and 0 < r["registry_only_hr"] < 1
    # fusion is closer to the posted HR than registry-only (the whole point)
    assert r["fusion_fold"] < r["registry_only_fold"]
    lo, hi = r["posted_ci"]
    # fusion lands inside the posted 95% CI; registry-only does not
    assert lo <= r["fusion_hr"] <= hi
    assert not (lo <= r["registry_only_hr"] <= hi)
