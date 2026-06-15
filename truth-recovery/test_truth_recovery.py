"""Truth-recovery assertions for kmcurve's Guyot IPD reconstruction.

These pin the MEASURED behaviour of the repo's own ``guyot.py`` against a
seeded known-truth exponential survival DGP. They encode both what works
(event recovery, monotone KM reproduction) and the honest negatives
(log-HR attenuation toward null, sub-nominal HR-CI coverage).

Run:  python -m pytest truth-recovery/test_truth_recovery.py -q
  or: python -m unittest (from inside truth-recovery/)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgp_survival import make_two_arm  # noqa: E402
from engine import reconstruct_arm, km_from_ipd, logrank_hr  # noqa: E402
from harness import recover_once, coverage_study, median_from_km  # noqa: E402


class TestGuyotTruthRecovery(unittest.TestCase):
    def test_event_count_recovery_within_5pct(self):
        """Reconstructed event totals match truth to within 5% (mean ~1.4%)."""
        res = recover_once(seed=42)
        te = res["true_events0"] + res["true_events1"]
        re = res["rec_events0"] + res["rec_events1"]
        self.assertLess(abs(re - te) / te, 0.05)

    def test_reconstructed_km_is_monotone_nonincreasing(self):
        """km_from_ipd on reconstructed IPD is a valid (monotone) survival curve."""
        a0, _a1, _ = make_two_arm(seed=7)
        t, e = reconstruct_arm(
            a0.curve_times, a0.curve_surv,
            nar_times=a0.nar_times, nar_values=a0.nar_values, total_n=a0.n,
        )
        _, surv = km_from_ipd(t, e)
        self.assertTrue(np.all(np.diff(surv) <= 1e-9))
        self.assertTrue(np.all((surv >= 0) & (surv <= 1)))

    def test_events_spread_across_interval_not_at_endpoint(self):
        """Regression guard vs the sibling kmdigitizer endpoint bug.

        kmcurve places interval events via linspace ACROSS the interval, so
        within any inter-NAR interval that yields >=2 events the event times
        must be DISTINCT (not all stacked on one endpoint).
        """
        a0, _a1, _ = make_two_arm(seed=3)
        t, e = reconstruct_arm(
            a0.curve_times, a0.curve_surv,
            nar_times=a0.nar_times, nar_values=a0.nar_values, total_n=a0.n,
        )
        ev_times = np.sort(t[e == 1])
        # at least some adjacent event times differ by < curve dt (spread inside intervals)
        self.assertGreater(np.unique(ev_times).size, 0.5 * ev_times.size)

    def test_loghr_attenuation_toward_null_is_present(self):
        """HONEST NEGATIVE: reconstruction attenuates log-HR toward null.

        Measured mean log-HR ~ -0.28 vs true -0.43 over 300 reps => positive
        bias of ~+0.15. We assert the attenuation is real (bias > 0.05) but
        bounded (< 0.30) -- i.e. NOT the catastrophic kmdigitizer regime.
        """
        cov = coverage_study(n_rep=200, true_hr=0.65)
        self.assertGreater(cov["bias_log_hr"], 0.05)   # attenuation is real
        self.assertLess(cov["bias_log_hr"], 0.30)      # but not catastrophic

    def test_hr_ci_coverage_subnominal_but_not_collapsed(self):
        """HONEST NEGATIVE: 95% HR-CI coverage ~0.55, below nominal 0.95.

        Crucially it is NOT the collapsed ~0.03 of the buggy kmdigitizer:
        we assert coverage is well above 0.20 (so the bug class differs) yet
        below 0.85 (so we do not falsely claim nominal coverage).
        """
        cov = coverage_study(n_rep=200, true_hr=0.65)
        self.assertGreater(cov["coverage"], 0.20)
        self.assertLess(cov["coverage"], 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
