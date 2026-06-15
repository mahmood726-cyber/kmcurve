"""Truth-recovery engine re-export for kmcurve.

The genuine methods engine lives in ``ipd_km_pipeline/guyot.py`` (a
self-contained Guyot IPD-reconstruction implementation, pure numpy/scipy, no
``document``/DOM dependency). We import its pure functions VERBATIM here so the
harness exercises the repo's own code rather than a fork.

Pure functions re-exported:
  - reconstruct_ipd_guyot / reconstruct_arm : Guyot IPD reconstruction
  - ipd_to_arrays                           : records -> (time, event) arrays
  - km_from_ipd                             : KM estimate from IPD
  - logrank_hr                              : Mantel-Haenszel log-rank HR
"""
import os
import sys

# Make ipd_km_pipeline importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.join(os.path.dirname(_HERE), "ipd_km_pipeline")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from guyot import (  # noqa: E402
    IPDRecord,
    reconstruct_ipd_guyot,
    reconstruct_arm,
    ipd_to_arrays,
    km_from_ipd,
    logrank_hr,
)

__all__ = [
    "IPDRecord",
    "reconstruct_ipd_guyot",
    "reconstruct_arm",
    "ipd_to_arrays",
    "km_from_ipd",
    "logrank_hr",
]
