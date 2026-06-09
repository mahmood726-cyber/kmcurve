#!/usr/bin/env python3
"""
Benchmark harness: end-to-end accuracy over a ground-truth trial corpus
=======================================================================

Phase-0 foundation for scaling KMcurve to "best in world, hundreds of PDFs":
you cannot claim an improvement (or "best") without measuring HR accuracy
against published values across many trials. This runner does exactly that and
-- just as importantly -- produces a FAILURE TAXONOMY so feature work is driven
by what actually breaks rather than by guesswork.

For each trial/panel in ``benchmark/ground_truth.json`` it:
  1. runs the full vector pipeline (pdf_to_ipd),
  2. estimates the hazard ratio (intensive vs standard) via log-rank,
  3. compares to the VERIFIED published HR / 95% CI,
  4. classifies the outcome and, on failure, the reason.

Aggregates: n panels, % within published CI, median |relative HR error|,
HR RMSE, and a failure-reason histogram. Designed to scale to hundreds of
trials -- per-trial exceptions are caught and bucketed, never fatal.

Ground-truth HRs MUST be transcribed from a cited source, never from memory
(see ground_truth.json 'provenance').

Run:  python ipd_km_pipeline/benchmark_harness.py [--json out.json]
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

import vector_km as V
from guyot import logrank_hr

# failure taxonomy ----------------------------------------------------------- #
OK = "within_ci"
OUTSIDE_CI = "outside_ci"
F_NO_PANEL = "panel_not_detected"
F_LOW_CALIB = "calibration_failed"
F_FEW_ARMS = "fewer_than_two_arms"
F_NO_IDENTITY = "arm_identity_unresolved"
F_HR_NAN = "hr_not_computable"
F_EXCEPTION = "exception"


@dataclass
class PanelEval:
    trial_id: str
    panel_index: int
    outcome: str
    hr_published: float
    ci: tuple
    hr_recon: Optional[float] = None
    rel_err: Optional[float] = None
    in_ci: Optional[bool] = None
    separation_confidence: Optional[float] = None
    status: str = ""
    detail: str = ""


def _resolve_pdf(rel: str) -> Optional[Path]:
    try:
        from project_paths import repo_path

        p = repo_path(*Path(rel).parts)
    except Exception:
        p = Path(rel)
    return p if p.exists() else None


def evaluate_trial(trial: dict) -> List[PanelEval]:
    pdf = _resolve_pdf(trial["pdf"])
    gt_panels = {p["index"]: p for p in trial["panels"]}
    evals: List[PanelEval] = []

    if pdf is None:
        for idx, gp in gt_panels.items():
            evals.append(PanelEval(trial["trial_id"], idx, gp["outcome"], gp["hr"],
                                   (gp["ci_lower"], gp["ci_upper"]),
                                   status=F_EXCEPTION, detail="pdf not found"))
        return evals

    itok = trial.get("intensive_token", "intensive").lower()
    stok = trial.get("standard_token", "standard").lower()

    try:
        km_panels = {pr.panel.index: pr for pr in
                     V.extract_km_from_pdf(str(pdf), trial["page_index"], monotone="increasing")}
        ipd_panels = {pi.panel_index: pi for pi in V.pdf_to_ipd(str(pdf), trial["page_index"])}
    except Exception as exc:  # whole-page failure -> bucket every panel
        for idx, gp in gt_panels.items():
            evals.append(PanelEval(trial["trial_id"], idx, gp["outcome"], gp["hr"],
                                   (gp["ci_lower"], gp["ci_upper"]),
                                   status=F_EXCEPTION, detail=f"{type(exc).__name__}: {exc}"))
        return evals

    for idx, gp in gt_panels.items():
        ev = PanelEval(trial["trial_id"], idx, gp["outcome"], gp["hr"],
                       (gp["ci_lower"], gp["ci_upper"]))
        kp = km_panels.get(idx)
        ev.separation_confidence = kp.separation_confidence if kp else None
        pi = ipd_panels.get(idx)
        try:
            if pi is None or kp is None:
                ev.status = F_NO_PANEL
            elif kp.x_fit.r2 < 0.999 or kp.y_fit.r2 < 0.999:
                ev.status = F_LOW_CALIB
                ev.detail = f"xR2={kp.x_fit.r2:.4f} yR2={kp.y_fit.r2:.4f}"
            elif len([a for a in pi.arms if a.time.size]) < 2:
                ev.status = F_FEW_ARMS
            else:
                inten = next((a for a in pi.arms if itok in a.label.lower()), None)
                std = next((a for a in pi.arms if stok in a.label.lower()), None)
                if inten is None or std is None:
                    ev.status = F_NO_IDENTITY
                    ev.detail = f"labels={[a.label for a in pi.arms]}"
                else:
                    hr = logrank_hr(inten.time, inten.event, std.time, std.event)["hr"]
                    if not np.isfinite(hr):
                        ev.status = F_HR_NAN
                    else:
                        ev.hr_recon = float(hr)
                        ev.rel_err = abs(hr - gp["hr"]) / gp["hr"]
                        ev.in_ci = gp["ci_lower"] <= hr <= gp["ci_upper"]
                        ev.status = OK if ev.in_ci else OUTSIDE_CI
        except Exception as exc:
            ev.status = F_EXCEPTION
            ev.detail = f"{type(exc).__name__}: {exc}"
        evals.append(ev)
    return evals


def run(registry_path: Optional[str] = None) -> dict:
    if registry_path is None:
        registry_path = str(Path(__file__).resolve().parent / "benchmark" / "ground_truth.json")
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    trials = reg["trials"]

    all_evals: List[PanelEval] = []
    for tr in trials:
        all_evals.extend(evaluate_trial(tr))

    scored = [e for e in all_evals if e.hr_recon is not None]
    rel_errs = [e.rel_err for e in scored]
    in_ci = [e for e in scored if e.in_ci]

    taxonomy: dict = {}
    for e in all_evals:
        taxonomy[e.status] = taxonomy.get(e.status, 0) + 1

    report = {
        "n_trials": len(trials),
        "n_panels": len(all_evals),
        "n_scored": len(scored),
        "pct_within_ci": (100.0 * len(in_ci) / len(scored)) if scored else 0.0,
        "median_rel_hr_err": (statistics.median(rel_errs) if rel_errs else None),
        "hr_rmse": (float(np.sqrt(np.mean([(e.hr_recon - e.hr_published) ** 2 for e in scored])))
                    if scored else None),
        "failure_taxonomy": taxonomy,
        "panels": [asdict(e) for e in all_evals],
    }

    # ---- print ----
    print("=" * 78)
    print("KMcurve benchmark harness -- HR vs published ground truth")
    print("=" * 78)
    print(f"{'trial':<10}{'panel':<6}{'outcome':<34}{'HR pub':>7}{'HR rec':>8}{'relerr':>8}{'conf':>6}  status")
    print("-" * 78)
    for e in all_evals:
        hr_r = f"{e.hr_recon:.2f}" if e.hr_recon is not None else "  -"
        re_ = f"{e.rel_err*100:.1f}%" if e.rel_err is not None else "  -"
        cf = f"{e.separation_confidence:.2f}" if e.separation_confidence is not None else " -"
        print(f"{e.trial_id:<10}{e.panel_index:<6}{e.outcome[:33]:<34}{e.hr_published:>7.2f}{hr_r:>8}{re_:>8}{cf:>6}  {e.status}")
    print("-" * 78)
    mre = report["median_rel_hr_err"]
    print(f"panels={report['n_panels']}  scored={report['n_scored']}  "
          f"within-CI={report['pct_within_ci']:.0f}%  "
          f"median|relHRerr|={mre*100:.1f}%" if mre is not None else
          f"panels={report['n_panels']}  scored={report['n_scored']}")
    print(f"failure taxonomy: {report['failure_taxonomy']}")
    print("=" * 78)
    return report


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
    rep = run()
    if out:
        Path(out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"wrote {out}")
