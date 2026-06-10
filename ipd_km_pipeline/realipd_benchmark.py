#!/usr/bin/env python3
"""
True-IPD external-accuracy benchmark for the raster digitization pipeline.
=========================================================================

The corpus benchmark (``raster_benchmark.py``) validates reconstructed *event
counts* against a figure's own at-risk table -- figure-INTERNAL ground truth.
What it cannot answer is the question that matters for meta-analysis: **how
close is the reconstructed hazard ratio to the TRUE HR computed from real
patient-level data?** There is no published-figure corpus with attached IPD,
so we borrow the sibling project's gold standard.

`C:\\Projects\\registry-ipd` ships ~40 open true-IPD datasets (Rdatasets /
KMsurv / asaur / cBioPortal-TCGA) with per-dataset arm/time/status configs.
registry-ipd's own ``HEADTOHEAD.md`` measures *registry-native* reconstruction
and a *simulated* digitization (coordinates sampled off the true curve with
Gaussian pixel noise). This benchmark closes the loop with the REAL thing:

    true IPD  ->  render the true KM curve to an actual raster image
              ->  run THIS pipeline (dark-cloud extract -> arm-separate -> Guyot)
              ->  reconstruct pseudo-IPD  ->  Cox/log-rank HR  ->  vs the TRUE HR

Both the truth and the reconstruction use the SAME estimator (``guyot.logrank_hr``,
Mantel-Haenszel O-E/V), so the fold-error isolates *pipeline-induced* error, not
an estimator difference. Calibration is supplied EXACTLY (we render the axes, so
the pixel<->value map is known) -- this isolates curve-extraction + arm-separation
+ Guyot reconstruction, which is precisely what registry-ipd's ``digitize()``
noise model approximates. It does NOT exercise the OCR auto-calibration (a
separate, separately-measured failure mode). Honest scope: clean single-style
rendered curves with known calibration are an *upper bound* on real-figure
accuracy, and the direct real-pipeline analogue of the simulated head-to-head.

Run:  python realipd_benchmark.py [--registry C:\\Projects\\registry-ipd]
                                  [--datasets gbsg,veteran,...] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import cv2

import raster_km as R
from guyot import logrank_hr

# --- dataset configs (mirrors registry-ipd/validate/goldstandard.js CONFIGS) ---
# Only the classic single-CSV survival datasets are mirrored here; each maps a
# CSV to a 2-arm time-to-event contrast. eventVal (optional): status value that
# counts as an event (default: status in {1, "1"} is an event).


@dataclass
class DSConfig:
    ds: str
    label: str
    time: str
    status: str
    arm: str
    exp: str
    ctl: str
    event_val: Optional[str] = None  # status value meaning "event" (else !=0)


CONFIGS: List[DSConfig] = [
    # classic single-CSV datasets -- field mappings ported verbatim from
    # registry-ipd/validate/goldstandard.js CONFIGS (the verified source).
    DSConfig("gbsg", "GBSG breast cancer (RFS, hormonal Rx)", "rfstime", "status", "hormon", "1", "0"),
    DSConfig("veteran", "Veterans lung cancer (OS, treatment)", "time", "status", "trt", "2", "1"),
    DSConfig("rotterdam", "Rotterdam breast cancer (OS, hormonal Rx)", "dtime", "death", "hormon", "1", "0"),
    DSConfig("pbc", "PBC (OS, D-penicillamine vs placebo)", "time", "status", "trt", "2", "1", event_val="2"),
    DSConfig("diabetic", "Diabetic retinopathy (vision loss, laser)", "time", "status", "trt", "1", "0"),
    DSConfig("cancer", "NCCTG lung cancer (OS, by sex)", "time", "status", "sex", "2", "1", event_val="2"),
    DSConfig("nwtco", "NWTSG Wilms tumour (relapse, histology)", "edrel", "rel", "histol", "2", "1"),
    DSConfig("bmt", "Bone marrow transplant (DFS, risk group)", "t2", "d3", "group", "3", "1"),
    DSConfig("tongue", "Tongue cancer (death, ploidy)", "time", "delta", "type", "2", "1"),
    DSConfig("kidtran", "Kidney transplant (graft survival, by sex)", "time", "delta", "gender", "2", "1"),
    DSConfig("mgus2", "MGUS cohort (death, by sex)", "futime", "death", "sex", "M", "F"),
    DSConfig("flchain", "Free light chain cohort (death, by sex)", "futime", "death", "sex", "M", "F"),
    DSConfig("nafld1", "NAFLD cohort (death, by sex)", "futime", "status", "male", "1", "0"),
    DSConfig("prostateSurvival", "Prostate cancer (cancer death, grade)", "survTime", "status", "grade", "poor", "mode", event_val="1"),
    DSConfig("melanoma", "Melanoma (cancer death, ulceration)", "time", "status", "ulcer", "1", "0", event_val="1"),
    DSConfig("pharmacoSmoking", "Smoking-cessation RCT (relapse)", "ttr", "relapse", "grp", "patchOnly", "combination"),
    DSConfig("gehan", "Gehan leukemia RCT (remission, 6-MP vs control)", "time", "cens", "treat", "6-MP", "control"),
    DSConfig("aidssi", "AIDS cohort (time to AIDS, CCR5 genotype)", "time", "status", "ccr5", "WM", "WW", event_val="1"),
    DSConfig("ebmt3", "EBMT transplant (RFS, T-cell depletion)", "rfstime", "rfsstat", "tcd", "TCD", "No TCD"),
    DSConfig("larynx", "Larynx cancer (death, stage)", "time", "delta", "stage", "3", "1"),
    DSConfig("burn", "Burn infection (time to infection, treatment)", "T3", "D3", "Z1", "1", "0"),
    DSConfig("pneumon", "Infant pneumonia (smoking)", "chldage", "hospital", "smoke", "1", "0"),
    DSConfig("bfeed", "Breastfeeding duration (smoking)", "duration", "delta", "smoke", "1", "0"),
    DSConfig("hepatoCellular", "Hepatocellular carcinoma (OS, vascular invasion)", "OS", "Death", "Vascularinvasion", "1", "0"),
    DSConfig("alloauto", "Leukemia transplant (DFS, allo vs auto)", "time", "delta", "type", "1", "2"),
    DSConfig("retinopathy", "Diabetic retinopathy (laser, vision loss)", "futime", "status", "trt", "1", "0"),
    DSConfig("ebmt1", "EBMT transplant (OS, risk score)", "srv", "srvstat", "score", "High risk", "Low risk"),
]
# TCGA / cBioPortal cohorts (late vs early stage -> strong contrast, true HR 1.6-7.6).
# Uniform config: columns patientId,time,status,sex,stage_group,subtype.
_CBIO = ["acc", "blca", "brca", "cesc", "coadread", "esca", "hnsc", "kirc", "kirp",
         "lihc", "luad", "lusc", "meso", "paad", "sarc", "skcm", "stad", "ucec"]
CONFIGS += [DSConfig(f"cbio_{c}", f"TCGA-{c.upper()} (OS, late vs early stage)",
                     "time", "status", "stage_group", "late", "early") for c in _CBIO]
CONFIG_BY_DS = {c.ds: c for c in CONFIGS}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion (honest at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def _num(x: str) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def load_true_ipd(csv_path: Path, cfg: DSConfig):
    """Parse a gold-standard CSV into (exp_t, exp_e, ctl_t, ctl_e) arrays."""
    rows = csv_path.read_text().strip().splitlines()
    head = [h.strip().strip('"') for h in rows[0].split(",")]
    idx = {h: i for i, h in enumerate(head)}
    for col in (cfg.time, cfg.status, cfg.arm):
        if col not in idx:
            raise KeyError(f"{cfg.ds}: column {col!r} not in {head}")
    arms = {cfg.exp: ([], []), cfg.ctl: ([], [])}
    for line in rows[1:]:
        cells = line.split(",")
        if len(cells) <= max(idx[cfg.time], idx[cfg.status], idx[cfg.arm]):
            continue
        a = cells[idx[cfg.arm]].strip().strip('"')
        if a not in arms:
            continue
        t = _num(cells[idx[cfg.time]])
        s_raw = cells[idx[cfg.status]].strip().strip('"')
        if t is None or t < 0:
            continue
        if cfg.event_val is not None:
            ev = 1 if s_raw == cfg.event_val else 0
        else:
            sv = _num(s_raw)
            ev = 1 if (sv is not None and sv != 0) else 0
        arms[a][0].append(t)
        arms[a][1].append(ev)
    et = np.asarray(arms[cfg.exp][0], float); ee = np.asarray(arms[cfg.exp][1], int)
    ct = np.asarray(arms[cfg.ctl][0], float); ce = np.asarray(arms[cfg.ctl][1], int)
    return et, ee, ct, ce


def km_steps(t: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier step function -> (times incl. 0, survival incl. 1.0)."""
    if t.size == 0:
        return np.array([0.0]), np.array([1.0])
    ev_times = np.unique(t[e == 1])
    times = [0.0]; surv = [1.0]; s = 1.0
    for tt in ev_times:
        n = int(np.sum(t >= tt)); d = int(np.sum((t == tt) & (e == 1)))
        if n > 0:
            s *= (1.0 - d / n)
        times.append(float(tt)); surv.append(s)
    return np.asarray(times), np.asarray(surv)


@dataclass
class Geometry:
    plot: "R.PlotBox"
    x_refs: List[Tuple[float, float]]
    y_refs: List[Tuple[float, float]]
    t_max: float


def render_km(arms_steps: List[Tuple[np.ndarray, np.ndarray]], t_max: float,
              W: int = 900, H: int = 650, margin: int = 70) -> Tuple[np.ndarray, Geometry]:
    """Render KM step curves to a grayscale raster with KNOWN axis geometry.

    Curves are anti-aliased black steps on white inside a known plot rectangle
    (no ticks/text/legend inside the box), so the only dark pixels the pipeline
    sees within the box are the curves -- a clean, calibration-known raster."""
    img = np.full((H, W), 255, np.uint8)
    x0, x1 = margin, W - margin // 2
    y_top, y_bot = margin // 2, H - margin

    def to_px(t, s):
        px = x0 + (t / t_max) * (x1 - x0)
        py = y_bot - s * (y_bot - y_top)
        return int(round(px)), int(round(py))

    for (ts, ss) in arms_steps:
        for i in range(len(ts) - 1):
            x_a, y_a = to_px(ts[i], ss[i])
            x_b, _ = to_px(ts[i + 1], ss[i])
            x_b2, y_b2 = to_px(ts[i + 1], ss[i + 1])
            cv2.line(img, (x_a, y_a), (x_b, y_a), 0, 2, cv2.LINE_AA)   # horizontal hold
            cv2.line(img, (x_b, y_a), (x_b2, y_b2), 0, 2, cv2.LINE_AA)  # vertical drop
        # tail: hold last survival out to t_max
        x_a, y_a = to_px(ts[-1], ss[-1])
        x_e, _ = to_px(t_max, ss[-1])
        cv2.line(img, (x_a, y_a), (x_e, y_a), 0, 2, cv2.LINE_AA)

    geom = Geometry(
        plot=R.PlotBox(x0, y_top, x1, y_bot),
        x_refs=[(float(x0), 0.0), (float(x1), float(t_max))],
        y_refs=[(float(y_bot), 0.0), (float(y_top), 1.0)],
        t_max=t_max,
    )
    return img, geom


def _km_survival_at(grid: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """KM survival sampled at the given time grid (step, right-continuous)."""
    ts, ss = km_steps(t, e)
    out = np.ones_like(grid, float)
    for i, g in enumerate(grid):
        k = np.searchsorted(ts, g, side="right") - 1
        out[i] = ss[max(k, 0)]
    return out


def _extract_arms(img: np.ndarray, geom: Geometry) -> List[np.ndarray]:
    """Replicate the public raster extraction, returning per-arm calibrated
    (time, survival) clouds -- the SAME steps as ``raster_km.raster_to_ipd`` but
    stopping before Guyot, so the benchmark can map each arm to exp/ctl and then
    reconstruct it with that arm's at-risk table (the figure-external structured
    input a real risk table / registry record supplies)."""
    from manual_calibration import axis_from_clicks, calibrate_points
    from vector_km import separate_arms

    xfit = axis_from_clicks(geom.x_refs)
    yfit = axis_from_clicks(geom.y_refs)
    cloud_px = R.column_curve_points(R.dark_curve_cloud(img, geom.plot), n_curves=2)
    if cloud_px.size == 0:
        return []
    t, v = calibrate_points(cloud_px, xfit, yfit, monotone="none")
    gap = max(0.02 * (float(np.ptp(v)) or 1.0), 1e-6)
    arms_xy = separate_arms(np.column_stack([t, v]), n_arms=2, gap=gap)
    out = []
    for ac in arms_xy:
        if ac.size == 0:
            continue
        ac = ac[ac[:, 0].argsort()]
        surv = np.minimum.accumulate(np.clip(ac[:, 1], 0.0, 1.0))
        out.append(np.column_stack([ac[:, 0], surv]))
    return out


def _at_risk_table(t: np.ndarray, t_max: float, k: int = 6):
    """The number-at-risk table a real KM figure prints (k labelled times)."""
    times = np.linspace(0.0, t_max, k)
    counts = np.array([int(np.sum(t >= tt)) for tt in times], float)
    return times, counts


def _hr_from(arm_exp_xy, arm_ctl_xy, n_exp, n_ctl,
             nar_exp=None, nar_ctl=None) -> float:
    """Reconstruct both mapped arms (optionally censoring-informed) -> log-rank HR."""
    from guyot import reconstruct_arm
    et, ee = reconstruct_arm(arm_exp_xy[:, 0], arm_exp_xy[:, 1],
                             nar_times=nar_exp[0] if nar_exp else None,
                             nar_values=nar_exp[1] if nar_exp else None,
                             total_n=n_exp)
    ct, ce = reconstruct_arm(arm_ctl_xy[:, 0], arm_ctl_xy[:, 1],
                             nar_times=nar_ctl[0] if nar_ctl else None,
                             nar_values=nar_ctl[1] if nar_ctl else None,
                             total_n=n_ctl)
    return logrank_hr(et, ee, ct, ce)["hr"]


def _hr_qp(arm_exp_xy, arm_ctl_xy, n_exp, n_ctl, e_exp, e_ctl, t_max) -> float:
    """Reconstruct both mapped arms with the Titman-QP backend -> log-rank HR."""
    from qp_reconstruct import reconstruct_arm_qp
    et, ee = reconstruct_arm_qp(arm_exp_xy[:, 0], arm_exp_xy[:, 1],
                                total_n=n_exp, total_events=e_exp, follow_up_max=t_max)
    ct, ce = reconstruct_arm_qp(arm_ctl_xy[:, 0], arm_ctl_xy[:, 1],
                                total_n=n_ctl, total_events=e_ctl, follow_up_max=t_max)
    if et.size == 0 or ct.size == 0:
        return float("nan")
    return logrank_hr(et, ee, ct, ce)["hr"]


def _fold(recon_hr: float, true_hr: float) -> Optional[float]:
    if (np.isfinite(recon_hr) and np.isfinite(true_hr)
            and recon_hr > 0 and true_hr > 0):
        return max(recon_hr, true_hr) / min(recon_hr, true_hr)
    return None


def recon_and_score(csv_path: Path, cfg: DSConfig) -> dict:
    """Render the true KM, extract via the raster pipeline, score recon HR vs
    true HR -- both curve-only and censoring-informed (with the at-risk table)."""
    et, ee, ct, ce = load_true_ipd(csv_path, cfg)
    n_exp, n_ctl = int(et.size), int(ct.size)
    if n_exp < 5 or n_ctl < 5:
        return {"ds": cfg.ds, "status": "too_small", "n_exp": n_exp, "n_ctl": n_ctl}

    true_hr = logrank_hr(et, ee, ct, ce)["hr"]
    t_max = float(max(et.max(), ct.max()))
    img, geom = render_km([km_steps(et, ee), km_steps(ct, ce)], t_max)

    arms_xy = _extract_arms(img, geom)
    if len(arms_xy) < 2:
        return {"ds": cfg.ds, "label": cfg.label, "status": "extract_fail",
                "n_arms": len(arms_xy), "true_hr": round(true_hr, 4),
                "n_exp": n_exp, "n_ctl": n_ctl}

    # Map extracted arms -> (exp, ctl) by closest KM survival to each true arm.
    grid = np.linspace(0, t_max, 50)
    true_exp_s = _km_survival_at(grid, et, ee)
    true_ctl_s = _km_survival_at(grid, ct, ce)
    ext_s = [np.interp(grid, a[:, 0], a[:, 1], left=1.0) for a in arms_xy[:2]]
    if np.abs(ext_s[0] - true_exp_s).mean() <= np.abs(ext_s[0] - true_ctl_s).mean():
        exp_xy, ctl_xy = arms_xy[0], arms_xy[1]
    else:
        exp_xy, ctl_xy = arms_xy[1], arms_xy[0]

    # Three backends: curve-only Guyot; censoring-informed Guyot (printed risk
    # table); Titman-QP (the "N (events)" cell -> events-informed, ported from
    # registry-ipd). All consume the SAME extracted curve, so differences are
    # purely the reconstruction backend.
    nar_exp = _at_risk_table(et, t_max)
    nar_ctl = _at_risk_table(ct, t_max)
    e_exp, e_ctl = int(ee.sum()), int(ce.sum())
    hr_curve = _hr_from(exp_xy, ctl_xy, n_exp, n_ctl)
    hr_nar = _hr_from(exp_xy, ctl_xy, n_exp, n_ctl, nar_exp, nar_ctl)
    hr_qp = _hr_qp(exp_xy, ctl_xy, n_exp, n_ctl, e_exp, e_ctl, t_max)
    fold_curve = _fold(hr_curve, true_hr)
    fold_nar = _fold(hr_nar, true_hr)
    fold_qp = _fold(hr_qp, true_hr)

    return {
        "ds": cfg.ds, "label": cfg.label, "status": "scored",
        "n_exp": n_exp, "n_ctl": n_ctl,
        "true_hr": round(true_hr, 4),
        "recon_hr_curveonly": round(hr_curve, 4) if np.isfinite(hr_curve) else None,
        "recon_hr_nar": round(hr_nar, 4) if np.isfinite(hr_nar) else None,
        "recon_hr_qp": round(hr_qp, 4) if np.isfinite(hr_qp) else None,
        "fold_curveonly": round(fold_curve, 4) if fold_curve else None,
        "fold_nar": round(fold_nar, 4) if fold_nar else None,
        "fold_qp": round(fold_qp, 4) if fold_qp else None,
        "abs_log_err_nar": round(abs(np.log(fold_nar)), 4) if fold_nar else None,
        "true_events": int(ee.sum() + ce.sum()),
    }


def fusion_and_score(csv_path: Path, cfg: DSConfig, n_anchors: int = 8) -> dict:
    """NAR-FUSION experiment. The two projects are mirror images: registry-ipd
    has the curve EXACTLY (AACT KM-estimate anchors) but no number-at-risk;
    kmcurve recovers the at-risk table (OCR) but its curve is pixel-noisy.
    Fusing them -- registry-exact anchors + kmcurve's OCR'd NAR -- should beat
    either alone. We hold the backend fixed (Guyot) and vary only the INPUTS:

      registry-only : exact KM anchors, NO at-risk table  (registry-ipd today)
      kmcurve-only  : noisy extracted curve + at-risk table  (kmcurve today)
      fusion        : exact KM anchors + at-risk table  (the proposed union)
    """
    et, ee, ct, ce = load_true_ipd(csv_path, cfg)
    n_exp, n_ctl = int(et.size), int(ct.size)
    if n_exp < 5 or n_ctl < 5:
        return {"ds": cfg.ds, "status": "too_small"}
    true_hr = logrank_hr(et, ee, ct, ce)["hr"]
    t_max = float(max(et.max(), ct.max()))

    # kmcurve's NOISY curve: render the true KM and run the real extraction.
    img, geom = render_km([km_steps(et, ee), km_steps(ct, ce)], t_max)
    arms_xy = _extract_arms(img, geom)
    if len(arms_xy) < 2:
        return {"ds": cfg.ds, "label": cfg.label, "status": "extract_fail",
                "true_hr": round(true_hr, 4)}
    grid = np.linspace(0, t_max, 50)
    true_exp_s = _km_survival_at(grid, et, ee)
    true_ctl_s = _km_survival_at(grid, ct, ce)
    ext_s = [np.interp(grid, a[:, 0], a[:, 1], left=1.0) for a in arms_xy[:2]]
    if np.abs(ext_s[0] - true_exp_s).mean() <= np.abs(ext_s[0] - true_ctl_s).mean():
        ext_exp, ext_ctl = arms_xy[0], arms_xy[1]
    else:
        ext_exp, ext_ctl = arms_xy[1], arms_xy[0]

    # registry-exact anchors: the true KM sampled at n_anchors posted timepoints.
    at = np.linspace(0, t_max, n_anchors)
    exact_exp = np.column_stack([at, _km_survival_at(at, et, ee)])
    exact_ctl = np.column_stack([at, _km_survival_at(at, ct, ce)])
    nar_exp = _at_risk_table(et, t_max)
    nar_ctl = _at_risk_table(ct, t_max)

    e_exp, e_ctl = int(ee.sum()), int(ce.sum())
    hr_reg = _hr_from(exact_exp, exact_ctl, n_exp, n_ctl)                  # exact, no NAR
    hr_km = _hr_from(ext_exp, ext_ctl, n_exp, n_ctl, nar_exp, nar_ctl)    # noisy + NAR
    hr_fuse = _hr_from(exact_exp, exact_ctl, n_exp, n_ctl, nar_exp, nar_ctl)  # exact + NAR (Guyot)
    hr_fuse_qp = _hr_qp(exact_exp, exact_ctl, n_exp, n_ctl, e_exp, e_ctl, t_max)  # exact + QP(events)
    return {
        "ds": cfg.ds, "label": cfg.label, "status": "scored",
        "n_exp": n_exp, "n_ctl": n_ctl, "true_hr": round(true_hr, 4),
        "fold_registry_only": _r(_fold(hr_reg, true_hr)),
        "fold_kmcurve_only": _r(_fold(hr_km, true_hr)),
        "fold_fusion": _r(_fold(hr_fuse, true_hr)),
        "fold_fusion_qp": _r(_fold(hr_fuse_qp, true_hr)),
    }


def _r(x):
    return round(x, 4) if x else None


def run_fusion(registry_dir: Path, datasets: Optional[List[str]] = None) -> dict:
    realipd = registry_dir / "realipd"
    want = datasets or [c.ds for c in CONFIGS]
    results = []
    for ds in want:
        cfg = CONFIG_BY_DS.get(ds)
        csv = realipd / f"{ds}.csv"
        if cfg is None or not csv.exists():
            continue
        try:
            results.append(fusion_and_score(csv, cfg))
        except Exception as exc:
            results.append({"ds": ds, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:120]})
    scored = [r for r in results if r.get("status") == "scored"]
    summary = {"n_scored": len(scored)}
    for key in ("fold_registry_only", "fold_kmcurve_only", "fold_fusion", "fold_fusion_qp"):
        st = _stats(scored, key)
        if st:
            summary[key] = st
    # paired: does fusion beat each single-source path on the SAME datasets?
    summary["paired_fusion_vs_registry"] = _paired(scored, "fold_fusion", "fold_registry_only")
    summary["paired_fusion_vs_kmcurve"] = _paired(scored, "fold_fusion", "fold_kmcurve_only")
    return {"summary": summary, "results": results}


def _print_fusion(out: dict) -> None:
    print(f"\n{'dataset':<12}{'trueHR':>9}{'registry-only':>15}{'kmcurve-only':>14}"
          f"{'FUSION':>9}")
    print(f"{'':<12}{'':>9}{'(exact,no NAR)':>15}{'(noisy+NAR)':>14}{'(exact+NAR)':>9}")
    print("-" * 60)
    for r in out["results"]:
        if r.get("status") == "scored":
            print(f"{r['ds']:<12}{r['true_hr']:>9}{str(r['fold_registry_only']):>15}"
                  f"{str(r['fold_kmcurve_only']):>14}{str(r['fold_fusion']):>9}")
        else:
            print(f"{r['ds']:<12}  {r.get('status')}")
    s = out["summary"]
    print("-" * 60)
    print(f"scored {s['n_scored']}")
    for key, lbl in (("fold_registry_only", "registry-only (exact anchors, NO NAR)"),
                     ("fold_kmcurve_only", "kmcurve-only  (noisy curve + NAR)    "),
                     ("fold_fusion", "FUSION        (exact anchors + NAR)  "),
                     ("fold_fusion_qp", "FUSION+QP     (exact anchors + QP)   ")):
        st = s.get(key)
        if st:
            ci = st["within_20pct_ci"]
            print(f"  {lbl}: median {st['median']}  p90 {st['p90']}"
                  f"  within20% {st['within_20pct']}/{st['n']} (95% CI {ci[0]:.0%}-{ci[1]:.0%})")
    for k, lbl in (("paired_fusion_vs_registry", "fusion vs registry-only"),
                   ("paired_fusion_vs_kmcurve", "fusion vs kmcurve-only ")):
        pr = s.get(k)
        if pr:
            print(f"  paired {lbl} (n={pr['n']}): fusion better {pr['a_better']}, "
                  f"other better {pr['b_better']}, median fold ratio {pr['median_ratio_b_over_a']}")


def run(registry_dir: Path, datasets: Optional[List[str]] = None) -> dict:
    realipd = registry_dir / "realipd"
    if not realipd.is_dir():
        raise FileNotFoundError(
            f"registry-ipd gold-standard not found at {realipd}. "
            f"Pass --registry <path to registry-ipd> (re-download cmd in its "
            f"validate/goldstandard.js header).")
    want = datasets or [c.ds for c in CONFIGS]
    results = []
    for ds in want:
        cfg = CONFIG_BY_DS.get(ds)
        if cfg is None:
            results.append({"ds": ds, "status": "no_config"}); continue
        csv = realipd / f"{ds}.csv"
        if not csv.exists():
            results.append({"ds": ds, "status": "csv_missing"}); continue
        try:
            results.append(recon_and_score(csv, cfg))
        except Exception as exc:
            results.append({"ds": ds, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:120]})

    scored = [r for r in results if r.get("status") == "scored" and r.get("fold_nar")]
    summary = {"n_total": len(results), "n_scored": len(scored)}
    if scored:
        summary["censoring_informed"] = _stats(scored, "fold_nar")
        summary["curve_only"] = _stats(scored, "fold_curveonly")
        summary["qp"] = _stats(scored, "fold_qp")
        # paired: does QP beat Guyot+NAR on the SAME datasets?
        summary["paired_qp_vs_nar"] = _paired(scored, "fold_qp", "fold_nar")
    return {"summary": summary, "results": results}


def _stats(scored: list, key: str):
    xs = sorted(r[key] for r in scored if r.get(key))
    if not xs:
        return None
    k = sum(x <= 1.2 for x in xs)
    return {
        "median": round(xs[len(xs) // 2], 4),
        "p90": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 4),
        "within_20pct": k, "n": len(xs),
        "within_20pct_ci": wilson_ci(k, len(xs)),
    }


def _paired(scored: list, key_a: str, key_b: str):
    """On datasets where both backends scored: which wins (lower fold), and the
    median paired fold ratio (b/a). Avoids comparing non-overlapping subsets."""
    pairs = [(r[key_a], r[key_b]) for r in scored if r.get(key_a) and r.get(key_b)]
    if not pairs:
        return None
    a_better = sum(1 for a, b in pairs if a < b)
    b_better = sum(1 for a, b in pairs if b < a)
    ratios = sorted(b / a for a, b in pairs)
    return {"n": len(pairs), "a_better": a_better, "b_better": b_better,
            "median_ratio_b_over_a": round(ratios[len(ratios) // 2], 3)}


def _print_table(out: dict) -> None:
    print(f"\n{'dataset':<12}{'n(e/c)':>11}{'trueHR':>9}{'fold(curve)':>12}"
          f"{'fold(+NAR)':>11}{'fold(QP)':>10}")
    print("-" * 66)
    for r in out["results"]:
        if r.get("status") == "scored":
            nec = f"{r['n_exp']}/{r['n_ctl']}"
            print(f"{r['ds']:<12}{nec:>11}{r['true_hr']:>9}"
                  f"{str(r['fold_curveonly']):>12}{str(r['fold_nar']):>11}"
                  f"{str(r['fold_qp']):>10}")
        else:
            extra = (" · " + r["error"]) if r.get("error") else ""
            print(f"{r['ds']:<12}{'':>11}  {r.get('status')}{extra}")
    s = out["summary"]
    print("-" * 66)
    if s.get("n_scored"):
        print(f"scored {s['n_scored']}/{s['n_total']}")
        for key, lbl in (("curve_only", "curve-only Guyot  "),
                         ("censoring_informed", "Guyot + NAR table "),
                         ("qp", "Titman-QP (events)")):
            st = s.get(key)
            if st:
                ci = st["within_20pct_ci"]
                print(f"  {lbl}: median {st['median']}  p90 {st['p90']}"
                      f"  within20% {st['within_20pct']}/{st['n']} "
                      f"(95% CI {ci[0]:.0%}-{ci[1]:.0%})")
        pr = s.get("paired_qp_vs_nar")
        if pr:
            print(f"  paired QP vs Guyot+NAR (n={pr['n']}): QP better {pr['a_better']}, "
                  f"NAR better {pr['b_better']}, median fold ratio {pr['median_ratio_b_over_a']}")
    else:
        print(f"scored 0/{s['n_total']} -- no datasets reconstructed 2 arms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=r"C:\Projects\registry-ipd",
                    help="path to the registry-ipd repo (for realipd/*.csv)")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset ids (default: all configured)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--fusion", action="store_true",
                    help="run the NAR-fusion experiment (registry anchors + kmcurve NAR)")
    args = ap.parse_args()
    ds_list = args.datasets.split(",") if args.datasets else None
    if args.fusion:
        out = run_fusion(Path(args.registry), ds_list)
        _print_fusion(out)
    else:
        out = run(Path(args.registry), ds_list)
        _print_table(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
