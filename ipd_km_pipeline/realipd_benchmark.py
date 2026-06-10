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
    DSConfig("gbsg", "GBSG breast cancer (RFS, hormonal Rx)", "rfstime", "status", "hormon", "1", "0"),
    DSConfig("veteran", "Veterans lung cancer (OS, treatment)", "time", "status", "trt", "2", "1"),
    DSConfig("rotterdam", "Rotterdam breast cancer (OS, hormonal Rx)", "dtime", "death", "hormon", "1", "0"),
    DSConfig("pbc", "PBC (OS, D-penicillamine vs placebo)", "time", "status", "trt", "2", "1", event_val="2"),
    DSConfig("diabetic", "Diabetic retinopathy (vision loss, laser)", "time", "status", "trt", "1", "0"),
    DSConfig("cancer", "NCCTG lung cancer (OS, by sex)", "time", "status", "sex", "2", "1", event_val="2"),
    DSConfig("colon", "Colon cancer (OS, Lev+5FU vs Obs)", "time", "status", "rx", "Lev+5FU", "Obs"),
    DSConfig("nwtco", "NWTSG Wilms tumour (relapse, histology)", "edrel", "rel", "histol", "2", "1"),
    DSConfig("bmt", "Bone marrow transplant (DFS, risk group)", "t2", "d3", "group", "3", "1"),
    DSConfig("tongue", "Tongue cancer (death, ploidy)", "time", "delta", "type", "2", "1"),
]
CONFIG_BY_DS = {c.ds: c for c in CONFIGS}


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

    # Curve-only (no at-risk table) vs censoring-informed (the printed risk table).
    nar_exp = _at_risk_table(et, t_max)
    nar_ctl = _at_risk_table(ct, t_max)
    hr_curve = _hr_from(exp_xy, ctl_xy, n_exp, n_ctl)
    hr_nar = _hr_from(exp_xy, ctl_xy, n_exp, n_ctl, nar_exp, nar_ctl)
    fold_curve = _fold(hr_curve, true_hr)
    fold_nar = _fold(hr_nar, true_hr)

    return {
        "ds": cfg.ds, "label": cfg.label, "status": "scored",
        "n_exp": n_exp, "n_ctl": n_ctl,
        "true_hr": round(true_hr, 4),
        "recon_hr_curveonly": round(hr_curve, 4) if np.isfinite(hr_curve) else None,
        "recon_hr_nar": round(hr_nar, 4) if np.isfinite(hr_nar) else None,
        "fold_curveonly": round(fold_curve, 4) if fold_curve else None,
        "fold_nar": round(fold_nar, 4) if fold_nar else None,
        "abs_log_err_nar": round(abs(np.log(fold_nar)), 4) if fold_nar else None,
        "true_events": int(ee.sum() + ce.sum()),
    }


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
        def _stats(key):
            xs = sorted(r[key] for r in scored if r.get(key))
            if not xs:
                return None
            return {
                "median": round(xs[len(xs) // 2], 4),
                "p90": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 4),
                "within_20pct": sum(x <= 1.2 for x in xs),
                "n": len(xs),
            }
        summary["censoring_informed"] = _stats("fold_nar")
        summary["curve_only"] = _stats("fold_curveonly")
    return {"summary": summary, "results": results}


def _print_table(out: dict) -> None:
    print(f"\n{'dataset':<12}{'n(e/c)':>11}{'trueHR':>9}{'HR(c-only)':>12}"
          f"{'HR(+NAR)':>10}{'fold(c)':>9}{'fold(N)':>9}")
    print("-" * 76)
    for r in out["results"]:
        if r.get("status") == "scored":
            nec = f"{r['n_exp']}/{r['n_ctl']}"
            print(f"{r['ds']:<12}{nec:>11}{r['true_hr']:>9}"
                  f"{str(r['recon_hr_curveonly']):>12}{str(r['recon_hr_nar']):>10}"
                  f"{str(r['fold_curveonly']):>9}{str(r['fold_nar']):>9}")
        else:
            extra = (" · " + r["error"]) if r.get("error") else ""
            print(f"{r['ds']:<12}{'':>11}  {r.get('status')}{extra}")
    s = out["summary"]
    print("-" * 76)
    if s.get("n_scored"):
        ci, co = s.get("censoring_informed"), s.get("curve_only")
        print(f"scored {s['n_scored']}/{s['n_total']}")
        if co:
            print(f"  curve-only        : median fold {co['median']}  p90 {co['p90']}"
                  f"  within20% {co['within_20pct']}/{co['n']}")
        if ci:
            print(f"  censoring-informed: median fold {ci['median']}  p90 {ci['p90']}"
                  f"  within20% {ci['within_20pct']}/{ci['n']}")
    else:
        print(f"scored 0/{s['n_total']} -- no datasets reconstructed 2 arms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=r"C:\Projects\registry-ipd",
                    help="path to the registry-ipd repo (for realipd/*.csv)")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset ids (default: all configured)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()
    ds_list = args.datasets.split(",") if args.datasets else None
    out = run(Path(args.registry), ds_list)
    _print_table(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
