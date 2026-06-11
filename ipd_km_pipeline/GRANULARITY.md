# Granularity-manifold pooling — kmcurve's role in mixed-granularity synthesis

*Where kmcurve fits in registry-ipd's `SYNTHESIS-VISION.md`: one meta-analysis that pools trials at their
own data granularity. This is the kmcurve-side demonstration (`granularity_pool.py`).*

## The idea (from registry-ipd)

Drop the AD-vs-IPD binary. Every trial sits on a data-completeness manifold and is **partially identified**
by what it posted: a curve alone point-identifies median/RMST/S(t) but only *partially* identifies the HR
(the event/censoring split is latent); a curve **+ a total-event count** point-identifies the HR (Titman-QP);
a curve + number-at-risk fully identifies it (Guyot). The **censoring lever** (event count / at-risk table)
is what moves a trial up the manifold — and **kmcurve's figure OCR is that lever** (it reads the "N(events)"
/ numbers-at-risk table off the published figure).

## What this demonstrates (`granularity_pool.py`)

Pool trials four ways with DerSimonian–Laird — spanning the manifold from most to least identified for the
HR estimand — against the true-IPD pool the patient data would give. Two modes:

### Simulation (`python granularity_pool.py --k 16 --tau2 0.06`)

k trials with true logHR ~ N(μ, τ²), exponential IPD:

| pool | per-trial input | k=16, true μ=−0.357 (HR 0.70), τ²=0.06 |
|---|---|---|
| `true_ipd` | logHR from simulated IPD (gold) | μ −0.417 (HR 0.66), τ² 0.150 |
| `event_pinned` | Titman-QP + each arm's total events | μ **−0.415 (HR 0.66)**, τ² **0.149** |
| `abstract_hr_only` | the reported HR, printed to 2 dp (no curve) | μ **−0.417 (HR 0.66)**, τ² 0.150 |
| `curve_only` | curve-only Guyot (no events) | μ **−0.166 (HR 0.85)**, τ² **0.012** |

In the simulation the censoring lever recovers the true-IPD pool on **both** μ and τ²; curve-only attenuates
both, so the synthesis reads the trials as more homogeneous and the effect smaller than the truth.

### Real data (`python granularity_pool.py --real`) — the gold-standard true-IPD datasets

A coherent protective-direction panel of 9 registry-ipd gold-standard datasets, each run through kmcurve's
**actual raster reconstruction pipeline** (`realipd_benchmark.recon_and_score`), pooled vs the true-IPD pool:

| pool | k=9 protective panel | d-μ vs true | per-trial mean \|recon−true\| logHR |
|---|---|---:|---:|
| `true_ipd` | μ −0.544 (HR 0.58), τ² 0.123 | — | — |
| `event_pinned` | μ −0.507 (HR 0.60) | +0.037 | 0.134 |
| `abstract_hr_only` | μ −0.545 (HR 0.58) | −0.002 | 0.007 |
| `curve_only` | μ **−0.322 (HR 0.72)** | **+0.222** | **0.372** |

**The pooled-EFFECT half replicates on real data; the τ² half does not.** Curve-only reliably attenuates the
pooled μ toward the null (true HR 0.58 → reconstructed 0.72) while event-pinned and abstract-HR-only recover
it; per-trial, the curve-only logHR error dwarfs the others on **every** dataset (the panel-independent
metric, robust to how effect directions balance). The completed manifold says something non-obvious: for the
HR estimand a **bare reported HR pins the pooled effect**, while a **full curve *without* the censoring lever
biases it** — more raw data, worse answer. (For time-dependent estimands — RMST, absolute risk, non-PH
checks — the ordering inverts: the curve rungs carry everything, `abstract_hr_only` carries nothing.)

Locked by `test_granularity_pool.py` (sim seeded; real tests use the deterministic raster pipeline, skip if
registry-ipd absent). The strong-effect datasets (nwtco HR 19.6 → curve 1.05, prostateSurvival 8.2 → 1.7) are
excluded from the default panel only so one outlier doesn't dominate τ²; on them the curve-only collapse is
far more dramatic — `python granularity_pool.py --real --datasets nwtco,prostateSurvival,ebmt1,melanoma`.

## Honest scope / what is NOT claimed

- **τ² recovery on real data holds only OUTSIDE the strong-effect/small-sample corner.** The simulation's
  clean τ² recovery does not carry over for a *specific* reason: per-trial QP reconstruction error rises with
  effect **strength** (corr(|err|, |true logHR|) ≈ +0.7) and **small sample** (corr(|err|, events) ≈ −0.5),
  so a strong-effect/few-events trial is attenuated toward the centre, shrinking the between-trial spread. On
  the default panel a **single** such trial — `gehan` (HR 0.19, 30 events: true logHR −1.64 → recon −0.91) —
  drives the *entire* τ² gap: drop it and `d-τ²` collapses −0.046 → −0.008 (k=8). So τ² *is* recoverable when
  no trial sits in that corner (precisely the OA-wall population kmcurve can't fully digitize anyway). This is
  **attenuation bias, not additive reconstruction variance** — a variance-subtraction "debias" is wrong-signed
  and would make it worse. μ-recovery is the robust real-data claim; τ²-recovery is conditional on the panel.
- **Measured aside (sim):** with *both* arms' curves fixed at the same anchors, the log-rank HR is nearly
  insensitive to the *total* event count, so the curve-only under-identification surfaces as **bias**
  (Guyot's zero-censoring assumption), not a variance band. The QP's value here is removing that bias.
- The real-data panel pools **clinically unrelated** gold-standard datasets — a *methods* benchmark to
  measure granularity-induced pooling bias, **not a clinical synthesis**. The Rubin `s²+r²`
  reconstruction-variance propagation (which matters when reconstruction noise dominates) lives in
  registry-ipd (`honest_pooling_sim.js`, `phase2_real_pooling.js`); do not duplicate it — cite it.

## Next

- Replace the unrelated-datasets panel with a genuinely **same-indication** IPD set (if one becomes
  available) so the pool is a real clinical synthesis, not just a bias measurement.
- The τ²-gap mechanism is now characterised (strong-effect/small-n attenuation, not variance — see above), so
  a variance-subtraction debias is ruled out. The remaining lever is upstream: a better arm-separation /
  reconstruction for the strong-effect/small-n corner (i.e. lever 3, the U-Net) would shrink the per-trial
  attenuation that drives the gap — the τ² half follows the digitization-accuracy frontier, not a pooling fix.
