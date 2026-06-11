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

Pool k simulated trials (true logHR ~ N(μ, τ²), known IPD) three ways with DerSimonian–Laird, against the
true-IPD pool the patient data would give:

| pool | per-trial input | k=16, true μ=−0.357 (HR 0.70), τ²=0.06 |
|---|---|---|
| `true_ipd` | Cox logHR from simulated IPD (gold) | μ −0.417 (HR 0.66), τ² 0.150 |
| `curve_only` | curve-only Guyot (no events) | μ **−0.166 (HR 0.85)**, τ² **0.012** |
| `event_pinned` | Titman-QP + each arm's total events | μ **−0.415 (HR 0.66)**, τ² **0.149** |

**The censoring lever recovers the true-IPD pool — both the pooled effect AND the heterogeneity τ².**
Curve-only does not just mis-estimate one trial: pooled, its attenuation toward HR=1 **shrinks μ, τ² and the
prediction interval**, so the synthesis reads the trials as more homogeneous and the effect smaller than the
truth. The event count is what lets a reconstructed pool stand in for an IPD meta-analysis. Locked by
`test_granularity_pool.py` (seeded). Run: `python granularity_pool.py --k 16 --tau2 0.06`.

## Honest scope / what is NOT claimed

- **Measured aside:** with *both* arms' curves fixed at the same anchors, the log-rank HR is nearly
  insensitive to the *total* event count, so the curve-only under-identification surfaces as **bias**
  (Guyot's zero-censoring assumption), not a variance band. The QP's value here is removing that bias.
- This is a **simulation** with exponential survival + light random censoring — a clean illustration of the
  mechanism, not a real-data result. The real-data half (and the Rubin `s²+r²` reconstruction-variance
  propagation, which matters when reconstruction noise dominates) lives in registry-ipd
  (`honest_pooling_sim.js`, `phase2_real_pooling.js`); do not duplicate it — cite it.
- Single seed shown; the ordering (event-pinned ≈ true-IPD ≫ curve-only) is what the test asserts, not the
  exact numbers.

## Next

Repeat on the `realipd_benchmark.py` gold-standard datasets (real true-IPD) instead of the simulator, so the
pool-recovery claim rests on real reconstructions; and add an `abstract_HR_only` granularity tier (a logHR
likelihood with no time structure) to complete the manifold (curve+NAR ▸ curve+events ▸ curve-only ▸ HR-only).
