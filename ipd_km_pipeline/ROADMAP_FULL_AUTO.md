# Roadmap: toward fully-automatic (no-2-click) KM extraction

Goal: push the raster path from semi-auto (auto where confident, 2-click else)
toward **confidence-gated full-auto** -- fully automatic on the figures it is
confident about, human only on the flagged minority. NOT chasing literal 100%.

## Honest success odds (assessment 2026-06-09)

| target | estimate |
|---|---|
| full-auto on a controlled subset (single-panel, clear labels, survival) | 80-90% |
| **confidence-gated auto across real OA figures (the transformational target)** | **70-85% coverage achievable** |
| near-human (>95%) fully unattended on ARBITRARY figures (the "holy grail") | 15-25% without a major ML+data investment; ~40-50% with one |

Why hard: full-auto = PRODUCT of ~6 stages (locate x panel-detect x calibrate x
extract x arm-separate x at-risk). 0.95^6 ~= 74%. The crux is **axis
calibration** (OCR of tick values) -- the dominant failure (7/18 calib_fail).
Plus a hard floor (genuinely illegible axes) and fragility (a 2px box shift
flips circrep's reconstructed events).

## Current state (this is what works now, PRs #1-#16)

- Vector path: exact (R^2=1.0), validated HR 0.90 on ADVANCE. Niche (premium PDFs).
- Raster path end-to-end: locate -> render -> dual-engine OCR auto-calibrate ->
  grid-aware multi-panel detect -> text-masked extraction -> Guyot. Fails closed
  to 2-click. **Corpus benchmark: success 4/18** (event-exact where ground truth
  exists; circrep [0,12]). Remaining: calib_fail 7, no_box 4, no_km_located 3.
- Benchmark is SLOW now (more panels -> more per-panel Tesseract OCR).

## The four levers for the new session (1,2,3,4)

1. **VLM-assisted axis calibration** (biggest lever). Use a Claude/GPT-4V-class
   model to read tick VALUES + axis semantics (survival vs CI, %, units, log),
   then classical CV for sub-pixel tick POSITIONS. The VLM leapfrogs the OCR
   crux; do NOT trust the VLM for precise geometry (hallucination) -- structure
   only. Target: turn most of the 7 calib_fail into successes.

   **STATUS (2026-06-10): shipped + piloted.** `vlm_calibrate.py` (API path +
   agent/replay path, shared ingest core), wired as an opt-in fallback in
   `raster_km` (`KM_VLM_CALIBRATE=1`), 7/7 unit tests, fail-closed. Pilot on 22
   corpus axis-pairs (read + adversarial-verify workflow): combined OCR u VLM
   **12/22 vs 8/22 OCR-alone (+4 rescues, R^2>=0.9999)**. KEY FINDING: VLM
   *reading* is solved (every verified read correct); the bottleneck MOVED to CV
   tick-count vs labelled-tick-count mismatch. Over-count from split peaks is
   FIXED (`_merge_split_peaks`); the residual is under-count (a missing extreme
   tick), which is genuinely ambiguous and MUST stay fail closed (arithmetic
   values x even positions => any alignment fits R^2=1, so guessing silently
   miscalibrates). See CORPUS_FINDINGS.md -> "VLM-assisted calibration pilot".
   **Refined next step:** higher-recall tick detection (lower projection
   threshold + spurious-peak guard) to recover missed extreme ticks safely.

2. **Labelled corpus (200-1000 figures)**. You cannot push past the long tail
   -- or even MEASURE fractional gains -- on n=18. Reuse acquire_corpus.py
   (Europe PMC) + figure_locator.py; hand-label panel counts, axis type/range,
   arm count, and (where reported) per-arm events/HR. This is the gating
   resource for everything else.

   **STATUS (2026-06-10): labelling MACHINE shipped; corpus growth network-gated.**
   `autolabel.py` (LABEL_SCHEMA + order-invariant `compare_labels` + confidence
   gate) + `render_figures.py` + `build_corpus.py`; 9/9 tests. Instead of
   hand-labelling, a VLM double-reads structured labels and a confidence gate
   auto-accepts vs flags-for-human. Seed corpus: 15 figs -> 8 KM (18 panels) +
   7 non-KM (locator FPs); 15/15 auto-accepted at mean conf 1.000; PMC2247136
   (4 panels, 2/4/2/4 arms) independently verified correct. BLOCKERS: (a)
   acquisition is network-blocked here (`getaddrinfo failed`, 200/200) -- wired,
   grows when networked; (b) **locator precision is the yield ceiling** (7/15
   located non-KM regions) -> improving `figure_locator` is the top next step;
   (c) lever 3 needs PIXEL MASKS not structured labels -> bridge = weak-label
   from CV extraction on auto-accepted figures (self-distillation). See
   CORPUS_FINDINGS.md -> "Labelled corpus via confidence-gated auto-labelling".

3. **ML curve segmentation** (small U-Net) for color/dashed/overlapping arms +
   censoring marks + CI bands -- robust where the current pixel heuristics are
   fragile (the circrep 2px-flip problem). Trains on lever 2's corpus.

   **STATUS (2026-06-10): scaffolded (not trained).** `unet_segment.py` (tiny
   2-level U-Net + synthetic KM generator + train/infer path) with
   `test_unet_smoke.py` proving the train->infer path on synthetic data (skips
   if torch absent). Real training is GATED on lever 2's labelled corpus -- that
   is the hard blocker. torch (CPU) added as the only new dependency. See
   CORPUS_FINDINGS.md -> "ML curve segmentation scaffold".

4. **Calibrated confidence** that PREDICTS reconstruction error (not just
   calibration R^2), so auto-accept vs human-flag is reliable. Validate the
   confidence against actual error on lever 2's corpus. This is what makes
   "confidence-gated auto" trustworthy at scale.

   **STATUS (2026-06-10): shipped (discrimination validated).**
   `confidence.py::calibration_confidence` scores from verify-agreement (0.35) +
   tick count (0.30) + semantic plausibility incl. scale-consistency (0.25) +
   R^2 floor (0.10) -> weighted geometric mean + `auto_accept` gate; surfaced in
   `ingest_vlm_answer` meta. `validate_confidence.py` perturbation study:
   **discrimination=1.000**, auto-accept precision correct=100% /
   wrong-scale=0% / verify-disagree=0%. Blind spot (quantified): a CORRELATED
   identical misread that stays linear+plausible (err 0.195, conf 1.0) is
   uncatchable -> needs an EXTERNAL value cross-check (at-risk N / caption
   follow-up), the natural next safeguard. Probability calibration (vs ranking)
   deferred to lever 2's labelled corpus. See CORPUS_FINDINGS.md -> "Calibrated
   confidence".

## Also queued (smaller)

- Speed up the benchmark: dedup/vectorise per-panel Tesseract tick OCR (RapidOCR
  is already LRU-cached per figure; Tesseract per-tick per-panel is the slow part).
- Locator precision: downweight body-text KM mentions (require figure-caption
  context), e.g. fonc1744027 located a methods page.
- x-axis label classification on imperfect/too-tall boxes (several calib_fail
  have x-ticks=0 because the box or x-band is slightly off).

## Key files

- `raster_km.py` -- the raster pipeline (detect_plot_boxes, auto_calibrate_axes,
  pdf_raster_to_ipd, extract_at_risk_raster, _detect_panels_morph).
- `figure_locator.py` -- KM figure location by caption + content.
- `acquire_corpus.py` -- Europe PMC / Unpaywall PDF acquisition.
- `raster_benchmark.py` -- end-to-end benchmark vs at-risk-table reported events.
- `vector_km.py`, `guyot.py`, `manual_calibration.py` -- vector path + IPD + 2-click.
- `CORPUS_FINDINGS.md` -- the full measured history + corrections.
