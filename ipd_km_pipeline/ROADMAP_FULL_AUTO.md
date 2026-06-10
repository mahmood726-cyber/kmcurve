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
   **11/22 vs 8/22 OCR-alone (+3 rescues, R^2=1.0)**. KEY FINDING: VLM *reading*
   is solved (every verified read correct); the bottleneck MOVED to CV
   tick-count vs labelled-tick-count mismatch. Count-mismatch MUST fail closed
   (arithmetic values x even positions => any alignment fits R^2=1, so guessing
   silently miscalibrates). See CORPUS_FINDINGS.md -> "VLM-assisted calibration
   pilot". **Refined next step:** major-tick-aware `detect_tick_positions` (by
   tick length / label proximity) so CV count matches the VLM read count.

2. **Labelled corpus (200-1000 figures)**. You cannot push past the long tail
   -- or even MEASURE fractional gains -- on n=18. Reuse acquire_corpus.py
   (Europe PMC) + figure_locator.py; hand-label panel counts, axis type/range,
   arm count, and (where reported) per-arm events/HR. This is the gating
   resource for everything else.

3. **ML curve segmentation** (small U-Net) for color/dashed/overlapping arms +
   censoring marks + CI bands -- robust where the current pixel heuristics are
   fragile (the circrep 2px-flip problem). Trains on lever 2's corpus.

4. **Calibrated confidence** that PREDICTS reconstruction error (not just
   calibration R^2), so auto-accept vs human-flag is reliable. Validate the
   confidence against actual error on lever 2's corpus. This is what makes
   "confidence-gated auto" trustworthy at scale.

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
