# Corpus robustness findings (Phase-1 input)

**Date:** 2026-06-09. **Harness:** `corpus_scan.py` over OA PDFs fetched by
`acquire_corpus.py` (Europe PMC rendered PDFs of NCBI-PMC OA KM/RCT papers).

## Result (n = 9 OA PDFs + 1 publisher positive control)

| status | count | meaning |
|---|---|---|
| `extracted` | 0 / 9 | calibrated panel + >=2 separated arms |
| `raster_figure` | 8 / 9 | figure is a flattened raster image (no vector curves) |
| `vector_no_panel` | 1 / 9 | vector content but no axis frame detected |

**Positive control:** the bundled ADVANCE *publisher* PDF (NEJMoa0802987)
scans as `extracted` — auto-located page 7, 4 panels, calibration R²=1.0, 2
arms, 2 at-risk rows. So the 8/9 `raster_figure` verdict is a **real signal,
not a scanner bug**: the scanner succeeds when the figure is vector and reports
raster when it is not.

## What this means

- The **vector path is exact but niche.** It fires on publisher PDFs that
  preserve vector figures (NEJM, etc.) and on ~0% of Europe-PMC *rendered* OA
  PDFs, which rasterize their figures during rendering.
- Therefore **the raster fallback is the make-or-break coverage component**,
  not a nice-to-have. The legacy KMcurve pipeline targeted raster (correct)
  but its OCR axis calibration scored 0% (`OCR_INVESTIGATION_RESULTS.md`) —
  that calibration step is the real unsolved problem for the common case.
- Secondary gap: panel detection requires a full axis-frame rectangle; the one
  vector PDF that failed (`vector_no_panel`) likely uses tick-only axes.

## Source-diverse scan (update 2026-06-09): raster dominates ALL OA sources

We tested the "acquisition source is a confound" hypothesis directly by adding
publisher-OA (Unpaywall) and preprint (bioRxiv) fetchers.

| source | n | raster_figure | extracted |
|---|---|---|---|
| Europe-PMC rendered | 9 | 8 | 0 |
| Publisher OA (Unpaywall) | 9 | 9 | 0 |
| bioRxiv/medRxiv | 0 | — | — (acquisition blocked) |
| **combined OA** | **18** | **17** | **0** |

**The hypothesis was wrong in the optimistic direction.** Publisher OA PDFs
(mostly Frontiers / open-access mega-journals here) rasterise figures just as
PMC-rendered ones do — 17/18 real OA PDFs are raster, 0 vector-extractable.
The vector path works on a *specific premium subset* (e.g. NEJM/ADVANCE), not
on typical OA literature. bioRxiv could not be fetched (Europe PMC PPR results
were mostly Preprints.org/ResearchSquare, and bioRxiv blocks direct
`.full.pdf` bot access) — preprint acquisition needs a dedicated approach.

Caveat the other way: the Unpaywall sample skewed to Frontiers journals (known
rasterisers); a journal-stratified sample would refine the exact rate. But the
direction is unambiguous across 18 PDFs and two acquisition methods.

**Conclusion: the raster path is not a fallback — for OA coverage it is the
primary path.** Built in `raster_km.py`: render → dark-curve pixel cloud →
(reuses) 2-click calibration + continuity arm separation + Guyot. Validated
end-to-end on a synthetic 2-curve image (recovers drawn survival within 0.05).

## Raster auto-calibration: OCR tested, NOT reliable (2026-06-09)

Installed Tesseract 5.4 and built tick OCR (`ocr_tick_values`,
`auto_calibrate_axis`). **Position detection is exact** (`detect_plot_box` /
`detect_tick_positions` recover the axis frame and tick pixels perfectly), but
**OCR of the tick VALUES is unreliable**: benchmarked ~15-60% per-digit
accuracy, erratic across font sizes (18-40 px), and RANSAC can even return a
confidently-wrong line from coincidentally-collinear misreads. Best
preprocessing found (LANCZOS upscale + adaptive mean threshold + tight glyph
bbox + multi-PSM) helped isolated digits (7/8) but not in-context labels.

This reproduced the legacy 0%-OCR finding for TESSERACT-ALONE.

## UPDATE: dual-engine OCR makes raster auto-calibration RELIABLE (no clicks)

Tesseract-alone is unreliable, but **RapidOCR (ONNX) + Tesseract are
complementary**: RapidOCR reads dense x-axis labels Tesseract misses; Tesseract
reads y-axis labels RapidOCR misses. `auto_calibrate_axes` merges both engines'
(position, value) detections per axis and robust-fits with RANSAC, which also
rejects stray numbers (e.g. at-risk counts that fall in the label band).

**Validated on the rasterised ADVANCE figure (a real KM figure rendered to an
image, known truth x 0-66 / y 25->0): 4/4 panels auto-calibrated with NO
clicks, all axes R^2 >= 0.999, y endpoints within 0.1.** RANSAC correctly
rejected the at-risk "447" contaminating Panel A's x-label band.

`auto_calibrate_axes` still FAILS CLOSED (needs >=4 consistent ticks AND inlier
R^2 > 0.97, else raises -> 2-click), so it never emits a confidently-wrong
calibration. Prereqs: `rapidocr-onnxruntime` (pip) + the Tesseract binary
(system). When either is absent it degrades to the single available engine or
2-click. **Net: no-click raster calibration is now reliable on real figures;
2-click remains the guaranteed fallback.**

## FULL raster PDF -> IPD -> HR validated end-to-end on a real OA PDF

`pdf_raster_to_ipd()` chains: figure_locator -> render -> detect_plot_box ->
dual-engine auto-calibrate -> text-masked dark-curve cloud (in-plot legend
excluded via OCR boxes) -> continuity arm separation -> monotone survival ->
at-risk OCR (`extract_at_risk_raster`, leading integer of each "N (events)"
cell) -> Guyot IPD.

Validated on `10.1253/circrep.cr-25-0304` ("Figure 1. Comparison of
Kaplan-Meier plots", a flattened raster KM the vector path cannot touch):
- arms correctly separated: Acoramidis stays ~1.0, PS-matched placebo drops to
  0.76 (matches the printed curves);
- at-risk N parsed exactly: Acoramidis 25, placebo 50;
- reconstructed events: **Acoramidis 0, placebo 12 -- EXACTLY the published
  at-risk table's "0 (12)"**;
- HR strongly favours acoramidis (consistent with the paper's P=0.0107).

Real-world hardening this required: exclude in-plot legend/annotation TEXT from
the curve cloud (OCR boxes) or it spikes the curve; extract the LEADING integer
of "N (events)" at-risk cells; tolerate "Month 18"-style and irregular ticks.
(HR magnitude is somewhat noisy run-to-run with small N + few events + OCR
stochasticity, but event counts are exact and direction/significance robust.)

## Raster pipeline benchmark over the corpus (honest robustness numbers)

`raster_benchmark.py` runs the full raster end-to-end on all 18 corpus PDFs and
validates reconstructed events against the at-risk table's REPORTED events
(figure-internal ground truth -- the parenthetical "0 (12)"). Result:

| outcome | n | meaning |
|---|---|---|
| success (events > 0) | 1 | circrep: recon events [0, 12] == reported [0, 12] (0 error) |
| extract_suspect | 2 | 2 arms but 0 total events -> silent extraction failure (flat curves) |
| calib_fallback | 5 | auto-calibration failed closed -> recoverable with 2-click |
| extract_fail | 7 | located + (maybe) calibrated but <2 arms separated |
| no_km_located | 3 | no KM caption (1 is genuinely a flowchart) |

**Honest read: full no-click raster is PROVEN (circrep: events exact) but NOT
yet robust -- ~1/18 fully-automatic true success today.** The benchmark
pinpoints the gaps: (a) curve-pixel extraction + arm separation on real figures
(9/18 = extract_fail + extract_suspect) is the #1 weakness; (b) calibration
robustness (5/18 fail closed -- the 2-click fallback rescues these). Event
accuracy where it succeeds AND has ground truth is perfect (0 error). This is
the measured target list for the next round of raster hardening.

### Sub-project: grid-robust panel detection (built + validated; a negative result)

Hypothesis: panel detection was the binding constraint (bctt etc. are grids the
single-box detector merged). Built `_detect_panels_morph` (cv2 morphological
line extraction -> connected components -> pair each y-axis with the x-axis at
its bottom-left corner), which handles N x M grids. `detect_plot_boxes` now
prefers whichever of {projection, morph} finds MORE panels, so grids are gained
WITHOUT regressing the 1-2 panel cases.

Validated (deterministic): circrep 1/1, fonc1807364 2/2, **bctt 6/6** (the
3x2 grid the old detector failed), synthetic 2x2 -> 4. circrep end-to-end
preserved exactly ([0,12]). Also fixed two bugs the sub-project surfaced:
auto y-scale detection (0-1 vs 0-100%; fonc's flat [0,0] was a value_scale=1.0
artifact) and a RapidOCR LRU-1 cache (multi-panel figures re-ran OCR per panel).

**RESULT (corrected after the full benchmark finished): success DOUBLED 2 -> 4.**
An earlier mid-investigation note called this a "negative result" -- that was
premature (written before the full-corpus re-run completed). The combined
grid detection + auto y-scale actually flipped two figures to success:
- bctt (the 3x2 grid) -- now detected as 6 panels and reconstructs;
- fonc1807364 (2-panel, 0-100% axis) -- auto y-scale fixed the flat [0,0];
and circrep stayed exact ([0,12], 0 event error). Final taxonomy: success 4,
calib_fail 7, no_box 4, no_km_located 3, extract_suspect 0 (was 1-2).

The biggest single win was **auto y-scale detection** (percent vs proportion);
grid detection enabled the multi-panel successes. Remaining gaps: calib_fail
(7) and no_box (4) -- and the full benchmark is now SLOW (more panels -> more
per-panel Tesseract OCR; dedup/vectorise per-panel tick OCR next).

### CORRECTION (instrumented benchmark): calibration, not extraction, is the gap

The earlier "extraction is the dominant gap" was an ARTIFACT of an
un-instrumented benchmark (every empty-arms result fell into `extract_fail`).
After making the benchmark classify per-panel failure modes, the true
breakdown on 18 PDFs is:

| status | n |
|---|---|
| calib_fail (auto-calibration fails closed) | **8** |
| no_box (panel detection found nothing) | 4 |
| no_km_located | 3 |
| success | 2 |
| extract_suspect (flat curves) | 1 |

**So calibration robustness is the #1 lever (8/18); panel detection is #2
(4/18); curve EXTRACTION is barely a problem (1/18) once calibration passes.**
Diagnosed root causes among calib_fail: detection sometimes returns a too-wide
box (e.g. bctt x[77,1198] -> y "ticks" are table numbers, R^2=0.146), and OCR
reads too few/noisy ticks on some figures (x R^2~0.80). Several calib_fails are
therefore DOWNSTREAM of detection quality. Lesson: instrument the failure
taxonomy before deciding what to harden.

### Update: multi-panel detection added (`detect_plot_boxes`)

Real KM figures are often multi-panel (A/B side-by-side); the old single-box
detector merged panels -> flat [0,0]. `detect_plot_boxes` now finds each panel
and `pdf_raster_to_ipd` processes them independently (failing closed per
panel). Re-run taxonomy: success 1->2 (fonc.1807364, a 2-panel figure, now
extracts; events [9,8]), extract_suspect 2->1. calib_fallback folded into
extract_fail (now 12) because per-panel calibration fails closed *inside*
pdf_raster_to_ipd instead of raising -- so **curve-pixel extraction + arm
separation is now unambiguously THE dominant gap (13/18)**. Marginal
end-to-end gain (1->2 true success), but multi-panel is necessary
infrastructure and clarifies the next target: robust curve extraction on real
figures (thin/grey/coloured/dashed arms, censoring marks, near-overlapping
high-survival curves). NB the benchmark no longer separates "calibration
failed" from "extraction failed" -- an instrumentation gap to close (inspect
per-panel errors) next.

## Revised Phase-1 priority (data-driven)

1. **Raster fallback, properly built** — CV curve extraction (already partly in
   the legacy pipeline) + semi-automatic calibration (2-click, the accepted
   standard) and/or improved tick OCR; route via `manual_calibration.figure_is_vector`.
2. **Source-diverse acquisition** — add publisher/Unpaywall + bioRxiv/medRxiv
   fetchers; report vector-vs-raster rate per source so we know the real
   addressable fraction of each path.
3. **Panel detection robustness** — handle tick-only axes (no frame box).
4. Then the verified-HR accuracy corpus can grow across both paths.

## VLM-assisted calibration pilot (lever 1) — 2026-06-10

Built `vlm_calibrate.py`: classical CV owns geometry (`detect_tick_positions`),
a vision model owns reading (tick VALUES + axis semantics), never trusted for
pixel coordinates. Two paths share one ingest core: the Claude vision API
(forced structured tool call) and an agent/replay path. Wired as an opt-in
fallback in `raster_km._panel_to_ipd` (`KM_VLM_CALIBRATE=1`) after the OCR
`auto_calibrate_axes` fails closed. Fail-closed throughout.

**Pilot** (22 corpus axis-pairs with CV-detectable ticks on both axes;
`render_corpus_crops.py` + an adversarial read/verify workflow — two independent
VLM reads per crop, accept only on agreement; `eval_vlm_calibration.py`):

| metric | value |
|---|---|
| boxes | 22 |
| both-reader agreement (verify gate) | 11/22 |
| OCR `auto_calibrate_axes` calibrates | 8/22 |
| VLM calibrates | 4/22 |
| **VLM rescues (OCR fail → VLM ok)** | **3** (R²=1.0000 each) |
| **combined OCR ∪ VLM** | **11/22 vs 8/22 OCR-alone (+37%)** |

**Findings:**
- **Reading is solved.** Every verified VLM read was correct (R²=1.0000 on all 4
  fits + 3 rescues). The VLM trivially reads label classes that defeat OCR —
  `Baseline / Month 6 / Month 9 …` and `2.5, 7.5, 12.5`.
- **The bottleneck moved** from "OCR can't read tick labels" to **CV tick-count
  vs labelled-tick-count mismatch**: minor ticks make CV over-count (e.g. 11 CV
  ticks vs 6 labels), faint ticks make it under-count (4 vs 5). The `bctt`
  cluster fails here even though the VLM read perfectly.
- **Count-mismatch must fail closed — it is NOT safe to guess.** Because tick
  values are arithmetic and positions evenly spaced, *both sequences are linear
  in index*, so ANY even decimation/alignment fits with R²=1. R² cannot
  disambiguate a wrong alignment → guessing the correspondence would silently
  miscalibrate. The strict count-match gate is a correctness requirement.
- VLM is **complementary, not strictly better**: it rescues boxes OCR fails, and
  fails some boxes OCR handles. The right production posture is the wired
  fallback (OCR first, VLM on OCR-failure), which is what shipped.

**Next for lever 1 (well-scoped):** make `detect_tick_positions` isolate MAJOR
(labelled) ticks — by tick LENGTH (major ticks are drawn longer) and/or
proximity to a detected label glyph — so the CV count matches the VLM read count.
That converts the count-mismatch cluster without any unsafe alignment guessing.
