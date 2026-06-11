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
| VLM calibrates (after split-peak merge) | 5/22 |
| **VLM rescues (OCR fail → VLM ok)** | **4** (R²≥0.9999 each) |
| **combined OCR ∪ VLM** | **12/22 vs 8/22 OCR-alone (+50%)** |

(Pre-merge: VLM 4/22, 3 rescues. The `_merge_split_peaks` post-process —
collapsing a single tick that split into two adjacent CV peaks — added `bctt`
box0 (11→6 X, 7→5 Y ticks). It never invents a missing tick, so genuine
under-detection still fails closed.)

**Findings:**
- **Reading is solved.** Every verified VLM read was correct (R²≥0.9999 on all 5
  fits). The VLM trivially reads label classes that defeat OCR —
  `Baseline / Month 6 / Month 9 …` and `2.5, 7.5, 12.5`.
- **The bottleneck moved** from "OCR can't read tick labels" to **CV tick-count
  vs labelled-tick-count mismatch**, of two kinds: (a) one tick split into two
  adjacent CV peaks — over-count, now FIXED by `_merge_split_peaks`; (b) a
  faint/extreme tick missed — under-count (4 vs 5), the residual `bctt` cluster,
  which is genuinely ambiguous (which end is missing flips the calibration) and
  correctly stays fail-closed.
- **Count-mismatch must fail closed — it is NOT safe to guess.** Because tick
  values are arithmetic and positions evenly spaced, *both sequences are linear
  in index*, so ANY even decimation/alignment fits with R²=1. R² cannot
  disambiguate a wrong alignment → guessing the correspondence would silently
  miscalibrate. The strict count-match gate is a correctness requirement.
- VLM is **complementary, not strictly better**: it rescues boxes OCR fails, and
  fails some boxes OCR handles. The right production posture is the wired
  fallback (OCR first, VLM on OCR-failure), which is what shipped.

**Next for lever 1 (well-scoped):** the residual failures are all under-detection
(a missing extreme/faint tick → CV count < VLM count). Recovering the missing
tick safely needs an unambiguous anchor (e.g. plot-frame edge ↔ extreme labelled
value) — but the pilot showed the detected ticks do NOT reliably sit at the box
edges, so naive edge-anchoring is unsafe. Treat this as a hard floor unless a
geometric anchor can be made unambiguous; do NOT relax the count-match gate
(linear degeneracy → silent miscalibration). Higher-recall tick detection
(lower projection threshold with a spurious-peak guard) is the safer lever.

## Calibrated confidence (lever 4) — 2026-06-10

`confidence.py::calibration_confidence` predicts whether a calibration is
CORRECT (not just its R²), for the auto-accept-vs-human-flag gate. R² is a weak
predictor on its own (the linear degeneracy: any in-order alignment of arithmetic
values to evenly-spaced positions scores R²=1). The predictive signals are, by
weight: **verify-agreement** (two independent VLM reads agree — 0.35),
**tick count** (≥4 arithmetic ticks make a misread non-collinear so RANSAC drops
it — 0.30), **semantic plausibility** (survival/CI y in [0,1]/[0,100] spanning
most of the axis; time x from ~0; scale-consistency: percent↔~100, proportion↔~1
— 0.25), and an **R² floor** (0.10). Combined via weighted geometric mean +
an `auto_accept` gate (conf ≥ 0.80, ≥4 ticks, verify not False). Surfaced in
`ingest_vlm_answer` meta.

**Validation** (`validate_confidence.py` — perturbation study on the 5 verified
pilot calibrations as correct references; reconstruction error = max fit
deviation vs reference over the box, normalised):

| perturbation | mean conf | auto-accept | mean err |
|---|---|---|---|
| correct | 1.000 | 100% | 0.000 |
| wrong_scale (toggle is_percent) | 0.610 | 0% | 0.000* |
| verify_fail (reads disagree) | 0.515 | 0% | 0.000* |
| y_kind_other | 0.861 | 100% | 0.000 |
| x_scale_1.2 (correlated misread) | 1.000 | 100% | 0.195 |

- **Discrimination = 1.000** (correct always out-scores fit-corrupting
  perturbations); auto-accept precision: correct 100%, wrong_scale/verify_fail 0%.
- **verify-agreement is the workhorse** — the double-read is what makes
  auto-accept safe; a disagreement alone forces a human flag.
- **Honest limits.** (1) The fit-error proxy is 0 for `wrong_scale`/`verify_fail`
  because they corrupt the *scale/semantics*, not the fit slope — confidence
  still catches them via the scale-consistency + verify signals. (2) `y_kind_other`
  yields a correct FIT (err=0) — a semantic mislabel, not a calibration error. (3)
  **Blind spot:** a *correlated identical misread* that stays linear and plausible
  (`x_scale_1.2`, err=0.195) is uncatchable by confidence. Catching it needs an
  EXTERNAL value cross-check — the at-risk-table baseline N or the caption's
  follow-up time — which is the natural next safeguard (and ties into the at-risk
  OCR already in the raster pipeline).
- **Deferred:** mapping the score to a true probability (vs ranking) needs the
  labelled corpus (lever 2). v1 validates DISCRIMINATION, which is what the
  auto-accept gate requires.

### External cross-check closes lever 4's blind spot — 2026-06-10

`axis_cross_check.py::cross_check_calibration` adds the INDEPENDENT value source
that confidence alone lacks: it compares the calibrated x-max against the
caption's reported follow-up (`caption_max_followup`) and/or the at-risk-table
column times, and the y-span against the axis scale. An external contradiction
VETOES `auto_accept` regardless of confidence (threaded through
`ingest_vlm_answer` → `calibration_confidence(cross_check=...)`); checks with no
available reference are skipped (never assumed-pass).

Validation (`validate_confidence.py`): the correlated-misread blind spot
(`x_scale_1.2`, err 0.195) goes from **auto-accept 1.00 → 0.00** once a caption /
at-risk follow-up reference is supplied. This ties the safeguard into the
at-risk OCR already in the raster pipeline and the figure caption from
`figure_locator`. Tests: +5 (`test_axis_cross_check.py`); suite 20/20.

## ML curve segmentation scaffold (lever 3) — 2026-06-10

`unet_segment.py`: a tiny 2-level U-Net (`build_unet`) that labels each pixel
background / arm-1 / arm-2 / censor-mark, plus a synthetic KM generator
(`synthetic_km_sample`) and a train/infer path (`train`, `segment`). This
replaces the fragile dark-cloud pixel heuristics (`dark_curve_cloud` +
`column_curve_points`) where curves are coloured / dashed / overlapping or
near-coincident at high survival — the circrep "2px shift flips the events"
problem.

**Scaffold only — not a trained model.** `test_unet_smoke.py` proves the
train→infer path end-to-end on synthetic data (loss falls, train-sample pixel
accuracy > 0.9), and skips cleanly if torch is absent. A production model needs
the **labelled corpus (lever 2)** — that is the hard gate. The synthetic
generator doubles as a pre-training bootstrap until real labels exist.
torch (CPU) is the only new dependency.

## Labelled corpus via confidence-gated auto-labelling (lever 2) — 2026-06-10

Hand-labelling 200-1000 figures is intractable, so lever 2 reuses the levers-1/4
machinery to BOOTSTRAP the corpus: a VLM reads structured labels, two independent
reads adversarially verify, and a label confidence gates each figure into
auto-accepted vs needs-human-review. "Labelling" becomes "human-verify the
low-confidence minority" — the confidence-gated labelling the whole roadmap aims
at, applied to corpus construction.

- `autolabel.py` — `LABEL_SCHEMA` (panel count; per-panel y/x axis kind+unit+range,
  arm count, censoring, at-risk table) + `compare_labels` (ORDER-INVARIANT panel
  matching; structural-field agreement → confidence + `needs_human_review`;
  agreed-not-KM is a clean negative) + `corpus_record`.
- `render_figures.py` — full-figure PNGs from corpus PDFs. `build_corpus.py` —
  the double-read workflow output → `labels.jsonl` with provenance + the gate.
- Tests: `test_autolabel.py` (9) incl. panel-reorder + agreed-not-KM.

**Seed corpus (15 figures; double-read + verify workflow, 30 agents):**
- **8 KM figures (18 panels), 7 non-KM** (locator false-positives — useful
  negative labels). **15/15 auto-accepted, mean double-read confidence 1.000**,
  0 needs-review (the two reads agreed on every structural field after
  order-invariant matching).
- **Independently validated** PMC2247136 (by direct read): 4 panels with 2/4/2/4
  arms, survival-percent, months 0-30 — exactly the auto-label. The two workflow
  reads differed only on panel ORDER, which `compare_labels` correctly does not
  flag.

**Honest limits / next:**
- **Acquisition is network-blocked in this environment** — `acquire_corpus.py`
  hit `getaddrinfo failed` on all 200 NCBI/Europe-PMC fetches (DNS). The
  acquisition is wired and ready; the corpus grows when run with network, and the
  labelling machine scales to whatever it returns.
- **Locator precision is the corpus-quality ceiling:** 7/15 located regions were
  NOT KM figures (e.g. a methods page) — the auto-labeller correctly tags these
  `is_km_figure:false`, but improving `figure_locator` (require figure-caption
  context, downweight body-text KM mentions) is the highest-value next step for
  corpus yield. ADVANCE (NEJM, non-PMC) wasn't even located — premium PDFs need a
  different path.
- **Lever 3 needs PIXEL MASKS, not structured labels.** The bridge is
  weak-labelling: derive masks from the CV curve extraction on high-confidence /
  auto-accepted figures, then train the U-Net on those (self-distillation) — a
  clean next step now that confidence-gating identifies the trustworthy figures.

## figure_locator precision: caption-anchoring (lever 2 yield) — 2026-06-10

The lever-2 ceiling was locator precision: a page matched if the survival
keyword appeared ANYWHERE in its text, so a Methods sentence ("Kaplan-Meier
analysis was performed...") + any large image scored as a KM figure (7/15 located
regions were non-KM). Fix: anchor the keyword to a real FIGURE CAPTION.

- `_caption_blocks` extracts only lines whose "Figure N" sits at the START (a
  caption), rejecting inline cross-references ("... in Figure 2 ..."),
  parenthetical refs ("(Figure 3C). ..."), and subpanel refs ("Figure 2A,").
- A page is a strong candidate only when a survival keyword appears in such a
  caption block; body-text-only mentions are kept as weak fallbacks far below
  (score x0.25) and exposed via `caption_anchored`. `require_caption=True` drops
  body-only candidates entirely (high-precision corpus building).
- Tests: `test_figure_locator.py` (8) — caption vs inline/parenthetical/subpanel.

**Measured on the 18-PDF corpus (top hit):**
- **12/18 now caption-anchored**; all 8 previously-confirmed KM figures still
  anchored (no regression).
- **3 rescues** — `nci49035` (p0->p3), `fimmu1753591` (p6->p4),
  `fonc1744027` (p2->p5) were located on the WRONG page before (labelled
  `is_km:false`); the new locator anchors the real KM caption. All three
  **verified by direct read** to be genuine KM figures (OS curves, at-risk
  tables, censoring). KM-figure yield on the rendered set: **8/15 -> 11/15
  (53% -> 73%)**; caption-anchored precision **11/12** (residual FP:
  `fonc1784342`, a patient-selection flowchart whose multi-line caption mentions
  "survival").
- The 3 non-anchored top hits (`PMC1977864`, `PMC7845702` cost figure,
  `PMC8420689` flowchart) are correctly de-ranked (score 0.5, `caption_anchored:
  false`) — body-text mentions with no actual KM figure on the located page.

Net: caption-anchoring is the single change that lifts corpus yield, and it
composes with the lever-2 confidence gate (the residual flowchart FP is exactly
what `is_km_figure:false` + human-review catches).

## Lever 3 real-figure training: weak labels + the data-gate (measured) — 2026-06-10

With the network back, acquisition added **+40 PMC PDFs (58 total)**. `weak_labels.py`
derives per-pixel arm masks from the CV extraction (`dark_curve_cloud` ->
`column_curve_points` -> `separate_arms`) on caption-anchored KM figures, and
`train_on_figures.py` trains the lever-3 U-Net on those REAL figures (3-class,
class-weighted CE; dedup across corpus dirs; thickness-dilated strokes so curves
survive downsizing).

**Measured result (honest negative):**
- From 58 acquired PDFs, only **9 figures** yield a usable 2-arm weak mask
  (~16% — most PMC PDFs are vector, single-arm, lack a detectable plot box, or
  aren't caption-anchored).
- Trained on ~8 figures (22 boxes), the U-Net **does not learn** curve
  segmentation: loss stays ~flat (1.09 -> 1.086) and **held-out arm-IoU ~= 0.00**
  (pixel-acc 0.96 is meaningless — background is ~96%). Stroke thickness (1->3)
  and resolution (96x128 -> 128x160) tweaks did not help.
- **Conclusion: lever 3 is DATA-gated, not architecture-gated.** The synthetic
  smoke test already proved the train->infer path learns when given enough clean
  data; on ~8 real figures it cannot. The requirement is corpus SCALE — order
  hundreds of usable 2-arm figures, i.e. thousands of acquired PDFs (at ~16%
  usable yield). The pipeline (acquire -> caption-anchor locate -> weak-label ->
  train -> eval) is built, tested, and scales; only the figure count is missing.

This de-risks the roadmap: do NOT invest in U-Net architecture tuning yet —
invest in corpus growth (acquisition at scale + hand-verified masks for the
auto-accepted high-confidence figures). The weak-label bootstrap is the right
day-1 substitute, but a heuristic-distilled model cannot exceed the heuristic;
real gains need real masks on hard (coloured/dashed/overlapping) figures.

### Scaled re-run (2026-06-11): yield holds, the gate begins to crack

Acquisition grew the corpus ~58 → ~500 PMC PDFs (`acquire_corpus.py` after the
pagination/retry/reconcile hardening). Re-running `train_on_figures.py`:

| metric | 58 PDFs | ~500 PDFs |
|---|---:|---:|
| usable 2-arm figures | 9 | **82** (147 boxes) |
| yield | 15.5% | **16.4%** |
| train / held-out figs | 8 / 1 | 66 / 16 |
| held-out arm-IoU | ~0.00 | **0.073** |

- **Yield scales linearly** — ~16% holds at 10× scale (9→82), confirming the
  acquire → caption-anchor → weak-label pipeline scales as the negative result
  predicted (NOT architecture-limited).
- **First learning signal** — held-out arm-IoU moved 0.00 → 0.073. Still weak
  (loss ~flat 1.05–1.15; a useful model needs ~0.5+), but it is no longer
  flat-zero: with ~8× the training figures the U-Net begins to segment arms.
- **Direction confirmed, gate not yet cleared.** Extrapolating ~16% to the
  ~1500-PDF target → ~240 usable figures, the next milestone for a stronger
  signal. Also fixed `train_on_figures` observability (flushed per-25-PDF
  progress) — the build was silent for ~30 min due to block-buffered stdout.

## True-IPD external HR-accuracy: the first non-circular accuracy number — 2026-06-10

Every prior corpus number is figure-INTERNAL (reconstructed events vs the
figure's own at-risk table) or synthetic. The question that matters for
meta-analysis — *how close is the reconstructed HR to the TRUE HR from real
patient-level data?* — had no answer because no published-figure corpus ships
attached IPD. `realipd_benchmark.py` borrows the sibling project
`C:\Projects\registry-ipd`'s gold standard (~40 open true-IPD datasets:
Rdatasets / KMsurv / asaur / cBioPortal-TCGA, with arm/time/status configs):
render the TRUE KM curve to an actual raster → run THIS pipeline
(`raster_km` dark-cloud extract → `separate_arms` → Guyot) → log-rank HR vs the
TRUE HR. Same estimator (`guyot.logrank_hr`) for truth and recon, so the
fold-error is pipeline-induced. Calibration + at-risk table supplied exactly
(production OCRs both) → isolates curve-extraction + arm-separation +
reconstruction. **42 true-IPD datasets** (Rdatasets/KMsurv/asaur classics +
TCGA late-vs-early cohorts), +11 tests (`test_realipd_benchmark.py`,
`test_qp_reconstruct.py`).

| reconstruction backend | median HR fold-err | p90 | within 20% (95% CI) | input it needs |
|---|---:|---:|---:|---|
| **curve-only** Guyot | 1.42 | 4.46 | 14/42 (21–48%) | the curve |
| **Guyot + NAR table** | 1.09 (~9%) | 1.52 | 31/42 (59–85%) | + interior at-risk table |
| **Titman-QP (events)** | **1.041 (~4%)** | **1.26** | 35/42 (69–92%) | + total "N (events)" only |

(Paired on the same 42 datasets, QP beats Guyot+NAR 28/42, median fold ratio 1.017.)

**Three findings:**
- **The at-risk/event information is essential, not optional.** Curve-only is
  unusable — median 1.42, p90 4.46, only 14/42 within 20%, with HR *inversions*
  on well-separated arms (`diabetic` true 0.47 → curve-only 0.83). Root cause
  (`guyot.reconstruct_ipd_guyot` line ~127): with no number-at-risk, Guyot
  approximates the risk set as `survival × N`, i.e. assumes ZERO censoring,
  attenuating the HR and collapsing censored plateaus (`pbc` true tail 0.36 →
  recon 0.00).
- **Titman-QP is a drop-in backend upgrade (`qp_reconstruct.py`).** Ported from
  registry-ipd's `reconstructArmQP`: the curve fixes per-interval hazards, the
  total event count is a linear constraint, and the leftover censoring DOF is the
  closed-form min-norm QP. Fed only the "N (events)" totals that
  `extract_at_risk_raster` OCRs, it reaches median **1.041 / p90 1.26** — beating
  Guyot-with-the-full-NAR-table (1.09 / 1.52), and matching registry-ipd's own
  finding (QP ~1.05 vs Guyot ~1.14). **Port nuance:** the literal JS realization
  places `round(h_k·n)` events per interval, which silently loses events on a
  DENSE extracted curve (hundreds of pixel columns where `h_k·n < 0.5`); the
  Python port adds Guyot-style fractional-event *carry* so it generalises from
  registry-ipd's ~8 sparse anchors to kmcurve's dense pixel curve (before the fix
  QP undercounted, e.g. 180 requested → 101 placed, and lost to Guyot+NAR).
- **This validates the pipeline AND the lever-3 priority.** ~4% median HR error
  on true IPD is a genuine, competitive external number — the pipeline works when
  the anchors are present. It corroborates registry-ipd's central finding that the
  binding constraint is anchor/event information, not pixel reading. The residual
  error concentrates in arm-separation on monochrome curves (the `nwtco` extreme
  imbalance is the hardest) — exactly the failure mode lever 3 (ML segmentation)
  targets, which the corpus-scale acquisition is feeding.

Honest scope: clean single-style monochrome rendered curves with EXACT
calibration + at-risk table are an upper bound — real figures add coloured/
overlapping arms, censor ticks, and OCR error on both calibration and the at-risk
table. Shared artifact: `registry-ipd/validate/real_pipeline_headtohead_results.json`
+ a real-pipeline section appended to registry-ipd `HEADTOHEAD.md`.

**QP wired into production (opt-in).** `pdf_raster_to_ipd` now uses the Titman-QP
backend when `KM_QP_BACKEND=1` and the "N (events)" cells are OCR'd
(`at_risk_reported_events`), with a Guyot fallback on any failure. Off by default
so the event-exact circrep validation is unchanged; benchmark evidence (QP 1.04
vs Guyot 1.09) supports promoting it to default once at-risk-events OCR is
hardened on real figures.

## NAR fusion: the mirror-image union beats either project alone — 2026-06-10

`realipd_benchmark.py --fusion` tests the deepest cross-project idea: registry-ipd
has the curve EXACTLY (AACT anchors) but no number-at-risk; kmcurve OCRs the
at-risk table but its curve is pixel-noisy. Holding the backend fixed (Guyot) and
varying only the INPUTS across 42 true-IPD datasets:

| input regime | median fold | p90 | within 20% (95% CI) |
|---|---:|---:|---:|
| registry-only (exact anchors, NO NAR) | 1.37 | 4.26 | 16/42 (25–53%) |
| kmcurve-only (noisy curve + NAR) | 1.09 | 1.52 | 31/42 (59–85%) |
| **FUSION (exact anchors + NAR)** | **1.049** | 1.28 | 35/42 (69–92%) |
| **FUSION + QP** | 1.058 | **1.21** | 36/42 (72–93%) |

Paired: fusion beats registry-only **34/42** (CIs don't overlap → robust — the
identifiability trap dissolved; `cbio_kirp` 8.6→1.36, `prostateSurvival` 5.4→1.05)
and kmcurve-only **26/42** (median ratio 1.035; CIs overlap → favorable, not
significant at n=42). The exact curve cures kmcurve's pixel noise; the at-risk
table cures registry-ipd's missing censoring — fusion is strictly ≥ either alone.
Full write-up: `registry-ipd/FUSION.md` + `validate/nar_fusion_results.json`.

**Real-trial demonstration (`fusion_real_trial.py`) — RADIANT-4 (NCT01524783),
validated vs the PUBLISHED HR.** registry-ipd's harvested AACT record gives the
exact posted KM anchors + N=205/97; the figure's "N (events)" totals (107/77, what
`at_risk_reported_events` OCRs) supply the censoring AACT lacks; the posted Cox HR
**0.48 (95% CI 0.35–0.67)** is held-out ground truth:

| reconstruction | HR | fold vs posted | inside posted CI? |
|---|---:|---:|:--:|
| registry-only (anchors, NO censoring) | 0.83 | 1.73 | no |
| **FUSION (anchors + figure events)** | **0.56** | **1.17** | **yes** |

On a real trial with a real posted effect, registry-only falls OUTSIDE the
published CI (the trap) and the figure's event count pulls fusion INSIDE it —
the same worked example registry-ipd's VALIDATION.md documents (curve-only 0.68 →
censoring-informed 0.47 vs posted 0.48), now as a union of both projects' real
data. (Honest link: events taken from the AACT harvest; kmcurve OCRs the identical
"N (events)" totals from the figure in production. The RADIANT-4 Lancet primary
isn't OA here, so the figure-OCR step is exercised separately by the corpus
benchmark.) +1 test (`test_fusion_real_trial.py`).

**Why the fully-OA end-to-end demo is hard: a measured OPEN-ACCESS GAP.** Two
scans quantify it from both directions. `fusion_crossmatch.py` (figure→registry):
of 327 OA corpus PDFs, 185 cite an NCT → 267 unique NCTs → 37 with posted ctgov
results → **0** with a structured 2-arm KM curve. `fusion_pairfinder.py`
(registry→figure, over registry-ipd's 30 curve+HR gallery trials): 25 link a
primary publication but **0 have an open-access PRIMARY in PMC** (27/30 have *some*
OA paper citing the NCT, but they are reviews/secondary analyses without the
matching 2-arm figure + at-risk table). The trials that post structured registry
curves are predominantly industry RCTs whose primaries are paywalled, while the OA
literature citing them lacks the figure — **the two sources are anti-correlated on
openness.** So the end-to-end real-figure fusion is bounded by data *access*, not
capability (this is exactly why `fusion_real_trial.py` takes RADIANT-4's event
count from the AACT harvest rather than OCR-ing the paywalled Lancet figure), and
it is a direct, quantified reinforcement of registry-ipd's POLICY.md ask. Both
scans are cached/incremental — re-run as the corpus grows or OA status changes.
+9 tests (`test_fusion_crossmatch.py`, `test_fusion_pairfinder.py`).

## UPDATE: scaling the corpus to 663 PDFs finds the first dual-available pairs — 2026-06-11

The "0 of 327" cross-match result was **corpus-bounded, not capability-bounded**, so the corpus was
doubled and the (cached/incremental) `fusion_crossmatch.py` re-run over `corpus_pmc`:

| metric | 327-PDF scan | **663-PDF scan** |
|---|---:|---:|
| PDFs scanned | 327 | **663** |
| PDFs citing an NCT | 185 | **373** |
| unique NCTs | 267 | **592** |
| NCTs with posted ctgov results | 37 | **106** |
| **dual-available fusion candidates** | **0** | **2** |

The two candidates (both posted survival curve + OA figure citing the NCT):

| PDF | NCT | outcome | timepoints | groups | posted HR |
|---|---|---|---:|---:|:--:|
| PMC7530824 | NCT01658878 | Overall Survival rate | 5 | 15 | — |
| PMC13006393 | NCT03110107 | Progression-free Survival rate | 3 | 9 | — |

**The 0→2 jump confirms corpus size was the binding constraint.** But honesty about quality: both are
**early-phase multi-cohort dose-finding trials** (NCT01658878 = nivolumab Phase 1/2, 657 pts, 15 dose
groups; NCT03110107 = BMS-986218 first-in-human Phase 1/2, 376 pts, 9 groups) and **neither posts a
hazard ratio**. They are valid **OCR-pipeline demonstration targets** (a real figure to extract an
at-risk table from, paired with the exact registry anchors) but **not validation-grade**: with no posted
HR there is no held-out ground truth, and the many-cohort structure is not the clean 2-arm comparison the
fusion HR-recovery validation needs. A **2-arm + posted-HR** open-access pair is still absent at 663 PDFs
— the confirmatory-RCT open-access gap (paywalled industry primaries) persists, exactly as the
anti-correlation finding predicts. Next: grow the corpus further, or target OA primaries of 2-arm RCTs
specifically; meanwhile NCT01658878/NCT03110107 are the first end-to-end OCR-fusion demo candidates.
Reproduce: `python fusion_crossmatch.py --corpus corpus_pmc --out fusion_candidates.json`.

## UPDATE: corpus 1500 PDFs → 6 candidates, 4 with a posted HR (the gap closes) — 2026-06-11

The mirror kept downloading (658 → **1500** OA PMC PDFs). Re-running the cross-match surfaced the first
**validation-grade** fusion pairs — and made the scan tractable at this size by switching NCT extraction
from pdfplumber to **PyMuPDF/fitz (~1.3s vs ~15s per PDF, ~10× faster; identical NCT output verified on
12/12 cached PDFs)**, with a pdfplumber fallback. The cached pdf→NCT map stays valid across the switch.

| metric | 327 | 663 | **1500** |
|---|---:|---:|---:|
| PDFs scanned | 327 | 663 | **1500** |
| PDFs citing an NCT | 185 | 373 | **803** |
| unique NCTs | 267 | 592 | **1297** |
| NCTs with posted results | 37 | 106 | **254** |
| NCT-citation candidates | 0 | 2 | **6** |
| …**fusion-USABLE** (PDF is the primary + has an at-risk table) | 0 | 2 | **3** |
| …**usable AND posts a hazard ratio** | 0 | 0 | **1** |

**Correction (and a crossmatch fix).** A first pass over 1500 PDFs reported "6 candidates, 4 with an HR"
and flagged `NCT00636168` (HR 0.75) as a clean validation-grade pair. **Verifying the PDFs refuted that**:
`PMC9893404` is a *network meta-analysis* that merely **cites** NCT00636168 (17 cited NCTs, 26 NMA
signals, **no at-risk table**) — it does not contain the trial's KM figure. The cross-match's
NCT-citation test cannot tell a primary from a review, so it was over-counting. Added `classify_pdf`
(likely-primary = ≤3 cited NCTs and not an NMA, AND an at-risk table present) and a `fusion_usable` flag.

| PDF | NCT | outcome | tpts | posted HR | usable? |
|---|---|---|---:|---|:--:|
| **PMC9662922** | **NCT01942135** | **Survival probability** | 3 | **0.42 (0.32–0.56)** | ✅ primary + at-risk |
| PMC7530824 | NCT01658878 | OS rate | 5 | — | ✅ primary + at-risk, no HR |
| PMC13006393 | NCT03110107 | PFS rate | 3 | — | ✅ primary + at-risk, no HR |
| PMC9893404 | NCT00636168 | OS rate | 6 | 0.75 | ❌ NMA citing the trial |
| PMC9487257 | NCT01121393/466660 | Time to response | 14/7 | 0.28/0.82 | ❌ review, no at-risk |

So the honest count at 1500 PDFs is **3 fusion-usable primaries with an at-risk table, exactly 1 of which
posts a hazard ratio**: **`NCT01942135` / `PMC9662922`** (survival-probability curve, posted HR 0.42, OA
primary with an at-risk table) is the first genuinely usable **curve + HR + figure** pair — held-out
ground truth for an end-to-end NAR fusion. The bottleneck was corpus size *and* primary-vs-review
filtering; both now addressed. The narrower count also re-confirms the open-access gap (most HR-posting
trials' OA mentions are reviews, not primaries). Reproduce:
`python fusion_crossmatch.py --corpus corpus_pmc --out fusion_candidates.json`.
