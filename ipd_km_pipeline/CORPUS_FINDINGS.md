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

## Important caveat: acquisition source is a confound

"~89% raster" is specific to **Europe-PMC rendered** PDFs. Vector availability
depends heavily on source:
- Publisher PDFs (via DOI / Unpaywall) and bioRxiv/medRxiv PDFs are far more
  likely to be vector.
- A proper characterisation must scan multiple sources and report
  vector-availability per source. This is an explicit Phase-1 task, not a
  settled number.

## Revised Phase-1 priority (data-driven)

1. **Raster fallback, properly built** — CV curve extraction (already partly in
   the legacy pipeline) + semi-automatic calibration (2-click, the accepted
   standard) and/or improved tick OCR; route via `manual_calibration.figure_is_vector`.
2. **Source-diverse acquisition** — add publisher/Unpaywall + bioRxiv/medRxiv
   fetchers; report vector-vs-raster rate per source so we know the real
   addressable fraction of each path.
3. **Panel detection robustness** — handle tick-only axes (no frame box).
4. Then the verified-HR accuracy corpus can grow across both paths.
