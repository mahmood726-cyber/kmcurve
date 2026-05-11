# KMcurve

KMcurve is a research repository for extracting Kaplan-Meier and related time-to-event curves from PDFs. The maintained workflow is the structured Python stack in `ipd_km_pipeline/`; most root-level extractor variants and status notes are historical experiments.

## Supported Workflow

Use `ipd_km_pipeline/` as the canonical code path for:
- PDF rendering and image extraction
- panel detection
- raster curve extraction and dotted-line filtering
- OCR-based axis and numbers-at-risk experiments
- IPD reconstruction experiments

## Quick Start

Create an environment and install the maintained Python dependencies:

```bash
python3 -m pip install -r ipd_km_pipeline/requirements.txt
```

Run the sample end-to-end demo against the bundled fixture PDF:

```bash
python3 ipd_km_pipeline/pipeline_end_to_end.py
```

You can also point the demo at a different PDF:

```bash
python3 ipd_km_pipeline/pipeline_end_to_end.py --pdf path/to/study.pdf --page-num 6
```

## Tests

The repo now exposes a clean pytest surface from `tests/`:

```bash
python3 -m pytest -q
```

These smoke tests use repo-local fixtures and skip cleanly when optional CV dependencies such as OpenCV are not installed.

## Repo Layout

- `ipd_km_pipeline/`: maintained Python workflow
- `R/` and `km_pdf_vector_extract_ultra.R`: R/vector extraction experiments
- `e156-submission/`: published project-facing assets
- root-level `fully_automated_*`, `UNIFIED_VECTOR_PIPELINE.py`, and many dated markdown files: historical prototypes and research notes

## Methods

The maintained extractor (`ipd_km_pipeline/comprehensive_km_extractor.py`) follows the `SurvdigitizeR`-style raster pipeline rather than SVG-vector parsing of the mutool output:

1. **PDF → raster.** PyMuPDF (`fitz`) rasterises the chosen page to a high-resolution PNG.
2. **Axis calibration.** Axis tick text is read first via PDF text extraction (fast when the PDF carries a text layer) and falls back to OCR (`tesseract`) on rasterised tick regions when the layer is absent.
3. **Curve pixel extraction.** Pixels are converted to HSL and clustered with k-medoids to separate up to 6 overlapping curves by colour, with k-NN refinement on overlap regions. Dotted CI bands and reference lines are detected as a separate cluster.
4. **Numbers-at-risk.** OCR is applied to the at-risk table region and each label is anchored to its X-tick position.
5. **IPD reconstruction.** Event-time inference uses the Guyot-style algorithm: drops in `S(t)` are mapped to events, and the at-risk table anchors `n_i` so that `(t_i, event_i)` per patient can be emitted.

`probe.py` (run via the Overmind baseline `baseline_probes/probe_kmcurve.py`) computes deterministic structural signals (`n_curve_points`, `n_steps_detected`, `first_step_time`, `first_drop_mag`, `total_drop`, `n_nrisk_times`, `n_events`) against a fixed fixture so the numerical witness can detect regressions in the extractor.

## Limitations

- **Vector-SVG path parsing is intentionally disabled.** Earlier experiments tried to parse mutool's SVG output directly; the `H<num>` no-space tokenisation and rasterised "plot images traced as paths" defeated the approach. The raster pipeline is the canonical path; root-level `*.R` extractors and `km_pdf_vector_extract_ultra.R` are kept for provenance only.
- **OCR-bound axis calibration.** When the PDF text layer is missing or unreliable, tick reading falls back to OCR — confidence drops on stylised fonts, rotated labels, or low-DPI scans. The pipeline reports OCR confidence per axis and downstream code should treat low-confidence axes as a hard error rather than a number.
- **k-medoids needs the number of curves up front (or auto-detects 2-6).** Networks with 7+ curves, or two near-identical-colour curves, will be merged. The output JSON exposes the chosen `k` so users can re-run with a forced value.
- **CI bands are detected, not pooled.** The dotted-curve cluster is extracted as a separate trace; the pipeline does not currently combine survival + CI into a single posterior. Confidence reconstruction is out of scope.
- **No automatic risk-table absence handling.** If the PDF has no numbers-at-risk table, the IPD reconstruction falls back to a single-N anchor at `t=0`, which propagates uncertainty through to the IPD. Users should report the at-risk-table-present flag in their methods.
- **Python OpenCV is optional.** Smoke tests skip cleanly when OpenCV is not installed; some quality-improvement paths inside the pipeline require it.

## Conclusions

Use `ipd_km_pipeline/` as the supported path for PDF → KM-curve → IPD reconstruction. Treat any output where (a) OCR axis confidence is low, (b) the at-risk table was absent, or (c) the auto-detected `k` does not match the visible curve count as a manual-review case rather than an evidence-grade result. The root-level R and historical-prototype paths are retained for provenance and should not be wired into new workflows.

## Notes

- Generated artifacts, caches, agent state, and local secrets are gitignored.
- Sample fixtures live under `papers_to_process/` and `ipd_km_pipeline/artifacts/`.
- Historical docs remain in the repo for provenance, but `README.md` and `ipd_km_pipeline/` define the current supported developer path.
