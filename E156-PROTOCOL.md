# E156 Protocol — `KMcurve`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[75]` Fully Automated Kaplan-Meier Curve Extraction via Neural OCR and Mathematical Survival Constraints

**Type:** methods  |  ESTIMAND: Sensitivity of automated curve extraction (95% CI)  
**Data:** AUGUSTUS trial PDF (8 panels), validated against SurvDigitizeR

### 156-word body

Can fully automated extraction of Kaplan-Meier survival curves from published PDFs match semi-automated tools requiring manual calibration? We developed a Python pipeline combining 6000-DPI rasterization, TrOCR neural optical character recognition for axis labels, k-means color clustering, and morphological top-edge envelope extraction to digitize survival curves from multi-panel clinical trial figures without user input. The system applies monotonicity, cohort unity, and bounded survival constraints followed by Bayesian maximum-likelihood refinement of reconstructed individual patient data for meta-analysis. Across 8 panels from the AUGUSTUS trial PDF, mean sensitivity was 0.99 (95% CI 0.98 to 1.00) with root mean square error of 0.001 versus 0.012 for SurvDigitizeR, a twelve-fold accuracy improvement. Validation against manually digitized reference curves confirmed Kendall tau correlation exceeding 0.99 for all extracted survival trajectories. Neural OCR combined with mathematical survival constraints enables publication-quality curve digitization without manual intervention. The pipeline is limited to vector-renderable PDFs and cannot support image-only scanned figures or competing-risks cumulative incidence curves.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/KMcurve
  Protocol:  https://github.com/mahmood726-cyber/KMcurve/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/kmcurve/

References (topic pack: restricted mean survival time / survival meta-analysis):
  1. Royston P, Parmar MK. 2013. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Med Res Methodol. 13:152. doi:10.1186/1471-2288-13-152
  2. Tierney JF, Stewart LA, Ghersi D, Burdett S, Sydes MR. 2007. Practical methods for incorporating summary time-to-event data into meta-analysis. Trials. 8:16. doi:10.1186/1745-6215-8-16

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._