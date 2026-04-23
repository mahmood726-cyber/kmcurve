Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

Protocol: Fully Automated Kaplan-Meier Curve Extraction via Neural OCR and Mathematical Survival Constraints

This protocol describes the planned methods submission for KMcurve, a pipeline for automated digitization of Kaplan-Meier survival curves from published PDFs. The frozen benchmark is the AUGUSTUS trial PDF with eight panels, validated against SurvDigitizeR and manually digitized reference curves. The workflow combines high-resolution rasterization, TrOCR axis reading, color clustering, top-edge envelope extraction, and survival-constraint refinement before individual-patient-data reconstruction experiments. Primary reporting will focus on mean sensitivity with confidence intervals and root mean square error against the benchmark curves. Secondary analyses will summarize rank-order agreement and panel-level robustness rather than pooled clinical outcomes. Code, fixtures, and the static E156 bundle are archived repo-relatively for deterministic reviewer inspection. The current scope is limited to vector-renderable PDFs and does not support scanned image-only figures or competing-risks cumulative incidence curves.

Outside Notes

Type: protocol
Primary estimand: Sensitivity of automated curve extraction (95% CI)
App: KMcurve Pipeline v1.0
Code: https://github.com/mahmood726-cyber/KMcurve
Date: 2026-03-27
Validation: DRAFT

References

1. Royston P, Parmar MK. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Med Res Methodol. 2013;13:152. doi:10.1186/1471-2288-13-152.
2. Tierney JF, Stewart LA, Ghersi D, Burdett S, Sydes MR. Practical methods for incorporating summary time-to-event data into meta-analysis. Trials. 2007;8:16. doi:10.1186/1745-6215-8-16.
