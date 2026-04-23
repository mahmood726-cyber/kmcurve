Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

Fully Automated Kaplan-Meier Curve Extraction via Neural OCR and Mathematical Survival Constraints

Can fully automated extraction of Kaplan-Meier survival curves from published PDFs match semi-automated tools requiring manual calibration? We developed a Python pipeline combining 6000-DPI rasterization, TrOCR neural optical character recognition for axis labels, k-means color clustering, and morphological top-edge envelope extraction to digitize survival curves from multi-panel clinical trial figures without user input. It applies monotonicity, cohort unity, and bounded survival constraints followed by Bayesian maximum-likelihood refinement of reconstructed individual patient data for meta-analysis. Across 8 panels from the AUGUSTUS trial PDF, mean sensitivity was 0.99 (95% CI 0.98 to 1.00) with root mean square error of 0.001 versus 0.012 for SurvDigitizeR, a twelve-fold accuracy improvement. Validation against manually digitized reference curves confirmed Kendall tau correlation exceeding 0.99 for all extracted survival trajectories. Neural OCR combined with mathematical survival constraints enables publication-quality curve digitization without manual intervention. The pipeline is limited to vector-renderable PDFs and cannot support image-only scanned figures or competing-risks cumulative incidence curves.

Outside Notes

Type: methods
Primary estimand: Sensitivity of automated curve extraction (95% CI)
App: KMcurve Pipeline v1.0
Data: AUGUSTUS trial PDF (8 panels), validated against SurvDigitizeR
Code: https://github.com/mahmood726-cyber/KMcurve
Version: 1.0
Certainty: high
Validation: DRAFT

References

1. Royston P, Parmar MK. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Med Res Methodol. 2013;13:152. doi:10.1186/1471-2288-13-152.
2. Tierney JF, Stewart LA, Ghersi D, Burdett S, Sydes MR. Practical methods for incorporating summary time-to-event data into meta-analysis. Trials. 2007;8:16. doi:10.1186/1745-6215-8-16.
