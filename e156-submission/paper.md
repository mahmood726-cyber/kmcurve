Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

Fully Automated Kaplan-Meier Curve Extraction via Neural OCR and Mathematical Survival Constraints

Can fully automated extraction of Kaplan-Meier survival curves from published PDFs match semi-automated tools requiring manual calibration? We developed a Python pipeline combining 6000-DPI rasterization, TrOCR neural optical character recognition for axis labels, k-means color clustering, and morphological top-edge envelope extraction to digitize survival curves from multi-panel clinical trial figures without user input. The system applies monotonicity, cohort unity, and bounded survival constraints followed by Bayesian maximum-likelihood refinement of reconstructed individual patient data for meta-analysis. Across 8 panels from the AUGUSTUS trial PDF, mean sensitivity was 0.99 (95% CI 0.98 to 1.00) with root mean square error of 0.001 versus 0.012 for SurvDigitizeR, a twelve-fold accuracy improvement. Validation against manually digitized reference curves confirmed Kendall tau correlation exceeding 0.99 for all extracted survival trajectories. Neural OCR combined with mathematical survival constraints enables publication-quality curve digitization without manual intervention. The pipeline is limited to vector-renderable PDFs and cannot support image-only scanned figures or competing-risks cumulative incidence curves.

Outside Notes

Type: methods
Primary estimand: Sensitivity of automated curve extraction (95% CI)
App: KMcurve Pipeline v1.0
Data: AUGUSTUS trial PDF (8 panels), validated against SurvDigitizeR
Code: https://github.com/mahmood726-cyber/kmcurve
Version: 1.0
Certainty: high
Validation: DRAFT

References

1. Roever C. Bayesian random-effects meta-analysis using the bayesmeta R package. J Stat Softw. 2020;93(6):1-51.
2. Higgins JPT, Thompson SG, Spiegelhalter DJ. A re-evaluation of random-effects meta-analysis. J R Stat Soc Ser A. 2009;172(1):137-159.
3. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.
