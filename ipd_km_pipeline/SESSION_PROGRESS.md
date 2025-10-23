# K-M Curve IPD Extraction - Session Progress Report

**Date**: 2025-10-22
**Session Focus**: R package analysis and real-world PDF testing setup

---

## COMPLETED THIS SESSION

### 1. Comprehensive R Package Analysis

**File**: `R_PACKAGES_ANALYSIS.md` (full competitive analysis)

#### Key Findings:

**SurvdigitizeR (July 2024 - Latest)**:
- Uses HSL color space (same as us!)
- k-medoids clustering for curve separation
- Tesseract OCR for axis detection
- Average RMSE: 0.012 (vs 0.014 manual)
- **No dotted line detection** - identified as future work
- Cannot handle rotated images or background grids
- Requires manual parameter specification

**IPDfromKM (June 2021)**:
- Two-stage approach: coordinate extraction + IPD reconstruction
- Manual clicking for coordinate extraction (not automated)
- Modified-iKM algorithm for IPD reconstruction
- Incorporates numbers-at-risk data
- RMSE target: ≤0.05
- Comparison baseline for Guyot (2012) method

#### Our Competitive Advantages:

✅ **ONLY method with dotted line detection** (4-method combined approach)
✅ **Full automation** (no manual parameters, auto-detect curves, auto-calibrate)
✅ **Comprehensive validation** (numbers-at-risk matching + quality metrics)
✅ **Performance optimized** (vectorized operations, parallel batch processing)
✅ **Advanced features planned** (Bayesian IPD, neural network refinement)

---

### 2. PDF Download Infrastructure

**File**: `download_km_pdfs.py` (268 lines)

**Features**:
- PubMed Central E-utilities API integration
- Automated search for K-M curve papers (2020-2024)
- Batch download with rate limiting (NCBI compliance)
- Metadata extraction (title, journal, year, DOI)
- Organized file structure (by source: pmc, nejm, other)
- Progress tracking and error handling
- JSON metadata catalog

**Currently**: Downloading 40 PDFs from PMC for validation testing

---

### 3. Implementation Priorities Identified

#### Immediate Improvements (from R package analysis):

1. **Upgrade to k-medoids clustering** (SurvdigitizeR uses this)
   - More robust to outliers than k-means
   - Better for color-based clustering
   - Action: Update `raster_cv/auto_detect.py`

2. **Add rotation detection/correction**
   - Both competitors fail on rotated images
   - Use Hough transform to detect axis orientation
   - Apply correction before extraction

3. **Implement background grid removal**
   - SurvdigitizeR explicitly fails with background grids
   - Use Hough line detection
   - Filter before curve extraction

4. **Enhance censoring marker detection**
   - Both packages struggle with censoring marks (ticks, crosses, diamonds)
   - Implement dedicated pattern detection
   - Remove before curve fitting

---

## KEY INSIGHTS

### Where We Excel

1. **Dotted Line Detection** - No competitor addresses this critical issue
   - Density analysis (pixels per x-unit)
   - Continuity analysis (connected components)
   - Horizontal detection (y-range/x-range ratio)
   - FFT periodicity detection
   - Combined confidence scoring

2. **Full Automation** - Others require manual parameter specification
   - Auto-detect number of curves (silhouette analysis)
   - Auto-calibrate axes (OCR-based)
   - Auto-parse numbers-at-risk tables

3. **Batch Processing** - We have parallel processing, they don't
   - Multiprocessing pool
   - Progress tracking with tqdm
   - Comprehensive error handling
   - CSV/JSON output aggregation

### Accuracy Targets

| Method | RMSE | Notes |
|--------|------|-------|
| Manual digitization | 0.014 | Baseline |
| SurvdigitizeR | 0.012 | Best published |
| SurvdigitizeR (w/ censoring) | 0.016 | Degrades with marks |
| IPDfromKM | ≤0.05 | Recommended threshold |
| **Our Goal** | **<0.01** | Using Bayesian + validation dataset |

---

## NEXT STEPS

### Immediate (This Week)

1. ✅ **R package analysis** (completed)
2. 🔄 **Download 40 test PDFs** (in progress)
3. **Run batch processor on 40 PDFs**
   ```bash
   python batch_processor.py test_pdfs batch_results
   ```
4. **Analyze results**:
   - Success rate
   - Edge cases and failure modes
   - Comparison to expected behavior

### Short-term (This Month)

5. **Implement k-medoids clustering**
   - Research scikit-learn-extra or custom implementation
   - Update `raster_cv/auto_detect.py`
   - Test on diverse curve styles

6. **Add rotation detection/correction**
   - Implement Hough-based axis detection
   - Apply rotation transformation
   - Test on rotated PDFs

7. **Background grid removal**
   - Hough line detection for grids
   - Filter horizontal/vertical lines
   - Preserve actual K-M curves

8. **Enhanced censoring marker detection**
   - Pattern library (ticks, crosses, diamonds)
   - Template matching or shape detection
   - Remove before curve extraction

### Medium-term (2-3 Months)

9. **Validation metrics** (IPDfromKM-style)
   - RMSE, MAE, max absolute error
   - Kolmogorov-Smirnov test
   - Compare against ground truth (when available)

10. **Bayesian IPD reconstruction**
    - PyMC or Stan implementation
    - MCMC sampling for event times
    - Uncertainty quantification
    - Outperform Guyot (2012)

11. **Benchmark comparison study**
    - Run our pipeline on SurvdigitizeR test set
    - Run on IPDfromKM examples
    - Publish head-to-head comparison

### Long-term (6+ Months)

12. **Neural network refinement** (pending validation dataset)
13. **Figure type classifier** (K-M vs CIF vs RMST vs HR)
14. **Data access applications** (Project Data Sphere, Vivli, YODA, CSDR)
15. **Academic publication** + benchmark dataset release
16. **Open-source package** (PyPI release)

---

## FILES CREATED/MODIFIED THIS SESSION

1. **R_PACKAGES_ANALYSIS.md** (new) - Comprehensive competitive analysis
2. **download_km_pdfs.py** (new) - Automated PDF downloader
3. **SESSION_PROGRESS.md** (new) - This document

---

## TECHNICAL COMPARISONS

### Algorithm Comparison

| Feature | SurvdigitizeR | IPDfromKM | Our Pipeline |
|---------|---------------|-----------|--------------|
| **Image Processing** |
| Color space | HSL | N/A | HSL |
| Clustering | k-medoids | N/A | k-means → k-medoids |
| Curve detection | Manual count | Manual click | Auto-detect (silhouette) |
| **OCR** |
| Axis detection | ✅ Tesseract | ❌ Manual | ✅ Tesseract + validation |
| Numbers-at-risk | ❌ | ✅ | ✅ |
| **Preprocessing** |
| Dotted line filtering | ❌ | ❌ | ✅ 4-method approach |
| Rotation correction | ❌ | ❌ | 🔄 Planned |
| Grid removal | ❌ | ❌ | 🔄 Planned |
| Censoring markers | ⚠️ Degrades | ⚠️ Limited | 🔄 Enhanced planned |
| **IPD Reconstruction** |
| Method | N/A | iKM algorithm | 🔄 Bayesian MCMC planned |
| Numbers-at-risk integration | N/A | ✅ | ✅ |
| Uncertainty quantification | N/A | ❌ | 🔄 Planned |
| **Automation** |
| Parameter specification | ❌ Manual | ❌ Manual | ✅ Fully automated |
| Batch processing | ❌ | ❌ | ✅ Parallel |
| **Performance** |
| RMSE | 0.012 | ≤0.05 target | <0.01 target |
| Speed | N/A | N/A | Optimized (vectorized) |

---

## COMPETITIVE MESSAGING

### For Academic Paper

**Title**: "Fully Automated K-M Curve IPD Extraction with Robust Dotted Line Detection: A Validation Study"

**Key Claims**:
1. First method to address dotted line detection problem
2. Only fully automated pipeline (no manual parameter specification)
3. Outperforms SurvdigitizeR and IPDfromKM on diverse real-world PDFs
4. Comprehensive validation framework with numbers-at-risk matching
5. Open-source implementation for reproducibility

**Target Journals**:
- BMC Medical Research Methodology (where SurvdigitizeR and IPDfromKM published)
- PLOS ONE (open access, methods focus)
- Journal of Statistical Software (with R/Python package)

---

## VALIDATION STRATEGY

### Phase 1: Diverse PDF Testing (Current)
- Download 40 diverse K-M curves from PMC (2020-2024)
- Different journals, cancer types, figure styles
- Run batch processor and analyze:
  - Success rate
  - Extraction quality (manual spot-checks)
  - Edge cases and failures

### Phase 2: Comparison to Competitors
- Obtain SurvdigitizeR test set (if available)
- Run on same PDFs they used
- Direct RMSE comparison
- Identify scenarios where each method excels

### Phase 3: Ground Truth Validation
- Apply for IPD access (Project Data Sphere, Vivli)
- Generate K-M curves from real IPD
- Extract with our pipeline
- Reconstruct IPD
- Compare to original (gold standard)

### Phase 4: Large-Scale Testing
- Download 300 PDFs (later phase)
- Automated quality assessment
- Statistical analysis of success rate
- Publication-ready benchmark dataset

---

## RISK ASSESSMENT

### Technical Risks

1. **PMC PDF availability**
   - Risk: Many papers may not have PDFs available
   - Mitigation: Cast wider net (100+ papers to get 40 PDFs)
   - Status: Testing with current download run

2. **Real-world curve diversity**
   - Risk: Test PDFs may not cover all edge cases
   - Mitigation: Manual curation of challenging examples
   - Status: Will assess after batch processing

3. **IPD data access timeline**
   - Risk: Data access applications take 2-6 months
   - Mitigation: Proceed with PDF testing in parallel
   - Status: Applications not yet submitted (planned)

### Competitive Risks

1. **SurvdigitizeR improvements**
   - Risk: They may add dotted line detection
   - Mitigation: We're also adding k-medoids, rotation correction
   - Status: Monitoring their GitHub/publications

2. **Benchmark availability**
   - Risk: They may not share test datasets
   - Mitigation: Create our own benchmark from IPD data
   - Status: Planned for Phase 3

---

## SUCCESS METRICS

### Short-term (1 Month)

- [ ] Successfully extract curves from ≥35/40 test PDFs (87.5% success rate)
- [ ] Implement k-medoids clustering
- [ ] Add rotation detection/correction
- [ ] Identify and document top 5 failure modes

### Medium-term (3 Months)

- [ ] Success rate ≥95% on diverse PDFs
- [ ] RMSE <0.015 on manual validation
- [ ] Bayesian IPD reconstruction implemented
- [ ] Comparison paper draft complete

### Long-term (6-12 Months)

- [ ] IPD data access granted (≥100 trials)
- [ ] RMSE <0.01 vs ground truth
- [ ] Published benchmark dataset released
- [ ] Academic paper published
- [ ] PyPI package released

---

## CONCLUSION

**This session established our competitive position**:

1. ✅ **Comprehensive R package analysis** - We know exactly where we stand
2. ✅ **Clear competitive advantages** - Dotted line detection, full automation, batch processing
3. ✅ **Validation infrastructure** - PDF download pipeline ready
4. 🔄 **Real-world testing initiated** - 40 PDFs downloading

**Key insight**: We're already ahead on dotted line detection (critical gap in both competitors). By adding k-medoids clustering and rotation correction, we'll address their other strengths while maintaining our unique capabilities.

**Next critical task**: Run batch processor on 40 test PDFs and analyze results to validate our pipeline on real-world data.

**Timeline**: On track for publication-quality results within 3-6 months, pending IPD data access.
