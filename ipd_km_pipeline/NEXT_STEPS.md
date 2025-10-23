# Next Steps: Advanced Features Implementation Guide

This document provides detailed implementation guidelines for the remaining advanced features.

---

## REMAINING TASKS

### Task 5: Figure Type Classifier
### Task 6: Bayesian IPD Reconstruction
### Task 7: Neural Network Curve Refinement
### Task 8: Data Access Applications

---

## TASK 5: Figure Type Classifier (K-M vs CIF vs RMST)

**Status**: Design complete, implementation pending
**Priority**: MEDIUM
**Estimated effort**: 2-3 days

### Purpose
Automatically classify figure types before extraction to:
- Apply appropriate extraction algorithms
- Handle different y-axis interpretations
- Validate extracted data correctly

### Figure Types

1. **Kaplan-Meier (K-M)** - Current focus
   - Y-axis: Survival probability (0-1)
   - Descending or flat curves
   - Censoring marks (ticks)

2. **Cumulative Incidence Function (CIF)**
   - Y-axis: Cumulative incidence (0-1)
   - **Ascending curves** (opposite of K-M)
   - Competing risks

3. **Restricted Mean Survival Time (RMST)**
   - Y-axis: Mean survival time
   - Area under K-M curve

4. **Hazard Ratio Plots**
   - Y-axis: Hazard ratio
   - Reference line at HR=1

### Implementation Approach

```python
# Pseudo-code structure
class FigureTypeClassifier:
    def classify(self, panel_image, ocr_text):
        features = self.extract_features(panel_image, ocr_text)

        # Rule-based classification
        if "cumulative incidence" in ocr_text.lower():
            return "CIF"
        elif "rmst" in ocr_text.lower() or "restricted mean" in ocr_text.lower():
            return "RMST"
        elif "hazard ratio" in ocr_text.lower():
            return "HR"

        # Curve shape analysis
        curve_trend = self.analyze_curve_trend(panel_image)
        if curve_trend == "ascending":
            return "CIF"
        elif curve_trend == "descending":
            return "KM"

        return "KM"  # Default

    def extract_features(self, panel_image, ocr_text):
        return {
            'has_censoring_marks': self.detect_censoring_marks(panel_image),
            'y_axis_range': self.get_y_axis_range(ocr_text),
            'has_hr_line': self.detect_horizontal_line_at_y1(panel_image),
            'text_features': self.extract_text_features(ocr_text)
        }
```

### Files to Create
- `classification/figure_classifier.py` (~250 lines)
- `classification/features.py` (~150 lines)
- `test_classification.py` (~100 lines)

---

## TASK 6: Bayesian IPD Reconstruction (Improving on Guyot 2012)

**Status**: Research design phase
**Priority**: HIGH
**Estimated effort**: 2-3 weeks

### Background

**Guyot et al. (2012)** "Enhanced secondary analysis of survival data"
- Uses integer-constrained optimization
- Matches digitized K-M curve + numbers-at-risk
- Widely used but has limitations

### Proposed Improvements

1. **Bayesian MCMC Approach**
   - Replace integer constraints with probabilistic sampling
   - Incorporate uncertainty from OCR errors
   - Better handling of tied event times

2. **Hierarchical Modeling**
   - Account for censoring patterns
   - Use informative priors from numbers-at-risk

3. **Ensemble Methods**
   - Combine multiple IPD reconstructions
   - Weight by validation metrics

### Implementation Design

```python
# Using PyMC for Bayesian inference
import pymc as pm
import numpy as np

class BayesianIPD Reconstructor:
    def __init__(self, survival_curve, numbers_at_risk):
        self.curve = survival_curve
        self.n_at_risk = numbers_at_risk

    def build_model(self):
        with pm.Model() as model:
            # Priors
            # Event times for each patient
            n_patients = self.n_at_risk[0]
            event_times = pm.Exponential('event_times', lam=1.0, shape=n_patients)

            # Censoring indicators
            censored = pm.Bernoulli('censored', p=0.3, shape=n_patients)

            # Observed data likelihood
            # Match to extracted curve
            predicted_survival = self.compute_km_from_times(event_times, censored)

            pm.Normal('obs', mu=predicted_survival,
                     sigma=0.01,  # From curve extraction uncertainty
                     observed=self.curve.survival)

            # Numbers-at-risk constraints
            for t_idx, (time, n) in enumerate(self.n_at_risk):
                predicted_n = pm.Deterministic(
                    f'n_at_risk_{t_idx}',
                    pm.math.sum(event_times > time)
                )
                pm.Poisson(f'obs_n_{t_idx}', mu=predicted_n, observed=n)

        return model

    def sample(self, model, n_samples=2000):
        with model:
            trace = pm.sample(n_samples, return_inferencedata=True)
        return trace

    def extract_ipd(self, trace):
        # Extract posterior mean event times
        event_times = trace.posterior['event_times'].mean(dim=['chain', 'draw'])
        censored = trace.posterior['censored'].mean(dim=['chain', 'draw']) > 0.5

        return pd.DataFrame({
            'time': event_times,
            'event': ~censored
        })
```

### Dependencies
- PyMC3 or PyMC (latest)
- ArviZ (diagnostics)
- Theano or JAX (backend)

### Files to Create
- `ipd_reconstruction/bayesian_ipd.py` (~500 lines)
- `ipd_reconstruction/guyot_baseline.py` (~300 lines - original method)
- `ipd_reconstruction/validation.py` (~200 lines)

### Validation Strategy
1. Generate synthetic IPD with known properties
2. Create K-M curves from synthetic IPD
3. Reconstruct IPD and compare to ground truth
4. Metrics: RMSE of event times, concordance index

---

## TASK 7: Neural Network Curve Refinement

**Status**: Awaiting validation dataset
**Priority**: MEDIUM
**Estimated effort**: 3-4 weeks

### Purpose
Use deep learning to:
- Denoise extracted curves
- Predict event times directly from image features
- Ensemble with traditional methods

### Architecture Design

```python
import torch
import torch.nn as nn

class CurveRefinementNet(nn.Module):
    """
    Takes noisy extracted curve + image features
    Outputs refined survival curve
    """
    def __init__(self):
        super().__init__()

        # Image encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256)
        )

        # Curve encoder (LSTM for time series)
        self.curve_encoder = nn.LSTM(
            input_size=2,  # (time, survival)
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Decoder (predicts refined curve)
        self.decoder = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100),  # 100 time points
            nn.Sigmoid()  # Survival probability 0-1
        )

    def forward(self, image, curve_data):
        img_features = self.image_encoder(image)
        curve_features, _ = self.curve_encoder(curve_data)
        curve_features = curve_features[:, -1, :]  # Last hidden state

        combined = torch.cat([img_features, curve_features], dim=1)
        refined_curve = self.decoder(combined)
        return refined_curve

# Training
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            image, noisy_curve, true_curve = batch

            optimizer.zero_grad()
            pred_curve = model(image, noisy_curve)
            loss = criterion(pred_curve, true_curve)
            loss.backward()
            optimizer.step()
```

### Training Data Requirements
- **100-200 trials with real IPD** (from data access applications)
- Generate K-M curves from IPD
- Add controlled noise/artifacts
- Train to denoise and refine

### Files to Create
- `neural/curve_refinement_net.py` (~400 lines)
- `neural/training.py` (~300 lines)
- `neural/data_augmentation.py` (~200 lines)

---

## TASK 8: Data Access Applications

**Status**: Template drafts needed
**Priority**: HIGH (enables tasks 6 & 7)
**Estimated effort**: 1 week drafting, 2-6 months for approvals

### Platforms to Apply

#### 1. Project Data Sphere
**URL**: https://www.projectdatasphere.org/
**Focus**: Oncology trials
**Available**: ~200 trials with IPD

**Application Requirements**:
- Research proposal (2-3 pages)
- Institutional affiliation
- Data use agreement
- Analysis plan

**Draft Proposal**:
```
Title: Validation of Automated K-M Curve IPD Extraction Methods

Background:
- Meta-analysis often requires IPD from published K-M curves
- Guyot (2012) method widely used but unvalidated on large scale
- Need gold-standard dataset for method validation

Objectives:
1. Create benchmark dataset of real IPD + published K-M curves
2. Validate automated extraction methods
3. Compare Bayesian vs integer-constrained reconstruction

Data Needed:
- Survival IPD from oncology trials
- Access to corresponding publications
- 100-200 trials (diverse cancer types)

Analysis Plan:
- Generate K-M curves from IPD
- Extract curves using automated pipeline
- Compare reconstructed IPD to ground truth
- Publish validation benchmark for community use
```

#### 2. Vivli
**URL**: https://vivli.org/
**Focus**: Multi-sponsor platform (7,500+ trials)

#### 3. YODA (Yale)
**URL**: https://yoda.yale.edu/
**Focus**: J&J, Medtronic, SI-BONE data

#### 4. CSDR
**URL**: https://www.clinicalstudydatarequest.com/
**Focus**: GSK, Roche, Bayer

---

## IMPLEMENTATION TIMELINE

### Week 1-2 (Immediate)
- ✅ Complete batch processing
- Draft data access applications
- Start figure type classifier

### Month 1
- Submit data access applications
- Complete figure type classifier
- Test batch processor on 20-50 PDFs

### Month 2-3
- Implement Guyot baseline method
- Design Bayesian IPD reconstruction
- Begin neural network architecture

### Month 4-6 (Pending data access)
- Receive IPD datasets
- Create validation framework
- Train neural networks
- Publish benchmark results

---

## SUCCESS METRICS

1. **Extraction Accuracy**: RMSE < 0.01 vs real IPD
2. **IPD Reconstruction**: Concordance index > 0.95
3. **Batch Processing**: >95% success rate on diverse PDFs
4. **Publication**: Methods paper + validation dataset release

---

## GETTING STARTED (Next Session)

1. Run batch processor on test PDFs
2. Draft Project Data Sphere application
3. Implement figure type classifier
4. Design Bayesian IPD model structure

All core automation is complete. Next phase focuses on validation and advanced algorithms!
