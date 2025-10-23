# IPD + KM Extraction & Reconstruction — Claude Code Playbook

**Purpose:** This file gives you step‑by‑step, copy‑pasteable instructions you can run inside Claude Code to build a *zero‑click* pipeline that (1) ingests survival datasets, (2) generates synthetic KM PDFs for training, (3) extracts curves/axes/at‑risk tables from PDF/PNG, and (4) reconstructs near‑IPD event/censor schedules with an integer‑constrained solver. It also includes a benchmark plan and a manifest format so you can scale to thousands of figures.

---

## 0) Project overview (what you’re building)

- A deterministic **PDF→image fallback ladder** (vector ⇒ native image ⇒ hi‑DPI render).
- A **layout detector** to segment panels, axes, legend, and numbers‑at‑risk table.
- Dual OCR: **digits‑only** (ticks/at‑risk) + **general** (labels/legend/units).
- **Curve vectorizer** with KM shape priors (non‑increasing, right‑continuous steps).
- **Censor marker** detection and association to curves.
- **Integer‑constrained reconstruction** (events/censors per interval; Efron ties; joint multi‑arm).
- **QC & retry ladder** to automatically pick the most reliable output.
- **Batch CLI** + artifacts (JSON/CSV/SVG) + **manifest** for provenance and licensing.

---

## 1) Repository scaffold (create this structure exactly)
```bash
mkdir -p ipd_km_pipeline/{pdf_io,raster_cv,layout,ocr,curve_vec,atrisk,reconstruct,qc,cli,benchmarks,synthetic,data,artifacts,scripts,docs,configs}
cd ipd_km_pipeline
```

Create **module stubs**:

```bash
touch pdf_io/__init__.py pdf_io/extract.py \
      raster_cv/__init__.py raster_cv/preproc.py raster_cv/vectorize.py \
      layout/__init__.py layout/detect.py \
      ocr/__init__.py ocr/digits.py ocr/general.py \
      curve_vec/__init__.py curve_vec/km_priors.py \
      atrisk/__init__.py atrisk/table_detect.py atrisk/parse.py \
      reconstruct/__init__.py reconstruct/solver.py reconstruct/metrics.py \
      qc/__init__.py qc/scores.py qc/retry.py \
      cli/__init__.py cli/main.py \
      scripts/make_manifest.py scripts/run_batch.py \
      configs/ocr_digits.toml configs/ocr_general.toml \
      benchmarks/README.md synthetic/README.md docs/README.md
```
---

## 2) Environment (Dockerfile + Makefile)

**Dockerfile** (save as `Dockerfile` in repo root):
```dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Optional: add PaddleOCR models or additional tesseract languages here

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    PyMuPDF==1.24.9 \
    opencv-python-headless==4.10.0.84 \
    pytesseract==0.3.10 \
    pillow==10.4.0 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    scikit-image==0.24.0 \
    layoutparser==0.3.4 \
    shapely==2.0.4 \
    pandas==2.2.2 \
    matplotlib==3.9.1 \
    seaborn==0.13.2 \
    tqdm==4.66.4 \
    pydantic==2.9.0 \
    orjson==3.10.7 \
    pyyaml==6.0.2 \
    rich==13.7.1 \
    pulp==2.8.0

CMD ["bash"]
```

**Makefile** (save as `Makefile`):

```makefile
.PHONY: build shell test bench run

build:
\tdocker build -t ipd-km:latest .

shell:
\tdocker run --rm -it -v $(PWD):/app ipd-km:latest bash

test:
\tpytest -q || true

run:
\tpython -m cli.main

bench:
\tpython scripts/run_batch.py --input benchmarks/list.csv --out artifacts/
```
---

## 3) Datasets to download (public IPD + training/benchmark sets)

> **Note:** Keep raw data outside the repo (e.g., `/data` folder mounted into Docker). Always store a MANIFEST row for each dataset with URL/DOI, license, and fetch date.

### 3.1 Classic open IPD (R/scikit textbooks)

- **KMsurv package** (survival datasets): download via CRAN; keep CSV exports.
  - Example names: `larynx`, `bladder`, `kidney`, `tongue`, `bmt`, `hodg`, `leukemia`, etc.
- **R `survival` datasets**: `veteran`, `ovarian`, `lung` (NCCTG), `rotterdam`.
- **GBSG2** (TH.data R package) and **Rotterdam** (survival package).
- **WHAS 100/500** (Worcester Heart Attack Study): in R and scikit-survival.
- **SUPPORT / SUPPORT2** (UCI repository).

**Action:** Export each to CSV/Parquet with columns:
```
study_id, subject_id, arm, time_start, time_end, event, endpoint, sex, age, stage, biomarker, source_url
```

### 3.2 Pan-cancer clinical IPD (large)

- **TCGA Clinical Data Resource (CDR)**: download curated endpoints (OS, DSS, DFI, PFI).
- **cBioPortal**: use the API to download clinical + survival for chosen studies; store `pmid|doi`.

### 3.3 Hospital/ICU

- **MIMIC‑IV**: after credentialing, export cohorts with time‑to‑death/discharge/ventilation.

### 3.4 Oncology trials platforms (registration required, but public sign‑up)

- **Project Data Sphere**: sign up and download trial datasets with survival endpoints; store DUA text in manifest.

### 3.5 Reconstructed (optional bridge)

- **KMDATA** and **Sage Synapse (syn25813713)**: keep clearly labeled as “reconstructed”.

Place all CSVs/Parquets under `/data/{source}/{study}/` and add rows to `data/MANIFEST.csv` (see §9).
---

## 4) Synthetic KM PDF generator (perfect ground truth)

Create `synthetic/generate.py`:
```python
import numpy as np, pandas as pd, json, random, os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def gen_one(outdir, seed=None, n=400, n_arms=2, scale=20.0, event_p=0.7):
    rng = np.random.default_rng(seed)
    arms = [f"A{i+1}" for i in range(n_arms)]
    df = pd.DataFrame({
        "time": rng.exponential(scale, n),
        "event": rng.binomial(1, event_p, n),
        "arm": rng.choice(arms, n)
    })
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    truth = {}
    for name, g in df.groupby("arm"):
        km = KaplanMeierFitter()
        km.fit(g.time, g.event, label=name)
        ax.step(km.survival_function_.index, km.survival_function_[name],
                where="post", linestyle=random.choice(["-","--","-.",":"]))
        truth[name] = {
            "time": km.survival_function_.index.tolist(),
            "S": km.survival_function_[name].tolist()
        }
    ax.set_xlabel(random.choice(["Time (months)","Follow-up (days)"]))
    ax.set_ylabel(random.choice(["Survival probability","Proportion alive"]))
    ax.grid(random.choice([True, False]), which=random.choice(["major","both"]))
    os.makedirs(outdir, exist_ok=True)
    base = f"km_{rng.integers(10**9)}"
    pdf_path = os.path.join(outdir, base + ".pdf")
    png_path = os.path.join(outdir, base + ".png")
    json_path = os.path.join(outdir, base + "_truth.json")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    with open(json_path, "w") as f:
        json.dump(truth, f)
    plt.close(fig)
    return pdf_path, png_path, json_path

if __name__ == "__main__":
    for i in range(1000):
        gen_one("synthetic/out", seed=i)
```

Run inside Docker shell:
```bash
make build
make shell
python synthetic/generate.py
```
---

## 5) PDF→image fallback ladder (pdf_io/extract.py)

- Try **vector**: PyMuPDF `page.get_drawings()` and `page.get_text("rawdict")` for text bboxes.
- Try **native image**: `page.get_images(full=True)` → `doc.extract_image(xref)`.
- Fallback **render**: Poppler `pdftocairo` at **600–1200 DPI** or PyMuPDF pixmap (Matrix(4,4)).
- Force RGB/GRAY (avoid CMYK), `alpha=False`.
- Return `[panel_images, meta]` with stable hashes.

---

## 6) Layout detection (layout/detect.py)

- Rule‑based to start: detect axes with Hough; panel is the interior.
- Find **numbers‑at‑risk table** under the x‑axis (look for grid lines/whitespace gutters).
- Optional: train a layout detector later (layoutparser/Detectron2) on synthetic labels.
---

## 7) OCR (ocr/digits.py, ocr/general.py)

- **Digits‑only**: Tesseract whitelist `0123456789.+-`, `--psm 7` for ticks & at‑risk.
- **General**: standard OCR for titles/labels/legend.
- At‑risk pipeline: Sauvola binarization → grid detection → per‑cell OCR → coerce to int + confidence → **row‑wise monotonic repair**.
---

## 8) Curve vectorization (raster_cv/preproc.py, raster_cv/vectorize.py)

- Preprocess: bilateral denoise, deskew (Hough), mask text/axes.
- Separate curves: cluster in Lab + (dx,dy).
- Edge → skeletonize → polyline tracing.
- **Snap to KM** (non‑increasing right‑continuous steps).
- Censor markers: classify `+ × • ◦ |`, assign to nearest curve, subtract before snapping.
- Legend mapping: style descriptor (color/dash/marker) ⇒ curve ↔ legend entry.
---

## 9) Manifest (data/MANIFEST.csv)

Headers:
```
study_id,source,source_url,license,fetch_date,endpoint,subjects,events,notes,pmid_doi,km_fig_ref,at_risk_available,synthetic,vector_pdf
```
Add a row per dataset/figure with provenance and licensing.
---

## 10) Axis calibration & units

- Map pixel↔time and pixel↔S from ticks; detect linear vs log.
- Snap to y=0 and y=1 gridlines when visible.
- Infer units (days/months/years) from x‑label; store multiplier.
- Export per‑stratum arrays: `time, S, n_risk, n_event, n_censor` (+ optional CI).
---

## 11) Integer‑constrained reconstruction (reconstruct/solver.py)

- Variables per interval `i`: events `E_i`, censors `C_i` (integers).
- Constraints:
  - `n_{i+1} = n_i - E_i - C_i`
  - step size ↔ `E_i / n_i` (Efron tie correction), with tolerance bands.
  - Totals and at‑risk recursion consistency.
- Objective: maximize KM likelihood or minimize deviation from S(t); small TV penalty on hazard.
- **Joint multi‑arm** fit so totals align across strata.
- Output: pseudo‑IPD or exact **event schedule** that re‑renders the KM.
---

## 12) QC & retry ladder (qc/scores.py, qc/retry.py)

- Scores: curve RMSE, axis residual, at‑risk monotonic pass/fail, censor plausibility.
- Retry presets:
  - DPI: 300 → 600 → 1200
  - Binarizer: Sauvola ↔ Wolf
  - OCR psm: 6 ↔ 7
- Choose the run with **highest composite QC**; flag low‑confidence strata.
---

## 13) CLI & batch (cli/main.py, scripts/run_batch.py)

Example commands:
```bash
python -m cli.main extract --in path/to.pdf --out artifacts/FIG123/
python -m cli.main batch  --list benchmarks/list.csv --out artifacts/
python -m cli.main recon  --in artifacts/FIG123/ --out artifacts/FIG123/recon.json
```

`benchmarks/list.csv`:
```
pdf_path,study_id,endpoint,notes
/path/to/figure1.pdf,TRIAL001,OS,"NEJM 2018 Fig 2A"
```
---

## 14) Benchmarks & acceptance targets

**Synthetic validation:**
- Panel IoU ≥ 0.98
- Tick mapping MAE ≤ 2 px
- Curve RMSE ≤ 0.02
- At‑risk digit accuracy ≥ 0.99

**Real mixed‑journal set (200 figs):**
- Zero‑click success ≥ 80–90%
- Re‑rendered KM deviation ≤ 0.01 for ≥ 70–80% strata
- At‑risk row monotonic validity ≥ 95%

**Reconstruction vs baselines:**
- Lower HR bias and median error than classic methods on trials with true IPD.
---

## 15) Example manifest rows

```
study_id,source,source_url,license,fetch_date,endpoint,subjects,events,notes,pmid_doi,km_fig_ref,at_risk_available,synthetic,vector_pdf
ROTTERDAM,CRAN-survival,https://cran.r-project.org/package=survival,LGPL,2025-10-22,OS,298,NA,"Classic breast cancer cohort",10.1007/BF00198734,Fig2A,true,false,true
GBSG2,CRAN-TH.data,https://cran.r-project.org/package=TH.data,GPL-2,2025-10-22,DFS,686,NA,"German Breast Cancer Study",10.1002/sim.4780030211,Fig3,false,false,true
SUPPORT,UCI,https://archive.ics.uci.edu/,CC-BY,2025-10-22,1y-mortality,9105,NA,"Prognostic cohort","",NA,false,false,NA
```
---

## 16) Quick start

```bash
make build
make shell
python synthetic/generate.py
python -m cli.main extract --in synthetic/out/<one_pdf>.pdf --out artifacts/sample/
python scripts/run_batch.py --input benchmarks/list.csv --out artifacts/
python -m cli.main recon --in artifacts/sample/ --out artifacts/sample/recon.json
```

**Remember:** keep raws immutable, record provenance in MANIFEST, prefer Efron ties, and always emit QC scores.
