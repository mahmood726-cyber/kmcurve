#!/usr/bin/env python3
"""
Analyze extraction quality and identify improvement opportunities.
"""
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))

# Get absolute paths
pipeline_dir = Path(__file__).parent
test_pdfs_dir = pipeline_dir / 'test_pdfs/medrxiv'
curves_dir = pipeline_dir / 'batch_results/curves'

# Get downloaded vs processed PDFs
downloaded_pdfs = sorted([p.stem for p in test_pdfs_dir.glob('*.pdf')])

curve_files = list(curves_dir.glob('*.csv'))
processed_pdfs = sorted(set([f.stem.rsplit('_p', 1)[0] for f in curve_files]))

unprocessed_pdfs = sorted(set(downloaded_pdfs) - set(processed_pdfs))

print("="*80)
print("K-M CURVE EXTRACTION - QUALITY ANALYSIS")
print("="*80)

print(f"\n1. DETECTION RATE")
print(f"   Downloaded PDFs: {len(downloaded_pdfs)}")
print(f"   Successfully processed: {len(processed_pdfs)} ({len(processed_pdfs)/len(downloaded_pdfs)*100:.1f}%)")
print(f"   No curves detected: {len(unprocessed_pdfs)} ({len(unprocessed_pdfs)/len(downloaded_pdfs)*100:.1f}%)")

print(f"\n2. UNPROCESSED PDFs (need investigation):")
for pdf in unprocessed_pdfs:
    print(f"   - {pdf}")

# Analyze curve quality
print(f"\n3. CURVE QUALITY ANALYSIS")
curve_stats = []
for curve_file in curve_files:
    df = pd.read_csv(curve_file)
    curve_stats.append({
        'file': curve_file.name,
        'n_points': len(df),
        'time_range': df['time'].max() - df['time'].min(),
        'survival_range': df['survival'].max() - df['survival'].min(),
        'time_monotonic': (df['time'].diff().dropna() >= 0).all(),
        'survival_bounded': (df['survival'] >= 0).all() and (df['survival'] <= 1).all()
    })

stats_df = pd.DataFrame(curve_stats)

print(f"\n   Total curves extracted: {len(curve_stats)}")
print(f"   Average points per curve: {stats_df['n_points'].mean():.0f} (median: {stats_df['n_points'].median():.0f})")
print(f"   Points range: {stats_df['n_points'].min():.0f} - {stats_df['n_points'].max():.0f}")

# Quality issues
time_issues = (~stats_df['time_monotonic']).sum()
survival_issues = (~stats_df['survival_bounded']).sum()

print(f"\n4. QUALITY ISSUES DETECTED")
print(f"   Non-monotonic time values: {time_issues} curves ({time_issues/len(curve_stats)*100:.1f}%)")
print(f"   Survival out of bounds [0,1]: {survival_issues} curves ({survival_issues/len(curve_stats)*100:.1f}%)")

# Analyze very short curves (likely errors)
short_curves = stats_df[stats_df['n_points'] < 50]
print(f"   Very short curves (<50 points): {len(short_curves)} curves ({len(short_curves)/len(curve_stats)*100:.1f}%)")

# Load calibration data from results.json
results_path = pipeline_dir / 'batch_results/results.json'
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    calibration_valid = 0
    calibration_total = 0
    for pdf_result in results:
        for figure in pdf_result.get('figures', []):
            calibration_total += 1
            if figure.get('calibration_valid'):
                calibration_valid += 1

    print(f"\n5. AXIS CALIBRATION")
    print(f"   Figures with valid calibration: {calibration_valid}/{calibration_total} ({calibration_valid/calibration_total*100:.1f}%)")
    print(f"   Figures with invalid calibration: {calibration_total - calibration_valid} ({(calibration_total - calibration_valid)/calibration_total*100:.1f}%)")

print(f"\n{'='*80}")
print("IMPROVEMENT PRIORITIES")
print("="*80)

print(f"\n1. INCREASE DETECTION RATE (currently {len(processed_pdfs)/len(downloaded_pdfs)*100:.1f}%)")
print("   - Investigate why 10 PDFs yielded no curves")
print("   - Improve panel detection algorithm")
print("   - Add figure type classification (K-M vs other plots)")

print(f"\n2. IMPROVE AXIS CALIBRATION (currently {calibration_valid/calibration_total*100:.1f}% valid)")
print("   - Enhance OCR accuracy (better preprocessing)")
print("   - Add manual calibration fallback")
print("   - Validate axis labels match expected units")

print(f"\n3. FIX DATA QUALITY ISSUES")
if time_issues > 0:
    print(f"   - Fix {time_issues} curves with non-monotonic time")
if survival_issues > 0:
    print(f"   - Fix {survival_issues} curves with out-of-bounds survival")
if len(short_curves) > 0:
    print(f"   - Investigate {len(short_curves)} curves with <50 points")

print(f"\n4. INCREASE RESOLUTION")
print(f"   - Current DPI: 300 (reduced for speed)")
print(f"   - Optimal DPI: 600-1200 for production")
print(f"   - Test higher resolution on sample PDFs")

print(f"\n5. FIX AUTO-DETECTION PERFORMANCE")
print(f"   - Current: Manually set n_curves=2 (bypassing auto-detect)")
print(f"   - Needed: Add pixel sampling to reduce O(n²) complexity")
print(f"   - This will allow automatic detection of 1, 2, 3+ curves")

print(f"\n{'='*80}\n")

# Save detailed stats
stats_path = pipeline_dir / 'batch_results/curve_quality_stats.csv'
stats_df.to_csv(stats_path, index=False)
print(f"Detailed statistics saved to: {stats_path}")
