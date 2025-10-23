#!/usr/bin/env python3
"""
Analyze OCR calibration performance from batch processing results.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

def analyze_ocr_results(results_path):
    """Analyze OCR calibration success rates and quality metrics."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Collect calibration statistics
    calibration_methods = []
    calibration_valid = []
    quality_scores = []
    validation_warnings = []
    n_curves_with_warnings = 0
    n_curves_decimated = 0

    # Analyze each figure and curve
    for pdf_result in results:
        for figure in pdf_result['figures']:
            # Check calibration
            is_valid = figure.get('calibration_valid', False)
            calibration_valid.append(is_valid)

            # Extract calibration method from reason string
            reason = figure.get('calibration_reason', 'unknown')
            if 'hough' in reason.lower():
                method = 'hough_v2'
            elif 'fallback' in reason.lower():
                method = 'fallback'
            else:
                method = 'other'
            calibration_methods.append(method)

            # Analyze curves
            for curve in figure.get('curves', []):
                quality_score = curve.get('quality_score', 0)
                quality_scores.append(quality_score)

                warnings = curve.get('validation_warnings', [])
                if warnings:
                    n_curves_with_warnings += 1
                    validation_warnings.extend(warnings)

                original_points = curve.get('original_points', 0)
                final_points = curve.get('n_points', 0)
                if original_points > final_points:
                    n_curves_decimated += 1

    # Calculate statistics
    total_figures = len(calibration_valid)
    ocr_success = sum(calibration_valid)
    ocr_success_rate = (ocr_success / total_figures * 100) if total_figures > 0 else 0

    method_counts = Counter(calibration_methods)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    min_quality = min(quality_scores) if quality_scores else 0
    max_quality = max(quality_scores) if quality_scores else 0

    warning_counts = Counter(validation_warnings)

    # Print report
    print("="*70)
    print("OCR CALIBRATION PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"\nCALIBRATION STATISTICS:")
    print(f"  Total figures processed: {total_figures}")
    print(f"  OCR successful (v2):     {ocr_success} ({ocr_success_rate:.1f}%)")
    print(f"  OCR fallback:            {method_counts.get('fallback', 0)} ({method_counts.get('fallback', 0)/total_figures*100:.1f}%)")
    print(f"\n  Breakdown by method:")
    for method, count in method_counts.most_common():
        print(f"    - {method}: {count} ({count/total_figures*100:.1f}%)")

    print(f"\nCURVE QUALITY SCORES:")
    print(f"  Total curves: {len(quality_scores)}")
    print(f"  Average quality score: {avg_quality:.1f}/100")
    print(f"  Min quality score: {min_quality:.1f}")
    print(f"  Max quality score: {max_quality:.1f}")
    print(f"  Curves with warnings: {n_curves_with_warnings} ({n_curves_with_warnings/len(quality_scores)*100:.1f}%)")
    print(f"  Curves decimated: {n_curves_decimated} ({n_curves_decimated/len(quality_scores)*100:.1f}%)")

    if warning_counts:
        print(f"\nTOP VALIDATION WARNINGS:")
        for warning, count in warning_counts.most_common(10):
            print(f"    - {warning}: {count}")

    print("="*70)

    # Calculate improvement vs baseline
    baseline_ocr = 0.0  # Original OCR success rate
    improvement = ocr_success_rate - baseline_ocr

    print(f"\nIMPROVEMENT OVER BASELINE:")
    print(f"  Baseline OCR success rate: {baseline_ocr:.1f}%")
    print(f"  Current OCR success rate:  {ocr_success_rate:.1f}%")
    print(f"  Improvement: +{improvement:.1f} percentage points")

    target_ocr = 90.0
    if ocr_success_rate >= target_ocr:
        print(f"  [SUCCESS] TARGET ACHIEVED: >={target_ocr}%")
    else:
        gap = target_ocr - ocr_success_rate
        print(f"  [WARNING] TARGET NOT YET MET: {target_ocr}% (gap: {gap:.1f}%)")

    print("="*70 + "\n")

    return {
        'total_figures': total_figures,
        'ocr_success': ocr_success,
        'ocr_success_rate': ocr_success_rate,
        'avg_quality': avg_quality,
        'n_curves_decimated': n_curves_decimated
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        results_path = Path("batch_results_v2") / "results.json"
    else:
        results_path = Path(sys.argv[1])

    if not results_path.exists():
        print(f"Error: {results_path} not found!")
        sys.exit(1)

    stats = analyze_ocr_results(results_path)
