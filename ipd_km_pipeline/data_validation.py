#!/usr/bin/env python3
"""
Data validation and quality control for extracted K-M curves.

Implements strict validation rules to achieve 95%+ data quality.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of curve validation."""
    is_valid: bool
    quality_score: float  # 0-100
    reason: str
    warnings: list
    curve_direction: str  # 'decreasing', 'increasing', 'mixed'


def validate_curve_data(
    curve_df: pd.DataFrame,
    calibration: Dict,
    min_points: int = 100,
    max_points: int = 500000,
    min_time_range: float = 5.0,
    min_survival_range: float = 0.15
) -> ValidationResult:
    """
    Validate extracted curve data meets quality thresholds.

    Args:
        curve_df: DataFrame with 'time' and 'survival' columns
        calibration: Dict with calibration info
        min_points: Minimum number of points required
        max_points: Maximum points before likely extraction error
        min_time_range: Minimum time span (months)
        min_survival_range: Minimum survival range (0-1 scale)

    Returns:
        ValidationResult with is_valid, quality_score, reason, warnings
    """
    warnings = []
    quality_score = 100.0  # Start at perfect, deduct for issues

    # Get basic statistics
    n_points = len(curve_df)
    time_range = curve_df['time'].max() - curve_df['time'].min()
    survival_range = curve_df['survival'].max() - curve_df['survival'].min()
    time_min = curve_df['time'].min()
    time_max = curve_df['time'].max()
    survival_min = curve_df['survival'].min()
    survival_max = curve_df['survival'].max()

    # CRITICAL CHECKS (immediate rejection)

    # Detect curve direction (increasing, decreasing, or mixed)
    # This is CRITICAL for handling both survival curves (decreasing) and cumulative incidence (increasing)
    survival_changes = curve_df['survival'].diff().dropna()
    n_increases = (survival_changes > 0.01).sum()
    n_decreases = (survival_changes < -0.01).sum()
    n_flat = len(survival_changes) - n_increases - n_decreases

    if n_decreases > n_increases:
        curve_direction = 'decreasing'  # Typical survival curve
    elif n_increases > n_decreases:
        curve_direction = 'increasing'  # Cumulative incidence, disease progression
    else:
        curve_direction = 'mixed'  # Unusual, may be data issue

    # Check 1: Sufficient points
    if n_points < min_points:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Too few points: {n_points} < {min_points}",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 2: Not absurdly many points (probable extraction error)
    if n_points > max_points:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Too many points: {n_points} > {max_points} (likely extraction error)",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 3: Non-zero time range
    if time_range < min_time_range:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Insufficient time range: {time_range:.2f} < {min_time_range} months",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 4: Reasonable survival range
    if survival_range < min_survival_range:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Insufficient survival range: {survival_range:.3f} < {min_survival_range}",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 5: Survival strictly bounded [0, 1]
    if survival_min < -0.01:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Survival below 0: min={survival_min:.3f}",
            warnings=warnings,
            curve_direction=curve_direction
        )

    if survival_max > 1.01:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Survival above 1: max={survival_max:.3f}",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 6: Time starts near 0
    if time_min < -1.0:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Negative start time: {time_min:.2f}",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # Check 7: Time monotonic (with small tolerance for noise)
    time_diffs = curve_df['time'].diff().dropna()
    non_monotonic_count = (time_diffs < -0.5).sum()  # Allow 0.5 month noise
    if non_monotonic_count > 0:
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            reason=f"Non-monotonic time: {non_monotonic_count} backwards steps",
            warnings=warnings,
            curve_direction=curve_direction
        )

    # QUALITY SCORING (for valid curves)

    # Deduct for borderline time range
    if time_range < 12.0:
        quality_score -= 10
        warnings.append(f"Short time range: {time_range:.1f} months")

    # Deduct for borderline survival range
    if survival_range < 0.3:
        quality_score -= 10
        warnings.append(f"Low survival range: {survival_range:.2f}")

    # Deduct for overly dense curves (need decimation)
    if n_points > 10000:
        quality_score -= 20
        warnings.append(f"Very dense curve: {n_points} points (recommend decimation)")
    elif n_points > 5000:
        quality_score -= 10
        warnings.append(f"Dense curve: {n_points} points")

    # Deduct for sparse curves
    if n_points < 200:
        quality_score -= 5
        warnings.append(f"Sparse curve: {n_points} points")

    # Check if using fallback calibration
    if calibration.get('fallback', False):
        quality_score -= 5
        warnings.append("Using fallback calibration (OCR failed)")

    # Bonus for good coverage
    if time_range > 36 and survival_range > 0.5:
        quality_score += 5

    # Ensure score is in [0, 100]
    quality_score = max(0.0, min(100.0, quality_score))

    return ValidationResult(
        is_valid=True,
        quality_score=quality_score,
        reason="Valid" if quality_score >= 80 else "Valid but low quality",
        warnings=warnings,
        curve_direction=curve_direction
    )


def decimate_curve(
    curve_df: pd.DataFrame,
    target_points: int = 1000,
    preserve_steps: bool = True
) -> pd.DataFrame:
    """
    Reduce curve density using Douglas-Peucker-like algorithm.

    Args:
        curve_df: DataFrame with 'time' and 'survival' columns
        target_points: Target number of points
        preserve_steps: Preserve sharp vertical steps (K-M characteristic)

    Returns:
        Decimated DataFrame
    """
    n_points = len(curve_df)

    # If already small enough, return as-is
    if n_points <= target_points:
        return curve_df.copy()

    # Simple uniform sampling for now (can be improved with Douglas-Peucker)
    step = max(1, n_points // target_points)
    decimated_indices = list(range(0, n_points, step))

    # Always include first and last point
    if 0 not in decimated_indices:
        decimated_indices.insert(0, 0)
    if (n_points - 1) not in decimated_indices:
        decimated_indices.append(n_points - 1)

    # If preserving steps, add points where survival drops sharply
    if preserve_steps and len(curve_df) > 2:
        survival_diffs = curve_df['survival'].diff().abs()
        large_steps = survival_diffs[survival_diffs > 0.05].index.tolist()

        # Add these critical points
        for idx in large_steps:
            if idx not in decimated_indices:
                decimated_indices.append(idx)

    # Sort and remove duplicates
    decimated_indices = sorted(set(decimated_indices))

    return curve_df.iloc[decimated_indices].reset_index(drop=True)


def filter_curves(
    curves_dir: str,
    output_dir: str,
    min_quality_score: float = 70.0,
    apply_decimation: bool = True
) -> Dict:
    """
    Filter and clean extracted curves based on quality thresholds.

    Args:
        curves_dir: Directory with raw curve CSVs
        output_dir: Directory for filtered/cleaned curves
        min_quality_score: Minimum quality score (0-100) to keep
        apply_decimation: Whether to decimate overly dense curves

    Returns:
        Dict with statistics
    """
    from pathlib import Path

    curves_path = Path(curves_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    curve_files = list(curves_path.glob("*.csv"))

    stats = {
        'total_curves': len(curve_files),
        'valid_curves': 0,
        'rejected_curves': 0,
        'decimated_curves': 0,
        'rejection_reasons': {}
    }

    for curve_file in curve_files:
        # Load curve
        df = pd.read_csv(curve_file)

        # Dummy calibration (replace with actual from results.json)
        calibration = {'fallback': True}

        # Validate
        result = validate_curve_data(df, calibration)

        if not result.is_valid:
            stats['rejected_curves'] += 1
            reason = result.reason
            stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
            continue

        if result.quality_score < min_quality_score:
            stats['rejected_curves'] += 1
            stats['rejection_reasons']['Low quality score'] = stats['rejection_reasons'].get('Low quality score', 0) + 1
            continue

        # Valid curve - optionally decimate
        if apply_decimation and len(df) > 2000:
            df = decimate_curve(df, target_points=1000)
            stats['decimated_curves'] += 1

        # Save cleaned curve
        output_file = output_path / curve_file.name
        df.to_csv(output_file, index=False)
        stats['valid_curves'] += 1

    # Calculate percentage
    stats['valid_percentage'] = (stats['valid_curves'] / stats['total_curves'] * 100) if stats['total_curves'] > 0 else 0

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_validation.py <input_curves_dir> <output_curves_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Filtering curves from {input_dir} to {output_dir}...")

    stats = filter_curves(input_dir, output_dir, min_quality_score=70.0, apply_decimation=True)

    print(f"\n{'='*70}")
    print("CURVE FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Total curves: {stats['total_curves']}")
    print(f"Valid curves: {stats['valid_curves']} ({stats['valid_percentage']:.1f}%)")
    print(f"Rejected curves: {stats['rejected_curves']}")
    print(f"Decimated curves: {stats['decimated_curves']}")

    if stats['rejection_reasons']:
        print(f"\nRejection reasons:")
        for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    print(f"{'='*70}\n")
