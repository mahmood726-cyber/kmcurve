#!/usr/bin/env python3
"""
Batch processing script for K-M curve extraction from multiple PDFs.

Processes entire directories of PDFs in parallel, extracting all K-M curves
and generating a consolidated dataset.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from pdf_io.extract import extract_images_from_pdf
from layout.detect import detect_panels
from raster_cv.extract import extract_curves, extract_curves_knn, extract_curves_morphological
from ocr.axis_reader import auto_calibrate_axes, validate_axis_calibration, get_fallback_calibration
from ocr.axis_reader_v2 import auto_calibrate_axes_v2, validate_axis_calibration_v2, get_fallback_calibration_v2
from ocr.numbers_at_risk import parse_at_risk_table, validate_at_risk_data
from ocr.generic_pdf_text_extractor import extract_axes_generic, AxisInfo
from data_validation import validate_curve_data, decimate_curve
import numpy as np
from sklearn.isotonic import IsotonicRegression


def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


def consolidate_curve_pixels(
    points_xy: np.ndarray,
    method: str = 'median',
    bin_width: int = 1
) -> np.ndarray:
    """
    Consolidate scattered curve pixels into a clean function (one y per x).

    K-M curve extraction produces thousands of scattered pixels from anti-aliasing,
    line width, etc. This consolidates them into a proper step function.

    Args:
        points_xy: Array of (x, y) pixel coordinates (N x 2)
        method: 'median' (robust to outliers), 'mean', or 'mode'
        bin_width: Width of x-bins in pixels (default 1)

    Returns:
        Clean curve as (x, y) array with ~one point per x-coordinate
    """
    if len(points_xy) == 0 or len(points_xy) < 2:
        return points_xy

    # Sort by x coordinate
    sorted_idx = np.argsort(points_xy[:, 0])
    sorted_points = points_xy[sorted_idx]

    # Determine x range
    x_min = int(np.floor(sorted_points[0, 0]))
    x_max = int(np.ceil(sorted_points[-1, 0]))

    curve_points = []

    # Group by x-coordinate bins and consolidate y-values
    for x_bin in range(x_min, x_max + 1, bin_width):
        # Get all points in this x-bin
        in_bin = (sorted_points[:, 0] >= x_bin) & (sorted_points[:, 0] < x_bin + bin_width)
        bin_points = sorted_points[in_bin]

        if len(bin_points) == 0:
            continue

        # Consolidate y-values to single representative value
        x_center = x_bin + bin_width / 2.0

        if method == 'median':
            # Median is robust to outliers (anti-aliasing, noise)
            y_value = np.median(bin_points[:, 1])
        elif method == 'mean':
            y_value = np.mean(bin_points[:, 1])
        elif method == 'mode':
            # Most common y-value (for discrete steps)
            y_rounded = np.round(bin_points[:, 1]).astype(int)
            if len(y_rounded) > 0:
                y_value = float(np.bincount(y_rounded).argmax())
            else:
                y_value = bin_points[0, 1]
        else:
            y_value = np.median(bin_points[:, 1])

        curve_points.append([x_center, y_value])

    return np.array(curve_points) if curve_points else points_xy


def convert_axis_info_to_calibration(x_axis: Optional[AxisInfo], y_axis: Optional[AxisInfo]) -> Dict:
    """
    Convert AxisInfo objects from generic_pdf_text_extractor to calibration dict format.

    Args:
        x_axis: X-axis information (or None if extraction failed)
        y_axis: Y-axis information (or None if extraction failed)

    Returns:
        Calibration dict in the format expected by batch_processor
    """
    # Use default fallback values if extraction failed
    if x_axis is None:
        x_range = (0, 60)
        x_unit = 'months'
        x_label = 'Time (months)'
        x_confidence = 0.0
        x_method = 'fallback'
    else:
        x_range = (x_axis.min_value, x_axis.max_value)
        x_unit = x_axis.unit
        x_label = x_axis.label
        x_confidence = x_axis.confidence
        x_method = x_axis.method

    if y_axis is None:
        y_range = (0, 1)
        y_unit = 'probability'
        y_label = 'Survival probability'
        y_confidence = 0.0
        y_method = 'fallback'
    else:
        y_range = (y_axis.min_value, y_axis.max_value)
        y_unit = y_axis.unit
        y_label = y_axis.label
        y_confidence = y_axis.confidence
        y_method = y_axis.method

    combined_confidence = (x_confidence + y_confidence) / 2.0

    return {
        'x_range': x_range,
        'y_range': y_range,
        'x_unit': x_unit,
        'y_unit': y_unit,
        'x_label': x_label,
        'y_label': y_label,
        'x_confidence': x_confidence,
        'y_confidence': y_confidence,
        'combined_confidence': combined_confidence,
        'x_method': x_method,
        'y_method': y_method
    }


def clean_curve_with_isotonic(df: pd.DataFrame, subsample: int = 5, min_step: float = 0.01) -> pd.DataFrame:
    """
    Clean noisy curve data using isotonic regression to enforce monotonicity.

    This addresses the issue where pixel extraction captures multiple sources
    (main curve + confidence bands/grid lines), creating non-monotonic noise.

    Args:
        df: DataFrame with 'time' and 'survival' columns
        subsample: Subsampling factor (higher = more smoothing)
        min_step: Minimum survival change to keep (removes near-duplicates)

    Returns:
        Cleaned DataFrame with monotonic decreasing survival
    """
    if len(df) < 3:
        return df

    # Sort by time
    df_sorted = df.sort_values('time').reset_index(drop=True)
    t = df_sorted['time'].values
    s = df_sorted['survival'].values

    # Subsample to reduce noise
    if subsample > 1 and len(t) > subsample:
        t = t[::subsample]
        s = s[::subsample]

    # Apply isotonic regression (decreasing)
    try:
        iso = IsotonicRegression(increasing=False)
        s_clean = iso.fit_transform(t, s)

        # Clip to valid range [0, 1]
        s_clean = np.clip(s_clean, 0, 1)

        # Remove near-duplicate points (keep only significant steps)
        unique_t = []
        unique_s = []
        prev_s = None

        for time, surv in zip(t, s_clean):
            if prev_s is None or abs(surv - prev_s) > min_step:
                unique_t.append(time)
                unique_s.append(surv)
                prev_s = surv

        # Return cleaned dataframe
        return pd.DataFrame({
            'time': np.array(unique_t),
            'survival': np.array(unique_s)
        })

    except Exception as e:
        # If cleaning fails, return original sorted data
        print(f"Warning: Isotonic cleaning failed: {e}")
        return df_sorted


class BatchProcessor:
    """Batch processor for K-M curve extraction."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        dpi: int = 600,
        n_workers: int = None,
        use_knn_tracing: bool = False,
        use_morphological: bool = False,
        remove_grid: bool = False
    ):
        """
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output data
            dpi: Resolution for PDF rendering
            n_workers: Number of parallel workers (default: CPU count - 1)
            use_knn_tracing: Use k-NN tracing algorithm (SurvdigitizeR method)
            use_morphological: Use morphological line detection (Canny + contours)
            remove_grid: Remove grid lines before extraction (recommended for R plots)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.use_knn_tracing = use_knn_tracing
        self.use_morphological = use_morphological
        self.remove_grid = remove_grid

        # Morphological takes precedence if both are set
        if self.use_morphological and self.use_knn_tracing:
            print("⚠️  Both --use-morphological and --use-knn specified. Using morphological.")
            self.use_knn_tracing = False

        # Create output directories
        (self.output_dir / 'curves').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    def discover_pdfs(self, pattern: str = "**/*.pdf") -> List[Path]:
        """Find all PDF files in input directory and subdirectories."""
        pdfs = list(self.input_dir.glob(pattern))
        print(f"Found {len(pdfs)} PDF files")
        return pdfs

    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF file."""
        try:
            results = {
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.stem,
                'status': 'success',
                'figures': [],
                'errors': []
            }

            # Try all pages
            pdf_file = str(pdf_path)
            page_num = 0

            while True:
                try:
                    # Extract page
                    page_results = extract_images_from_pdf(
                        pdf_path=pdf_file,
                        page_num=page_num,
                        dpi=self.dpi,
                        force_render=False
                    )

                    if not page_results:
                        break  # No more pages

                    page_image = page_results[0]['image']

                    # Detect K-M panels
                    panels = detect_panels(page_image)

                    # Fallback: If no panels detected, use full page as single panel
                    # This handles simple R plots without borders
                    if not panels or len(panels) == 0:
                        width, height = page_image.size
                        panels = [{
                            'bbox': (0, 0, width, height),
                            'confidence': 0.5,
                            'axes': {'x_axis': None, 'y_axis': None},
                            'at_risk_region': None
                        }]

                    for panel_idx, panel in enumerate(panels):
                        figure_id = f"{pdf_path.stem}_p{page_num+1}_f{panel_idx+1}"

                        try:
                            # Extract panel
                            x, y, w, h = panel['bbox']
                            panel_img = page_image.crop((x, y, x + w, y + h))

                            # Extract curves FIRST (needed for heuristic calibration validation)
                            # TEMP FIX: Manually specify n_curves=2 to avoid slow auto-detect
                            # TODO: Add pixel sampling or limit to auto_detect_n_curves()
                            if self.use_morphological:
                                # Use morphological line detection (Canny + contours)
                                # Most accurate - detects by STRUCTURE not color
                                print("  Using morphological curve detection (Canny + contours)...")
                                curves = extract_curves_morphological(
                                    panel_img,
                                    n_curves=2,  # Manually set to speed up
                                    min_curve_length=100,
                                    exclude_dotted=True
                                )
                            elif self.use_knn_tracing:
                                # Use k-NN tracing algorithm (SurvdigitizeR method)
                                # This is more accurate but slower
                                print("  Using k-NN curve tracing (SurvdigitizeR method)...")
                                curves = extract_curves_knn(
                                    panel_img,
                                    n_curves=2,  # Manually set to speed up
                                    k_neighbors=20,  # SurvdigitizeR default
                                    exclude_dotted=True
                                )
                            else:
                                # Use original color clustering method
                                curves = extract_curves(
                                    panel_img,
                                    n_curves=2,  # Manually set to speed up (most K-M plots have 2 curves)
                                    exclude_dotted=True,
                                    remove_grid=True  # ALWAYS remove grid lines (was: self.remove_grid)
                                )

                            if not curves:
                                continue

                            # CRITICAL FIX 1: Remove edge pixels (axes, labels) BEFORE consolidation
                            # Axes are always at panel edges, curves are in the interior
                            # This prevents axes/grid noise from contaminating curve data
                            edge_margin = 30  # pixels from panel edge
                            for curve in curves:
                                points = curve['points']

                                # Filter out pixels near panel edges
                                interior_mask = (
                                    (points[:, 0] >= edge_margin) &  # Not too far left
                                    (points[:, 0] < w - edge_margin) &  # Not too far right
                                    (points[:, 1] >= edge_margin) &  # Not too far top
                                    (points[:, 1] < h - edge_margin)  # Not too far bottom
                                )

                                curve['points'] = points[interior_mask]
                                curve['n_points'] = len(curve['points'])

                            # CRITICAL FIX 2: Consolidate scattered pixels to curve functions
                            # This converts clouds of pixels into clean step functions
                            # Must be done AFTER dotted filter and edge filtering, BEFORE calibration
                            # NOTE: k-NN and morphological already produce cleaner curves, but consolidation still helps
                            if not self.use_knn_tracing and not self.use_morphological:
                                # Only consolidate for original HSL clustering method
                                for curve in curves:
                                    curve['points'] = consolidate_curve_pixels(curve['points'])
                                    curve['n_points'] = len(curve['points'])

                            # Collect all curve pixels for heuristic calibration
                            # This allows heuristic method to validate against actual curve data
                            all_curve_pixels = None
                            if curves and len(curves) > 0:
                                # Use first curve for validation (usually most reliable)
                                all_curve_pixels = curves[0]['points']

                            # HYBRID AXIS CALIBRATION (PDF text → OCR fallback)
                            # Priority 1: Try PDF text extraction (100% accurate for vector PDFs, instant)
                            # Priority 2: Fall back to OCR (Hough + multi-OCR + heuristics)
                            calibration = None
                            is_valid = False
                            reason = ""

                            try:
                                # Try PDF text extraction first (generic extractor works with ANY PDF)
                                x_axis, y_axis = extract_axes_generic(pdf_file, page_num)

                                # DEBUG: Log what PDF text extraction returned
                                pdf_name = pdf_file.name if hasattr(pdf_file, 'name') else Path(pdf_file).name
                                print(f"\n=== DEBUG CALIBRATION for {pdf_name} page {page_num} ===")
                                print(f"PDF text extraction returned:")
                                print(f"  X-axis: {x_axis}")
                                print(f"  Y-axis: {y_axis}")
                                if x_axis:
                                    print(f"  X confidence: {x_axis.confidence:.2f}, range: {x_axis.min_value}-{x_axis.max_value}")
                                if y_axis:
                                    print(f"  Y confidence: {y_axis.confidence:.2f}, range: {y_axis.min_value}-{y_axis.max_value}")

                                # Check if we got both axes with reasonable confidence
                                if x_axis and y_axis and x_axis.confidence >= 0.5 and y_axis.confidence >= 0.5:
                                    # Convert AxisInfo objects to calibration dict
                                    calibration = convert_axis_info_to_calibration(x_axis, y_axis)
                                    is_valid = True
                                    reason = f"PDF text extraction: x={calibration['x_range']}, y={calibration['y_range']}"
                                    print(f"  >>> USING PDF TEXT CALIBRATION: {reason}")
                                else:
                                    # PDF text extraction failed or low confidence
                                    calibration = None
                                    print(f"  >>> PDF text extraction LOW CONFIDENCE, will try OCR fallback")
                            except Exception as e:
                                # PDF text extraction failed, will try OCR
                                calibration = None
                                print(f"  >>> PDF text extraction EXCEPTION: {type(e).__name__}: {str(e)}")
                                import traceback
                                traceback.print_exc()

                            # If PDF text extraction failed, fall back to OCR
                            if calibration is None:
                                print(f"  >>> Falling back to OCR calibration...")
                                try:
                                    calibration = auto_calibrate_axes_v2(
                                        panel_img,
                                        panel['bbox'],
                                        curve_pixels=all_curve_pixels,  # Pass curve pixels for heuristic validation
                                        pdf_path=pdf_file,  # Pass PDF path for axis_reader_v2's PDF text extraction
                                        page_num=page_num
                                    )
                                    is_valid, reason = validate_axis_calibration_v2(calibration)
                                    print(f"  >>> OCR calibration: x={calibration['x_range']}, y={calibration['y_range']}, valid={is_valid}")

                                    # If validation fails, use safe fallback
                                    if not is_valid:
                                        calibration = get_fallback_calibration_v2()
                                        is_valid = False
                                        reason = f"OCR failed ({reason}), using fallback: x={calibration['x_range']}, y={calibration['y_range']}"
                                        print(f"  >>> USING FALLBACK: {reason}")
                                except Exception as e:
                                    # If OCR completely fails, use safe fallback
                                    calibration = get_fallback_calibration_v2()
                                    is_valid = False
                                    reason = f"OCR error ({str(e)}), using fallback"
                                    print(f"  >>> USING FALLBACK due to exception: {reason}")

                            # Parse at-risk table if available
                            at_risk_data = None
                            if 'at_risk_region' in panel:
                                try:
                                    at_risk_data = parse_at_risk_table(
                                        page_image,
                                        panel['at_risk_region']
                                    )
                                except Exception as e:
                                    pass  # Continue even if at-risk parsing fails

                            # Save curve data
                            figure_data = {
                                'figure_id': figure_id,
                                'page': page_num + 1,
                                'panel_index': panel_idx,
                                'n_curves': len(curves),
                                'calibration': calibration,
                                'calibration_valid': is_valid,
                                'calibration_reason': reason,
                                'curves': []
                            }

                            for curve in curves:
                                # Transform to real coordinates
                                times = calibration['x_range'][0] + \
                                    (curve['points'][:, 0] / w) * \
                                    (calibration['x_range'][1] - calibration['x_range'][0])

                                survival = calibration['y_range'][1] - \
                                    (curve['points'][:, 1] / h) * \
                                    (calibration['y_range'][1] - calibration['y_range'][0])

                                # Create DataFrame
                                df = pd.DataFrame({
                                    'time': times,
                                    'survival': survival
                                })

                                # Apply isotonic cleaning to remove noise and enforce monotonicity
                                # This addresses pixel extraction capturing confidence bands/grid lines
                                original_points = len(df)
                                df = clean_curve_with_isotonic(df, subsample=3, min_step=0.005)

                                # Validate curve quality
                                # Use relaxed min_points for cleaned curves (they're sparse but high-quality)
                                min_points_threshold = 5 if len(df) < 50 else 100
                                validation_result = validate_curve_data(
                                    df,
                                    calibration,
                                    min_points=min_points_threshold,
                                    max_points=500000,
                                    min_time_range=5.0,
                                    min_survival_range=0.15
                                )

                                # Only save valid curves with quality score ≥70
                                if not validation_result.is_valid or validation_result.quality_score < 70:
                                    results['errors'].append({
                                        'figure_id': figure_id,
                                        'curve_id': curve['curve_id'],
                                        'error': 'Curve failed validation',
                                        'reason': validation_result.reason,
                                        'quality_score': validation_result.quality_score,
                                        'warnings': validation_result.warnings
                                    })
                                    continue

                                # Apply decimation if curve is too dense (>1000 points)
                                # CRITICAL FIX: Lower threshold from 2000 to 1000, and target from 1000 to 500
                                # This ensures 50k-100k point curves are reduced to manageable size
                                # For very dense curves (>10k), disable preserve_steps to prevent re-adding too many points
                                original_points = len(df)
                                if original_points > 1000:
                                    preserve_steps = original_points < 10000  # Only preserve steps for moderately dense curves
                                    df = decimate_curve(df, target_points=500, preserve_steps=preserve_steps)

                                # Save validated and optimized curve
                                csv_path = self.output_dir / 'curves' / f"{figure_id}_curve{curve['curve_id']+1}.csv"
                                df.to_csv(csv_path, index=False)

                                figure_data['curves'].append({
                                    'curve_id': curve['curve_id'],
                                    'n_points': len(df),  # After decimation
                                    'original_points': original_points,  # Before decimation
                                    'quality_score': validation_result.quality_score,
                                    'csv_path': str(csv_path),
                                    'dotted_detection': curve.get('dotted_detection', {}),
                                    'validation_warnings': validation_result.warnings,
                                    'curve_direction': validation_result.curve_direction  # Add curve direction
                                })

                            results['figures'].append(figure_data)

                        except Exception as e:
                            results['errors'].append({
                                'figure_id': figure_id,
                                'error': str(e),
                                'traceback': traceback.format_exc()
                            })

                    page_num += 1

                except ValueError as e:
                    # PyMuPDF raises ValueError when page doesn't exist
                    if "not found" in str(e).lower() or "page" in str(e).lower():
                        break  # End of document
                    else:
                        results['errors'].append({
                            'page': page_num,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        })
                        page_num += 1
                except Exception as e:
                    results['errors'].append({
                        'page': page_num,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    page_num += 1

            return results

        except Exception as e:
            return {
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.stem,
                'status': 'failed',
                'figures': [],
                'errors': [{
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }]
            }

    def process_batch(self, pdfs: List[Path]) -> List[Dict]:
        """Process multiple PDFs in parallel."""
        print(f"Processing {len(pdfs)} PDFs with {self.n_workers} workers...")

        if self.n_workers == 1:
            # Single-threaded for debugging
            results = []
            for pdf in tqdm(pdfs):
                results.append(self.process_single_pdf(pdf))
        else:
            # Multi-threaded
            with Pool(self.n_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_single_pdf, pdfs),
                    total=len(pdfs)
                ))

        return results

    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """Generate summary statistics."""
        total_pdfs = len(results)
        successful_pdfs = sum(1 for r in results if r['status'] == 'success')
        total_figures = sum(len(r['figures']) for r in results)
        total_curves = sum(
            sum(len(f['curves']) for f in r['figures'])
            for r in results
        )
        total_errors = sum(len(r['errors']) for r in results)

        summary = {
            'total_pdfs': total_pdfs,
            'successful_pdfs': successful_pdfs,
            'failed_pdfs': total_pdfs - successful_pdfs,
            'total_figures': total_figures,
            'total_curves': total_curves,
            'total_errors': total_errors,
            'success_rate': successful_pdfs / total_pdfs if total_pdfs > 0 else 0
        }

        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total PDFs processed: {total_pdfs}")
        print(f"Successful: {successful_pdfs} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed: {total_pdfs - successful_pdfs}")
        print(f"Total figures extracted: {total_figures}")
        print(f"Total curves extracted: {total_curves}")
        print(f"Total errors: {total_errors}")
        print("="*70)

        return summary

    def save_results(self, results: List[Dict], summary: Dict):
        """Save results to JSON and generate report."""
        # Convert NumPy types to native Python types
        results_clean = convert_numpy_types(results)
        summary_clean = convert_numpy_types(summary)

        # Save full results
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"\nFull results saved to: {results_path}")

        # Save summary
        summary_path = self.output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_clean, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        # Create CSV catalog of all extracted curves
        catalog_data = []
        for result in results:
            for figure in result['figures']:
                for curve in figure['curves']:
                    catalog_data.append({
                        'pdf_name': result['pdf_name'],
                        'figure_id': figure['figure_id'],
                        'page': figure['page'],
                        'curve_id': curve['curve_id'],
                        'n_points': curve['n_points'],
                        'csv_path': curve['csv_path'],
                        'x_unit': figure['calibration']['x_unit'],
                        'y_unit': figure['calibration']['y_unit'],
                        'calibration_valid': figure['calibration_valid']
                    })

        catalog_df = pd.DataFrame(catalog_data)
        catalog_path = self.output_dir / 'curve_catalog.csv'
        catalog_df.to_csv(catalog_path, index=False)
        print(f"Curve catalog saved to: {catalog_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch process K-M curves from PDFs')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('output_dir', help='Directory for output data')
    parser.add_argument('--dpi', type=int, default=600, help='PDF rendering DPI')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--pattern', default='**/*.pdf', help='PDF file pattern (supports recursive search)')
    parser.add_argument('--use-knn', action='store_true', help='Use k-NN tracing algorithm (SurvdigitizeR method)')
    parser.add_argument('--use-morphological', action='store_true', help='Use morphological line detection (Canny + contours)')
    parser.add_argument('--remove-grid', action='store_true', help='Remove grid lines before extraction (recommended for R plots)')

    args = parser.parse_args()

    # Create processor
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        n_workers=args.workers,
        use_knn_tracing=args.use_knn,
        use_morphological=args.use_morphological,
        remove_grid=args.remove_grid
    )

    # Discover PDFs
    pdfs = processor.discover_pdfs(pattern=args.pattern)

    if not pdfs:
        print("No PDF files found!")
        return

    # Process
    results = processor.process_batch(pdfs)

    # Generate summary
    summary = processor.generate_summary_report(results)

    # Save results
    processor.save_results(results, summary)

    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()
