#!/usr/bin/env python3
"""
Batch processing script for K-M curve extraction from multiple PDFs.

Processes entire directories of PDFs in parallel, extracting all K-M curves
and generating a consolidated dataset.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from pdf_io.extract import extract_images_from_pdf
from layout.detect import detect_panels
from raster_cv.extract import extract_curves
from ocr.axis_reader import auto_calibrate_axes, validate_axis_calibration, get_fallback_calibration
from ocr.axis_reader_v2 import auto_calibrate_axes_v2, validate_axis_calibration_v2, get_fallback_calibration_v2
from ocr.numbers_at_risk import parse_at_risk_table, validate_at_risk_data
from data_validation import validate_curve_data, decimate_curve
import numpy as np


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


class BatchProcessor:
    """Batch processor for K-M curve extraction."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        dpi: int = 600,
        n_workers: int = None
    ):
        """
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output data
            dpi: Resolution for PDF rendering
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.n_workers = n_workers or max(1, cpu_count() - 1)

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

                    for panel_idx, panel in enumerate(panels):
                        figure_id = f"{pdf_path.stem}_p{page_num+1}_f{panel_idx+1}"

                        try:
                            # Extract panel
                            x, y, w, h = panel['bbox']
                            panel_img = page_image.crop((x, y, x + w, y + h))

                            # Extract curves
                            # TEMP FIX: Manually specify n_curves=2 to avoid slow auto-detect
                            # TODO: Add pixel sampling or limit to auto_detect_n_curves()
                            curves = extract_curves(
                                panel_img,
                                n_curves=2,  # Manually set to speed up (most K-M plots have 2 curves)
                                exclude_dotted=True
                            )

                            if not curves:
                                continue

                            # Auto-calibrate axes using IMPROVED v2 method (Hough + multi-OCR)
                            # This replaces the old method that had 0% success rate
                            try:
                                calibration = auto_calibrate_axes_v2(
                                    panel_img,
                                    panel['bbox']
                                )
                                is_valid, reason = validate_axis_calibration_v2(calibration)

                                # If validation fails, use safe fallback
                                if not is_valid:
                                    calibration = get_fallback_calibration_v2()
                                    is_valid = False  # Keep track that we used fallback
                                    reason = f"OCR failed ({reason}), using fallback: x={calibration['x_range']}, y={calibration['y_range']}"
                            except Exception as e:
                                # If OCR completely fails, use safe fallback
                                calibration = get_fallback_calibration_v2()
                                is_valid = False
                                reason = f"OCR error ({str(e)}), using fallback"

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

                                # Validate curve quality
                                validation_result = validate_curve_data(
                                    df,
                                    calibration,
                                    min_points=100,
                                    max_points=200000,
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

    args = parser.parse_args()

    # Create processor
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        n_workers=args.workers
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
