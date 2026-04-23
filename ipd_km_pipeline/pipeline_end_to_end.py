#!/usr/bin/env python3
"""
End-to-end K-M curve extraction pipeline.

This script is a repo-local sample runner for the maintained `ipd_km_pipeline`
workflow. It now defaults to bundled fixtures instead of OneDrive-specific paths.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pdf_io.extract import extract_images_from_pdf
from layout.detect import detect_panels
from project_paths import artifact_path, ensure_dir, sample_pdf_path
from raster_cv.extract import extract_curves, visualize_extracted_curves


def transform_pixels_to_survival(
    curve_pixels: np.ndarray,
    panel_width: int,
    panel_height: int,
    x_range: tuple,  # (min_time, max_time) in months
    y_range: tuple   # (min_survival, max_survival) typically (0, 1)
) -> tuple:
    """
    Transform pixel coordinates to time and survival probability.

    Args:
        curve_pixels: (N, 2) array of (x, y) pixel coordinates
        panel_width: width of panel in pixels
        panel_height: height of panel in pixels
        x_range: (min_time, max_time) from axis labels
        y_range: (min_prob, max_prob) from axis labels

    Returns:
        (times, survival_probs) as numpy arrays
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # X: left edge = x_min, right edge = x_max
    times = x_min + (curve_pixels[:, 0] / panel_width) * (x_max - x_min)

    # Y: top edge = y_max, bottom edge = y_min (inverted)
    survival_probs = y_max - (curve_pixels[:, 1] / panel_height) * (y_max - y_min)

    # Clip to valid range
    survival_probs = np.clip(survival_probs, y_min, y_max)

    return times, survival_probs


def aggregate_survival_data(times: np.ndarray, probs: np.ndarray, time_step: float = 1.0) -> pd.DataFrame:
    """
    Aggregate survival data to regular time intervals.

    For each time bin, take the median survival probability.
    """
    # Create time bins
    min_time = np.floor(times.min())
    max_time = np.ceil(times.max())
    bins = np.arange(min_time, max_time + time_step, time_step)

    # Bin the data
    bin_indices = np.digitize(times, bins)

    # Aggregate by bin
    aggregated_times = []
    aggregated_probs = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            aggregated_times.append(bins[i-1])
            aggregated_probs.append(np.median(probs[mask]))

    return pd.DataFrame({
        'time': aggregated_times,
        'survival': aggregated_probs
    })


def parse_args(argv=None):
    """Parse command line arguments for the sample pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the sample KM extraction pipeline on a bundled or user-supplied PDF."
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        help="Path to the PDF to process. Defaults to the bundled NEJM sample PDF.",
    )
    parser.add_argument(
        "--page-num",
        type=int,
        default=6,
        help="0-indexed PDF page number to process. Default: 6 (page 7).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Render DPI used when rasterizing the page. Default: 600.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated images/CSVs. Defaults to ipd_km_pipeline/artifacts/end_to_end.",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Skip vector/native extraction attempts and force raster rendering.",
    )
    return parser.parse_args(argv)


def resolve_paths(args):
    """Resolve repo-relative defaults for the sample pipeline."""
    if args.pdf_path:
        pdf_path = Path(args.pdf_path).expanduser().resolve()
    else:
        pdf_path = sample_pdf_path()

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = artifact_path("end_to_end")

    ensure_dir(output_dir)
    return pdf_path, output_dir


def main(argv=None):
    args = parse_args(argv)
    pdf_path, output_dir = resolve_paths(args)

    print("="*70)
    print("End-to-End K-M Curve Extraction Pipeline")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")

    # Step 1: Extract page from PDF
    print("\n[1/5] Extracting page from PDF...")
    page_num = args.page_num

    results = extract_images_from_pdf(
        pdf_path=str(pdf_path),
        page_num=page_num,
        dpi=args.dpi,
        force_render=args.force_render,
    )

    if not results:
        print("Error: No images extracted from PDF")
        return

    page_image = results[0]['image']
    print(f"    Extracted page {page_num + 1} at {page_image.size[0]}x{page_image.size[1]} pixels")

    # Step 2: Detect K-M panels
    print("\n[2/5] Detecting K-M curve panels...")
    panels = detect_panels(page_image)
    print(f"    Found {len(panels)} panel(s)")

    # Process first panel (Panel A)
    if len(panels) == 0:
        print("Error: No panels detected")
        return

    panel = panels[0]  # Use first panel
    x, y, w, h = panel['bbox']
    print(f"    Panel A bbox: x={x}, y={y}, w={w}, h={h}")

    # Crop panel
    panel_img = page_image.crop((x, y, x + w, y + h))

    # Step 3: Extract curves from panel
    print("\n[3/5] Extracting curves from panel...")
    curves = extract_curves(
        panel_img,
        n_curves=2,
        background_color=(255, 255, 255),
        lightness_threshold=0.90,
        saturation_threshold=0.01
    )

    print(f"    Extracted {len(curves)} curve(s):")
    for i, curve in enumerate(curves):
        print(f"      Curve {i+1}: {curve['n_points']} points")

    # Save visualization
    viz_path = output_dir / "curves_extracted.png"
    visualize_extracted_curves(panel_img, curves, str(viz_path))

    # Step 4: Manual axis calibration
    # (In production, this would use OCR to read axis labels)
    print("\n[4/5] Calibrating axes...")
    print("    Manual calibration (from visual inspection):")
    print("      X-axis: 0 to 66 months")
    print("      Y-axis: 5.0 to 10.0 (Mean Glycated Hemoglobin %)")

    x_range = (0, 66)  # months
    y_range = (5.0, 10.0)  # HbA1c %

    # Step 5: Transform to real coordinates and save data
    print("\n[5/5] Transforming to survival data...")

    survival_data = {}

    for i, curve in enumerate(curves):
        curve_id = i + 1
        pixels = curve['points']

        # Transform pixels to real coordinates
        times, values = transform_pixels_to_survival(
            pixels,
            panel_width=w,
            panel_height=h,
            x_range=x_range,
            y_range=y_range
        )

        # Aggregate to monthly intervals
        df = aggregate_survival_data(times, values, time_step=1.0)

        # Save to CSV
        csv_path = output_dir / f"curve_{curve_id}_data.csv"
        df.to_csv(csv_path, index=False)

        print(f"    Curve {curve_id}:")
        print(f"      Time range: {df['time'].min():.1f} - {df['time'].max():.1f} months")
        print(f"      HbA1c range: {df['survival'].min():.2f} - {df['survival'].max():.2f} %")
        print(f"      Data points: {len(df)}")
        print(f"      Saved: {csv_path}")

        survival_data[f"curve_{curve_id}"] = df

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - curves_extracted.png (visualization)")
    for i in range(len(curves)):
        print(f"  - curve_{i+1}_data.csv (extracted data)")

    print("\nNext steps:")
    print("  1. Add OCR module to automatically read axis labels")
    print("  2. Parse numbers-at-risk tables for validation")
    print("  3. Implement IPD reconstruction algorithm")
    print("  4. Add quality control checks")
    print("="*70)

    return survival_data


if __name__ == "__main__":
    data = main()
