#!/usr/bin/env python3
"""
Test dotted line filtering on the NEJM K-M curve.

This should filter out the horizontal dotted "Standard control" reference line
and keep only the solid "Intensive control" curve.
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pdf_io.extract import extract_images_from_pdf
from layout.detect import detect_panels
from raster_cv.extract import extract_curves
from raster_cv.dotted_line_filter import visualize_dotted_detection

def main():
    print("="*70)
    print("Testing Dotted Line Filtering")
    print("="*70)

    # Paths
    pdf_path = "C:/Users/user/OneDrive - NHS/Documents/KMcurve/papers_to_process/NEJMoa0802987.pdf"
    output_dir = "C:/Users/user/OneDrive - NHS/Documents/KMcurve/ipd_km_pipeline/artifacts/dotted_filter_test"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Extract page
    print("\n[1/4] Extracting page from PDF...")
    results = extract_images_from_pdf(
        pdf_path=pdf_path,
        page_num=6,
        dpi=600,
        force_render=False
    )

    if not results:
        print("Error: No images extracted")
        return

    page_image = results[0]['image']
    print(f"    Extracted page: {page_image.size[0]}x{page_image.size[1]} pixels")

    # Step 2: Detect panels
    print("\n[2/4] Detecting K-M panels...")
    panels = detect_panels(page_image)

    if len(panels) == 0:
        print("Error: No panels detected")
        return

    panel = panels[0]
    x, y, w, h = panel['bbox']
    print(f"    Panel A bbox: x={x}, y={y}, w={w}, h={h}")

    panel_img = page_image.crop((x, y, x + w, y + h))

    # Step 3: Extract curves WITHOUT filtering dotted lines
    print("\n[3/4] Extracting curves (including dotted lines)...")
    curves_all = extract_curves(
        panel_img,
        n_curves=None,  # Auto-detect
        exclude_dotted=False  # Keep all curves including dotted
    )

    print(f"    Extracted {len(curves_all)} total curve(s):")
    for i, curve in enumerate(curves_all):
        detection = curve.get('dotted_detection', {})
        is_dotted = detection.get('is_dotted', False)
        confidence = detection.get('confidence', 0.0)
        print(f"      Curve {i+1}: {curve['n_points']} points, "
              f"{'DOTTED' if is_dotted else 'SOLID'} (confidence: {confidence:.3f})")

        # Print diagnostic details
        if 'diagnostics' in detection:
            diag = detection['diagnostics']
            if 'density' in diag:
                print(f"        Density: {diag['density']['pixels_per_x_unit']:.2f} pixels/x-unit")
            if 'continuity' in diag:
                print(f"        Components: {diag['continuity']['num_components']}")
            if 'horizontal' in diag:
                print(f"        Horizontality: {diag['horizontal']['horizontality']:.4f}")

    # Save visualization with all curves
    viz_all_path = Path(output_dir) / "all_curves_labeled.png"
    visualize_dotted_detection(curves_all, w, h, str(viz_all_path))

    # Step 4: Extract curves WITH filtering dotted lines
    print("\n[4/4] Extracting curves (excluding dotted lines)...")
    curves_solid = extract_curves(
        panel_img,
        n_curves=None,  # Auto-detect
        exclude_dotted=True,  # Filter out dotted lines
        dotted_confidence_threshold=0.5
    )

    print(f"    Extracted {len(curves_solid)} solid curve(s):")
    for i, curve in enumerate(curves_solid):
        detection = curve.get('dotted_detection', {})
        is_dotted = detection.get('is_dotted', False)
        confidence = detection.get('confidence', 0.0)
        print(f"      Curve {i+1}: {curve['n_points']} points, "
              f"{'DOTTED' if is_dotted else 'SOLID'} (confidence: {confidence:.3f})")

    # Save visualization with filtered curves
    viz_filtered_path = Path(output_dir) / "solid_curves_only.png"
    visualize_dotted_detection(curves_solid, w, h, str(viz_filtered_path))

    # Summary
    print("\n" + "="*70)
    print("DOTTED LINE FILTERING TEST COMPLETE!")
    print("="*70)
    print(f"Total curves detected: {len(curves_all)}")
    print(f"Solid curves after filtering: {len(curves_solid)}")
    print(f"Dotted lines filtered out: {len(curves_all) - len(curves_solid)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - all_curves_labeled.png (all curves with dotted/solid labels)")
    print(f"  - solid_curves_only.png (solid curves only)")
    print("="*70)

    return curves_all, curves_solid


if __name__ == "__main__":
    all_curves, solid_curves = main()
