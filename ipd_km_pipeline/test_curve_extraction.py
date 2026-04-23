#!/usr/bin/env python3
"""
Test script for raster curve extraction on detected panels.

Tests HSL + k-medoids curve extraction on both K-M panels.
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.detect import detect_panels
from project_paths import artifact_path, ensure_dir, sample_render_path
from raster_cv.extract import extract_curves, visualize_extracted_curves


def main():
    page_img_path = sample_render_path()
    output_dir = ensure_dir(artifact_path("curve_extraction"))

    print("="*70)
    print("Curve Extraction Test")
    print("="*70)
    print(f"Input: {page_img_path}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Create output directory
    # Load image
    print("\n[1/4] Loading image...")
    image = Image.open(page_img_path)
    print(f"    Image size: {image.size}")

    # Detect panels
    print("\n[2/4] Detecting panels...")
    panels = detect_panels(image)
    print(f"    Found {len(panels)} panel(s)")

    # Extract curves from each panel
    print("\n[3/4] Extracting curves from panels...")

    all_results = []

    for i, panel in enumerate(panels[:2]):  # Process first 2 unique panels
        x, y, w, h = panel['bbox']
        print(f"\n    Panel {i+1}:")
        print(f"      BBox: x={x}, y={y}, w={w}, h={h}")

        # Crop panel from full page
        panel_img = image.crop((x, y, x + w, y + h))

        # Extract curves (assume 2 curves per panel for K-M)
        curves = extract_curves(
            panel_img,
            n_curves=2,
            background_color=(255, 255, 255),
            lightness_threshold=0.90,  # Allow slightly lighter pixels
            saturation_threshold=0.01   # Low threshold for grayscale curves
        )

        print(f"      Extracted {len(curves)} curve(s)")

        for j, curve in enumerate(curves):
            print(f"        Curve {j+1}:")
            print(f"          Points: {curve['n_points']}")
            print(f"          Color: RGB{curve['color']}")
            print(f"          X range: {np.min(curve['points'][:, 0]):.0f} - {np.max(curve['points'][:, 0]):.0f}")
            print(f"          Y range: {np.min(curve['points'][:, 1]):.0f} - {np.max(curve['points'][:, 1]):.0f}")

        # Visualize extracted curves
        viz_path = output_dir / f"panel_{i+1}_curves.png"
        visualize_extracted_curves(panel_img, curves, str(viz_path))
        print(f"      Visualization saved: {viz_path}")

        all_results.append({
            'panel_id': i + 1,
            'bbox': panel['bbox'],
            'curves': curves
        })

    # Summary
    print("\n[4/4] Creating summary...")

    total_curves = sum(len(r['curves']) for r in all_results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Panels processed: {len(all_results)}")
    print(f"Total curves extracted: {total_curves}")
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Check visualizations to verify curve extraction")
    print("  2. Implement OCR for axis labels (to get time/probability ranges)")
    print("  3. Convert pixel coordinates to survival probabilities")
    print("  4. Implement IPD reconstruction from survival curves")
    print("="*70)


if __name__ == "__main__":
    main()
