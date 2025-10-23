#!/usr/bin/env python3
"""
Test script for layout detection on extracted page.

Tests panel, axis, and at-risk table detection.
"""
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.detect import detect_panels, visualize_panels


def main():
    # Path to extracted page image
    page_img_path = "C:/Users/user/OneDrive - NHS/Documents/KMcurve/ipd_km_pipeline/artifacts/test_extraction/page_6/page6_00_render_307d2428fec305a8.png"
    output_dir = "C:/Users/user/OneDrive - NHS/Documents/KMcurve/ipd_km_pipeline/artifacts/layout_detection"

    print("="*70)
    print("Layout Detection Test")
    print("="*70)
    print(f"Input: {page_img_path}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load image
    print("\n[1/3] Loading image...")
    image = Image.open(page_img_path)
    print(f"    Image size: {image.size}")
    print(f"    Image mode: {image.mode}")

    # Detect panels
    print("\n[2/3] Detecting panels...")
    panels = detect_panels(image)

    print(f"    Found {len(panels)} panel(s)")

    for i, panel in enumerate(panels):
        x, y, w, h = panel['bbox']
        conf = panel['confidence']

        print(f"\n    Panel {i+1}:")
        print(f"      BBox: x={x}, y={y}, w={w}, h={h}")
        print(f"      Confidence: {conf:.3f}")
        print(f"      Aspect ratio: {w/h:.2f}")

        axes = panel['axes']
        if axes['x_axis']:
            print(f"      X-axis: {axes['x_axis']}")
        else:
            print(f"      X-axis: Not detected")

        if axes['y_axis']:
            print(f"      Y-axis: {axes['y_axis']}")
        else:
            print(f"      Y-axis: Not detected")

        if panel['at_risk_region']:
            ar_x, ar_y, ar_w, ar_h = panel['at_risk_region']
            print(f"      At-risk region: x={ar_x}, y={ar_y}, w={ar_w}, h={ar_h}")
        else:
            print(f"      At-risk region: Not detected")

    # Visualize panels
    print("\n[3/3] Creating visualization...")
    viz_path = Path(output_dir) / "panel_detection.png"
    visualize_panels(image, panels, str(viz_path))
    print(f"    Saved: {viz_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Panels detected: {len(panels)}")
    print(f"Visualization: {viz_path}")
    print("\nNext steps:")
    print("  1. Check visualization to verify panel detection")
    print("  2. Implement raster curve extraction (HSL + k-medoids)")
    print("  3. Implement OCR for axis labels")
    print("="*70)


if __name__ == "__main__":
    main()
