#!/usr/bin/env python3
"""
Test script for PDF-to-image extraction.

Tests on NEJMoa0802987.pdf pages 7 and 55 (the problematic pages).
"""
import sys
from pathlib import Path

# Add parent directory to path to import pdf_io
sys.path.insert(0, str(Path(__file__).parent))

from pdf_io.extract import extract_images_from_pdf, save_extracted_images
from project_paths import artifact_path, ensure_dir, sample_pdf_path


def test_page(pdf_path, page_num, output_dir):
    """Test extraction for a single page."""
    print(f"\n{'='*70}")
    print(f"Testing page {page_num}")
    print(f"{'='*70}")

    try:
        # Extract images from page
        results = extract_images_from_pdf(
            pdf_path=pdf_path,
            page_num=page_num,
            dpi=600,  # High DPI for quality
            force_render=False  # Try all methods
        )

        print(f"\n[+] Extraction successful!")
        print(f"    Methods tried and results:")

        for i, result in enumerate(results):
            print(f"\n    Result {i+1}:")
            print(f"      Method: {result['method']}")
            print(f"      Image size: {result['image'].size}")
            print(f"      Image mode: {result['image'].mode}")
            print(f"      Hash: {result['hash']}")

            if result['bbox']:
                print(f"      BBox: {result['bbox']}")

        # Save extracted images
        page_output_dir = Path(output_dir) / f"page_{page_num}"
        save_extracted_images(
            results,
            str(page_output_dir),
            prefix=f"page{page_num}"
        )

        return True

    except Exception as e:
        print(f"\n[!] Error extracting page {page_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    pdf_path = sample_pdf_path()
    output_dir = ensure_dir(artifact_path("test_extraction"))

    print("="*70)
    print("PDF-to-Image Extraction Test")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Test problematic pages mentioned in claude.md
    pages_to_test = [6, 54]  # 0-indexed: page 7 and 55

    results = {}
    for page_num in pages_to_test:
        success = test_page(str(pdf_path), page_num, str(output_dir))
        results[page_num] = success

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for page_num, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  Page {page_num+1} ({page_num} 0-indexed): {status}")

    print("\nImages saved to:")
    print(f"  {output_dir}")
    print("\nNext steps:")
    print("  1. Check extracted images visually")
    print("  2. Implement layout detection to find K-M curve panels")
    print("  3. Implement raster curve extraction with HSL+k-medoids")


if __name__ == "__main__":
    main()
