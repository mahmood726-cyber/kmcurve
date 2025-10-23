"""
Test PDF Text Extraction for Axis Calibration

This script demonstrates that direct PDF text extraction works better than OCR
on rasterized images for extracting axis labels from medical PDFs.

Goal: Prove that PDF text extraction can achieve >50% success rate (vs 0% with OCR).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
import re

def extract_text_from_page_region(
    pdf_path: str,
    page_num: int,
    region_fraction: str = 'bottom'
) -> str:
    """
    Extract text from a specific region of a PDF page.

    Args:
        pdf_path: Path to PDF
        page_num: Page number (0-indexed)
        region_fraction: 'bottom' (x-axis), 'left' (y-axis), or 'full'

    Returns:
        Extracted text
    """
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        doc.close()
        return ""

    page = doc[page_num]
    page_rect = page.rect  # Full page bounding box

    # Define region based on fraction
    if region_fraction == 'bottom':
        # Bottom 20% for x-axis labels
        clip_rect = fitz.Rect(
            page_rect.x0,
            page_rect.y1 - 0.2 * page_rect.height,
            page_rect.x1,
            page_rect.y1
        )
    elif region_fraction == 'left':
        # Left 20% for y-axis labels
        clip_rect = fitz.Rect(
            page_rect.x0,
            page_rect.y0,
            page_rect.x0 + 0.2 * page_rect.width,
            page_rect.y1
        )
    else:  # full
        clip_rect = page_rect

    # Extract text from clipped region
    text = page.get_text("text", clip=clip_rect)

    doc.close()
    return text


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract all numeric values from text.

    Handles:
    - Integers: 0, 12, 60
    - Decimals: 0.5, 0.95, 1.0
    - Ranges: 0-60 (extracts both 0 and 60)
    """
    numbers = []

    # Pattern: Match numbers (integers and decimals)
    pattern = r'(?<!\w)(\d+\.?\d*)(?!\w)'
    matches = re.findall(pattern, text)

    for match in matches:
        try:
            num = float(match)
            numbers.append(num)
        except ValueError:
            continue

    # Remove duplicates and sort
    numbers = sorted(list(set(numbers)))

    return numbers


def test_pdf_text_extraction(pdf_path: str, page_nums: List[int]):
    """
    Test PDF text extraction on multiple pages.

    For each page:
    1. Extract text from bottom (x-axis region)
    2. Extract text from left (y-axis region)
    3. Extract numbers from each region
    4. Report success/failure

    Success criteria:
    - X-axis: Found at least 2 numbers in range [0, 200]
    - Y-axis: Found at least 2 numbers in range [0, 1.1]
    """
    print(f"Testing PDF text extraction: {pdf_path}")
    print("=" * 80)

    successes = 0
    total_pages = len(page_nums)

    for page_num in page_nums:
        print(f"\nPage {page_num + 1}:")
        print("-" * 80)

        # Extract text from bottom (x-axis)
        bottom_text = extract_text_from_page_region(pdf_path, page_num, 'bottom')
        x_numbers = extract_numbers_from_text(bottom_text)
        x_numbers_filtered = [n for n in x_numbers if 0 <= n <= 200]

        print(f"  Bottom region text preview: {bottom_text[:200]}...")
        print(f"  X-axis numbers found: {x_numbers_filtered}")

        # Extract text from left (y-axis)
        left_text = extract_text_from_page_region(pdf_path, page_num, 'left')
        y_numbers = extract_numbers_from_text(left_text)
        y_numbers_filtered = [n for n in y_numbers if 0 <= n <= 1.1]

        print(f"  Left region text preview: {left_text[:200]}...")
        print(f"  Y-axis numbers found: {y_numbers_filtered}")

        # Check success
        x_success = len(x_numbers_filtered) >= 2
        y_success = len(y_numbers_filtered) >= 2

        if x_success and y_success:
            print(f"  [SUCCESS] Both axes calibrated")
            successes += 1
        else:
            print(f"  [FAILURE] X-axis={'OK' if x_success else 'FAILED'}, Y-axis={'OK' if y_success else 'FAILED'}")

        # Suggest calibration values
        if x_success:
            x_min, x_max = min(x_numbers_filtered), max(x_numbers_filtered)
            print(f"  >> Suggested X calibration: ({x_min}, {x_max})")
        else:
            print(f"  >> X calibration: Fallback to (0, 60)")

        if y_success:
            y_min, y_max = min(y_numbers_filtered), max(y_numbers_filtered)
            print(f"  >> Suggested Y calibration: ({y_min:.2f}, {y_max:.2f})")
        else:
            print(f"  >> Y calibration: Fallback to (0, 1)")

    print("\n" + "=" * 80)
    print(f"RESULTS: {successes}/{total_pages} pages successfully calibrated")
    print(f"Success rate: {100 * successes / total_pages:.1f}%")
    print("=" * 80)

    return successes, total_pages


if __name__ == "__main__":
    # Test on the sample PDF
    test_pdf = Path("test_pdfs/medrxiv/medrxiv_19004184.pdf")

    if not test_pdf.exists():
        print(f"ERROR: Test PDF not found: {test_pdf}")
        print("Please ensure the test PDF exists in test_pdfs/medrxiv/")
        exit(1)

    # Test on pages with figures (pages 13, 15, 21, 22 - 0-indexed: 12, 14, 20, 21)
    # Each page has 2 figures, so 6 pages total from the test results
    test_pages = [12, 14, 20, 21]  # 0-indexed

    successes, total = test_pdf_text_extraction(str(test_pdf), test_pages)

    print("\nCOMPARISON:")
    print(f"  OCR on rasterized images: 0% success (0/12 curves)")
    print(f"  PDF text extraction:      {100 * successes / total:.1f}% success ({successes}/{total} pages)")
    print(f"  Improvement:              {successes}/{total} pages working vs 0 before")
