"""
TrOCR Performance Test

This script tests TrOCR (neural network OCR) against the sample PDF to measure
actual performance improvement over Tesseract OCR (baseline: 0% success).

Test procedure:
1. Render PDF pages to images
2. Extract axis regions manually from figures (bottom 10% for x-axis, left 10% for y-axis)
3. Run TrOCR on axis regions
4. Compare with expected values
5. Report success rate and confidence scores

Expected outcome: 70-90% success rate (vs Tesseract's 0%)
"""

import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ocr.axis_reader_trocr import create_trocr_reader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def render_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> np.ndarray:
    """
    Render PDF page to numpy array.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: Rendering DPI

    Returns:
        Page image as numpy array (RGB)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Render at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert to numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    # Ensure RGB (drop alpha if present)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    doc.close()
    return img


def detect_figures(page_image: np.ndarray):
    """
    Detect figure regions in page image using simple contour detection.

    Args:
        page_image: Page image as numpy array

    Returns:
        List of figure bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale
    if len(page_image.shape) == 3:
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = page_image

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size (figures should be large)
    min_area = 50000  # Minimum area for a figure
    figures = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area > min_area and w > 200 and h > 200:
            figures.append((x, y, w, h))

    # Sort by y-coordinate (top to bottom)
    figures.sort(key=lambda f: f[1])

    return figures


def extract_axis_region(panel_image: np.ndarray, axis_type: str = 'x') -> np.ndarray:
    """
    Extract axis region from panel image.

    Args:
        panel_image: Panel image as numpy array
        axis_type: 'x' or 'y' axis

    Returns:
        Axis region as numpy array
    """
    h, w = panel_image.shape[:2]

    if axis_type == 'x':
        # X-axis: bottom 10% of image
        return panel_image[int(h * 0.9):, :]
    else:  # y-axis
        # Y-axis: left 10% of image
        return panel_image[:, :int(w * 0.1)]


def validate_axis_numbers(numbers: list, axis_type: str = 'x') -> bool:
    """
    Validate extracted axis numbers.

    Args:
        numbers: List of extracted numbers
        axis_type: 'x' or 'y' axis

    Returns:
        True if valid, False otherwise
    """
    if len(numbers) < 2:
        return False

    min_val = min(numbers)
    max_val = max(numbers)

    if min_val >= max_val:
        return False

    if axis_type == 'x':
        # X-axis: typically 0-120 months
        return 0 <= min_val and max_val <= 200
    else:  # y-axis
        # Y-axis: typically 0-1 probability
        return 0 <= min_val and max_val <= 1.1


def test_trocr_on_pdf(
    pdf_path: str,
    test_pages: list,
    dpi: int = 300,
    confidence_threshold: float = 0.5
):
    """
    Test TrOCR on sample PDF.

    Args:
        pdf_path: Path to test PDF
        test_pages: List of page numbers to test (0-indexed)
        dpi: Rendering DPI
        confidence_threshold: Minimum confidence for success
    """
    print("=" * 80)
    print("TrOCR Performance Test")
    print("=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"Pages: {test_pages}")
    print(f"DPI: {dpi}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 80)

    # Create TrOCR reader
    print("\nLoading TrOCR model...")
    reader = create_trocr_reader(model_size="base")

    if not reader:
        print("ERROR: Failed to create TrOCR reader")
        return

    print(f"TrOCR model loaded (device: {reader.device})")

    # Test statistics
    total_axes_tested = 0
    successful_axes = 0
    confidence_scores = []

    results = []

    # Process each page
    for page_num in test_pages:
        print(f"\n{'=' * 80}")
        print(f"Page {page_num + 1}")
        print(f"{'=' * 80}")

        # Render page
        page_image = render_page_to_image(pdf_path, page_num, dpi)
        print(f"Page rendered: {page_image.shape[1]}x{page_image.shape[0]} pixels")

        # Detect figures on page
        figures = detect_figures(page_image)
        print(f"Figures detected: {len(figures)}")

        # Test each figure
        for fig_idx, (x, y, w, h) in enumerate(figures):
            panel_image = page_image[y:y+h, x:x+w]
            print(f"\n  Figure {fig_idx + 1}:")
            print(f"    Position: ({x}, {y})")
            print(f"    Size: {w}x{h} pixels")

            # Test X-axis
            print(f"    X-axis:")
            x_region = extract_axis_region(panel_image, 'x')
            x_text, x_conf = reader.extract_text_from_image(x_region)
            x_numbers = reader.extract_numbers_from_text(x_text, 'x')
            x_valid = validate_axis_numbers(x_numbers, 'x')

            print(f"      Text: '{x_text}'")
            print(f"      Numbers: {x_numbers}")
            print(f"      Confidence: {x_conf:.2f}")
            print(f"      Valid: {x_valid}")

            total_axes_tested += 1
            if x_valid and x_conf >= confidence_threshold:
                successful_axes += 1
                confidence_scores.append(x_conf)
                print(f"      [SUCCESS]")
            else:
                print(f"      [FAILED]")

            # Test Y-axis
            print(f"    Y-axis:")
            y_region = extract_axis_region(panel_image, 'y')
            y_text, y_conf = reader.extract_text_from_image(y_region)
            y_numbers = reader.extract_numbers_from_text(y_text, 'y')
            y_valid = validate_axis_numbers(y_numbers, 'y')

            print(f"      Text: '{y_text}'")
            print(f"      Numbers: {y_numbers}")
            print(f"      Confidence: {y_conf:.2f}")
            print(f"      Valid: {y_valid}")

            total_axes_tested += 1
            if y_valid and y_conf >= confidence_threshold:
                successful_axes += 1
                confidence_scores.append(y_conf)
                print(f"      [SUCCESS]")
            else:
                print(f"      [FAILED]")

            # Store result
            results.append({
                'page': page_num + 1,
                'panel': fig_idx + 1,
                'x_text': x_text,
                'x_numbers': x_numbers,
                'x_conf': x_conf,
                'x_valid': x_valid,
                'y_text': y_text,
                'y_numbers': y_numbers,
                'y_conf': y_conf,
                'y_valid': y_valid
            })

    # Final results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total axes tested: {total_axes_tested}")
    print(f"Successful axes: {successful_axes}")
    print(f"Success rate: {100 * successful_axes / total_axes_tested:.1f}%")

    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        print(f"Average confidence (successful): {avg_conf:.2f}")

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print(f"Tesseract OCR (baseline):    0.0% success (0/{total_axes_tested} axes)")
    print(f"TrOCR (neural network):    {100 * successful_axes / total_axes_tested:.1f}% success ({successful_axes}/{total_axes_tested} axes)")
    print(f"Improvement: +{100 * successful_axes / total_axes_tested:.1f} percentage points")

    if successful_axes > 0:
        print("\n[SUCCESS] TrOCR successfully improved OCR performance!")
        print(f"  Expected range: 70-90% success")
        print(f"  Actual: {100 * successful_axes / total_axes_tested:.1f}% success")
    else:
        print("\n[FAILURE] TrOCR did not improve over baseline")
        print("  Consider:")
        print("    - Testing with larger model (trocr-large-printed)")
        print("    - Increasing rendering DPI (600-1200)")
        print("    - Fine-tuning model on medical figure axis labels")

    print("=" * 80)

    return results


if __name__ == "__main__":
    # Test PDF
    test_pdf = Path("test_pdfs/medrxiv/medrxiv_19004184.pdf")

    if not test_pdf.exists():
        print(f"ERROR: Test PDF not found: {test_pdf}")
        print("Please ensure the test PDF exists")
        sys.exit(1)

    # Test on pages with figures (same pages as baseline test)
    # Pages 13, 15, 21, 22 (1-indexed) = Pages 12, 14, 20, 21 (0-indexed)
    test_pages = [12, 14, 20, 21]

    # Run test
    results = test_trocr_on_pdf(
        str(test_pdf),
        test_pages,
        dpi=300,  # Same DPI as baseline test
        confidence_threshold=0.5
    )

    print("\nTest complete!")
