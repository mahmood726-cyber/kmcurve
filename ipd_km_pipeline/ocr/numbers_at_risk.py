#!/usr/bin/env python3
"""
Numbers-at-risk table parsing for K-M curves.

Extracts the "Number at risk" table typically shown below K-M curves.
This data is critical for:
- Validating extracted survival curves
- IPD reconstruction (Guyot method)
- Quality assurance
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtRiskData:
    """Parsed numbers-at-risk data for one treatment group."""
    group_label: str  # e.g., "Control", "Treatment", "Intensive", etc.
    time_points: List[float]  # Time points (months/years)
    n_at_risk: List[int]  # Number at risk at each time point
    confidence: float  # OCR confidence 0-1


def parse_at_risk_table(
    panel_image: Image.Image,
    at_risk_region: Tuple[int, int, int, int]
) -> List[AtRiskData]:
    """
    Parse numbers-at-risk table from image region.

    Args:
        panel_image: PIL Image of full panel/page
        at_risk_region: (x, y, width, height) of at-risk table

    Returns:
        List of AtRiskData, one per treatment group
    """
    # Crop to at-risk region
    x, y, w, h = at_risk_region
    table_img = panel_image.crop((x, y, x + w, y + h))

    # Preprocess for OCR
    table_img_processed = _preprocess_table(table_img)

    # Run OCR to get full table
    ocr_data = pytesseract.image_to_data(
        table_img_processed,
        config='--psm 6',  # Assume block of text
        output_type=pytesseract.Output.DICT
    )

    # Parse table structure
    groups = _parse_table_structure(ocr_data, table_img_processed)

    return groups


def _preprocess_table(image: Image.Image) -> Image.Image:
    """Preprocess at-risk table for better OCR."""
    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Binarize
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    return Image.fromarray(denoised)


def _parse_table_structure(ocr_data: Dict, table_img: Image.Image) -> List[AtRiskData]:
    """
    Parse the table structure from OCR output.

    Typical format:
        Number at risk
        Control:    100   85   70   55   40   25
        Treatment:  100   90   82   75   68   60
        Time:         0   12   24   36   48   60

    Or:
        No. at risk
        Group A     50    45    40    35
        Group B     52    48    44    40
                     0    10    20    30
    """
    # Extract all text and positions
    words = []
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if not word:
            continue

        words.append({
            'text': word,
            'left': ocr_data['left'][i],
            'top': ocr_data['top'][i],
            'width': ocr_data['width'][i],
            'height': ocr_data['height'][i],
            'conf': ocr_data['conf'][i]
        })

    # Cluster by vertical position (rows)
    rows = _cluster_into_rows(words)

    # Identify treatment group rows vs time row
    groups = []
    time_points = None

    for row in rows:
        row_text = ' '.join([w['text'] for w in row])

        # Check if this is a label row (contains "Number", "No.", "at risk")
        if re.search(r'\b(number|no\.?|at\s+risk)\b', row_text, re.IGNORECASE):
            continue  # Skip header

        # Extract numbers from row
        numbers = _extract_numbers_from_row(row)

        if not numbers:
            continue

        # Check if this is a group row (starts with label) or time row (all numbers)
        label_words = [w['text'] for w in row if not _is_number(w['text'])]

        if label_words:
            # This is a treatment group row
            group_label = ' '.join(label_words)
            n_at_risk = numbers

            groups.append(AtRiskData(
                group_label=group_label,
                time_points=[],  # Will fill later
                n_at_risk=n_at_risk,
                confidence=np.mean([w['conf'] for w in row if w['conf'] != -1]) / 100.0
            ))
        else:
            # This might be the time row (all numbers, no labels)
            if not time_points:
                time_points = numbers

    # Assign time points to all groups
    if time_points:
        for group in groups:
            # Match time points to n_at_risk counts
            # They should have same length, but handle mismatch
            min_len = min(len(time_points), len(group.n_at_risk))
            group.time_points = time_points[:min_len]
            group.n_at_risk = group.n_at_risk[:min_len]

    return groups


def _cluster_into_rows(words: List[Dict], tolerance: int = 10) -> List[List[Dict]]:
    """
    Cluster words into rows based on vertical position.

    Args:
        words: List of word dicts with 'top' position
        tolerance: Vertical tolerance in pixels for same row

    Returns:
        List of rows, each row is list of words
    """
    if not words:
        return []

    # Sort by top position
    sorted_words = sorted(words, key=lambda w: w['top'])

    rows = []
    current_row = [sorted_words[0]]
    current_top = sorted_words[0]['top']

    for word in sorted_words[1:]:
        if abs(word['top'] - current_top) <= tolerance:
            # Same row
            current_row.append(word)
        else:
            # New row
            rows.append(sorted(current_row, key=lambda w: w['left']))  # Sort by left position
            current_row = [word]
            current_top = word['top']

    # Add last row
    if current_row:
        rows.append(sorted(current_row, key=lambda w: w['left']))

    return rows


def _extract_numbers_from_row(row: List[Dict]) -> List[int]:
    """Extract all integer numbers from a row of words."""
    numbers = []
    for word in row:
        text = word['text'].strip()
        if _is_number(text):
            try:
                numbers.append(int(float(text)))
            except ValueError:
                continue
    return numbers


def _is_number(text: str) -> bool:
    """Check if text represents a number."""
    text = text.strip()
    try:
        float(text)
        return True
    except ValueError:
        return False


def validate_at_risk_data(
    at_risk_data: List[AtRiskData],
    expected_groups: int = 2
) -> Tuple[bool, str]:
    """
    Validate parsed at-risk data.

    Returns:
        (is_valid, reason)
    """
    if not at_risk_data:
        return False, "No at-risk data found"

    if len(at_risk_data) != expected_groups:
        return False, f"Expected {expected_groups} groups, found {len(at_risk_data)}"

    for group in at_risk_data:
        # Check we have data
        if not group.time_points or not group.n_at_risk:
            return False, f"Group '{group.group_label}' has no data"

        # Check lengths match
        if len(group.time_points) != len(group.n_at_risk):
            return False, f"Group '{group.group_label}' has mismatched lengths"

        # Check numbers at risk are decreasing (or at least non-increasing)
        for i in range(1, len(group.n_at_risk)):
            if group.n_at_risk[i] > group.n_at_risk[i-1]:
                return False, f"Group '{group.group_label}' has increasing n_at_risk"

        # Check time points are increasing
        for i in range(1, len(group.time_points)):
            if group.time_points[i] <= group.time_points[i-1]:
                return False, f"Group '{group.group_label}' has non-increasing time points"

    return True, "Valid"


def match_at_risk_to_curves(
    at_risk_data: List[AtRiskData],
    curves: List[Dict],
    tolerance: float = 0.05
) -> Dict:
    """
    Match at-risk groups to extracted curves.

    Args:
        at_risk_data: Parsed at-risk data
        curves: Extracted curve data
        tolerance: Tolerance for matching (0-1)

    Returns:
        Dict mapping curve_id to AtRiskData
    """
    # Simple matching: assume same order (top to bottom)
    # More sophisticated: could match by curve position or color

    mapping = {}
    for i, (curve, at_risk) in enumerate(zip(curves, at_risk_data)):
        mapping[curve['curve_id']] = at_risk

    return mapping


def compute_validation_metrics(
    curve_data: Dict,
    at_risk_data: AtRiskData
) -> Dict:
    """
    Compute validation metrics comparing curve to at-risk data.

    Metrics:
    - Consistency of event counts with survival probabilities
    - RMSE between expected and observed survival at time points
    """
    time_points = at_risk_data.time_points
    n_at_risk = at_risk_data.n_at_risk

    # Initial number at risk
    n_initial = n_at_risk[0] if n_at_risk else 100

    # Expected survival probability at each time point
    # S(t) = n_at_risk(t) / n_initial (approximate)
    expected_survival = [n / n_initial for n in n_at_risk]

    # Observed survival from curve at those time points
    # Interpolate curve data to time points
    curve_times = curve_data['times']
    curve_survival = curve_data['survival_probs']

    observed_survival = np.interp(time_points, curve_times, curve_survival)

    # Compute RMSE
    rmse = np.sqrt(np.mean((np.array(expected_survival) - observed_survival) ** 2))

    # Compute maximum absolute error
    max_error = np.max(np.abs(np.array(expected_survival) - observed_survival))

    return {
        'rmse': rmse,
        'max_error': max_error,
        'time_points': time_points,
        'expected_survival': expected_survival,
        'observed_survival': observed_survival.tolist()
    }
