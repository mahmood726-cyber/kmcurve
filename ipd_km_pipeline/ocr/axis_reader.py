#!/usr/bin/env python3
"""
OCR-based automatic axis label reading for K-M curves.

Extracts axis labels and tick marks to automatically determine:
- X-axis range (time in months/years)
- Y-axis range (survival probability 0-1 or percentage 0-100)
- Axis units and labels
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AxisInfo:
    """Information extracted from an axis."""
    min_value: float
    max_value: float
    unit: str  # 'months', 'years', 'probability', 'percentage', etc.
    label: str  # Full axis label text
    tick_values: List[float]  # Values at tick marks
    tick_positions: List[int]  # Pixel positions of ticks
    confidence: float  # OCR confidence 0-1


def extract_axis_labels(
    panel_image: Image.Image,
    axis_region: Tuple[int, int, int, int],
    axis_type: str = 'x'
) -> AxisInfo:
    """
    Extract axis labels and values using OCR.

    Args:
        panel_image: PIL Image of the K-M panel
        axis_region: (x, y, width, height) of axis region
        axis_type: 'x' or 'y'

    Returns:
        AxisInfo with extracted values
    """
    # Crop to axis region
    x, y, w, h = axis_region
    axis_img = panel_image.crop((x, y, x + w, y + h))

    # Preprocess for OCR
    axis_img_processed = preprocess_for_ocr(axis_img)

    # Run OCR
    ocr_config = '--psm 6'  # Assume uniform block of text
    ocr_data = pytesseract.image_to_data(
        axis_img_processed,
        config=ocr_config,
        output_type=pytesseract.Output.DICT
    )

    # Extract numbers and text
    numbers = extract_numbers_from_ocr(ocr_data)
    full_text = ' '.join([word for word in ocr_data['text'] if word.strip()])

    # Determine axis type and unit
    unit = detect_axis_unit(full_text, axis_type)

    # Parse tick values and positions
    tick_values, tick_positions = parse_tick_marks(
        numbers, ocr_data, axis_type
    )

    # Determine min/max
    if tick_values:
        min_value = min(tick_values)
        max_value = max(tick_values)
    else:
        # Fallback defaults
        if axis_type == 'x':
            min_value, max_value = 0, 60  # Default months
        else:
            min_value, max_value = 0, 1  # Default probability

    # Compute confidence
    confidences = [c for c in ocr_data['conf'] if c != -1]
    avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.5

    return AxisInfo(
        min_value=min_value,
        max_value=max_value,
        unit=unit,
        label=full_text,
        tick_values=tick_values,
        tick_positions=tick_positions,
        confidence=avg_confidence
    )


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR results.

    - Convert to grayscale
    - Increase contrast
    - Binarize
    - Denoise
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Binarize with Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # Convert back to PIL
    return Image.fromarray(denoised)


def extract_numbers_from_ocr(ocr_data: Dict) -> List[float]:
    """
    Extract all numbers from OCR output.

    Handles:
    - Integers: 0, 10, 100
    - Decimals: 0.5, 0.95, 1.0
    - Percentages: 50%, 95%
    """
    numbers = []

    for word in ocr_data['text']:
        # Clean the word
        word = word.strip()
        if not word:
            continue

        # Try to extract number
        # Remove % sign if present
        if '%' in word:
            word = word.replace('%', '')

        # Try parsing as float
        try:
            num = float(word)
            numbers.append(num)
        except ValueError:
            # Try regex extraction
            matches = re.findall(r'[-+]?\d*\.?\d+', word)
            for match in matches:
                try:
                    numbers.append(float(match))
                except ValueError:
                    continue

    return numbers


def detect_axis_unit(text: str, axis_type: str) -> str:
    """
    Detect the unit of the axis from label text.

    X-axis common units: months, years, days, weeks
    Y-axis common units: probability (0-1), percentage (0-100)
    """
    text_lower = text.lower()

    if axis_type == 'x':
        if 'month' in text_lower:
            return 'months'
        elif 'year' in text_lower:
            return 'years'
        elif 'day' in text_lower:
            return 'days'
        elif 'week' in text_lower:
            return 'weeks'
        else:
            return 'months'  # Default
    else:  # y-axis
        if '%' in text or 'percent' in text_lower:
            return 'percentage'
        elif 'probability' in text_lower or 'survival' in text_lower:
            return 'probability'
        else:
            return 'probability'  # Default


def parse_tick_marks(
    numbers: List[float],
    ocr_data: Dict,
    axis_type: str
) -> Tuple[List[float], List[int]]:
    """
    Parse tick mark values and their positions.

    Returns:
        (tick_values, tick_positions) where positions are in pixels
    """
    if not numbers:
        return [], []

    # Get positions of numbers
    tick_values = []
    tick_positions = []

    for i, word in enumerate(ocr_data['text']):
        try:
            num = float(word.strip().replace('%', ''))
            if axis_type == 'x':
                # X position (left coordinate)
                pos = ocr_data['left'][i]
            else:
                # Y position (top coordinate)
                pos = ocr_data['top'][i]

            tick_values.append(num)
            tick_positions.append(pos)
        except ValueError:
            continue

    # Sort by position
    if tick_values and tick_positions:
        sorted_pairs = sorted(zip(tick_positions, tick_values))
        tick_positions, tick_values = zip(*sorted_pairs)
        tick_positions = list(tick_positions)
        tick_values = list(tick_values)

    return tick_values, tick_positions


def auto_calibrate_axes(
    panel_image: Image.Image,
    panel_bbox: Tuple[int, int, int, int],
    x_axis_region: Optional[Tuple[int, int, int, int]] = None,
    y_axis_region: Optional[Tuple[int, int, int, int]] = None
) -> Dict:
    """
    Automatically calibrate both axes using OCR.

    Args:
        panel_image: PIL Image of panel
        panel_bbox: (x, y, width, height) of panel
        x_axis_region: Optional (x, y, w, h) of x-axis labels
        y_axis_region: Optional (x, y, w, h) of y-axis labels

    Returns:
        Dict with x_range, y_range, x_unit, y_unit, confidence
    """
    x, y, w, h = panel_bbox

    # If regions not provided, estimate them
    if x_axis_region is None:
        # X-axis labels are typically below the panel
        x_axis_region = (x, y + h, w, 50)  # 50 pixels below

    if y_axis_region is None:
        # Y-axis labels are typically to the left of panel
        y_axis_region = (max(0, x - 80), y, 80, h)  # 80 pixels to left

    # Extract axis information
    x_info = extract_axis_labels(panel_image, x_axis_region, axis_type='x')
    y_info = extract_axis_labels(panel_image, y_axis_region, axis_type='y')

    # Normalize y-axis if it's in percentage
    if y_info.unit == 'percentage':
        y_min = y_info.min_value / 100.0
        y_max = y_info.max_value / 100.0
    else:
        y_min = y_info.min_value
        y_max = y_info.max_value

    return {
        'x_range': (x_info.min_value, x_info.max_value),
        'y_range': (y_min, y_max),
        'x_unit': x_info.unit,
        'y_unit': y_info.unit,
        'x_label': x_info.label,
        'y_label': y_info.label,
        'x_confidence': x_info.confidence,
        'y_confidence': y_info.confidence,
        'combined_confidence': (x_info.confidence + y_info.confidence) / 2.0
    }


def validate_axis_calibration(
    calibration: Dict,
    expected_x_range: Optional[Tuple[float, float]] = None,
    expected_y_range: Optional[Tuple[float, float]] = None
) -> Tuple[bool, str]:
    """
    Validate that OCR-extracted axis ranges are reasonable.

    Returns:
        (is_valid, reason)
    """
    x_min, x_max = calibration['x_range']
    y_min, y_max = calibration['y_range']

    # Check x-axis
    if x_min >= x_max:
        return False, f"Invalid x-axis range: {x_min} >= {x_max}"

    if x_min < 0:
        return False, f"Negative x-axis minimum: {x_min}"

    # Check y-axis
    if y_min >= y_max:
        return False, f"Invalid y-axis range: {y_min} >= {y_max}"

    # For survival/probability, should be 0-1
    if calibration['y_unit'] == 'probability':
        if y_min < -0.1 or y_max > 1.1:
            return False, f"Probability out of range [0, 1]: [{y_min}, {y_max}]"

    # Check confidence
    if calibration['combined_confidence'] < 0.3:
        return False, f"Low OCR confidence: {calibration['combined_confidence']:.2f}"

    # If expected ranges provided, check they're close
    if expected_x_range:
        exp_x_min, exp_x_max = expected_x_range
        if abs(x_min - exp_x_min) > 0.2 * exp_x_max or abs(x_max - exp_x_max) > 0.2 * exp_x_max:
            return False, f"X-axis range {(x_min, x_max)} differs significantly from expected {expected_x_range}"

    if expected_y_range:
        exp_y_min, exp_y_max = expected_y_range
        if abs(y_min - exp_y_min) > 0.1 or abs(y_max - exp_y_max) > 0.1:
            return False, f"Y-axis range {(y_min, y_max)} differs significantly from expected {expected_y_range}"

    return True, "Valid"


def get_fallback_calibration() -> Dict:
    """
    Get safe default calibration values for K-M curves when OCR fails.

    Uses standard ranges that work for most K-M plots:
    - X-axis: 0-60 months (typical survival study duration)
    - Y-axis: 0-1 (survival probability)

    Returns:
        Dict with safe default calibration
    """
    return {
        'x_range': (0, 60),  # 0-60 months (5 years)
        'y_range': (0, 1),   # 0-1 survival probability
        'x_unit': 'months',
        'y_unit': 'probability',
        'x_label': 'Time (months)',
        'y_label': 'Survival probability',
        'x_confidence': 0.0,
        'y_confidence': 0.0,
        'combined_confidence': 0.0,
        'fallback': True  # Flag to indicate this is a fallback
    }
