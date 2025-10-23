#!/usr/bin/env python3
"""
Dotted and dashed line detection and filtering for K-M curves.

K-M plots often use dotted lines for:
- Reference lines (e.g., horizontal standard/control baselines)
- Confidence intervals
- Secondary outcomes

This module identifies and filters out dotted/dashed lines to extract only
the solid survival curves.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from scipy import ndimage, signal


def is_dotted_or_dashed(
    curve_pixels: np.ndarray,
    panel_width: int,
    panel_height: int,
    method: str = 'combined'
) -> Tuple[bool, float, Dict]:
    """
    Determine if a curve is dotted/dashed vs solid.

    Args:
        curve_pixels: (N, 2) array of (x, y) pixel coordinates
        panel_width: width of panel in pixels
        panel_height: height of panel in pixels
        method: 'density', 'continuity', 'fft', or 'combined'

    Returns:
        Tuple of (is_dotted, confidence, diagnostic_info)
        - is_dotted: True if curve is dotted/dashed
        - confidence: 0-1 score (higher = more confident)
        - diagnostic_info: dict with metrics
    """
    if len(curve_pixels) < 50:
        return False, 0.0, {'reason': 'too_few_pixels'}

    diagnostics = {}

    # Method 1: Pixel density along curve
    density_score, density_info = _check_pixel_density(
        curve_pixels, panel_width, panel_height
    )
    diagnostics['density'] = density_info

    # Method 2: Continuity analysis (gap detection)
    continuity_score, continuity_info = _check_continuity(
        curve_pixels, panel_width, panel_height
    )
    diagnostics['continuity'] = continuity_info

    # Method 3: Horizontal line detection (common for reference lines)
    horizontal_score, horizontal_info = _check_horizontal(curve_pixels)
    diagnostics['horizontal'] = horizontal_info

    # Method 4: FFT periodicity detection
    fft_score, fft_info = _check_periodicity_fft(curve_pixels)
    diagnostics['fft'] = fft_info

    # Combine scores
    if method == 'density':
        is_dotted = density_score > 0.5
        confidence = density_score
    elif method == 'continuity':
        is_dotted = continuity_score > 0.5
        confidence = continuity_score
    elif method == 'fft':
        is_dotted = fft_score > 0.5
        confidence = fft_score
    else:  # combined
        # Weighted combination
        combined_score = (
            0.3 * density_score +
            0.4 * continuity_score +
            0.2 * horizontal_score +
            0.1 * fft_score
        )
        is_dotted = combined_score > 0.5
        confidence = combined_score

    diagnostics['combined_score'] = combined_score if method == 'combined' else None

    return is_dotted, confidence, diagnostics


def _check_pixel_density(
    curve_pixels: np.ndarray,
    panel_width: int,
    panel_height: int
) -> Tuple[float, Dict]:
    """
    Check pixel density. Dotted lines have lower density.

    Solid curves typically have 10-50 pixels per x-unit.
    Dotted lines have 2-10 pixels per x-unit.
    """
    # Create a raster image of just this curve
    img = np.zeros((panel_height, panel_width), dtype=np.uint8)
    for x, y in curve_pixels:
        if 0 <= int(x) < panel_width and 0 <= int(y) < panel_height:
            img[int(y), int(x)] = 255

    # Calculate pixels per x-unit (horizontal density)
    x_min = int(np.min(curve_pixels[:, 0]))
    x_max = int(np.max(curve_pixels[:, 0]))
    x_range = max(x_max - x_min, 1)

    density = len(curve_pixels) / x_range

    # Thresholds
    if density < 3:
        score = 0.9  # Very likely dotted
    elif density < 8:
        score = 0.6  # Probably dotted
    elif density < 15:
        score = 0.3  # Possibly dotted
    else:
        score = 0.1  # Probably solid

    return score, {
        'pixels_per_x_unit': density,
        'total_pixels': len(curve_pixels),
        'x_range': x_range,
        'threshold': 8
    }


def _check_continuity(
    curve_pixels: np.ndarray,
    panel_width: int,
    panel_height: int
) -> Tuple[float, Dict]:
    """
    Check for gaps in the curve. Dotted lines have many gaps.

    Uses connected component analysis on a rasterized version.
    """
    # Create raster image
    img = np.zeros((panel_height, panel_width), dtype=np.uint8)
    for x, y in curve_pixels:
        if 0 <= int(x) < panel_width and 0 <= int(y) < panel_height:
            img[int(y), int(x)] = 255

    # Dilate slightly to connect very close pixels (within 2-3 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_dilated = cv2.dilate(img, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_dilated, connectivity=8
    )

    # Exclude background (label 0)
    num_components = num_labels - 1

    if num_components == 0:
        return 0.0, {'num_components': 0}

    # Get component sizes
    component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background

    # Metrics
    avg_component_size = np.mean(component_sizes)
    max_component_size = np.max(component_sizes)
    total_pixels = len(curve_pixels)

    # If many small components, likely dotted
    # If one large component, likely solid
    if num_components > 20:
        score = 0.9
    elif num_components > 10:
        score = 0.7
    elif num_components > 5:
        score = 0.5
    elif num_components > 2:
        score = 0.3
    else:
        score = 0.1

    # Adjust based on component size distribution
    if max_component_size > 0.8 * total_pixels:
        # One dominant component - likely solid
        score *= 0.3

    return score, {
        'num_components': num_components,
        'avg_component_size': avg_component_size,
        'max_component_size': max_component_size,
        'total_pixels': total_pixels
    }


def _check_horizontal(curve_pixels: np.ndarray) -> Tuple[float, Dict]:
    """
    Check if curve is horizontal. Reference lines are often horizontal dotted lines.
    """
    if len(curve_pixels) < 10:
        return 0.0, {'reason': 'too_few_pixels'}

    # Calculate y-variance relative to x-range
    y_std = np.std(curve_pixels[:, 1])
    y_range = np.max(curve_pixels[:, 1]) - np.min(curve_pixels[:, 1])
    x_range = np.max(curve_pixels[:, 0]) - np.min(curve_pixels[:, 0])

    # If y-range is very small relative to x-range, it's horizontal
    if x_range > 0:
        horizontality = y_range / x_range
    else:
        horizontality = 1.0

    # Score based on horizontality
    if horizontality < 0.01:
        score = 0.9  # Very horizontal - likely reference line
    elif horizontality < 0.05:
        score = 0.6  # Quite horizontal
    elif horizontality < 0.1:
        score = 0.3  # Somewhat horizontal
    else:
        score = 0.0  # Not horizontal

    return score, {
        'y_std': y_std,
        'y_range': y_range,
        'x_range': x_range,
        'horizontality': horizontality
    }


def _check_periodicity_fft(curve_pixels: np.ndarray) -> Tuple[float, Dict]:
    """
    Use FFT to detect periodic patterns in dotted lines.

    Dotted/dashed lines have periodic gaps that show up as peaks in FFT.
    """
    if len(curve_pixels) < 100:
        return 0.0, {'reason': 'too_few_pixels'}

    # Sort pixels by x-coordinate
    sorted_idx = np.argsort(curve_pixels[:, 0])
    sorted_pixels = curve_pixels[sorted_idx]

    # Create a 1D signal: number of pixels in each x-bin
    x_min = int(np.min(sorted_pixels[:, 0]))
    x_max = int(np.max(sorted_pixels[:, 0]))
    x_range = x_max - x_min

    if x_range < 20:
        return 0.0, {'reason': 'x_range_too_small'}

    # Bin pixels by x
    bins = np.arange(x_min, x_max + 1)
    hist, _ = np.histogram(sorted_pixels[:, 0], bins=bins)

    # Apply FFT
    fft = np.fft.fft(hist)
    power = np.abs(fft) ** 2

    # Look for strong periodic components (excluding DC and very high frequencies)
    # Typical dash patterns: 5-20 pixel period
    n = len(power)
    freq = np.fft.fftfreq(n, d=1.0)

    # Focus on frequencies corresponding to 5-30 pixel periods
    period_range = (5, 30)
    freq_range = (1.0 / period_range[1], 1.0 / period_range[0])

    relevant_mask = (np.abs(freq) > freq_range[0]) & (np.abs(freq) < freq_range[1])
    if np.sum(relevant_mask) == 0:
        return 0.0, {'reason': 'no_relevant_frequencies'}

    relevant_power = power[relevant_mask]
    max_relevant_power = np.max(relevant_power)
    dc_power = power[0]

    # If there's strong periodic power relative to DC, it's dotted
    if dc_power > 0:
        periodicity_ratio = max_relevant_power / dc_power
    else:
        periodicity_ratio = 0

    # Score based on periodicity
    if periodicity_ratio > 0.3:
        score = 0.8
    elif periodicity_ratio > 0.15:
        score = 0.5
    elif periodicity_ratio > 0.05:
        score = 0.2
    else:
        score = 0.0

    return score, {
        'periodicity_ratio': periodicity_ratio,
        'max_relevant_power': max_relevant_power,
        'dc_power': dc_power
    }


def filter_solid_curves(
    curves: List[Dict],
    panel_width: int,
    panel_height: int,
    method: str = 'combined',
    confidence_threshold: float = 0.5,
    exclude_dotted: bool = True
) -> List[Dict]:
    """
    Filter curves to keep only solid curves (or only dotted curves).

    Args:
        curves: List of curve dicts from extract_curves()
        panel_width: width of panel
        panel_height: height of panel
        method: detection method ('combined', 'density', etc.)
        confidence_threshold: minimum confidence to classify as dotted
        exclude_dotted: If True, exclude dotted curves. If False, keep only dotted.

    Returns:
        Filtered list of curves
    """
    filtered_curves = []

    for curve in curves:
        points = curve['points']
        is_dotted, confidence, diagnostics = is_dotted_or_dashed(
            points, panel_width, panel_height, method=method
        )

        # Add diagnostic info to curve
        curve['dotted_detection'] = {
            'is_dotted': is_dotted,
            'confidence': confidence,
            'diagnostics': diagnostics
        }

        # Apply filter
        if exclude_dotted:
            if not is_dotted or confidence < confidence_threshold:
                filtered_curves.append(curve)
        else:
            if is_dotted and confidence >= confidence_threshold:
                filtered_curves.append(curve)

    return filtered_curves


def visualize_dotted_detection(
    curves: List[Dict],
    panel_width: int,
    panel_height: int,
    output_path: str
):
    """
    Visualize which curves are detected as dotted vs solid.

    Solid curves: green
    Dotted curves: red
    """
    from PIL import Image

    img = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

    for curve in curves:
        points = curve['points'].astype(int)
        detection = curve.get('dotted_detection', {})
        is_dotted = detection.get('is_dotted', False)
        confidence = detection.get('confidence', 0.0)

        # Color based on detection
        if is_dotted:
            color = (255, 0, 0)  # Red for dotted
        else:
            color = (0, 255, 0)  # Green for solid

        # Draw points
        for x, y in points:
            if 0 <= x < panel_width and 0 <= y < panel_height:
                cv2.circle(img, (x, y), radius=2, color=color, thickness=-1)

        # Add label
        if len(points) > 0:
            label_x = int(np.max(points[:, 0])) + 10
            label_y = int(np.mean(points[:, 1]))
            label = f"{'DOTTED' if is_dotted else 'SOLID'} ({confidence:.2f})"
            cv2.putText(
                img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    result_img = Image.fromarray(img)
    result_img.save(output_path)
    print(f"Dotted detection visualization saved: {output_path}")
