"""
Panel and layout detection for K-M curves.

Strategy:
1. Detect rectangular panels using edge detection + contour finding
2. Find axes within panels using Hough line detection
3. Locate at-risk tables below x-axis
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional


def detect_panels(
    image: Image.Image,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.5,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.5
) -> List[Dict]:
    """
    Detect K-M curve panels in page image.

    Args:
        image: PIL Image (RGB or grayscale)
        min_area_ratio: Minimum panel area as fraction of page area
        max_area_ratio: Maximum panel area as fraction of page area
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio

    Returns:
        List of dicts with:
            - bbox: (x, y, width, height) in pixels
            - confidence: float 0-1
            - axes: dict with x_axis and y_axis line coordinates
            - at_risk_region: optional bbox for numbers-at-risk table
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    # Calculate total page area
    page_area = gray.shape[0] * gray.shape[1]

    # Step 1: Edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Step 2: Find contours
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    panels = []

    # Step 3: Filter contours by size and shape
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Filter by area
        area_ratio = area / page_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Filter by aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # Filter by rectangularity (contour area vs bounding rect area)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / area if area > 0 else 0
        if rectangularity < 0.7:  # Should be fairly rectangular
            continue

        # Calculate confidence based on multiple factors
        confidence = _calculate_panel_confidence(
            gray, x, y, w, h, rectangularity, aspect_ratio
        )

        if confidence < 0.5:
            continue

        # Detect axes within panel
        axes = _detect_axes(gray[y:y+h, x:x+w])

        # Detect at-risk table region (below x-axis)
        at_risk_region = _detect_at_risk_region(
            gray, x, y, w, h, axes
        )

        panels.append({
            'bbox': (x, y, w, h),
            'confidence': confidence,
            'axes': axes,
            'at_risk_region': at_risk_region
        })

    # Sort panels by confidence (highest first)
    panels.sort(key=lambda p: p['confidence'], reverse=True)

    return panels


def _calculate_panel_confidence(
    gray: np.ndarray,
    x: int, y: int, w: int, h: int,
    rectangularity: float,
    aspect_ratio: float
) -> float:
    """
    Calculate confidence score for a potential panel.

    Factors:
    - Rectangularity (0.7-1.0 is good)
    - Aspect ratio (closer to 1.0-1.5 is typical for K-M curves)
    - Border strength (strong edges indicate panel border)
    """
    confidence = 0.0

    # Rectangularity score (30%)
    rect_score = min(rectangularity / 0.95, 1.0) * 0.3
    confidence += rect_score

    # Aspect ratio score (30%)
    # Prefer aspect ratios between 0.8 and 1.8
    ideal_aspect = 1.3
    aspect_diff = abs(aspect_ratio - ideal_aspect)
    aspect_score = max(0, 1.0 - aspect_diff / 2.0) * 0.3
    confidence += aspect_score

    # Border strength score (40%)
    # Check if the panel has strong borders
    border_score = _check_border_strength(gray, x, y, w, h) * 0.4
    confidence += border_score

    return min(confidence, 1.0)


def _check_border_strength(
    gray: np.ndarray,
    x: int, y: int, w: int, h: int,
    border_thickness: int = 5
) -> float:
    """
    Check if panel has strong black borders.

    Returns score 0-1 based on how dark the borders are.
    """
    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        return 0.0

    # Sample pixels along borders
    top_border = gray[y:y+border_thickness, x:x+w]
    bottom_border = gray[y+h-border_thickness:y+h, x:x+w]
    left_border = gray[y:y+h, x:x+border_thickness]
    right_border = gray[y:y+h, x+w-border_thickness:x+w]

    # Calculate mean darkness (lower = darker = better)
    borders = [top_border, bottom_border, left_border, right_border]
    mean_values = [np.mean(b) for b in borders if b.size > 0]

    if not mean_values:
        return 0.0

    avg_darkness = np.mean(mean_values)

    # Score: darker borders = higher score
    # Typical black border: ~0-50 (gray level)
    # Typical white background: ~240-255
    score = 1.0 - (avg_darkness / 255.0)

    return score


def _detect_axes(panel_gray: np.ndarray) -> Dict:
    """
    Detect x-axis and y-axis lines within a panel using Hough transform.

    Returns dict with:
        - x_axis: (x1, y1, x2, y2) or None
        - y_axis: (x1, y1, x2, y2) or None
    """
    # Apply edge detection
    edges = cv2.Canny(panel_gray, threshold1=50, threshold2=150)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=min(panel_gray.shape) // 4,
        maxLineGap=10
    )

    x_axis = None
    y_axis = None

    if lines is not None:
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

            # Horizontal: angle close to 0 or 180
            if angle < 15 or angle > 165:
                horizontal_lines.append((x1, y1, x2, y2, abs(x2 - x1)))

            # Vertical: angle close to 90
            elif 75 < angle < 105:
                vertical_lines.append((x1, y1, x2, y2, abs(y2 - y1)))

        # X-axis is typically near the bottom
        if horizontal_lines:
            horizontal_lines.sort(key=lambda l: l[1], reverse=True)  # Sort by y
            x1, y1, x2, y2, length = horizontal_lines[0]
            x_axis = (x1, y1, x2, y2)

        # Y-axis is typically on the left
        if vertical_lines:
            vertical_lines.sort(key=lambda l: l[0])  # Sort by x
            x1, y1, x2, y2, length = vertical_lines[0]
            y_axis = (x1, y1, x2, y2)

    return {
        'x_axis': x_axis,
        'y_axis': y_axis
    }


def _detect_at_risk_region(
    gray: np.ndarray,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    axes: Dict
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect numbers-at-risk table region below x-axis.

    Returns (x, y, width, height) or None.
    """
    if axes['x_axis'] is None:
        return None

    # x-axis coordinates (relative to panel)
    _, y1, _, y2 = axes['x_axis']
    x_axis_y = max(y1, y2)

    # At-risk region is typically 50-200 pixels below x-axis
    region_y_start = panel_y + x_axis_y + 10
    region_y_end = min(
        region_y_start + 200,
        gray.shape[0]
    )

    if region_y_start >= gray.shape[0]:
        return None

    # Region width matches panel width
    region_x = panel_x
    region_w = panel_w
    region_h = region_y_end - region_y_start

    return (region_x, region_y_start, region_w, region_h)


def visualize_panels(
    image: Image.Image,
    panels: List[Dict],
    output_path: str
):
    """
    Draw detected panels, axes, and at-risk regions on image and save.

    Args:
        image: Original PIL image
        panels: List of panel dicts from detect_panels()
        output_path: Path to save visualization
    """
    img_array = np.array(image).copy()

    # Convert grayscale to RGB for colored annotations
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    for i, panel in enumerate(panels):
        x, y, w, h = panel['bbox']
        confidence = panel['confidence']

        # Draw panel bounding box (green)
        cv2.rectangle(
            img_array,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            thickness=3
        )

        # Add panel label
        label = f"Panel {i+1} ({confidence:.2f})"
        cv2.putText(
            img_array,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Draw axes (blue)
        axes = panel['axes']
        if axes['x_axis']:
            x1, y1, x2, y2 = axes['x_axis']
            cv2.line(
                img_array,
                (x + x1, y + y1),
                (x + x2, y + y2),
                (255, 0, 0),
                thickness=3
            )

        if axes['y_axis']:
            x1, y1, x2, y2 = axes['y_axis']
            cv2.line(
                img_array,
                (x + x1, y + y1),
                (x + x2, y + y2),
                (255, 0, 0),
                thickness=3
            )

        # Draw at-risk region (red)
        if panel['at_risk_region']:
            ar_x, ar_y, ar_w, ar_h = panel['at_risk_region']
            cv2.rectangle(
                img_array,
                (ar_x, ar_y),
                (ar_x + ar_w, ar_y + ar_h),
                (0, 0, 255),
                thickness=2
            )

    # Save visualization
    result_img = Image.fromarray(img_array)
    result_img.save(output_path)
    print(f"Visualization saved: {output_path}")
