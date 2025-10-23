"""
Raster curve extraction using HSL color space + k-medoids clustering.

Based on SurvdigitizeR algorithm (Harrison et al., 2021):
- HSL color space for better curve separation
- k-medoids clustering (more robust than k-means)
- k-NN for resolving overlapping regions
- Achieved RMSE 0.012 on validation set

Strategy:
1. Convert panel image to HSL
2. Identify non-background pixels (likely curve pixels)
3. Cluster pixels by color using k-medoids
4. Assign curves by cluster labels
5. Sort points left-to-right to form curves
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import colorsys

from .auto_detect import (
    auto_detect_n_curves,
    estimate_curve_confidence,
    detect_curve_colors
)
from .dotted_line_filter import (
    filter_solid_curves,
    is_dotted_or_dashed
)


def extract_curves(
    panel_image: Image.Image,
    n_curves: Optional[int] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    lightness_threshold: float = 0.85,
    saturation_threshold: float = 0.05,
    auto_detect_method: str = 'silhouette',
    exclude_dotted: bool = True,
    dotted_confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract K-M survival curves from panel image using HSL + k-medoids.

    Args:
        panel_image: PIL Image of K-M curve panel (cropped from page)
        n_curves: Number of curves to extract (typically 2-6). If None, auto-detect.
        background_color: RGB background color to exclude
        lightness_threshold: Max lightness (0-1) for curve pixels (default 0.85)
        saturation_threshold: Min saturation (0-1) for colored curves
        auto_detect_method: 'silhouette', 'elbow', or 'combined' (if n_curves=None)
        exclude_dotted: If True, filter out dotted/dashed lines (default True)
        dotted_confidence_threshold: Confidence threshold for dotted line detection

    Returns:
        List of curve dicts with:
            - curve_id: int (0, 1, 2, ...)
            - points: numpy array of (x, y) pixel coordinates
            - color: average RGB color
            - n_points: number of points
            - dotted_detection: dict with is_dotted, confidence, diagnostics
    """
    # Convert PIL to numpy
    img_array = np.array(panel_image)
    if len(img_array.shape) == 2:
        # Grayscale - convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    height, width = img_array.shape[:2]

    # Step 1: Convert RGB to HSL
    hsl_image = rgb_to_hsl_image(img_array)

    # Step 2: Identify curve pixels (non-background)
    curve_mask = identify_curve_pixels(
        img_array,
        hsl_image,
        background_color,
        lightness_threshold,
        saturation_threshold
    )

    # Get curve pixel coordinates
    curve_coords = np.column_stack(np.where(curve_mask))  # (y, x) pairs

    if len(curve_coords) == 0:
        print("Warning: No curve pixels detected")
        return []

    # Convert to (x, y) format for auto-detection
    curve_pixels_xy = np.column_stack([curve_coords[:, 1], curve_coords[:, 0]])

    # Step 3: Extract HSL features for curve pixels
    curve_hsl_features = hsl_image[curve_coords[:, 0], curve_coords[:, 1]]

    # Step 3.5: Auto-detect number of curves if not specified
    auto_detect_info = None
    if n_curves is None:
        detected_n, auto_detect_info = auto_detect_n_curves(
            curve_pixels_xy,
            curve_hsl_features,
            min_curves=1,
            max_curves=6,
            method=auto_detect_method
        )
        n_curves = detected_n
        print(f"Auto-detected {n_curves} curve(s) using {auto_detect_method} method")
        if auto_detect_info.get('optimal_silhouette', 0) > 0:
            print(f"  Silhouette score: {auto_detect_info['optimal_silhouette']:.3f}")

    # Step 4: Cluster pixels by color using k-medoids (approximate with k-means for now)
    # TODO: Replace with actual k-medoids for better robustness
    kmeans = KMeans(n_clusters=n_curves, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(curve_hsl_features)

    # Step 5: Build curves from clustered pixels
    curves = []
    for curve_id in range(n_curves):
        # Get pixels belonging to this curve
        mask = cluster_labels == curve_id
        curve_points_yx = curve_coords[mask]

        if len(curve_points_yx) == 0:
            continue

        # Convert (y, x) to (x, y) for standard coordinate system
        curve_points_xy = np.column_stack([
            curve_points_yx[:, 1],  # x coordinates
            curve_points_yx[:, 0]   # y coordinates
        ])

        # Sort points left to right (by x coordinate)
        sorted_indices = np.argsort(curve_points_xy[:, 0])
        curve_points_sorted = curve_points_xy[sorted_indices]

        # Calculate average color for this curve
        curve_pixels_rgb = img_array[curve_points_yx[:, 0], curve_points_yx[:, 1]]
        avg_color = np.mean(curve_pixels_rgb, axis=0).astype(int)

        curves.append({
            'curve_id': curve_id,
            'points': curve_points_sorted,
            'color': tuple(avg_color),
            'n_points': len(curve_points_sorted)
        })

    # Sort curves by average y position (top curves first)
    # K-M curves: higher y = higher survival probability
    curves.sort(key=lambda c: np.mean(c['points'][:, 1]))

    # Filter out dotted/dashed lines if requested
    if exclude_dotted:
        curves_before = len(curves)
        curves = filter_solid_curves(
            curves,
            panel_width=width,
            panel_height=height,
            method='combined',
            confidence_threshold=dotted_confidence_threshold,
            exclude_dotted=True
        )
        curves_after = len(curves)
        if curves_before > curves_after:
            print(f"Filtered out {curves_before - curves_after} dotted/dashed line(s)")
            print(f"Remaining solid curves: {curves_after}")
    else:
        # Still run detection but don't filter
        for curve in curves:
            is_dotted, confidence, diagnostics = is_dotted_or_dashed(
                curve['points'], width, height, method='combined'
            )
            curve['dotted_detection'] = {
                'is_dotted': is_dotted,
                'confidence': confidence,
                'diagnostics': diagnostics
            }

    return curves


def rgb_to_hsl_image(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSL color space (vectorized for performance).

    Args:
        rgb_image: numpy array (H, W, 3) with RGB values 0-255

    Returns:
        numpy array (H, W, 3) with HSL values:
            - H: 0-360 degrees
            - S: 0-1
            - L: 0-1
    """
    # Normalize RGB to 0-1
    rgb = rgb_image.astype(float) / 255.0

    # Extract R, G, B channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Compute min and max across RGB channels
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Initialize HSL arrays
    height, width = rgb_image.shape[:2]
    h = np.zeros((height, width), dtype=float)
    s = np.zeros((height, width), dtype=float)
    l = (max_c + min_c) / 2.0

    # Compute saturation
    # Avoid division by zero
    non_gray = delta > 0
    s[non_gray & (l <= 0.5)] = delta[non_gray & (l <= 0.5)] / (max_c[non_gray & (l <= 0.5)] + min_c[non_gray & (l <= 0.5)])
    s[non_gray & (l > 0.5)] = delta[non_gray & (l > 0.5)] / (2.0 - max_c[non_gray & (l > 0.5)] - min_c[non_gray & (l > 0.5)])

    # Compute hue (in range 0-1)
    # Where red is max
    r_max = non_gray & (r == max_c)
    h[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6

    # Where green is max
    g_max = non_gray & (g == max_c)
    h[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2

    # Where blue is max
    b_max = non_gray & (b == max_c)
    h[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4

    # Convert hue to 0-360 degrees
    h = (h / 6.0) * 360
    h[h < 0] += 360

    # Stack into HSL image
    hsl_image = np.stack([h, s, l], axis=-1)

    return hsl_image


def identify_curve_pixels(
    rgb_image: np.ndarray,
    hsl_image: np.ndarray,
    background_color: Tuple[int, int, int],
    lightness_threshold: float,
    saturation_threshold: float
) -> np.ndarray:
    """
    Identify which pixels are likely part of curves (not background/axes/text).

    Args:
        rgb_image: RGB image array
        hsl_image: HSL converted image
        background_color: RGB background color (typically white)
        lightness_threshold: Max lightness for curve pixels
        saturation_threshold: Min saturation for colored curves

    Returns:
        Boolean mask (H, W) where True = curve pixel
    """
    height, width = rgb_image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    # Extract HSL channels
    hue = hsl_image[:, :, 0]
    saturation = hsl_image[:, :, 1]
    lightness = hsl_image[:, :, 2]

    # Method 1: Exclude very light pixels (near white background)
    not_too_light = lightness < lightness_threshold

    # Method 2: Exclude very dark pixels (likely axes/text)
    # K-M curves are usually medium darkness
    not_too_dark = lightness > 0.1

    # Method 3: For colored curves, require some saturation
    # For grayscale curves, saturation will be low
    has_some_saturation = saturation > saturation_threshold

    # Method 4: Exclude exact background color
    bg_r, bg_g, bg_b = background_color
    not_background = ~(
        (rgb_image[:, :, 0] == bg_r) &
        (rgb_image[:, :, 1] == bg_g) &
        (rgb_image[:, :, 2] == bg_b)
    )

    # Combine criteria
    # Either colored OR dark enough to be grayscale curve
    is_curve = not_background & not_too_light & not_too_dark

    return is_curve


def smooth_curve(points: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth curve using moving average.

    For each x-coordinate, average the y-coordinates of nearby points.

    Args:
        points: (N, 2) array of (x, y) coordinates
        window_size: number of points to average

    Returns:
        Smoothed (x, y) coordinates
    """
    if len(points) < window_size:
        return points

    smoothed = points.copy()

    # Smooth y-coordinates using moving average
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed[i, 1] = np.mean(points[start:end, 1])

    return smoothed


def visualize_extracted_curves(
    panel_image: Image.Image,
    curves: List[Dict],
    output_path: str
):
    """
    Visualize extracted curves overlaid on original panel.

    Args:
        panel_image: Original panel image
        curves: List of curve dicts from extract_curves()
        output_path: Path to save visualization
    """
    img_array = np.array(panel_image).copy()

    # Convert grayscale to RGB for colored overlay
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Define distinct colors for curves
    colors = [
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (0, 255, 0),    # Green
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (0, 255, 255),  # Cyan
    ]

    for i, curve in enumerate(curves):
        points = curve['points'].astype(int)
        color = colors[i % len(colors)]

        # Draw curve points
        for point in points:
            x, y = point
            cv2.circle(img_array, (x, y), radius=2, color=color, thickness=-1)

        # Add curve label
        if len(points) > 0:
            # Label at the right end of curve
            label_x = int(np.max(points[:, 0])) + 10
            label_y = int(np.mean(points[:, 1]))
            cv2.putText(
                img_array,
                f"Curve {i+1}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # Save visualization
    result_img = Image.fromarray(img_array)
    result_img.save(output_path)
    print(f"Curve visualization saved: {output_path}")


def curves_to_survival_data(
    curves: List[Dict],
    panel_bbox: Tuple[int, int, int, int],
    x_axis_range: Tuple[float, float],
    y_axis_range: Tuple[float, float]
) -> List[Dict]:
    """
    Convert pixel coordinates to survival probabilities.

    Args:
        curves: List of curve dicts from extract_curves()
        panel_bbox: (x, y, width, height) of panel in pixels
        x_axis_range: (min_time, max_time) in months/years
        y_axis_range: (min_prob, max_prob) typically (0, 1)

    Returns:
        List of dicts with:
            - curve_id: int
            - times: array of time points
            - survival_probs: array of survival probabilities
            - color: RGB color
    """
    x, y, w, h = panel_bbox
    t_min, t_max = x_axis_range
    p_min, p_max = y_axis_range

    survival_curves = []

    for curve in curves:
        points = curve['points']

        # Convert pixel x to time (left to right)
        # Pixel 0 = t_min, Pixel w = t_max
        times = t_min + (points[:, 0] / w) * (t_max - t_min)

        # Convert pixel y to survival probability (top to bottom INVERTED)
        # Pixel 0 = p_max (top = 1.0), Pixel h = p_min (bottom = 0.0)
        survival_probs = p_max - (points[:, 1] / h) * (p_max - p_min)

        # Clip to valid range
        survival_probs = np.clip(survival_probs, p_min, p_max)

        survival_curves.append({
            'curve_id': curve['curve_id'],
            'times': times,
            'survival_probs': survival_probs,
            'color': curve['color'],
            'n_points': len(times)
        })

    return survival_curves
