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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
    dotted_confidence_threshold: float = 0.5,
    remove_grid: bool = False
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
        remove_grid: If True, remove grid lines before extraction (recommended for R plots)

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

    # Step 0.5: Remove grid lines if requested
    if remove_grid:
        print("  Removing grid lines using Hough transform...")
        img_array = remove_grid_lines(img_array, angle_tolerance=2.0)

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


def consolidate_to_function(
    points_xy: np.ndarray,
    method: str = 'median',
    bin_width: int = 1
) -> np.ndarray:
    """
    Consolidate scattered curve pixels into a clean function (one y per x).

    For K-M curves, we often extract thousands of scattered pixels due to
    anti-aliasing, line width, shadows, etc. This function consolidates them
    into a proper step function with one y-value per x-coordinate.

    Args:
        points_xy: Array of (x, y) pixel coordinates (possibly thousands)
        method: 'median', 'mean', or 'mode' for aggregating y-values
        bin_width: Width of x-bins for grouping points (pixels)

    Returns:
        Clean curve as (x, y) array with one point per x-coordinate
    """
    if len(points_xy) == 0:
        return np.array([])

    # Sort by x
    sorted_idx = np.argsort(points_xy[:, 0])
    sorted_points = points_xy[sorted_idx]

    # Group by x-coordinate bins
    x_min = int(np.floor(sorted_points[0, 0]))
    x_max = int(np.ceil(sorted_points[-1, 0]))

    curve_points = []

    for x in range(x_min, x_max + 1, bin_width):
        # Get all points in this x-bin
        in_bin = (sorted_points[:, 0] >= x) & (sorted_points[:, 0] < x + bin_width)
        bin_points = sorted_points[in_bin]

        if len(bin_points) == 0:
            continue

        # Consolidate y-values
        x_center = x + bin_width / 2
        if method == 'median':
            y_value = np.median(bin_points[:, 1])
        elif method == 'mean':
            y_value = np.mean(bin_points[:, 1])
        elif method == 'mode':
            # For mode, use most common y-value (binned)
            y_counts = np.bincount(bin_points[:, 1].astype(int))
            y_value = np.argmax(y_counts)
        else:
            raise ValueError(f"Unknown method: {method}")

        curve_points.append([x_center, y_value])

    return np.array(curve_points)


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
    # Normalize RGB to 0-1 (use float32 to reduce memory by 50%)
    rgb = rgb_image.astype(np.float32) / 255.0

    # Extract R, G, B channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Compute min and max across RGB channels
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Initialize HSL arrays (use float32 to reduce memory)
    height, width = rgb_image.shape[:2]
    h = np.zeros((height, width), dtype=np.float32)
    s = np.zeros((height, width), dtype=np.float32)
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


def calculate_knn_coherence_scores(
    pixel_coords: np.ndarray,
    pixel_groups: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """
    Calculate k-NN coherence scores for all pixels (SurvdigitizeR algorithm).

    Formula: knn(p) = Σ(i=1 to k) [comp(p, nᵢ) / dist(p, nᵢ)²]

    Where:
    - comp(p, nᵢ) = +1 if pixel and neighbor in same group, -1 if different
    - dist(p, nᵢ) = Euclidean distance between pixel and neighbor

    High score = tightly integrated within cluster (real curve pixel)
    Low score = near different groups (artifact, grid line, overlap)

    Args:
        pixel_coords: (N, 2) array of (x, y) coordinates
        pixel_groups: (N,) array of group IDs (0, 1, 2, ...)
        k: Number of nearest neighbors (default 20, from SurvdigitizeR)

    Returns:
        scores: (N,) array of coherence scores
    """
    if len(pixel_coords) == 0:
        return np.array([])

    # Find k nearest neighbors for each pixel
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(pixel_coords)))
    nbrs.fit(pixel_coords)
    distances, indices = nbrs.kneighbors(pixel_coords)

    # Remove self (first neighbor at distance 0)
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        # Edge case: only 1 pixel
        return np.zeros(len(pixel_coords))

    # Get group IDs of neighbors
    neighbor_groups = pixel_groups[indices]  # (N, k)

    # Calculate comp(p, nᵢ): +1 if same group, -1 if different
    same_group = (neighbor_groups == pixel_groups[:, np.newaxis]).astype(int)
    comp = 2 * same_group - 1  # Converts: 0 → -1, 1 → +1

    # Calculate weights: 1 / dist²
    # Avoid division by zero (min distance = 0.1 pixels)
    distances = np.maximum(distances, 0.1)
    weights = 1.0 / (distances ** 2)

    # Calculate coherence score: Σ(comp * weight) / k
    scores = np.sum(comp * weights, axis=1) / distances.shape[1]

    return scores


def trace_curve_with_coherence(
    start_x: int,
    start_y: int,
    group_id: int,
    pixel_dict: Dict[Tuple[int, int], float],
    group_dict: Dict[Tuple[int, int], int],
    visited: set,
    max_iterations: int = 100000
) -> List[Tuple[int, int]]:
    """
    Trace a K-M curve from starting point using k-NN coherence scores.

    Algorithm (from SurvdigitizeR overlap_detect.R):
    1. Start at top-left pixel of curve
    2. Check pixels: down (y-1) and right (x+1)
    3. If both exist: choose one with higher coherence score
    4. If one exists: go that direction
    5. If neither exists: GUESS by summing surrounding scores
    6. Repeat until reaching curve endpoint

    This creates continuous curves and fills gaps automatically.

    Args:
        start_x, start_y: Starting pixel coordinates
        group_id: Group ID to trace
        pixel_dict: Maps (x, y) → coherence_score
        group_dict: Maps (x, y) → group_id
        visited: Set of already-visited pixels
        max_iterations: Safety limit to avoid infinite loops

    Returns:
        curve_path: List of (x, y) coordinates forming the curve
    """
    curve_path = [(start_x, start_y)]
    visited.add((start_x, start_y))

    x, y = start_x, start_y

    # Determine curve endpoint (bottom-right region)
    # In image coordinates: y increases downward, so max_y is at bottom
    group_pixels = [(px, py) for (px, py), g in group_dict.items() if g == group_id]
    if not group_pixels:
        return curve_path

    max_x = max(px for px, py in group_pixels)
    max_y = max(py for px, py in group_pixels)  # Bottom of curve (lowest survival)

    iterations = 0
    while x <= max_x and y <= max_y and iterations < max_iterations:
        iterations += 1

        # K-M curves step down and right
        # In image coordinates: y=0 is TOP, y increases DOWNWARD
        # So to trace downward (decreasing survival), we check y+1
        down = (x, y + 1)  # Move down in image (lower survival)
        right = (x + 1, y)  # Move right (later time)

        down_exists = (
            down in group_dict and
            group_dict[down] == group_id and
            down not in visited
        )
        right_exists = (
            right in group_dict and
            group_dict[right] == group_id and
            right not in visited
        )

        if down_exists and right_exists:
            # Both exist - choose one with higher coherence score
            if pixel_dict[down] > pixel_dict[right]:
                x, y = down
            else:
                x, y = right

        elif down_exists:
            x, y = down

        elif right_exists:
            x, y = right

        else:
            # Neither exists - FILL GAP by guessing based on surrounding scores
            # (This is the key innovation from SurvdigitizeR)

            # Calculate score_down: sum of scores below (y > current) and to the right
            score_down = 0.0
            for py in range(y + 1, min(y + 11, max_y + 1)):
                if (x, py) in pixel_dict and group_dict.get((x, py)) == group_id:
                    score_down += pixel_dict[(x, py)]
            for px in range(x + 1, min(x + 11, max_x + 1)):
                if (px, y + 1) in pixel_dict and group_dict.get((px, y + 1)) == group_id:
                    score_down += pixel_dict[(px, y + 1)]

            # Calculate score_right: sum of scores below (y > current) and to the right
            score_right = 0.0
            for py in range(y + 1, min(y + 11, max_y + 1)):
                if (x + 1, py) in pixel_dict and group_dict.get((x + 1, py)) == group_id:
                    score_right += pixel_dict[(x + 1, py)]
            for px in range(x + 1, min(x + 11, max_x + 1)):
                if (px, y) in pixel_dict and group_dict.get((px, y)) == group_id:
                    score_right += pixel_dict[(px, y)]

            # Choose direction with higher surrounding score
            if score_down > score_right:
                y = y + 1  # Move down in image
            else:
                x = x + 1  # Move right

        # Add to curve path
        curve_path.append((x, y))
        visited.add((x, y))

    return curve_path


def extract_curves_knn(
    panel_image: Image.Image,
    n_curves: Optional[int] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    lightness_threshold: float = 0.85,
    saturation_threshold: float = 0.05,
    k_neighbors: int = 20,
    auto_detect_method: str = 'silhouette',
    exclude_dotted: bool = True,
    dotted_confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract K-M curves using SurvdigitizeR's k-NN tracing algorithm.

    This is the improved version that fixes the curve pixel extraction problem.
    Instead of just clustering pixels by color, we:
    1. Cluster by color (initial grouping)
    2. Calculate k-NN coherence scores (identify artifacts)
    3. Trace curves pixel-by-pixel using coherence guidance
    4. Fill gaps automatically

    Args:
        panel_image: PIL Image of K-M curve panel
        n_curves: Number of curves (None = auto-detect)
        background_color: RGB background color
        lightness_threshold: Max lightness for curve pixels
        saturation_threshold: Min saturation (not used in k-NN version)
        k_neighbors: Number of neighbors for k-NN (default 20)
        auto_detect_method: Method for auto-detecting n_curves
        exclude_dotted: Filter out dotted lines
        dotted_confidence_threshold: Threshold for dotted detection

    Returns:
        List of curve dicts with curve_id, points, color, n_points
    """
    # Convert PIL to numpy
    img_array = np.array(panel_image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    height, width = img_array.shape[:2]

    # Step 1: Convert RGB to HSL
    hsl_image = rgb_to_hsl_image(img_array)

    # Step 2: Identify candidate curve pixels (lightness filtering)
    curve_mask = identify_curve_pixels(
        img_array, hsl_image, background_color,
        lightness_threshold, saturation_threshold
    )

    curve_coords = np.column_stack(np.where(curve_mask))  # (y, x)

    if len(curve_coords) == 0:
        print("Warning: No curve pixels detected")
        return []

    # Convert to (x, y) format
    pixel_coords_xy = curve_coords[:, ::-1]  # (x, y)

    # CRITICAL FIX: Remove pixels near edges (axes, borders, grid lines)
    # Curves are typically in the interior, not at edges
    edge_margin = 30  # pixels from edge
    interior_mask = (
        (pixel_coords_xy[:, 0] >= edge_margin) &  # Not too far left
        (pixel_coords_xy[:, 0] < width - edge_margin) &  # Not too far right
        (pixel_coords_xy[:, 1] >= edge_margin) &  # Not too far top
        (pixel_coords_xy[:, 1] < height - edge_margin)  # Not too far bottom
    )

    pixel_coords_xy = pixel_coords_xy[interior_mask]
    curve_coords = curve_coords[interior_mask]

    if len(pixel_coords_xy) == 0:
        print("Warning: No curve pixels after edge filtering")
        return []

    print(f"  After edge filtering: {len(pixel_coords_xy)} pixels (removed edges)")

    # Step 3: Extract HSL features
    curve_hsl_features = hsl_image[curve_coords[:, 0], curve_coords[:, 1]]

    # Step 4: Auto-detect number of curves if not specified
    if n_curves is None:
        detected_n, auto_detect_info = auto_detect_n_curves(
            pixel_coords_xy, curve_hsl_features,
            min_curves=1, max_curves=6, method=auto_detect_method
        )
        n_curves = detected_n
        print(f"Auto-detected {n_curves} curve(s) using {auto_detect_method}")

    # Step 5: Cluster pixels by color (k-means for now)
    kmeans = KMeans(n_clusters=n_curves, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(curve_hsl_features)

    # Step 6: Calculate k-NN coherence scores
    print(f"Calculating k-NN coherence scores (k={k_neighbors})...")
    coherence_scores = calculate_knn_coherence_scores(
        pixel_coords_xy, cluster_labels, k=k_neighbors
    )

    # Create lookup dictionaries
    pixel_dict = {
        tuple(pixel_coords_xy[i]): coherence_scores[i]
        for i in range(len(pixel_coords_xy))
    }
    group_dict = {
        tuple(pixel_coords_xy[i]): cluster_labels[i]
        for i in range(len(pixel_coords_xy))
    }

    # Step 7: Calculate average coherence per group
    group_avg_scores = {}
    for group_id in range(n_curves):
        group_mask = cluster_labels == group_id
        if np.any(group_mask):
            group_avg_scores[group_id] = np.mean(coherence_scores[group_mask])
        else:
            group_avg_scores[group_id] = 0.0

    # Sort groups by average coherence
    # TRY: Trace LOWEST coherence first (might be actual curve, not grid lines)
    sorted_groups = sorted(
        group_avg_scores.items(),
        key=lambda x: x[1],
        reverse=False  # Changed from True - trace low coherence first
    )

    print(f"Group coherence scores: {group_avg_scores}")

    # Step 8: Trace each curve using coherence guidance
    curves = []
    visited = set()

    for group_id, avg_score in sorted_groups:
        # Find starting point: leftmost-topmost pixel of this group
        group_pixels = [
            (x, y) for (x, y), g in group_dict.items()
            if g == group_id
        ]

        if not group_pixels:
            continue

        # Strategy: Find pixels in the leftmost region, then pick the topmost
        # This avoids picking a grid line pixel at the left edge
        min_x = min(px for px, py in group_pixels)
        leftmost_region = [
            (px, py) for px, py in group_pixels
            if px <= min_x + 20  # Within 20 pixels of left edge
        ]

        # Among leftmost pixels, pick the TOP-most (smallest y)
        # K-M curves start at early time (left) with high survival (top)
        start_x, start_y = min(leftmost_region, key=lambda p: p[1])

        if (start_x, start_y) in visited:
            continue

        print(f"Tracing curve {group_id} from ({start_x}, {start_y})...")

        # Trace curve using k-NN coherence
        curve_path = trace_curve_with_coherence(
            start_x, start_y, group_id,
            pixel_dict, group_dict, visited
        )

        # Convert to (x, y) numpy array
        curve_points_xy = np.array(curve_path)

        if len(curve_points_xy) < 10:
            print(f"  Skipped: curve too short ({len(curve_points_xy)} points)")
            continue

        # Calculate average color
        # Convert (x, y) back to (y, x) for image indexing
        curve_pixels_yx = curve_points_xy[:, ::-1]
        curve_rgb = img_array[curve_pixels_yx[:, 0], curve_pixels_yx[:, 1]]
        avg_color = np.mean(curve_rgb, axis=0).astype(int)

        curves.append({
            'curve_id': group_id,
            'points': curve_points_xy,
            'color': tuple(avg_color),
            'n_points': len(curve_points_xy),
            'avg_coherence': avg_score
        })

        print(f"  Traced {len(curve_points_xy)} points (avg coherence: {avg_score:.2f})")

    # Sort curves by average y position (top first)
    curves.sort(key=lambda c: np.mean(c['points'][:, 1]))

    # Filter out dotted/dashed lines if requested
    if exclude_dotted:
        curves_before = len(curves)
        curves = filter_solid_curves(
            curves, panel_width=width, panel_height=height,
            method='combined', confidence_threshold=dotted_confidence_threshold,
            exclude_dotted=True
        )
        curves_after = len(curves)
        if curves_before > curves_after:
            print(f"Filtered out {curves_before - curves_after} dotted line(s)")

    return curves


def remove_grid_lines(img_array: np.ndarray, angle_tolerance: float = 2.0) -> np.ndarray:
    """
    Remove grid lines from image using Hough Line Transform.

    Grid lines are perfectly horizontal (0°) or vertical (90°) straight lines.
    This removes them while preserving curves which are NOT perfectly straight.

    Args:
        img_array: RGB or grayscale image as numpy array
        angle_tolerance: Tolerance for horizontal/vertical detection (degrees)

    Returns:
        Image with grid lines removed (same format as input)
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Create binary mask of dark pixels (potential lines)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,  # Distance resolution in pixels
        theta=np.pi/180,  # Angle resolution in radians
        threshold=50,  # Minimum votes
        minLineLength=30,  # Minimum line length
        maxLineGap=5  # Maximum gap between segments
    )

    if lines is None:
        return img_array

    # Create mask for grid lines
    mask = np.zeros_like(gray)

    angle_tol_rad = np.deg2rad(angle_tolerance)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle of line
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            # Vertical line (90 degrees)
            angle = np.pi / 2
        else:
            angle = np.arctan(abs(dy / dx))

        # Check if line is horizontal (angle ≈ 0) or vertical (angle ≈ 90°)
        is_horizontal = angle < angle_tol_rad
        is_vertical = abs(angle - np.pi/2) < angle_tol_rad

        if is_horizontal or is_vertical:
            # Draw line on mask (thicker to ensure full removal)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)

    # Dilate mask slightly to ensure complete grid removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Remove grid lines by setting those pixels to white (background)
    result = img_array.copy()
    if len(result.shape) == 3:
        # RGB image
        result[mask_dilated > 0] = [255, 255, 255]
    else:
        # Grayscale
        result[mask_dilated > 0] = 255

    return result


def extract_curves_morphological(
    panel_image: Image.Image,
    n_curves: Optional[int] = None,
    min_curve_length: int = 100,
    exclude_dotted: bool = True,
    dotted_confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract K-M curves using hybrid color + morphological approach.

    This combines the best of both worlds:
    1. Use HSL color clustering to find curve pixels (not grid lines)
    2. Use morphological operations to clean up and separate curves
    3. Extract contours from cleaned binary mask

    Args:
        panel_image: PIL Image of K-M curve panel
        n_curves: Number of curves to extract (if None, return all)
        min_curve_length: Minimum contour length in pixels
        exclude_dotted: If True, filter out dotted/dashed lines
        dotted_confidence_threshold: Confidence threshold for dotted detection

    Returns:
        List of curve dicts (same format as extract_curves())
    """
    # Convert PIL to numpy
    img_array = np.array(panel_image)
    if len(img_array.shape) == 2:
        gray = img_array
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # Step 1: Identify curve pixels using inverse threshold
    # K-M curves are darker than background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Remove small noise with morphological opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Step 3: Connect curve segments with morphological closing
    # Use horizontal kernel since K-M curves are mostly horizontal
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 4: Find contours
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        print("Warning: No contours detected")
        return []

    # Step 4: Filter contours by curve-like properties
    curve_candidates = []

    for contour in contours:
        # Get contour length
        length = cv2.arcLength(contour, closed=False)

        if length < min_curve_length:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by aspect ratio (curves should be reasonably wide)
        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 1.5:  # Relaxed from 2.0 - allow narrower curves
            continue

        # Filter by width - curves should span significant horizontal distance
        if w < width * 0.3:  # Relaxed from 0.5 - allow shorter curves
            continue

        # Check for continuity - no huge gaps
        contour_points = contour.reshape(-1, 2)
        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]

        # Sort by x coordinate
        sorted_indices = np.argsort(x_coords)
        x_sorted = x_coords[sorted_indices]
        y_sorted = y_coords[sorted_indices]

        # Check for large x-gaps (discontinuities)
        if len(x_sorted) > 1:
            x_gaps = np.diff(x_sorted)
            max_gap = np.max(x_gaps) if len(x_gaps) > 0 else 0

            # Skip if there's a huge gap (clearly fragmented)
            if max_gap > 100:  # Relaxed from 50 - allow some gaps
                continue

        # Filter by smoothness - K-M curves should not be too jagged
        # Calculate average y-change per x-step
        if len(x_sorted) > 10:
            y_changes = np.abs(np.diff(y_sorted))
            avg_y_change = np.mean(y_changes)

            # Skip if extremely jagged (likely noise or text)
            if avg_y_change > 10.0:  # Relaxed from 5.0 - allow more variation
                continue

        # This contour passes all filters
        curve_candidates.append({
            'contour': contour,
            'length': length,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'points': contour_points
        })

    if len(curve_candidates) == 0:
        print("Warning: No curve-like contours found after filtering")
        return []

    # Step 5: Sort by length (longest curves first)
    curve_candidates.sort(key=lambda c: c['length'], reverse=True)

    # Step 6: Select top n_curves if specified
    if n_curves is not None:
        curve_candidates = curve_candidates[:n_curves]

    # Step 7: Convert to standard curve format
    curves = []
    for i, candidate in enumerate(curve_candidates):
        points = candidate['points']

        # Sort points left-to-right (by x coordinate)
        sorted_indices = np.argsort(points[:, 0])
        points = points[sorted_indices]

        # Get average color from original image at curve points
        if len(img_array.shape) == 3:
            colors = []
            for x, y in points:
                if 0 <= y < height and 0 <= x < width:
                    colors.append(img_array[y, x])
            avg_color = tuple(np.mean(colors, axis=0).astype(int)) if colors else (0, 0, 0)
        else:
            avg_color = (0, 0, 0)

        curves.append({
            'curve_id': i,
            'points': points,
            'color': avg_color,
            'n_points': len(points),
            'length': candidate['length'],
            'bounding_box': (candidate['x'], candidate['y'], candidate['width'], candidate['height'])
        })

    # Step 8: Sort by vertical position (topmost first - smallest y)
    curves.sort(key=lambda c: np.mean(c['points'][:, 1]))

    # Reassign curve IDs after sorting
    for i, curve in enumerate(curves):
        curve['curve_id'] = i

    # Step 9: Filter dotted lines if requested
    if exclude_dotted:
        curves_before = len(curves)
        curves = filter_solid_curves(
            curves, panel_width=width, panel_height=height,
            method='combined', confidence_threshold=dotted_confidence_threshold,
            exclude_dotted=True
        )
        curves_after = len(curves)
        if curves_before > curves_after:
            print(f"Filtered out {curves_before - curves_after} dotted line(s)")

    return curves


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
