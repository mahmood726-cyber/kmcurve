#!/usr/bin/env python3
"""
Automatic detection of the number of curves in a K-M plot panel.

Uses silhouette analysis and elbow method to determine the optimal number
of distinct survival curves present in the image.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Tuple, Dict, List
import colorsys


def auto_detect_n_curves(
    curve_pixels: np.ndarray,
    curve_features: np.ndarray,
    min_curves: int = 1,
    max_curves: int = 6,
    method: str = 'silhouette'
) -> Tuple[int, Dict]:
    """
    Automatically detect the optimal number of curves.

    Args:
        curve_pixels: (N, 2) array of pixel coordinates
        curve_features: (N, F) array of features (HSL, spatial, etc.)
        min_curves: Minimum number of curves to consider
        max_curves: Maximum number of curves to consider
        method: 'silhouette', 'elbow', or 'combined'

    Returns:
        Tuple of (optimal_n_curves, diagnostic_info)
    """
    if len(curve_pixels) < 100:
        return 1, {'reason': 'Too few pixels', 'n_pixels': len(curve_pixels)}

    # Normalize features for better clustering
    features_normalized = _normalize_features(curve_features)

    # Try different numbers of clusters
    results = {}
    for n in range(min_curves, max_curves + 1):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)

        # Compute metrics
        if n > 1:
            silhouette_avg = silhouette_score(features_normalized, labels)
            silhouette_samples_vals = silhouette_samples(features_normalized, labels)
        else:
            silhouette_avg = 0
            silhouette_samples_vals = np.zeros(len(labels))

        inertia = kmeans.inertia_

        # Check cluster sizes
        cluster_sizes = [np.sum(labels == i) for i in range(n)]
        min_cluster_size = min(cluster_sizes)

        results[n] = {
            'silhouette': silhouette_avg,
            'inertia': inertia,
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min_cluster_size,
            'silhouette_samples': silhouette_samples_vals
        }

    # Select optimal number based on method
    if method == 'silhouette':
        optimal_n = _select_by_silhouette(results, min_curves, max_curves)
    elif method == 'elbow':
        optimal_n = _select_by_elbow(results, min_curves, max_curves)
    else:  # combined
        optimal_n = _select_combined(results, min_curves, max_curves)

    diagnostic_info = {
        'method': method,
        'all_results': results,
        'optimal_n': optimal_n,
        'optimal_silhouette': results[optimal_n]['silhouette'] if optimal_n > 1 else 0,
        'optimal_cluster_sizes': results[optimal_n]['cluster_sizes']
    }

    return optimal_n, diagnostic_info


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    return (features - mean) / std


def _select_by_silhouette(results: Dict, min_n: int, max_n: int) -> int:
    """
    Select optimal number of curves based on silhouette score.

    Chooses the number with the highest silhouette score,
    but penalizes very small clusters.
    """
    best_n = min_n
    best_score = -1

    for n in range(min_n, max_n + 1):
        if n == 1:
            continue

        score = results[n]['silhouette']
        min_cluster_size = results[n]['min_cluster_size']

        # Penalize if any cluster is very small (< 1% of pixels)
        total_pixels = sum(results[n]['cluster_sizes'])
        if min_cluster_size < 0.01 * total_pixels:
            score *= 0.5  # Penalty

        if score > best_score:
            best_score = score
            best_n = n

    return best_n


def _select_by_elbow(results: Dict, min_n: int, max_n: int) -> int:
    """
    Select optimal number of curves using elbow method on inertia.

    Finds the point where adding more clusters doesn't significantly
    reduce inertia.
    """
    inertias = [results[n]['inertia'] for n in range(min_n, max_n + 1)]

    # Compute rate of change in inertia
    deltas = []
    for i in range(1, len(inertias)):
        delta = inertias[i-1] - inertias[i]
        deltas.append(delta)

    # Find elbow: where delta decreases significantly
    if len(deltas) < 2:
        return min_n

    # Compute second derivative (change in rate of change)
    second_deriv = []
    for i in range(1, len(deltas)):
        second_deriv.append(deltas[i-1] - deltas[i])

    # Elbow is where second derivative is maximum
    if second_deriv:
        elbow_idx = np.argmax(second_deriv)
        return min_n + elbow_idx + 1
    else:
        return min_n


def _select_combined(results: Dict, min_n: int, max_n: int) -> int:
    """
    Combined method: uses both silhouette and elbow.

    Prefers silhouette, but validates with elbow method.
    """
    silhouette_optimal = _select_by_silhouette(results, min_n, max_n)
    elbow_optimal = _select_by_elbow(results, min_n, max_n)

    # If they agree, use that
    if silhouette_optimal == elbow_optimal:
        return silhouette_optimal

    # If they differ by 1, prefer the higher silhouette
    if abs(silhouette_optimal - elbow_optimal) == 1:
        return silhouette_optimal

    # If they differ significantly, prefer silhouette but cap at elbow + 1
    if silhouette_optimal > elbow_optimal + 1:
        return elbow_optimal + 1
    else:
        return silhouette_optimal


def estimate_curve_confidence(
    curve_pixels: np.ndarray,
    curve_features: np.ndarray,
    labels: np.ndarray,
    n_curves: int
) -> Dict[int, float]:
    """
    Estimate confidence for each detected curve.

    Returns a dict mapping curve_id to confidence score (0-1).
    Higher confidence means the curve is well-separated and substantial.
    """
    confidences = {}

    # Compute silhouette scores per cluster
    if n_curves > 1:
        silhouette_vals = silhouette_samples(curve_features, labels)
    else:
        silhouette_vals = np.ones(len(labels))

    total_pixels = len(labels)

    for curve_id in range(n_curves):
        mask = labels == curve_id

        # Size factor: larger curves are more confident
        size = np.sum(mask)
        size_score = min(1.0, size / (0.1 * total_pixels))

        # Separation factor: how well separated from other curves
        if n_curves > 1:
            separation_score = np.mean(silhouette_vals[mask])
            # Silhouette ranges from -1 to 1, map to 0 to 1
            separation_score = (separation_score + 1) / 2
        else:
            separation_score = 1.0

        # Compactness factor: spatial coherence
        curve_coords = curve_pixels[mask]
        if len(curve_coords) > 10:
            # Measure spread in y-direction (curves should be functions of x)
            x_bins = np.digitize(curve_coords[:, 0], bins=np.linspace(
                curve_coords[:, 0].min(), curve_coords[:, 0].max(), 20
            ))
            y_spreads = []
            for bin_id in range(1, 21):
                bin_mask = x_bins == bin_id
                if np.sum(bin_mask) > 0:
                    y_spread = np.std(curve_coords[bin_mask, 1])
                    y_spreads.append(y_spread)

            avg_spread = np.mean(y_spreads) if y_spreads else 0
            # Lower spread is better (more compact curve)
            compactness_score = 1.0 / (1.0 + avg_spread / 10.0)
        else:
            compactness_score = 0.5

        # Combined confidence (weighted average)
        confidence = (
            0.4 * size_score +
            0.4 * separation_score +
            0.2 * compactness_score
        )

        confidences[curve_id] = confidence

    return confidences


def detect_curve_colors(
    curve_pixels: np.ndarray,
    rgb_image: np.ndarray,
    labels: np.ndarray,
    n_curves: int
) -> Dict[int, Dict]:
    """
    Detect the dominant color of each curve for visualization.

    Returns dict mapping curve_id to color info (RGB, HSL, hex).
    """
    colors = {}

    for curve_id in range(n_curves):
        mask = labels == curve_id
        curve_coords = curve_pixels[mask]

        # Sample RGB values at curve pixels
        rgb_values = []
        for x, y in curve_coords[:1000]:  # Sample up to 1000 pixels
            if 0 <= int(y) < rgb_image.shape[0] and 0 <= int(x) < rgb_image.shape[1]:
                rgb = rgb_image[int(y), int(x), :3]
                rgb_values.append(rgb)

        if rgb_values:
            # Compute median RGB
            rgb_values = np.array(rgb_values)
            median_rgb = np.median(rgb_values, axis=0).astype(int)

            # Convert to HSL
            r, g, b = median_rgb / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)

            # Convert to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(*median_rgb)

            colors[curve_id] = {
                'rgb': tuple(median_rgb),
                'hsl': (h * 360, s, l),
                'hex': hex_color,
                'name': _color_name_from_hsl(h * 360, s, l)
            }
        else:
            colors[curve_id] = {
                'rgb': (0, 0, 0),
                'hsl': (0, 0, 0),
                'hex': '#000000',
                'name': 'black'
            }

    return colors


def _color_name_from_hsl(h: float, s: float, l: float) -> str:
    """Simple color naming based on HSL values."""
    if l < 0.2:
        return 'black'
    if l > 0.9:
        return 'white'
    if s < 0.1:
        return 'gray'

    # Hue-based names
    if h < 15 or h >= 345:
        return 'red'
    elif h < 45:
        return 'orange'
    elif h < 75:
        return 'yellow'
    elif h < 165:
        return 'green'
    elif h < 255:
        return 'blue'
    elif h < 285:
        return 'purple'
    elif h < 345:
        return 'pink'
    else:
        return 'red'
