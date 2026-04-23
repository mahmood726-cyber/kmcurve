#!/usr/bin/env python3
"""
Mathematical KM Curve Extraction

Uses the mathematical properties of Kaplan-Meier curves:
1. Step functions (piecewise constant)
2. Monotonically decreasing
3. Vertical drops at event times
4. Horizontal segments between events

This approach ENFORCES the KM mathematical structure instead of
just smoothing pixels.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import curve_fit, minimize
from sklearn.isotonic import IsotonicRegression
import cv2


class MathematicalKMExtractor:
    """
    Extract KM curves by enforcing mathematical constraints.

    Mathematical Properties of KM Curves:
    - S(t) is a step function
    - S(t) is monotonically decreasing
    - S(t) has horizontal segments (constant survival)
    - S(t) has vertical drops (events)
    - 0 <= S(t) <= 1
    - S(t) is right-continuous
    """

    def __init__(self):
        self.min_step_height = 0.01  # Minimum drop in survival
        self.min_segment_length = 5  # Minimum points per segment

    def extract_step_function(self, time, survival):
        """
        Convert noisy pixels to true KM step function.

        Strategy:
        1. Detect vertical steps (event times)
        2. Detect horizontal plateaus (constant survival)
        3. Reconstruct as piecewise constant function
        4. Enforce mathematical constraints
        """

        # Sort by time
        idx = np.argsort(time)
        t = time[idx]
        s = survival[idx]

        # STEP 1: Detect steps using derivative
        steps = self.detect_steps(t, s)

        # STEP 2: Segment into plateaus
        segments = self.segment_plateaus(t, s, steps)

        # STEP 3: Reconstruct as step function
        km_curve = self.reconstruct_step_function(segments)

        return km_curve

    def detect_steps(self, time, survival):
        """
        Detect vertical steps (event times) in KM curve.

        KM curves have vertical drops where events occur.
        Use first derivative to find rapid changes.
        """

        if len(time) < 5:
            return []

        # Calculate discrete derivative
        dt = np.diff(time)
        ds = np.diff(survival)

        # Avoid division by zero
        dt[dt == 0] = 1e-10

        # Derivative (rate of change)
        derivative = ds / dt

        # Steps are where derivative is large and negative
        # (survival drops rapidly)
        threshold = np.percentile(np.abs(derivative), 75)  # Top 25% changes

        step_indices = np.where(
            (derivative < -threshold) |  # Large negative change
            (np.abs(ds) > self.min_step_height)  # Or absolute drop
        )[0]

        return step_indices

    def segment_plateaus(self, time, survival, step_indices):
        """
        Segment curve into horizontal plateaus.

        Between steps, survival should be constant (horizontal line).
        """

        segments = []

        # Add boundaries
        boundaries = np.concatenate([[0], step_indices + 1, [len(time)]])

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end - start < 2:
                continue

            # Extract segment
            seg_time = time[start:end]
            seg_survival = survival[start:end]

            # Plateau value = median of segment (robust to noise)
            plateau_survival = np.median(seg_survival)

            segments.append({
                'time_start': seg_time[0],
                'time_end': seg_time[-1],
                'survival': plateau_survival,
                'n_points': len(seg_time)
            })

        return segments

    def reconstruct_step_function(self, segments):
        """
        Reconstruct KM curve as true step function.

        Mathematical form:
        S(t) = constant for t in [t_i, t_{i+1})
        S(t) drops at t_{i+1}
        """

        if not segments:
            return pd.DataFrame({'time': [], 'survival': []})

        # Create step function
        times = []
        survivals = []

        for i, seg in enumerate(segments):
            # Add horizontal segment
            times.append(seg['time_start'])
            survivals.append(seg['survival'])

            # Add step (end of segment)
            if i < len(segments) - 1:
                # Vertical drop to next segment
                times.append(seg['time_end'])
                survivals.append(segments[i + 1]['survival'])

        # Final point
        times.append(segments[-1]['time_end'])
        survivals.append(segments[-1]['survival'])

        # Enforce constraints
        times = np.array(times)
        survivals = np.array(survivals)

        # 1. Monotonic decreasing
        for i in range(1, len(survivals)):
            if survivals[i] > survivals[i-1]:
                survivals[i] = survivals[i-1]

        # 2. Bounded [0, 1]
        survivals = np.clip(survivals, 0, 1)

        # 3. Remove near-duplicates
        unique_t = []
        unique_s = []
        prev_s = None

        for t, s in zip(times, survivals):
            if prev_s is None or abs(s - prev_s) > 0.005:  # 0.5% threshold
                unique_t.append(t)
                unique_s.append(s)
                prev_s = s

        return pd.DataFrame({
            'time': unique_t,
            'survival': unique_s
        })

    def fit_parametric_survival(self, time, survival, dist='weibull'):
        """
        Fit parametric survival distribution.

        KM curves often follow known distributions:
        - Weibull (most common)
        - Exponential
        - Lognormal
        - Log-logistic

        This enforces smooth mathematical form.
        """

        if dist == 'weibull':
            # Weibull survival: S(t) = exp(-(t/lambda)^k)
            def weibull_surv(t, lam, k):
                return np.exp(-(t / lam) ** k)

            # Fit
            try:
                popt, _ = curve_fit(
                    weibull_surv,
                    time,
                    survival,
                    p0=[np.median(time), 1.0],  # Initial guess
                    bounds=([0, 0.1], [np.inf, 10])  # Reasonable bounds
                )

                # Generate fitted curve
                t_fit = np.linspace(time.min(), time.max(), 100)
                s_fit = weibull_surv(t_fit, *popt)

                return pd.DataFrame({'time': t_fit, 'survival': s_fit})

            except Exception as e:
                print(f"Warning: Weibull fit failed: {e}")
                return None

        elif dist == 'exponential':
            # Exponential survival: S(t) = exp(-lambda * t)
            def exp_surv(t, lam):
                return np.exp(-lam * t)

            try:
                popt, _ = curve_fit(
                    exp_surv,
                    time,
                    survival,
                    p0=[1/np.median(time)],
                    bounds=([0], [np.inf])
                )

                t_fit = np.linspace(time.min(), time.max(), 100)
                s_fit = exp_surv(t_fit, *popt)

                return pd.DataFrame({'time': t_fit, 'survival': s_fit})

            except Exception as e:
                print(f"Warning: Exponential fit failed: {e}")
                return None

        return None

    def enforce_km_constraints(self, df):
        """
        Enforce all KM mathematical constraints on extracted curve.

        Constraints:
        1. Monotonic decreasing
        2. Bounded [0, 1]
        3. Right-continuous (step function)
        4. Non-negative step sizes
        """

        df = df.sort_values('time').reset_index(drop=True)

        # 1. Monotonic decreasing
        for i in range(1, len(df)):
            if df.loc[i, 'survival'] > df.loc[i-1, 'survival']:
                df.loc[i, 'survival'] = df.loc[i-1, 'survival']

        # 2. Bounded
        df['survival'] = np.clip(df['survival'], 0, 1)

        # 3. Remove zero-length steps
        df['time_diff'] = df['time'].diff()
        df['surv_diff'] = df['survival'].diff().abs()

        if df.empty:
            return df[['time', 'survival']] if 'time' in df.columns else df

        # Keep points with either time change OR survival change
        keep = (df['time_diff'] > 1e-6) | (df['surv_diff'] > 1e-6)
        keep.iloc[0] = True  # Always keep first point  # sentinel:skip-line P1-empty-dataframe-access  (guarded by df.empty above)

        df = df[keep][['time', 'survival']].reset_index(drop=True)

        return df


def extract_km_with_math_constraints(pixel_df, axis_info):
    """
    Main entry point: Extract KM curve using mathematical constraints.

    Args:
        pixel_df: DataFrame with pixel coordinates (x, y)
        axis_info: Axis calibration (x_min, x_max, y_min, y_max)

    Returns:
        DataFrame with (time, survival) enforcing KM mathematics
    """

    # Calibrate pixels to coordinates
    df = calibrate_pixels(pixel_df, axis_info)

    # Extract step function
    extractor = MathematicalKMExtractor()
    km_curve = extractor.extract_step_function(
        df['time'].values,
        df['survival'].values
    )

    # Enforce final constraints
    km_curve = extractor.enforce_km_constraints(km_curve)

    return km_curve


def calibrate_pixels(pixel_df, axis_info):
    """Convert pixel coordinates to (time, survival) coordinates."""

    # Get axis bounds
    x_min, x_max = axis_info['x_min'], axis_info['x_max']
    y_min, y_max = axis_info['y_min'], axis_info['y_max']

    # Get pixel bounds
    px_x_min, px_x_max = pixel_df['x'].min(), pixel_df['x'].max()
    px_y_min, px_y_max = pixel_df['y'].min(), pixel_df['y'].max()

    # Linear calibration
    time = x_min + (pixel_df['x'] - px_x_min) / (px_x_max - px_x_min) * (x_max - x_min)

    # Y-axis is inverted in images (top = high survival)
    survival = y_max - (pixel_df['y'] - px_y_min) / (px_y_max - px_y_min) * (y_max - y_min)

    return pd.DataFrame({'time': time, 'survival': survival})


# ============================================================================
# EDGE DETECTION FOR STEP FUNCTIONS
# ============================================================================

def detect_km_steps_with_edges(img_panel):
    """
    Use edge detection to find vertical steps in KM curves.

    KM curves have:
    - Horizontal edges (tops/bottoms of plateaus)
    - Vertical edges (event times)

    We can detect these directly!
    """

    # Convert to grayscale
    if len(img_panel.shape) == 3:
        gray = cv2.cvtColor(img_panel, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_panel

    # Detect edges
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Separate horizontal and vertical edges
    kernel_h = np.ones((1, 15), np.uint8)  # Horizontal kernel
    kernel_v = np.ones((15, 1), np.uint8)  # Vertical kernel

    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)

    return horizontal_edges, vertical_edges


def reconstruct_from_edges(horizontal_edges, vertical_edges):
    """
    Reconstruct KM curve from detected edges.

    Strategy:
    1. Vertical edges = event times (where survival drops)
    2. Horizontal edges = survival levels (plateaus)
    3. Reconstruct step function from these
    """

    # Find vertical edge locations (event times)
    v_locs = np.where(vertical_edges.any(axis=0))[0]

    # Find horizontal edge locations (survival levels)
    h_locs = np.where(horizontal_edges.any(axis=1))[0]

    # Reconstruct step function
    # ... (detailed implementation)

    pass


if __name__ == "__main__":
    # Test
    print("Mathematical KM Extraction - Testing")
    print("="*60)

    # Generate test data (noisy step function)
    np.random.seed(42)
    true_steps = [0, 10, 20, 35, 50, 70, 100]
    true_survival = [1.0, 0.95, 0.85, 0.70, 0.55, 0.40, 0.25]

    # Add noise
    time = []
    survival = []
    for i in range(len(true_steps) - 1):
        n_points = np.random.randint(10, 30)
        t_seg = np.linspace(true_steps[i], true_steps[i+1], n_points)
        s_seg = true_survival[i] + np.random.normal(0, 0.05, n_points)

        time.extend(t_seg)
        survival.extend(s_seg)

    time = np.array(time)
    survival = np.array(survival)

    # Extract with mathematical constraints
    extractor = MathematicalKMExtractor()
    result = extractor.extract_step_function(time, survival)

    print(f"Input: {len(time)} noisy points")
    print(f"Output: {len(result)} clean step function points")
    print()
    print("Extracted curve:")
    print(result)
