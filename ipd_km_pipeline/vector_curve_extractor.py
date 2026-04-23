#!/usr/bin/env python3
"""
Vector Curve Extractor
======================

Extracts Kaplan-Meier curves from PDF vector paths with perfect calibration.

Approach:
1. Extract perfect axis calibration from embedded PDF text
2. Extract vector paths (curves) from PDF drawings
3. Convert path coordinates to calibrated (time, survival) values
4. Detect events from survival level changes

This is the UNDERLYING REALITY - no rasterization, no approximation!

Author: Claude Code
Date: 2025-10-24
"""

import numpy as np
import pandas as pd
import fitz  # PyMuPDF


def extract_axis_calibration(page, page_height_threshold=0.7, x_axis_left_threshold=100):
    """
    Extract perfect axis calibration from embedded PDF text.

    Args:
        page: PyMuPDF page object
        page_height_threshold: Y-position threshold for X-axis labels (default 0.7)
        x_axis_left_threshold: X-position threshold for Y-axis labels (default 100)

    Returns:
        dict: {
            'x_slope': float,
            'x_intercept': float,
            'y_slope': float,
            'y_intercept': float,
            'x_labels': list,
            'y_labels': list
        }
    """
    # Extract all numeric text with positions
    text_items = []
    for block in page.get_text('dict')['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    bbox = span['bbox']
                    text = span['text'].strip()
                    try:
                        val = float(text)
                        x = (bbox[0] + bbox[2]) / 2
                        y = (bbox[1] + bbox[3]) / 2
                        text_items.append({'value': val, 'x': x, 'y': y})
                    except:
                        pass

    if not text_items:
        raise ValueError("No numeric labels found in PDF")

    # Get approximate page height
    page_rect = page.rect
    page_height = page_rect.height if hasattr(page_rect, 'height') else 400

    # Separate X-axis (bottom of page) and Y-axis (left of page)
    # X-axis: bottom of page AND not too far left (avoid Y-axis labels)
    x_vals = [item for item in text_items
              if item['y'] > page_height * 0.85 and item['x'] > x_axis_left_threshold]

    # Y-axis: left of page
    y_vals = [item for item in text_items if item['x'] < x_axis_left_threshold]

    if not x_vals or not y_vals:
        raise ValueError("Could not find axis labels")

    # Sort by position
    x_vals.sort(key=lambda x: x['x'])
    y_vals.sort(key=lambda x: x['y'])

    # Calculate linear calibration (pixel → value)
    # X-axis: time
    x_m = (x_vals[-1]['value'] - x_vals[0]['value']) / (x_vals[-1]['x'] - x_vals[0]['x'])
    x_b = x_vals[0]['value'] - x_m * x_vals[0]['x']

    # Y-axis: survival
    y_m = (y_vals[-1]['value'] - y_vals[0]['value']) / (y_vals[-1]['y'] - y_vals[0]['y'])
    y_b = y_vals[0]['value'] - y_m * y_vals[0]['y']

    return {
        'x_slope': x_m,
        'x_intercept': x_b,
        'y_slope': y_m,
        'y_intercept': y_b,
        'x_labels': x_vals,
        'y_labels': y_vals
    }


def extract_vector_curves(page, min_segments=20):
    """
    Extract vector curve paths from PDF.

    Args:
        page: PyMuPDF page object
        min_segments: Minimum line segments to consider as curve (default 20)

    Returns:
        list of dict: [{
            'path_index': int,
            'n_segments': int,
            'color': tuple or None,
            'width': float,
            'points': list of {'x': float, 'y': float}
        }]
    """
    paths = page.get_drawings()

    curves = []

    for i, path in enumerate(paths):
        items = path['items']

        # Count line segments
        n_lines = sum(1 for item in items if item[0] == 'l')

        if n_lines >= min_segments:
            # Extract all points from line segments
            raw_points = []
            for item in items:
                if item[0] == 'l':  # line segment
                    _, pt1, pt2 = item
                    raw_points.append({'x': pt1.x, 'y': pt1.y})
                    raw_points.append({'x': pt2.x, 'y': pt2.y})

            # Remove consecutive duplicates
            unique_points = []
            prev = None
            for pt in raw_points:
                if prev is None or abs(pt['x'] - prev['x']) > 0.01 or abs(pt['y'] - prev['y']) > 0.01:
                    unique_points.append(pt)
                    prev = pt

            curves.append({
                'path_index': i,
                'n_segments': n_lines,
                'color': path.get('color'),
                'width': path.get('width', 0),
                'points': unique_points
            })

    return curves


def calibrate_curve(curve_points, calibration):
    """
    Convert pixel coordinates to calibrated (time, survival) values.

    Args:
        curve_points: list of {'x': float, 'y': float}
        calibration: dict from extract_axis_calibration()

    Returns:
        pd.DataFrame with columns ['time', 'survival']
    """
    x_m = calibration['x_slope']
    x_b = calibration['x_intercept']
    y_m = calibration['y_slope']
    y_b = calibration['y_intercept']

    calibrated = []
    for pt in curve_points:
        time = x_m * pt['x'] + x_b
        survival = y_m * pt['y'] + y_b

        # Apply mathematical constraints
        survival = np.clip(survival, 0, 1)

        calibrated.append({'time': time, 'survival': survival})

    df = pd.DataFrame(calibrated)

    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)

    # Enforce monotonicity (survival can only decrease or stay same)
    for i in range(1, len(df)):
        if df.loc[i, 'survival'] > df.loc[i-1, 'survival']:
            df.loc[i, 'survival'] = df.loc[i-1, 'survival']

    return df


def classify_km_segments(curve_df, min_drop=0.0001, time_epsilon=0.5):
    """
    Classify K-M curve segments as vertical (events) vs horizontal (constant survival).

    K-M curves are step functions with two types of segments:
    - Horizontal: Time increases, survival constant (NOT an event)
    - Vertical: Time constant (dt ≈ 0), survival decreases (IS an event)

    This is the CRITICAL FIX: We must only count vertical drops as events!

    Args:
        curve_df: DataFrame with ['time', 'survival']
        min_drop: Minimum survival drop to count as event (default 0.0001)
        time_epsilon: Maximum time change for "vertical" segment (default 0.5 time units)

    Returns:
        pd.DataFrame with events (only vertical drops)
    """
    if len(curve_df) < 2:
        return pd.DataFrame(columns=['time', 'survival'])

    events = []

    if curve_df.empty:
        return events

    # Start point
    events.append({
        'time': curve_df.iloc[0]['time'],  # sentinel:skip-line P1-empty-dataframe-access  (guarded by curve_df.empty above)
        'survival': curve_df.iloc[0]['survival']  # sentinel:skip-line P1-empty-dataframe-access  (guarded by curve_df.empty above)
    })

    # Analyze each segment
    for i in range(len(curve_df) - 1):
        t1 = curve_df.iloc[i]['time']
        s1 = curve_df.iloc[i]['survival']
        t2 = curve_df.iloc[i + 1]['time']
        s2 = curve_df.iloc[i + 1]['survival']

        dt = abs(t2 - t1)
        ds = s1 - s2  # Positive if survival decreased

        # A segment is "vertical" (an event) if:
        # 1. Survival drops (ds > min_drop)
        # 2. Time barely changes (dt < time_epsilon)
        is_vertical = (ds > min_drop) and (dt < time_epsilon)

        if is_vertical:
            # This is an EVENT!
            events.append({
                'time': t2,
                'survival': s2
            })

    return pd.DataFrame(events)


def detect_events(curve_df, survival_decimals=5, min_change=0.0001):
    """
    Detect events from survival level changes.

    DEPRECATED: This old approach counts ALL points as events.
    Use classify_km_segments() instead for correct event detection.

    Kept for backward compatibility only.

    Args:
        curve_df: DataFrame with ['time', 'survival']
        survival_decimals: Rounding precision for survival (default 5)
        min_change: Minimum survival change to count as event (default 0.0001)

    Returns:
        pd.DataFrame with detected events
    """
    # Round to avoid floating point noise
    curve_df['surv_rounded'] = np.round(curve_df['survival'], decimals=survival_decimals)

    # Keep points where survival changes
    events = []
    prev_surv = None

    for i in range(len(curve_df)):
        surv = curve_df.loc[i, 'surv_rounded']

        # Keep first point and any point where survival changed
        if prev_surv is None or abs(surv - prev_surv) > min_change:
            events.append({
                'time': curve_df.loc[i, 'time'],
                'survival': curve_df.loc[i, 'survival']
            })
            prev_surv = surv

    return pd.DataFrame(events)


def extract_km_curves_from_pdf(pdf_path, page_num=0, min_segments=20):
    """
    Main function: Extract all KM curves from a PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed, default 0)
        min_segments: Minimum line segments for curve detection (default 20)

    Returns:
        dict: {
            'calibration': dict,
            'curves': list of {
                'curve_index': int,
                'path_index': int,
                'color': tuple or None,
                'width': float,
                'events': pd.DataFrame with ['time', 'survival']
            }
        }
    """
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} not found in PDF")

    page = doc[page_num]

    # Step 1: Get perfect calibration
    calibration = extract_axis_calibration(page)

    # Step 2: Extract vector curves
    vector_curves = extract_vector_curves(page, min_segments=min_segments)

    # Step 3: Process each curve
    results = []

    for i, curve in enumerate(vector_curves):
        # Calibrate coordinates
        calibrated_df = calibrate_curve(curve['points'], calibration)

        # Classify segments: Only vertical drops are events!
        events_df = classify_km_segments(calibrated_df, min_drop=0.001)

        results.append({
            'curve_index': i,
            'path_index': curve['path_index'],
            'color': curve['color'],
            'width': curve['width'],
            'n_segments': curve['n_segments'],
            'n_events': len(events_df),
            'events': events_df
        })

    doc.close()

    return {
        'calibration': calibration,
        'curves': results
    }


def format_color(color_tuple):
    """Helper: Format color tuple for display."""
    if color_tuple is None:
        return "black"
    elif isinstance(color_tuple, (list, tuple)) and len(color_tuple) == 3:
        r, g, b = color_tuple
        if r == 0 and g == 0 and b == 0:
            return "black"
        elif r > 0.5 and g < 0.5 and b < 0.5:
            return "red"
        elif r < 0.5 and g > 0.5 and b < 0.5:
            return "green"
        elif r < 0.5 and g < 0.5 and b > 0.5:
            return "blue"
        else:
            return f"RGB({r:.2f},{g:.2f},{b:.2f})"
    else:
        return str(color_tuple)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vector_curve_extractor.py <pdf_path> [page_num]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print('='*70)
    print('VECTOR CURVE EXTRACTION')
    print('='*70)
    print(f'PDF: {pdf_path}')
    print(f'Page: {page_num}')
    print()

    try:
        result = extract_km_curves_from_pdf(pdf_path, page_num)

        cal = result['calibration']
        print(f'Calibration:')
        print(f'  X-axis: {len(cal["x_labels"])} labels')
        print(f'  Y-axis: {len(cal["y_labels"])} labels')
        print()

        curves = result['curves']
        print(f'Extracted {len(curves)} curves:')
        print()

        for curve in curves:
            color_str = format_color(curve['color'])
            print(f'Curve {curve["curve_index"]} (Path {curve["path_index"]}, {color_str}):')
            print(f'  Segments: {curve["n_segments"]}')
            print(f'  Events: {curve["n_events"]}')

            events = curve['events']
            if len(events) > 0:
                print(f'  Time range: {events["time"].min():.1f} - {events["time"].max():.1f}')
                print(f'  Survival range: {events["survival"].min():.3f} - {events["survival"].max():.3f}')
                print()
                print('  First 5 events:')
                for idx, row in events.head(5).iterrows():
                    print(f'    time={row["time"]:.1f}, survival={row["survival"]:.3f}')
            print()

    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print('='*70)
    print('Alhamdulillah! Vector extraction complete.')
    print('='*70)
