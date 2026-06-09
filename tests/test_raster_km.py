"""Synthetic end-to-end test for the raster fallback path.

Builds a known 2-curve survival image in-memory (no PDF / no pypdfium needed),
runs the raster CV extraction + 2-click calibration + continuity separation +
Guyot, and checks the recovered curves match what was drawn. Validates the
raster path on a controlled input; real-PDF robustness is iterative.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

import raster_km as R  # noqa: E402 (sys.path via conftest)


def _make_two_curve_image():
    """White image with two dark survival curves; returns (img, plot, truth)."""
    H, W = 300, 400
    img = np.full((H, W), 255, dtype=np.uint8)
    plot = R.PlotBox(x0=50, y0=20, x1=350, y1=270)
    # survival s -> y_px = y1 - s*(y1-y0); time fraction along x
    # arm A: 1.0 -> 0.40 ; arm B: 1.0 -> 0.70 (A drops faster => more events)
    finalA, finalB = 0.40, 0.70
    for xpx in range(plot.x0, plot.x1):
        frac = (xpx - plot.x0) / (plot.x1 - plot.x0)
        sA = 1.0 - (1.0 - finalA) * frac
        sB = 1.0 - (1.0 - finalB) * frac
        for s in (sA, sB):
            ypx = int(round(plot.y1 - s * (plot.y1 - plot.y0)))
            img[ypx, xpx] = 0
            img[min(ypx + 1, H - 1), xpx] = 0  # 2px thick line
    return img, plot, (finalA, finalB)


X_REFS = [(50.0, 0.0), (350.0, 30.0)]      # x px -> months
Y_REFS = [(20.0, 1.0), (270.0, 0.0)]       # y px -> survival


def test_dark_cloud_and_column_points_recover_two_curves():
    img, plot, _ = _make_two_curve_image()
    cloud = R.dark_curve_cloud(img, plot)
    assert cloud.shape[0] > 200
    cols = R.column_curve_points(cloud, n_curves=2)
    # roughly two points per column across the plot width
    n_cols = plot.x1 - plot.x0
    assert cols.shape[0] > 1.5 * n_cols * 0.5


def test_calibrated_survival_matches_drawn_curves():
    from manual_calibration import axis_from_clicks, calibrate_points
    from vector_km import separate_arms

    img, plot, (finalA, finalB) = _make_two_curve_image()
    xfit, yfit = axis_from_clicks(X_REFS), axis_from_clicks(Y_REFS)
    cols = R.column_curve_points(R.dark_curve_cloud(img, plot), n_curves=2)
    t, v = calibrate_points(cols, xfit, yfit, monotone="none")
    # survival is 0-1, so the split gap must be small (default 1.0 is for %)
    arms = separate_arms(np.column_stack([t, v]), n_arms=2, gap=0.02)
    finals = sorted(float(a[a[:, 0].argmax(), 1]) for a in arms if a.size)
    # recovered final survivals should match the drawn 0.40 / 0.70 (±0.05)
    assert finals[0] == pytest.approx(finalA, abs=0.05)
    assert finals[1] == pytest.approx(finalB, abs=0.05)


def _make_axes_image():
    """White image with a y-axis, x-axis, and ticks at known positions."""
    H, W = 300, 400
    img = np.full((H, W), 255, dtype=np.uint8)
    ax_x0, ax_y1 = 50, 270           # y-axis column, x-axis row
    img[20:271, ax_x0] = 0           # y-axis vertical line
    img[ax_y1, 50:351] = 0           # x-axis horizontal line
    x_ticks = [50, 110, 170, 230, 290, 350]   # values 0,6,12,18,24,30
    for xt in x_ticks:
        img[ax_y1 + 1: ax_y1 + 7, xt] = 0      # downward tick marks
    y_ticks = [270, 210, 150, 90, 30]          # values 0,.25,.5,.75,1.0
    for yt in y_ticks:
        img[yt, ax_x0 - 6: ax_x0] = 0          # leftward tick marks
    return img, x_ticks, y_ticks


def test_auto_detect_plot_box():
    img, _, _ = _make_axes_image()
    box = R.detect_plot_box(img)
    assert box is not None
    assert abs(box.x0 - 50) <= 2 and abs(box.y1 - 270) <= 2
    assert abs(box.x1 - 350) <= 3 and abs(box.y0 - 20) <= 3


def test_auto_detect_ticks_and_calibrate():
    img, x_ticks, _ = _make_axes_image()
    box = R.detect_plot_box(img)
    xs = R.detect_tick_positions(img, box, axis="x")
    assert len(xs) == len(x_ticks)
    fit = R.auto_axis_fit(xs, [0, 6, 12, 18, 24, 30])
    assert fit.r2 > 0.999
    assert abs(fit.value(50) - 0.0) < 0.5 and abs(fit.value(350) - 30.0) < 0.5


def test_raster_to_ipd_orders_event_counts():
    img, plot, _ = _make_two_curve_image()
    nar = [{"times": np.array([0, 30.0]), "counts": np.array([500, 200.0])},
           {"times": np.array([0, 30.0]), "counts": np.array([500, 350.0])}]
    arms = R.raster_to_ipd(img, plot, X_REFS, Y_REFS, n_arms=2,
                           value_is_cumulative_incidence=False, value_scale=1.0, nar=nar)
    assert len(arms) == 2
    # the faster-dropping arm (lower final survival) must have more events
    by_events = sorted(arms, key=lambda a: a["n_events"])
    assert by_events[-1]["n_events"] > by_events[0]["n_events"]
