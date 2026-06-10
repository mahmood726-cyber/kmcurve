#!/usr/bin/env python3
"""Network-free tests for vlm_calibrate ingest (roadmap lever 1).

Builds a synthetic figure whose axis tick marks ``detect_tick_positions`` finds
at known pixel positions, feeds a recorded VLM answer, and asserts the resulting
AxisFit. Also checks the fail-closed paths (count mismatch, log axis) and the
crop/packet builders. No Anthropic API call -- ``calibrate_via_api`` is exercised
only via its shared ``ingest_vlm_answer`` core.
"""

import numpy as np
import pytest

from raster_km import PlotBox, detect_tick_positions
import vlm_calibrate as VC


def _synthetic_plot():
    """White image with a plot box + 1px dark tick marks at known positions.

    x ticks at px 100,200,300,400,500 ; y ticks at px 50,150,250,350,450.
    """
    img = np.full((520, 620), 255, np.uint8)
    box = PlotBox(x0=100, y0=50, x1=500, y1=450)
    xticks = [100, 200, 300, 400, 500]
    yticks = [50, 150, 250, 350, 450]
    for x in xticks:                       # vertical marks just below the x-axis
        img[box.y1 + 1: box.y1 + 7, x] = 0
    for y in yticks:                       # horizontal marks just left of the y-axis
        img[y, box.x0 - 7: box.x0 - 1] = 0
    return img, box, xticks, yticks


def test_detect_positions_match_synthetic():
    img, box, xticks, yticks = _synthetic_plot()
    assert sorted(detect_tick_positions(img, box, "x")) == [float(x) for x in xticks]
    assert sorted(detect_tick_positions(img, box, "y")) == [float(y) for y in yticks]


def test_ingest_fits_known_calibration():
    img, box, _, _ = _synthetic_plot()
    answer = {
        # x: 0..12 months left->right ; y: 100..0 percent top->bottom
        "x_axis": {"tick_values": [0, 3, 6, 9, 12], "is_log": False,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [100, 75, 50, 25, 0], "is_log": False,
                   "is_percent": True, "axis_kind": "cumulative_incidence"},
    }
    x_fit, y_fit, meta = VC.ingest_vlm_answer(img, box, answer)

    assert x_fit.slope == pytest.approx(0.03, abs=1e-9)
    assert x_fit.intercept == pytest.approx(-3.0, abs=1e-9)
    assert y_fit.slope == pytest.approx(-0.25, abs=1e-9)
    assert y_fit.intercept == pytest.approx(112.5, abs=1e-9)
    assert x_fit.r2 == pytest.approx(1.0, abs=1e-9)
    assert y_fit.r2 == pytest.approx(1.0, abs=1e-9)
    # endpoints map back to the printed values
    assert x_fit.value(100) == pytest.approx(0.0, abs=1e-9)
    assert x_fit.value(500) == pytest.approx(12.0, abs=1e-9)
    assert y_fit.value(50) == pytest.approx(100.0, abs=1e-9)
    assert y_fit.value(450) == pytest.approx(0.0, abs=1e-9)
    # semantics surfaced for the downstream pipeline
    assert meta["value_scale"] == 100.0
    assert meta["value_is_cumulative_incidence"] is True


def test_vlm_calibrate_axes_replay_returns_two_fits():
    img, box, _, _ = _synthetic_plot()
    answer = {
        "x_axis": {"tick_values": [0, 3, 6, 9, 12], "is_log": False,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [1.0, 0.75, 0.5, 0.25, 0.0], "is_log": False,
                   "is_percent": False, "axis_kind": "survival"},
    }
    x_fit, y_fit = VC.vlm_calibrate_axes(img, box, answer=answer)
    assert x_fit.r2 == pytest.approx(1.0, abs=1e-9)
    # y survival 1.0 (top) .. 0.0 (bottom)
    assert y_fit.value(50) == pytest.approx(1.0, abs=1e-9)
    assert y_fit.value(450) == pytest.approx(0.0, abs=1e-9)


def test_count_mismatch_fails_closed():
    img, box, _, _ = _synthetic_plot()
    answer = {
        "x_axis": {"tick_values": [0, 3, 6, 9, 12], "is_log": False,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [100, 50, 0], "is_log": False,   # only 3 vs 5 ticks
                   "is_percent": True, "axis_kind": "cumulative_incidence"},
    }
    with pytest.raises(ValueError, match="count mismatch"):
        VC.ingest_vlm_answer(img, box, answer)


def test_log_axis_fails_closed():
    img, box, _, _ = _synthetic_plot()
    answer = {
        "x_axis": {"tick_values": [1, 10, 100, 1000, 10000], "is_log": True,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [100, 75, 50, 25, 0], "is_log": False,
                   "is_percent": True, "axis_kind": "cumulative_incidence"},
    }
    with pytest.raises(ValueError, match="log-scaled"):
        VC.ingest_vlm_answer(img, box, answer)


def test_too_few_values_fails_closed():
    img, box, _, _ = _synthetic_plot()
    answer = {
        "x_axis": {"tick_values": [6], "is_log": False,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [100, 75, 50, 25, 0], "is_log": False,
                   "is_percent": True, "axis_kind": "cumulative_incidence"},
    }
    with pytest.raises(ValueError):
        VC.ingest_vlm_answer(img, box, answer)


def test_merge_split_peaks():
    # one tick split into two adjacent peaks (6px) amid ~60px real spacing -> merged
    pos = [93.5, 150.5, 157.0, 219.0, 281.5, 288.0, 345.0]
    merged = VC._merge_split_peaks(pos)
    assert len(merged) == 5  # 153.75, 284.75, 348 collapsed from the split pairs
    # clean evenly-spaced peaks are untouched
    assert VC._merge_split_peaks([100, 200, 300, 400, 500]) == [100, 200, 300, 400, 500]
    # genuine under-detection is NOT invented (still 4, will fail count-match upstream)
    assert VC._merge_split_peaks([85, 144, 203, 262]) == [85, 144, 203, 262]


def test_split_peaks_then_fit():
    # build a synthetic image where one x tick is split into two adjacent columns
    img = np.full((520, 620), 255, np.uint8)
    box = PlotBox(x0=100, y0=50, x1=500, y1=450)
    # x ticks at 100,200,300,400,500 but 300 rendered as a 2-col split (297 & 303)
    for x in [100, 200, 297, 303, 400, 500]:
        img[box.y1 + 1: box.y1 + 7, x] = 0
    for y in [50, 150, 250, 350, 450]:
        img[y, box.x0 - 7: box.x0 - 1] = 0
    answer = {
        "x_axis": {"tick_values": [0, 3, 6, 9, 12], "is_log": False,
                   "is_percent": False, "axis_kind": "time"},
        "y_axis": {"tick_values": [1, 0.75, 0.5, 0.25, 0], "is_log": False,
                   "is_percent": False, "axis_kind": "survival"},
    }
    # raw detection finds 6 x peaks; merge collapses the split -> 5 -> matches VLM
    assert len(detect_tick_positions(img, box, "x")) == 6
    x_fit, y_fit, _ = VC.ingest_vlm_answer(img, box, answer)
    assert x_fit.r2 == pytest.approx(1.0, abs=1e-6)


def test_build_packet_and_crops():
    img, box, _, _ = _synthetic_plot()
    x_crop = VC.axis_label_crop(img, box, "x")
    y_crop = VC.axis_label_crop(img, box, "y")
    assert x_crop.size > 0 and y_crop.size > 0
    packet = VC.build_packet(img, box)
    assert packet["x_axis_b64"] and packet["y_axis_b64"]
    assert packet["schema"] == VC.CALIBRATION_SCHEMA
