#!/usr/bin/env python3
"""Tests for weak-label mask derivation (lever 3 real-figure bridge)."""
import numpy as np
from raster_km import PlotBox
import weak_labels as WL


def _two_curve_figure():
    """White image, plot box, two monotone-decreasing dark curves (2 arms)."""
    img = np.full((300, 400), 255, np.uint8)
    box = PlotBox(x0=60, y0=40, x1=360, y1=260)
    # draw axis frame so it's a plausible plot
    img[box.y0:box.y1, box.x0] = 0; img[box.y1, box.x0:box.x1] = 0
    y1 = float(box.y0 + 10); y2 = float(box.y0 + 40)
    for x in range(box.x0 + 2, box.x1 - 2):
        y1 = min(y1 + 0.7, box.y1 - 2); y2 = min(y2 + 1.1, box.y1 - 2)
        img[int(y1), x] = 0
        img[int(y2), x] = 0
    return img, box


def test_weak_mask_has_two_arms():
    img, box = _two_curve_figure()
    mask = WL.weak_arm_mask(img, box, n_arms=2)
    classes = set(np.unique(mask).tolist())
    assert 1 in classes and 2 in classes      # both arms painted
    assert mask.shape == (box.y1 - box.y0, box.x1 - box.x0)
    # background dominates (curves are thin)
    assert (mask == 0).mean() > 0.8


def test_figure_sample_and_resize():
    img, box = _two_curve_figure()
    crop, mask = WL.figure_sample(img, box, n_arms=2)
    assert crop.shape == mask.shape
    assert 0.0 <= crop.min() and crop.max() <= 1.0
    ri, rm = WL.resize_sample(crop, mask, 96, 128)
    assert ri.shape == (96, 128) and rm.shape == (96, 128)
    assert set(np.unique(rm).tolist()) <= {0, 1, 2}


def test_empty_plot_returns_bg_mask():
    img = np.full((120, 160), 255, np.uint8)
    box = PlotBox(x0=20, y0=20, x1=140, y1=100)
    mask = WL.weak_arm_mask(img, box, n_arms=2)
    assert (mask == 0).all()
