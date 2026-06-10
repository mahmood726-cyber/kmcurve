#!/usr/bin/env python3
"""
Weak segmentation labels from CV extraction (lever 3 bridge to REAL training).
==============================================================================

The lever-3 U-Net (`unet_segment.py`) was trained only on synthetic data because
pixel-precise masks for real figures don't exist (hand-labelling them is the
expensive part of lever 2). This module bridges that gap: it derives WEAK
per-pixel arm masks for real KM figures from the existing CV extraction
(`dark_curve_cloud` -> `column_curve_points` -> `separate_arms`), so the U-Net can
train on real figures instead of synthetic ones.

These are *weak* labels -- as good (and as noisy) as the heuristic that made
them. The point is NOT to beat the heuristic on its own training figures (a model
distilled from a heuristic can't exceed it on average) but to (a) prove the
real-figure train->infer path, (b) provide a fast learned segmenter, and (c)
bootstrap until hand-labelled masks exist. Restricting weak-labelling to the
lever-2 AUTO-ACCEPTED / high-confidence figures keeps the labels clean.

Classes: 0 background, 1 arm-1, 2 arm-2 (3-class; censoring left to future work).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

N_CLASSES = 3


def weak_arm_mask(gray: np.ndarray, plot, text_boxes=None, n_arms: int = 2,
                  thickness: int = 1) -> np.ndarray:
    """Per-pixel arm mask (plot-box-local) from the CV curve extraction.

    Returns an int64 (H_box, W_box) array with 0=bg, 1=arm1, 2=arm2. Uses the
    same dark-cloud -> per-column-points -> velocity-continuity separation the
    raster pipeline uses, then paints each arm's points (dilated by ``thickness``)
    onto the mask.
    """
    from raster_km import dark_curve_cloud, column_curve_points
    from vector_km import separate_arms

    H = int(plot.y1 - plot.y0)
    W = int(plot.x1 - plot.x0)
    mask = np.zeros((max(H, 1), max(W, 1)), np.int64)

    cloud = dark_curve_cloud(gray, plot, exclude_boxes=text_boxes)
    pts = column_curve_points(cloud, n_curves=n_arms)
    if pts.size == 0:
        return mask
    arms = separate_arms(pts, n_arms=n_arms, col_dt=2.0, gap=8.0)
    for ai, arm in enumerate(arms[:n_arms]):
        cls = ai + 1
        for x, y in arm:
            ix = int(round(x - plot.x0)); iy = int(round(y - plot.y0))
            for dy in range(-thickness, thickness + 1):
                for dx in range(-thickness, thickness + 1):
                    yy, xx = iy + dy, ix + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        mask[yy, xx] = cls
    return mask


def figure_sample(gray: np.ndarray, plot, text_boxes=None, n_arms: int = 2,
                  thickness: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """(image, weak_mask) for one plot box. Image is the box crop in [0,1].

    ``thickness`` dilates painted curves so a 1px stroke survives downsizing for
    training (a thin line is otherwise lost by nearest-neighbour resize)."""
    crop = gray[int(plot.y0):int(plot.y1), int(plot.x0):int(plot.x1)].astype(np.float32) / 255.0
    mask = weak_arm_mask(gray, plot, text_boxes, n_arms, thickness=thickness)
    return crop, mask


def resize_sample(img: np.ndarray, mask: np.ndarray, h: int = 96, w: int = 128
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Resize a sample for tractable CPU training. Image bilinear, mask nearest
    (labels must not be interpolated). Needs PIL."""
    from PIL import Image

    im = Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    mk = Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
    return np.asarray(im, np.float32) / 255.0, np.asarray(mk, np.int64)
