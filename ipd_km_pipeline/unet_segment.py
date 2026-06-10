#!/usr/bin/env python3
"""
ML curve segmentation -- a small U-Net (roadmap lever 3).
=========================================================

The raster path separates arms with pixel heuristics (`dark_curve_cloud` +
`column_curve_points` + velocity continuity in `vector_km.separate_arms`). These
are fragile where curves are coloured / dashed / overlapping or near-coincident
at high survival -- the circrep "2px box shift flips the reconstructed events"
problem. A learned per-pixel segmentation is robust there: it labels each pixel
as background / arm-1 / arm-2 / censor-mark / CI-band directly from local texture
and colour, instead of thresholding a single dark cloud.

This module is the **architecture + training/inference path + a synthetic data
generator** so the pipeline is provably trainable end-to-end. **It is NOT a
trained model** -- a production model needs the labelled corpus (lever 2). The
synthetic generator doubles as: (a) the smoke-test fixture, and (b) a
pre-training bootstrap until real labels exist.

Requires torch (CPU is fine for this small net). Import is lazy so the rest of
the pipeline doesn't depend on torch.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# class ids for the segmentation mask
BG, ARM1, ARM2, CENSOR = 0, 1, 2, 3
N_CLASSES = 4


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def build_unet(n_classes: int = N_CLASSES, in_ch: int = 1):
    """A tiny 2-level U-Net. Returns an ``nn.Module``. Lazy torch import."""
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.ReLU(inplace=True))

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = block(in_ch, 16)
            self.e2 = block(16, 32)
            self.pool = nn.MaxPool2d(2)
            self.bott = block(32, 64)
            self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d2 = block(64, 32)
            self.u1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.d1 = block(32, 16)
            self.head = nn.Conv2d(16, n_classes, 1)

        def forward(self, x):
            import torch
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b = self.bott(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(b), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)

    return TinyUNet()


def _pad_to(mult: int, *arrs):
    """Pad (H,W) arrays at bottom/right so H,W are multiples of ``mult``."""
    H, W = arrs[0].shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    return [np.pad(a, ((0, ph), (0, pw)), mode="edge") for a in arrs], (H, W)


# --------------------------------------------------------------------------- #
# Synthetic data (smoke-test fixture + pre-training bootstrap)
# --------------------------------------------------------------------------- #
def synthetic_km_sample(h: int = 64, w: int = 96, seed: int = 0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Make a synthetic KM-like crop + per-pixel class mask.

    Two monotone-decreasing step curves (two arms) at different gray levels on a
    white background, plus a couple of censor ticks. Returns (image float32 in
    [0,1] shape (h,w), mask int64 shape (h,w)). Deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    img = np.ones((h, w), np.float32)
    mask = np.zeros((h, w), np.int64)

    def draw(arm_id, gray, start_frac, drop):
        y = int(h * start_frac)
        thick = 1
        for x in range(w):
            if rng.random() < drop and y < h - 2:
                y += rng.integers(1, 3)
            for dy in range(-thick, thick + 1):
                yy = min(max(y + dy, 0), h - 1)
                img[yy, x] = gray
                mask[yy, x] = arm_id

    draw(ARM1, 0.15, 0.10, 0.10)   # upper arm, dark
    draw(ARM2, 0.45, 0.18, 0.16)   # lower arm, mid-gray
    # a few censor ticks on arm1
    for _ in range(3):
        cx = int(rng.integers(5, w - 5))
        cy = int(np.argmax(mask[:, cx] == ARM1)) if (mask[:, cx] == ARM1).any() else h // 4
        img[max(cy - 2, 0):cy + 3, cx] = 0.0
        mask[max(cy - 2, 0):cy + 3, cx] = CENSOR
    return img, mask


def make_dataset(n: int = 8):
    return [synthetic_km_sample(seed=i) for i in range(n)]


# --------------------------------------------------------------------------- #
# Train / infer
# --------------------------------------------------------------------------- #
def train(model, samples: List[Tuple[np.ndarray, np.ndarray]],
          epochs: int = 30, lr: float = 1e-2):
    """Minimal CE training loop. Returns the per-epoch loss list."""
    import torch
    import torch.nn as nn

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xs, ys = [], []
    for img, msk in samples:
        (pi, pm), _ = _pad_to(4, img, msk)
        xs.append(torch.from_numpy(pi)[None, None])     # (1,1,H,W)
        ys.append(torch.from_numpy(pm)[None])           # (1,H,W)
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def segment(model, image: np.ndarray) -> np.ndarray:
    """Per-pixel class mask for a single (H,W) float image in [0,1]."""
    import torch

    (pi,), (H, W) = _pad_to(4, image.astype(np.float32))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(pi)[None, None])
        pred = out.argmax(1)[0].cpu().numpy()
    return pred[:H, :W]
