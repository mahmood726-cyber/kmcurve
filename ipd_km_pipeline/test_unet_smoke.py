#!/usr/bin/env python3
"""Smoke test for the lever-3 U-Net: proves the train->infer path works on
synthetic KM data. Skipped if torch is not installed."""
import numpy as np
import pytest

pytest.importorskip("torch")  # lever 3 dependency; skip cleanly if absent

import unet_segment as U


def test_synthetic_sample_shapes_and_classes():
    img, mask = U.synthetic_km_sample(64, 96, seed=1)
    assert img.shape == (64, 96) and mask.shape == (64, 96)
    classes = set(np.unique(mask).tolist())
    assert {U.ARM1, U.ARM2}.issubset(classes)      # both arms present
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_unet_trains_and_infers():
    import torch
    torch.manual_seed(0)
    model = U.build_unet()
    samples = U.make_dataset(6)
    losses = U.train(model, samples, epochs=40)
    # loss must decrease meaningfully -> the train path is wired correctly
    assert losses[-1] < 0.6 * losses[0], (losses[0], losses[-1])
    # inference returns a same-size class mask that learned the arms
    img, mask = samples[0]
    pred = U.segment(model, img)
    assert pred.shape == img.shape
    # after overfitting a tiny set, pixel accuracy on a train sample is high
    acc = float((pred == mask).mean())
    assert acc > 0.9, acc
