#!/usr/bin/env python3
"""Train the lever-3 U-Net on REAL KM figures via weak labels.

Builds (plot-box image, weak arm-mask) samples from the corpus's caption-anchored
KM figures (`weak_labels.figure_sample`), splits by figure (no leakage), trains a
3-class U-Net with class-weighted cross-entropy (curves are thin -> bg dominates),
and reports pixel accuracy + per-arm IoU on held-out figures.

This is the bridge from the synthetic-only scaffold to real-data training. The
labels are WEAK (heuristic-derived), so the metric measures whether the learned
segmenter reproduces the CV extraction on held-out figures end-to-end -- not a
production model (that needs hand-labelled masks). Requires torch.

Usage: python train_on_figures.py [--epochs 60] [--exclude-text]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figure_locator as FL
from raster_km import (detect_plot_box, detect_plot_boxes, render_page,
                       _rapidocr_text_boxes)
import weak_labels as WL

CORPUS_DIRS = ["corpus", "corpus_unpaywall", "corpus_pmc", "corpus_biorxiv"]
DPI = 150
N_CLASSES = WL.N_CLASSES


def _iter_pdfs():
    # dedup by stem: the same PDF can appear in multiple corpus dirs (e.g. a PMC
    # paper acquired into both corpus/ and corpus_pmc/) -- count it once.
    seen = set()
    for d in CORPUS_DIRS:
        for p in sorted(Path(d).glob("*.pdf")):
            key = p.stem.replace(".", "_").lower()
            if key in seen:
                continue
            seen.add(key)
            yield p


def build_samples(exclude_text: bool):
    """One sample per plot box that yields a 2-arm weak mask. Grouped by figure."""
    figures = []  # list of (fig_id, [ (img,mask), ... ])
    pdfs = list(_iter_pdfs())
    print(f"building weak masks from {len(pdfs)} corpus PDFs...", flush=True)
    for i, pdf in enumerate(pdfs, 1):
        if i % 25 == 0:
            print(f"  ...{i}/{len(pdfs)} PDFs scanned, {len(figures)} usable figures so far",
                  flush=True)
        try:
            cands = FL.locate_km_figures(str(pdf), require_caption=True)
        except Exception:
            continue
        if not cands:
            continue
        c = cands[0]
        try:
            g = render_page(str(pdf), c.page_index, dpi=DPI)
        except Exception:
            continue
        if c.bbox:
            sc = DPI / 72.0
            x0, t, x1, b = c.bbox
            g = g[int(t * sc):int(b * sc), int(x0 * sc):int(x1 * sc)]
        boxes = detect_plot_boxes(g) or ([detect_plot_box(g)] if detect_plot_box(g) else [])
        tboxes = _rapidocr_text_boxes(g) if exclude_text else None
        samples = []
        for box in boxes:
            if box is None or min(box.x1 - box.x0, box.y1 - box.y0) < 40:
                continue
            # thickness=3 so 1px curves survive the downsize to 128x160
            img, mask = WL.figure_sample(g, box, text_boxes=tboxes, n_arms=2, thickness=3)
            if 1 in mask and 2 in mask:                 # a usable 2-arm mask
                samples.append(WL.resize_sample(img, mask, 128, 160))
        if samples:
            figures.append((pdf.stem, samples))
            print(f"[sample] {pdf.name}: {len(samples)} box(es)", flush=True)
    return figures


def _iou(pred, mask, cls):
    p, m = (pred == cls), (mask == cls)
    inter = np.logical_and(p, m).sum()
    union = np.logical_or(p, m).sum()
    return float(inter / union) if union else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--exclude-text", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    figures = build_samples(args.exclude_text)
    n_fig = len(figures)
    print(f"\n{n_fig} figures with usable 2-arm weak masks "
          f"({sum(len(s) for _, s in figures)} boxes)")
    if n_fig < 4:
        print("too few figures for a train/held-out split; need >=4. "
              "Grow the corpus (acquire_corpus.py) and re-run.")
        return

    n_hold = max(1, n_fig // 5)
    train_figs, hold_figs = figures[:-n_hold], figures[-n_hold:]
    train = [s for _, ss in train_figs for s in ss]
    held = [s for _, ss in hold_figs for s in ss]
    print(f"train: {len(train)} boxes from {len(train_figs)} figs | "
          f"held-out: {len(held)} boxes from {len(hold_figs)} figs")

    X = torch.stack([torch.from_numpy(i)[None] for i, _ in train])   # (N,1,H,W)
    Y = torch.stack([torch.from_numpy(m) for _, m in train])         # (N,H,W)
    # inverse-frequency class weights (thin curves -> bg dominates)
    counts = np.bincount(Y.numpy().ravel(), minlength=N_CLASSES).astype(float)
    w = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    weights = torch.tensor(np.clip(w, 0.5, 50.0), dtype=torch.float32)
    print(f"class pixel counts {counts.astype(int).tolist()} -> weights "
          f"{[round(x,2) for x in weights.tolist()]}")

    from unet_segment import build_unet
    model = build_unet(n_classes=N_CLASSES, in_ch=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward(); opt.step()
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {loss.item():.4f}")

    # held-out evaluation
    model.eval()
    accs, iou1, iou2 = [], [], []
    with torch.no_grad():
        for img, mask in held:
            out = model(torch.from_numpy(img)[None, None])
            pred = out.argmax(1)[0].numpy()
            accs.append(float((pred == mask).mean()))
            iou1.append(_iou(pred, mask, 1)); iou2.append(_iou(pred, mask, 2))
    arm_iou = np.nanmean(iou1 + iou2)
    print(f"\nHELD-OUT: pixel-acc {np.mean(accs):.3f}  "
          f"arm-IoU {arm_iou:.3f}  (n={len(held)} boxes)")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/unet_real.pt")
    Path("artifacts/unet_real_metrics.json").write_text(json.dumps({
        "n_figures": n_fig, "train_boxes": len(train), "held_boxes": len(held),
        "held_pixel_acc": round(float(np.mean(accs)), 4),
        "held_arm_iou": round(float(arm_iou), 4),
    }, indent=2))
    print("-> artifacts/unet_real.pt + unet_real_metrics.json")


if __name__ == "__main__":
    main()
