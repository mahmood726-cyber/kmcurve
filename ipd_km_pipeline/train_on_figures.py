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

Training-axis results (held-out arm-IoU on the 295-figure cache, same split):
  - full-batch GD, lr 1e-2 / 60 ep ............... 0.050
  - full-batch GD, lr 3e-3 / cosine / 120 ep ..... 0.153
  - mini-batch SGD (bs 32) + augment / 150 ep .... 0.159  (pixel-acc 0.758 -> 0.802)
The mini-batch + augmentation gain is real but MODEST, and the IoU plateaus by
~epoch 40 (no longer climbing) -- the training axis is now largely exhausted. The
binding constraints from here are DATA (295 figures) and WEAK-LABEL quality (a
heuristic-distilled model cannot exceed the heuristic it learns from); the lever is
more corpus + hand-labelled masks, not more training tricks.

Usage: python train_on_figures.py --batch-size 32 --augment --eval-every 10   # best
       python train_on_figures.py --no-grow ...   # train on the current cache only
       python train_on_figures.py                  # legacy full-batch GD (bs 0)
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


def _extract_one(pdf, exclude_text: bool):
    """Weak 2-arm-mask samples for one PDF (None if the figure-locate/extract
    chain yields no usable 2-arm mask)."""
    try:
        cands = FL.locate_km_figures(str(pdf), require_caption=True)
    except Exception:
        return None
    if not cands:
        return None
    c = cands[0]
    try:
        g = render_page(str(pdf), c.page_index, dpi=DPI)
    except Exception:
        return None
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
    return samples or None


def _cache_path(exclude_text: bool) -> Path:
    Path("artifacts").mkdir(exist_ok=True)
    return Path("artifacts") / f"weak_samples_{'notext' if exclude_text else 'all'}.pkl"


def build_samples(exclude_text: bool, grow: bool = True):
    """One sample per plot box that yields a 2-arm weak mask, grouped by figure.

    grow=False loads the existing cache and returns its present-corpus figures
    WITHOUT extracting any new PDFs -- use it to train on the current cache
    without paying for a (slow) extraction of newly-acquired corpus PDFs.

    INCREMENTAL CACHE: the (image, mask) samples and the set of processed PDF
    stems are cached to artifacts/, so a re-run only extracts PDFs new since the
    last build (extraction over hundreds of PDFs is the slow part). The cache
    records BOTH yields (figures) and misses (PDFs with no usable mask) so a
    miss is never re-attempted. Delete the .pkl to force a full rebuild."""
    import pickle

    cache_file = _cache_path(exclude_text)
    cache = {"samples": {}, "misses": []}           # stem->samples ; [stems w/o mask]
    if cache_file.exists():
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception:
            cache = {"samples": {}, "misses": []}
    done = set(cache["samples"]) | set(cache["misses"])

    pdfs = list(_iter_pdfs())
    todo = [] if not grow else [p for p in pdfs if p.stem not in done]
    print(f"{len(pdfs)} corpus PDFs; {len(done)} cached, {len(todo)} new to extract"
          f"{' (grow disabled)' if not grow else ''}...", flush=True)
    for i, pdf in enumerate(todo, 1):
        if i % 25 == 0:
            print(f"  ...{i}/{len(todo)} new PDFs, "
                  f"{len(cache['samples'])} usable figures total", flush=True)
        samples = _extract_one(pdf, exclude_text)
        if samples:
            cache["samples"][pdf.stem] = samples
            print(f"[sample] {pdf.name}: {len(samples)} box(es)", flush=True)
        else:
            cache["misses"].append(pdf.stem)
        if i % 25 == 0:                              # periodic flush for long builds
            cache_file.write_bytes(pickle.dumps(cache))
    cache_file.write_bytes(pickle.dumps(cache))

    # return only the figures whose PDF is in the CURRENT corpus, stable order
    present = {p.stem for p in pdfs}
    return [(stem, s) for stem, s in cache["samples"].items() if stem in present]


def _augment_batch(xb, yb, torch, gen):
    """On-the-fly augmentation for one (image, mask) batch -- the data is the
    binding constraint (~295 figures), so this multiplies effective variety.
    SAFE transforms only: horizontal flip (KM arms keep their upper/lower order,
    so the class-1=upper weak-label convention is preserved -- a vertical flip
    would NOT, and is deliberately excluded) applied to image AND mask together;
    mild contrast + light gaussian noise on the IMAGE only (robustness to figure
    rendering, mask unchanged). xb: (B,1,H,W) float; yb: (B,H,W) long."""
    B = xb.shape[0]
    # horizontal flip ~half the batch (image + mask together -> labels stay aligned)
    flip = torch.rand(B, generator=gen) < 0.5
    if flip.any():
        xb[flip] = torch.flip(xb[flip], dims=[-1])
        yb[flip] = torch.flip(yb[flip], dims=[-1])
    # image-only intensity jitter: contrast around the white background (1.0),
    # then light gaussian noise; clamp back to the valid [0,1] range.
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.4   # 0.8-1.2
    xb = 1.0 + (xb - 1.0) * contrast
    xb = xb + torch.randn(xb.shape, generator=gen) * 0.03
    return xb.clamp_(0.0, 1.0), yb


def _iou(pred, mask, cls):
    p, m = (pred == cls), (mask == cls)
    inter = np.logical_and(p, m).sum()
    union = np.logical_or(p, m).sum()
    return float(inter / union) if union else float("nan")


def _eval_held(model, held, torch):
    """Held-out pixel accuracy and mean arm-IoU (classes 1 & 2)."""
    model.eval()
    accs, ious = [], []
    with torch.no_grad():
        for img, mask in held:
            pred = model(torch.from_numpy(img)[None, None]).argmax(1)[0].numpy()
            accs.append(float((pred == mask).mean()))
            ious.append(_iou(pred, mask, 1)); ious.append(_iou(pred, mask, 2))
    model.train()
    return float(np.mean(accs)), float(np.nanmean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="initial LR (1e-2 oscillated; 3e-3 + cosine converges)")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="mini-batch size; 0 = full-batch GD (the old default). "
                         "Mini-batch SGD gives many more steps/epoch -> faster "
                         "convergence on this data-limited problem.")
    ap.add_argument("--augment", action="store_true",
                    help="on-the-fly h-flip + intensity jitter (multiplies the "
                         "~295-figure effective dataset; mask kept aligned)")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="held-out eval cadence for best-checkpoint selection")
    ap.add_argument("--exclude-text", action="store_true")
    ap.add_argument("--no-grow", action="store_true",
                    help="train on the existing sample cache without extracting "
                         "newly-acquired corpus PDFs (fast; isolates training changes)")
    ap.add_argument("--metrics-out", default="artifacts/unet_real_metrics.json")
    ap.add_argument("--model-out", default="artifacts/unet_real.pt")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    figures = build_samples(args.exclude_text, grow=not args.no_grow)
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
    import copy
    model = build_unet(n_classes=N_CLASSES, in_ch=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing: the old fixed lr=1e-2 oscillated (loss bounced 1.05-1.15
    # without converging); a decaying LR lets it settle. Eval held-out IoU
    # periodically and KEEP THE BEST model, not the last epoch (which can be
    # past the optimum).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model.train()
    bs = args.batch_size if args.batch_size and args.batch_size < len(train) else 0
    gen = torch.Generator().manual_seed(0)             # deterministic shuffle/augment
    n = X.shape[0]
    print(f"optimiser: {'full-batch GD' if bs == 0 else f'mini-batch SGD (bs={bs})'}"
          f"{' + augment' if args.augment else ''}, {args.epochs} epochs")
    best_iou, best_state, best_acc = -1.0, None, 0.0
    for ep in range(args.epochs):
        if bs == 0:                                    # full-batch (legacy path)
            opt.zero_grad()
            xb, yb = (X, Y)
            if args.augment:
                xb, yb = _augment_batch(xb.clone(), yb.clone(), torch, gen)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        else:                                          # mini-batch SGD
            perm = torch.randperm(n, generator=gen)
            last = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                xb, yb = X[idx], Y[idx]
                if args.augment:
                    xb, yb = _augment_batch(xb.clone(), yb.clone(), torch, gen)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward(); opt.step()
                last = loss.item()
            loss = torch.tensor(last)
        sched.step()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            acc, iou = _eval_held(model, held, torch)
            if iou > best_iou:
                best_iou, best_acc, best_state = iou, acc, copy.deepcopy(model.state_dict())
            print(f"  epoch {ep:3d}  loss {loss.item():.4f}  "
                  f"held arm-IoU {iou:.3f}  (best {best_iou:.3f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)              # restore the best checkpoint
    print(f"\nHELD-OUT (best): pixel-acc {best_acc:.3f}  "
          f"arm-IoU {best_iou:.3f}  (n={len(held)} boxes)")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    Path(args.metrics_out).write_text(json.dumps({
        "n_figures": n_fig, "train_boxes": len(train), "held_boxes": len(held),
        "lr": args.lr, "epochs": args.epochs, "scheduler": "cosine",
        "batch_size": bs, "augment": bool(args.augment),
        "held_pixel_acc": round(best_acc, 4),
        "held_arm_iou": round(best_iou, 4),
    }, indent=2))
    print(f"-> {args.model_out} + {args.metrics_out}")


if __name__ == "__main__":
    main()
