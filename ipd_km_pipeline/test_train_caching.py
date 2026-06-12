"""Tests for the incremental weak-mask sample cache in train_on_figures.py.
Synthetic (no real PDF extraction): verifies the cache round-trips, re-runs only
extract NEW PDFs, misses aren't re-attempted, and the present-corpus filter."""
from pathlib import Path

import numpy as np
import pytest

import train_on_figures as T


def _fake_sample():
    img = np.zeros((128, 160), np.uint8)
    mask = np.zeros((128, 160), np.uint8)
    mask[10, 10] = 1; mask[20, 20] = 2          # a usable 2-arm mask
    return [(img, mask)]


def test_cache_incremental_and_misses(tmp_path, monkeypatch):
    cache_file = tmp_path / "ws.pkl"
    monkeypatch.setattr(T, "_cache_path", lambda exclude_text: cache_file)

    # corpus of 3 PDFs: a,b yield samples; c is a miss
    pdfs = [tmp_path / f"{s}.pdf" for s in ("a", "b", "c")]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(T, "_iter_pdfs", lambda: list(pdfs))

    calls = []

    def fake_extract(pdf, exclude_text):
        calls.append(pdf.stem)
        return _fake_sample() if pdf.stem in ("a", "b") else None

    monkeypatch.setattr(T, "_extract_one", fake_extract)

    figs = T.build_samples(exclude_text=False)
    assert sorted(s for s, _ in figs) == ["a", "b"]      # only yielding figures
    assert sorted(calls) == ["a", "b", "c"]              # all 3 extracted once

    # second run: nothing new -> ZERO extraction calls (full cache hit)
    calls.clear()
    figs2 = T.build_samples(exclude_text=False)
    assert calls == []                                   # cache hit, no re-extract
    assert sorted(s for s, _ in figs2) == ["a", "b"]


def test_cache_adds_only_new_pdf(tmp_path, monkeypatch):
    cache_file = tmp_path / "ws.pkl"
    monkeypatch.setattr(T, "_cache_path", lambda exclude_text: cache_file)
    pdfs = [tmp_path / "a.pdf"]
    pdfs[0].write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(T, "_iter_pdfs", lambda: list(pdfs))
    monkeypatch.setattr(T, "_extract_one", lambda pdf, et: _fake_sample())
    T.build_samples(exclude_text=False)

    # grow the corpus by one PDF -> only the new one is extracted
    new = tmp_path / "b.pdf"; new.write_bytes(b"%PDF-1.4"); pdfs.append(new)
    calls = []
    monkeypatch.setattr(T, "_extract_one",
                        lambda pdf, et: calls.append(pdf.stem) or _fake_sample())
    figs = T.build_samples(exclude_text=False)
    assert calls == ["b"]                                # only the new PDF
    assert sorted(s for s, _ in figs) == ["a", "b"]


def test_no_grow_skips_extraction(tmp_path, monkeypatch):
    """grow=False returns the cached present figures WITHOUT extracting new PDFs."""
    cache_file = tmp_path / "ws.pkl"
    monkeypatch.setattr(T, "_cache_path", lambda exclude_text: cache_file)
    pdfs = [tmp_path / "a.pdf"]
    pdfs[0].write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(T, "_iter_pdfs", lambda: list(pdfs))
    monkeypatch.setattr(T, "_extract_one", lambda pdf, et: _fake_sample())
    T.build_samples(exclude_text=False)                  # seed cache with 'a'

    # add a new PDF, then build with grow=False -> the new one is NOT extracted
    new = tmp_path / "b.pdf"; new.write_bytes(b"%PDF-1.4"); pdfs.append(new)
    calls = []
    monkeypatch.setattr(T, "_extract_one",
                        lambda pdf, et: calls.append(pdf.stem) or _fake_sample())
    figs = T.build_samples(exclude_text=False, grow=False)
    assert calls == []                                   # no extraction at all
    assert sorted(s for s, _ in figs) == ["a"]           # only the already-cached figure


class _FakeCand:
    def __init__(self, bbox, page_index=0):
        self.bbox = bbox
        self.page_index = page_index


def test_extract_one_empty_crop_returns_none(monkeypatch):
    """A degenerate caption bbox cropping to an empty image must yield a miss
    (None), not crash the whole corpus build (regression: cv2.morphologyEx on an
    empty src aborted the build at PDF ~2509/2709)."""
    monkeypatch.setattr(T.FL, "locate_km_figures",
                        lambda *a, **k: [_FakeCand(bbox=(10, 10, 10, 10))])
    monkeypatch.setattr(T, "render_page", lambda *a, **k: np.full((100, 100), 255, np.uint8))
    assert T._extract_one(Path("x.pdf"), exclude_text=False) is None


def test_extract_one_swallows_detect_exception(monkeypatch):
    """Any exception in the detect/sample chain becomes a miss, not a crash."""
    monkeypatch.setattr(T.FL, "locate_km_figures",
                        lambda *a, **k: [_FakeCand(bbox=None)])
    monkeypatch.setattr(T, "render_page", lambda *a, **k: np.full((100, 100), 255, np.uint8))
    def _boom(*a, **k):
        raise RuntimeError("cv2 morphologyEx assertion")
    monkeypatch.setattr(T, "detect_plot_boxes", _boom)
    assert T._extract_one(Path("x.pdf"), exclude_text=False) is None


def test_present_filter_drops_removed_pdf(tmp_path, monkeypatch):
    """A figure cached from a PDF no longer in the corpus is not returned."""
    cache_file = tmp_path / "ws.pkl"
    monkeypatch.setattr(T, "_cache_path", lambda exclude_text: cache_file)
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(T, "_iter_pdfs", lambda: list(pdfs))
    monkeypatch.setattr(T, "_extract_one", lambda pdf, et: _fake_sample())
    T.build_samples(exclude_text=False)

    monkeypatch.setattr(T, "_iter_pdfs", lambda: [pdfs[0]])   # b removed
    figs = T.build_samples(exclude_text=False)
    assert sorted(s for s, _ in figs) == ["a"]
