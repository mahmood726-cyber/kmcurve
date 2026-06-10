#!/usr/bin/env python3
"""Tests for figure_locator caption-anchoring (precision fix)."""
from figure_locator import _caption_blocks, _KM_RE


def test_caption_line_is_a_block():
    text = "Figure 2. Kaplan-Meier curves for overall survival by arm.\nMore detail here."
    blocks = _caption_blocks(text)
    assert any(_KM_RE.search(b) for b in blocks)


def test_inline_reference_is_not_a_caption():
    # "Figure 2" mid-sentence = cross-reference, must NOT count as a caption block
    text = "As shown in Figure 2, the Kaplan-Meier analysis revealed a benefit."
    assert _caption_blocks(text) == []


def test_body_text_km_without_figure_label_is_not_a_caption():
    text = "Kaplan-Meier analysis was performed using the log-rank test for survival."
    assert _caption_blocks(text) == []


def test_fig_abbrev_caption_detected():
    text = "Fig. 3 Progression-free survival in the intention-to-treat population."
    blocks = _caption_blocks(text)
    assert blocks and _KM_RE.search(blocks[0])


def test_caption_continuation_lines_joined():
    text = ("Figure 1. Overall survival\nKaplan-Meier estimates by treatment.\n"
            "Numbers at risk shown below.")
    blocks = _caption_blocks(text)
    # KM keyword on the continuation line is still captured
    assert any("Kaplan" in b for b in blocks)


def test_parenthetical_inline_ref_rejected():
    # "(Figure 3C). Overall ... survival" is a results sentence, not a caption
    text = "(Figure 3C). Overall, these results suggest improved survival outcomes."
    assert _caption_blocks(text) == []


def test_subpanel_reference_rejected():
    assert _caption_blocks("Figure 2A, the survival curve diverges early.") == []


def test_real_caption_with_colon_kept():
    text = "Figure 4: Overall survival by treatment group (Kaplan-Meier)."
    blocks = _caption_blocks(text)
    assert blocks and _KM_RE.search(blocks[0])
