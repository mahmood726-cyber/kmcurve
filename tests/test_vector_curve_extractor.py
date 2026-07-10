"""Regression tests for the superseded vector_curve_extractor axis calibrator.

The module is no longer on the active pipeline path (vector_km replaced it), but
its ``extract_axis_calibration`` is still importable and must fail closed rather
than raise an opaque ZeroDivisionError when an axis has fewer than two distinct
tick positions. Regression for KMC-CAL-2.
"""

import pytest

pytest.importorskip("fitz")  # module imports PyMuPDF at top level

import vector_curve_extractor as vce  # noqa: E402 (sys.path via tests/conftest.py)


class _Rect:
    def __init__(self, height):
        self.height = height


class _StubPage:
    """Minimal PyMuPDF-page stand-in yielding pre-baked numeric text spans."""

    def __init__(self, spans, height=400.0):
        self._spans = spans
        self.rect = _Rect(height)

    def get_text(self, kind):  # noqa: ARG002 - signature parity with fitz
        return {"blocks": [{"lines": [{"spans": self._spans}]}]}


def _span(text, x, y):
    return {"text": text, "bbox": (x - 2, y - 2, x + 2, y + 2)}


def test_single_x_tick_raises_valueerror_not_zerodiv():
    # Exactly one X-axis label (single distinct x) -> zero-width denominator.
    # Must raise a clear ValueError, never ZeroDivisionError.
    spans = [
        _span("12", 250, 360),   # x-axis label (y > 0.85*400, x > 100)
        _span("1.0", 40, 60),    # y-axis label (x < 100)
        _span("0.0", 40, 340),   # y-axis label
    ]
    with pytest.raises(ValueError):
        vce.extract_axis_calibration(_StubPage(spans))


def test_two_distinct_ticks_per_axis_calibrate():
    spans = [
        _span("0", 150, 360),    # x-axis ticks (x > 100, y > 340)
        _span("30", 300, 360),
        _span("1.0", 40, 60),    # y-axis ticks (x < 100)
        _span("0.0", 40, 340),
    ]
    cal = vce.extract_axis_calibration(_StubPage(spans))
    assert cal["x_slope"] != 0.0
    assert cal["y_slope"] != 0.0
