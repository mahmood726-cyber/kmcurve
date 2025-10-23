"""OCR module for automatic text extraction from K-M curves."""

from .axis_reader import (
    AxisInfo,
    extract_axis_labels,
    auto_calibrate_axes,
    validate_axis_calibration
)
from .numbers_at_risk import (
    AtRiskData,
    parse_at_risk_table,
    validate_at_risk_data,
    match_at_risk_to_curves,
    compute_validation_metrics
)

__all__ = [
    'AxisInfo',
    'extract_axis_labels',
    'auto_calibrate_axes',
    'validate_axis_calibration',
    'AtRiskData',
    'parse_at_risk_table',
    'validate_at_risk_data',
    'match_at_risk_to_curves',
    'compute_validation_metrics'
]
