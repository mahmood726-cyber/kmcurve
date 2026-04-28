"""
Metadata extraction module.

Combines color clustering with OCR to extract complete K-M curve metadata.
"""

from .color_legend_matcher import (
    match_curves_to_legend,
    match_risk_table_to_curves,
    create_complete_metadata
)

__all__ = [
    'match_curves_to_legend',
    'match_risk_table_to_curves',
    'create_complete_metadata'
]
