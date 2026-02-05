"""Image processing algorithms for the PID assignment (pixel-by-pixel)."""
from __future__ import annotations

from .box_filter import box_filter
from .canny import canny
from .freeman import freeman_chain_code
from .marr_hildreth import marr_hildreth
from .otsu import connected_components, count_objects, otsu_threshold
from .segmentation import intensity_segmentation
from .text import comparison_text
from .utils import EdgeResults, to_list
from .watershed import watershed_segment

__all__ = [
    "EdgeResults",
    "box_filter",
    "canny",
    "comparison_text",
    "connected_components",
    "count_objects",
    "freeman_chain_code",
    "intensity_segmentation",
    "marr_hildreth",
    "otsu_threshold",
    "to_list",
    "watershed_segment",
]
