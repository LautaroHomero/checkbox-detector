"""Stage 6: candidate -> checked / unchecked."""

import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import Box, Rect


def ink_ratio(ink: np.ndarray, bbox: Rect, margin: float) -> float:
    """Share of ink inside the box, ignoring a margin that holds the border."""
    inner = bbox.shrink(margin)
    crop = ink[inner.y1 : inner.y2, inner.x1 : inner.x2]
    return float((crop > 0).mean()) if crop.size else 0.0


def classify(rects: list[Rect], ink: np.ndarray, cfg: DetectionConfig) -> list[Box]:
    """Measured on the ink mask (not the line mask), so X, checkmarks and fills all count.

    TODO: send boxes whose ratio is close to the threshold to a second-stage
    classifier (e.g. a vision model on the crop) instead of a hard cut.
    """
    boxes = []
    for r in rects:
        ratio = ink_ratio(ink, r, cfg.interior_margin)
        boxes.append(Box(bbox=r, is_checked=ratio > cfg.checked_threshold, ink_ratio=ratio))
    return boxes
