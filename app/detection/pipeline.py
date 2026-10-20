"""Orchestrates the detection stages. ``detect_checkboxes`` is the module's entry point."""

import numpy as np

from app.detection.candidates import extract_lines, find_frames, find_holes, merge_candidates
from app.detection.classifier import classify
from app.detection.config import DEFAULT_CONFIG, DetectionConfig
from app.detection.models import Box
from app.detection.preprocess import binarize, to_gray


def detect_checkboxes(img: np.ndarray, cfg: DetectionConfig = DEFAULT_CONFIG) -> list[Box]:
    """Detect and classify checkboxes in a BGR (or grayscale) image."""
    gray = to_gray(img)
    ink = binarize(gray, cfg)
    lines = extract_lines(ink, cfg)
    rects = merge_candidates(find_holes(lines, cfg), find_frames(ink, cfg), gray.shape, cfg)
    return sort_reading_order(classify(rects, ink, cfg))


def sort_reading_order(boxes: list[Box]) -> list[Box]:
    """Sort boxes in rows top to bottom, left to right within a row."""
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: b.bbox.center_y):
        if rows:
            anchor = rows[-1][0].bbox
            if box.bbox.center_y - anchor.center_y <= anchor.height / 2:
                rows[-1].append(box)
                continue
        rows.append([box])
    return [box for row in rows for box in sorted(row, key=lambda b: b.bbox.x1)]
