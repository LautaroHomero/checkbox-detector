"""Stages 1-2: image -> gray -> ink mask."""

import cv2
import numpy as np

from app.detection.config import DetectionConfig


def to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale that keeps each pixel's brightest channel, so colored marks turn light."""
    if img.ndim == 2:
        return img
    return img[:, :, :3].max(axis=2)


def binarize(gray: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Ink mask (255 = ink): adaptive threshold OR a global dark cutoff."""
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adaptive_block,
        cfg.adaptive_c,
    )
    return np.where((adaptive > 0) | (gray < cfg.dark_cutoff), 255, 0).astype(np.uint8)
