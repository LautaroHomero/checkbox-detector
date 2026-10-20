"""Checkbox detection core: pure functions on numpy images, no HTTP or file I/O."""

from app.detection.config import DEFAULT_CONFIG, DetectionConfig
from app.detection.models import Box, Rect
from app.detection.pipeline import detect_checkboxes

__all__ = ["DEFAULT_CONFIG", "Box", "DetectionConfig", "Rect", "detect_checkboxes"]
