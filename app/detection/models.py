"""Domain types shared by every detection stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def iou(self, other: Rect) -> float:
        """Intersection over union, in [0, 1]."""
        ix = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def expand(self, px: int, max_w: int, max_h: int) -> Rect:
        """Grow by ``px`` on every side, clipped to an image of size max_w x max_h."""
        return Rect(
            max(0, self.x1 - px),
            max(0, self.y1 - px),
            min(max_w, self.x2 + px),
            min(max_h, self.y2 + px),
        )

    def shrink(self, ratio: float) -> Rect:
        """Remove ``ratio`` of the width/height from each side (0.25 keeps the central 50%)."""
        dx, dy = int(self.width * ratio), int(self.height * ratio)
        return Rect(self.x1 + dx, self.y1 + dy, self.x2 - dx, self.y2 - dy)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class Box:
    """A detected checkbox. ``bbox`` covers the box's outer border."""

    bbox: Rect
    is_checked: bool
    ink_ratio: float
