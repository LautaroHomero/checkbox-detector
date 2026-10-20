"""Stages 3-5: ink mask -> checkbox candidates.

Two complementary candidate sources:
- holes (source A): closed regions inside the line mask. Handles boxes that
  share their borders with a table grid.
- frames (source B): box-shaped ink blobs whose four sides are closed. Handles
  isolated boxes whose thick X leaves line fragments that split the hole.
"""

import cv2
import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import Rect


def extract_lines(ink: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Keep only horizontal and vertical straight runs of at least `line_kernel` px."""
    k = cfg.line_kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    return horizontal | vertical


def _has_box_shape(w: int, h: int, max_side: int, cfg: DetectionConfig) -> bool:
    """Whether a width x height fits the size and aspect limits of a box."""
    return (
        cfg.min_side <= w <= max_side
        and cfg.min_side <= h <= max_side
        and cfg.min_aspect <= w / h <= cfg.max_aspect
    )


def find_holes(lines: np.ndarray, cfg: DetectionConfig) -> list[Rect]:
    """Source A: box-shaped regions fully enclosed by lines. Returns their interiors."""
    height, width = lines.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - lines, connectivity=4)
    holes = []
    for x, y, w, h, area in stats[1:n]:  # label 0 is the line pixels themselves
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_edge or not _has_box_shape(w, h, cfg.max_side, cfg):
            continue
        if area / (w * h) < cfg.min_rectangularity:
            continue
        holes.append(Rect(int(x), int(y), int(x + w), int(y + h)))
    return holes


def _frame_coverage(ink: np.ndarray, r: Rect, band: int) -> float:
    """Closedness of the weakest side: share of positions along it with ink in its band."""
    sub = ink[r.y1 : r.y2, r.x1 : r.x2] > 0
    top = sub[:band, :].any(axis=0).mean()
    bottom = sub[-band:, :].any(axis=0).mean()
    left = sub[:, :band].any(axis=1).mean()
    right = sub[:, -band:].any(axis=1).mean()
    return float(min(top, bottom, left, right))


def find_frames(ink: np.ndarray, cfg: DetectionConfig) -> list[Rect]:
    """Source B: box-shaped ink blobs with four closed sides. Returns outer borders.

    Requiring all four sides rejects letters such as "O", whose band coverage
    drops at the corners.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # Frames are measured on the outer border, holes on the interior: allow for the border.
    max_side = cfg.max_side + 2 * cfg.border_pad
    frames = []
    for x, y, w, h, _ in stats[1:n]:
        if not _has_box_shape(w, h, max_side, cfg):
            continue
        r = Rect(int(x), int(y), int(x + w), int(y + h))
        if _frame_coverage(ink, r, cfg.frame_band) >= cfg.min_frame_coverage:
            frames.append(r)
    return frames


def merge_candidates(
    holes: list[Rect], frames: list[Rect], shape: tuple[int, int], cfg: DetectionConfig
) -> list[Rect]:
    """Combine holes and frames as outer-border rects, dropping duplicate frames."""
    height, width = shape
    merged = [h.expand(cfg.border_pad, width, height) for h in holes]
    covered = np.zeros(shape, dtype=bool)
    for r in merged:
        covered[r.y1 : r.y2, r.x1 : r.x2] = True
    for frame in frames:
        if covered[frame.y1 : frame.y2, frame.x1 : frame.x2].mean() <= cfg.max_frame_overlap:
            merged.append(frame)
            covered[frame.y1 : frame.y2, frame.x1 : frame.x2] = True
    return merged
