"""All tunable parameters of the pipeline, in one place.

Pixel values are tuned for the sample documents (boxes of roughly 22-50 px).
TODO: derive the pixel-based values from an estimated page scale (e.g. median
text height) so the detector is resolution-independent.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Every tunable parameter of the pipeline."""

    # Stage 2 - binarization
    adaptive_block: int = 41  # neighbourhood size of the adaptive threshold (odd)
    adaptive_c: int = 10  # how much darker than its neighbourhood a pixel must be to be ink
    dark_cutoff: int = 100  # gray level below which a pixel is always ink

    # Stage 3 - line mask
    line_kernel: int = 10  # minimum length (px) of a straight run to count as a line

    # Stages 4/5 - candidate shape filters (sizes refer to the box interior)
    min_side: int = 18  # px; drops small letter loops such as a bold "o"
    max_side: int = 80  # px; drops table cells
    min_aspect: float = 0.9  # width / height; drops tall, narrow table cells
    max_aspect: float = 1.45  # some forms draw boxes as wide as the row is tall
    min_rectangularity: float = 0.85  # hole area / its bbox area

    # Stage 5 - frames and merge
    min_frame_coverage: float = 0.85  # how closed each side of a frame must be
    frame_band: int = 3  # px; band along each side where the border is looked for
    border_pad: int = 3  # px; grows a hole's interior to the box's outer border
    max_frame_overlap: float = 0.3  # share of a frame already covered by accepted boxes = duplicate

    # Stage 6 - classification
    interior_margin: float = 0.25  # share of each side cropped away before measuring ink
    checked_threshold: float = 0.06  # ink ratio above which a box is checked


DEFAULT_CONFIG = DetectionConfig()
