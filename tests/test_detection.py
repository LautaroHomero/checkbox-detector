"""Detection tests on synthetic pages: each case from the design, drawn with OpenCV."""

import time

import cv2
import numpy as np
import pytest

from app.detection import Rect, detect_checkboxes
from app.detection.models import Box
from app.detection.pipeline import sort_reading_order

BLACK = (0, 0, 0)
WATERMARK_RED = (127, 127, 254)  # BGR; median color of the watermark in img-003


def blank_page(w: int = 400, h: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def draw_box(img: np.ndarray, x: int, y: int, side: int = 30, thickness: int = 2) -> Rect:
    cv2.rectangle(img, (x, y), (x + side, y + side), BLACK, thickness)
    return Rect(x, y, x + side + 1, y + side + 1)


def draw_x(img: np.ndarray, r: Rect, thickness: int = 2, inset: int = 6) -> None:
    cv2.line(img, (r.x1 + inset, r.y1 + inset), (r.x2 - inset, r.y2 - inset), BLACK, thickness)
    cv2.line(img, (r.x1 + inset, r.y2 - inset), (r.x2 - inset, r.y1 + inset), BLACK, thickness)


def single_match(boxes: list[Box], expected: Rect) -> Box:
    matches = [b for b in boxes if b.bbox.iou(expected) > 0.5]
    assert len(matches) == 1, f"expected one box at {expected}, got {boxes}"
    return matches[0]


# --- Rect ---------------------------------------------------------------------------------


def test_rect_iou():
    a = Rect(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(Rect(20, 20, 30, 30)) == 0.0
    assert a.iou(Rect(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_rect_expand_is_clipped_to_image():
    assert Rect(1, 1, 9, 9).expand(3, max_w=10, max_h=10) == Rect(0, 0, 10, 10)


# --- detection and classification -----------------------------------------------------------


def test_empty_box_is_detected_unchecked():
    img = blank_page()
    expected = draw_box(img, 50, 50)
    box = single_match(detect_checkboxes(img), expected)
    assert not box.is_checked


def test_box_with_thin_x_is_checked():
    img = blank_page()
    expected = draw_box(img, 50, 50)
    draw_x(img, expected, thickness=2)
    assert single_match(detect_checkboxes(img), expected).is_checked


def test_box_with_thick_x_is_rescued_by_frames():
    """A thick X splits the hole in the line mask; the frame source must still find it."""
    img = blank_page()
    expected = draw_box(img, 50, 50, side=30, thickness=3)
    draw_x(img, expected, thickness=6, inset=3)
    assert single_match(detect_checkboxes(img), expected).is_checked


def test_box_sharing_borders_with_a_table_grid():
    """The box's top, bottom and left edges are the table's own lines."""
    img = blank_page(500, 200)
    top, bottom = 60, 100
    for y in (top, bottom):
        cv2.line(img, (10, y), (490, y), BLACK, 2)
    for x in (10, 50, 300, 490):  # x=10..50 is the box; the rest are wide cells
        cv2.line(img, (x, top), (x, bottom), BLACK, 2)
    expected = Rect(10, top, 51, bottom + 1)
    draw_x(img, expected, thickness=2, inset=10)
    boxes = detect_checkboxes(img)
    assert single_match(boxes, expected).is_checked
    assert len(boxes) == 1  # the wide cells are not boxes


def test_red_watermark_does_not_check_a_box():
    img = blank_page()
    expected = draw_box(img, 50, 50)
    # A watermark is overlaid, not painted over: dark print stays dark (multiply-like blend).
    layer = blank_page()
    cv2.line(layer, (20, 120), (140, 20), WATERMARK_RED, 6)  # crosses the box diagonally
    img = np.minimum(img, layer)
    box = single_match(detect_checkboxes(img), expected)
    assert not box.is_checked


@pytest.mark.parametrize(
    "shape",
    [
        "letter_o",  # small closed loop, like a bold "o"
        "narrow_cell",  # tall, narrow rectangle (aspect < 0.9)
        "wide_field",  # input field, far wider than tall
        "large_rect",  # bigger than any box
    ],
)
def test_non_boxes_are_rejected(shape):
    img = blank_page()
    if shape == "letter_o":
        cv2.circle(img, (100, 100), 8, BLACK, 3)
    elif shape == "narrow_cell":
        cv2.rectangle(img, (100, 60), (138, 114), BLACK, 2)
    elif shape == "wide_field":
        cv2.rectangle(img, (50, 80), (250, 110), BLACK, 2)
    else:
        cv2.rectangle(img, (50, 40), (190, 180), BLACK, 2)
    assert detect_checkboxes(img) == []


# --- ordering ---------------------------------------------------------------------------------


def test_reading_order_is_rows_then_columns():
    def box(x: int, y: int) -> Box:
        return Box(Rect(x, y, x + 30, y + 30), is_checked=False, ink_ratio=0.0)

    # second row first, and a slightly misaligned box in the first row
    boxes = [box(10, 100), box(200, 12), box(10, 10), box(100, 104)]
    ordered = sort_reading_order(boxes)
    assert [(b.bbox.x1, b.bbox.y1) for b in ordered] == [(10, 10), (200, 12), (10, 100), (100, 104)]


# --- cost ---------------------------------------------------------------------------------


def test_page_full_of_boxes_stays_fast():
    """Cost must grow with the image, not quadratically with the number of boxes."""
    img = blank_page(2540, 2540)
    for i in range(70):
        for j in range(70):
            draw_box(img, 10 + j * 36, 10 + i * 36, side=28)
    start = time.perf_counter()
    boxes = detect_checkboxes(img)
    assert len(boxes) == 70 * 70
    assert time.perf_counter() - start < 5  # ~1 s on a laptop; was ~15 s with quadratic steps
