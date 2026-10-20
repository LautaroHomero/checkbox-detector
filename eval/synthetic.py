"""Find the detector's limits on synthetic forms whose ground truth is known exactly.

Usage (from the project root):
    python -m eval.synthetic

Each experiment changes one variable (box size, border, mark type, ...) and keeps
the rest fixed. Every page is a grid of labeled boxes; even boxes get a mark,
odd ones stay empty. For each value it prints detection recall, false positives
and classification accuracy, and saves a sheet with what was detected drawn on
top (green = checked, red = unchecked) to eval/output/synthetic/<experiment>.png.
"""

import cv2
import numpy as np

from app.detection import detect_checkboxes
from eval.evaluate import OUTPUT_DIR, evaluate_image
from eval.robustness import noise, rotate

ROWS, COLS = 4, 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Mark name -> whether a box with it should count as checked.
MARKS = {
    "x_thin": True,
    "x_thick": True,
    "check": True,
    "dot": True,
    "fill": True,
    "letter": False,
    "digit": False,
}


def draw_mark(img, x, y, side, mark, gray, size=0.8):
    """Draw ``mark`` inside the box at (x, y), spanning ``size`` of its interior."""
    color = (gray, gray, gray)
    pad = int(side * (1 - size) / 2) + 3
    a, b = (x + pad, y + pad), (x + side - pad, y + side - pad)
    thin = max(1, side // 15)
    if mark in ("x_thin", "x_thick"):
        t = thin if mark == "x_thin" else max(3, side // 6)
        cv2.line(img, a, b, color, t)
        cv2.line(img, (a[0], b[1]), (b[0], a[1]), color, t)
    elif mark == "check":
        pts = np.array([[a[0], y + side // 2], [x + side // 2 - 2, b[1]], [b[0], a[1]]], np.int32)
        cv2.polylines(img, [pts], False, color, thin + 1)
    elif mark == "dot":
        cv2.circle(img, (x + side // 2, y + side // 2), max(2, side // 7), color, -1)
    elif mark == "fill":
        cv2.rectangle(img, a, b, color, -1)
    else:
        text = "A" if mark == "letter" else "5"
        scale = side / 45
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thin + 1)
        cv2.putText(
            img, text, (x + (side - tw) // 2, y + (side + th) // 2), FONT, scale, color, thin + 1
        )


def make_page(side=30, border=2, mark="x_thin", mark_size=0.8, gray=0, table=False):
    """A form with ROWS x COLS labeled boxes. Returns the image and its ground truth."""
    col_w, row_h = side + 170, max(2 * side, 60)
    img = np.full((ROWS * row_h + 40, COLS * col_w + 40, 3), 255, np.uint8)
    color = (gray, gray, gray)
    gt = []
    for i in range(ROWS):
        y = 20 + i * row_h + (row_h - side) // 2
        if table:  # each row is a table row and each box is one of its cells
            for yy in (y, y + side):
                cv2.line(img, (20, yy), (20 + COLS * col_w, yy), color, border)
        for j in range(COLS):
            x = 20 + j * col_w + 10
            if table:
                cv2.line(img, (x, y), (x, y + side), color, border)
                cv2.line(img, (x + side, y), (x + side, y + side), color, border)
            else:
                cv2.rectangle(img, (x, y), (x + side, y + side), color, border)
            cv2.putText(img, "Option", (x + side + 10, y + side // 2 + 7), FONT, 0.6, color, 1)
            marked = (i * COLS + j) % 2 == 0 and mark != "none"
            if marked:
                draw_mark(img, x, y, side, mark, gray, mark_size)
            h = border // 2
            gt.append(
                {
                    "bbox": [x - h, y - h, x + side + h + 1, y + side + h + 1],
                    "is_checked": marked and MARKS[mark],
                    "ambiguous": False,
                }
            )
    return img, gt


EXPERIMENTS = {
    "box_size": {f"{s}px": {"side": s} for s in (10, 14, 18, 22, 30, 45, 60, 80, 100)},
    "border": {f"{t}px": {"border": t} for t in (1, 2, 3, 4, 5)},
    "mark_type": {m: {"mark": m} for m in ("none", *MARKS)},
    "mark_size": {f"{int(f * 100)}%": {"mark_size": f} for f in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0)},
    "ink_gray": {str(g): {"gray": g} for g in (0, 60, 120, 160, 190, 220)},
    "in_table": {"no": {"table": False}, "yes": {"table": True}},
    "noise": {f"sd {s}": {"noise": s} for s in (0, 10, 20, 30)},
    "rotation": {f"{d}°": {"rotation": d} for d in (0, 1, 2, 3, 5)},
}


def run_variant(params: dict):
    """Generate one page for a variant, run the detector and score it."""
    params = dict(params)
    sigma, degrees = params.pop("noise", 0), params.pop("rotation", 0)
    img, gt = make_page(**params)
    if sigma:
        img, gt = noise(sigma)(img, gt)
    if degrees:
        img, gt = rotate(degrees)(img, gt)
    boxes = detect_checkboxes(img)
    counts, _ = evaluate_image(boxes, gt)
    return img, boxes, counts


def first_row(img, boxes, label: str) -> np.ndarray:
    """The page's first row with the detections drawn, labeled with the variant name."""
    out = img.copy()
    for b in boxes:
        r = b.bbox
        cv2.rectangle(
            out, (r.x1, r.y1), (r.x2, r.y2), (40, 150, 40) if b.is_checked else (40, 40, 215), 2
        )
    row_h = (img.shape[0] - 40) // ROWS
    strip = out[: 20 + row_h + 10]
    header = np.full((28, strip.shape[1], 3), 235, np.uint8)
    cv2.putText(header, label, (8, 20), FONT, 0.6, (0, 0, 0), 1)
    return np.vstack([header, strip])


def main() -> None:
    out_dir = OUTPUT_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, variants in EXPERIMENTS.items():
        print(f"\n{name}\n{'value':<10} {'recall':>7} {'fp':>4} {'class.acc':>10}")
        strips = []
        for value, params in variants.items():
            img, boxes, c = run_variant(params)
            recall = c.tp / (c.tp + c.fn)
            accuracy = c.correct_class / c.tp if c.tp else 0.0
            print(f"{value:<10} {recall:>7.0%} {c.fp:>4} {accuracy:>10.0%}")
            strips.append(first_row(img, boxes, f"{name} = {value}"))
        width = max(s.shape[1] for s in strips)
        sheet = np.vstack(
            [
                cv2.copyMakeBorder(s, 0, 6, 0, width - s.shape[1], 0, value=(255, 255, 255))
                for s in strips
            ]
        )
        cv2.imwrite(str(out_dir / f"{name}.png"), sheet)


if __name__ == "__main__":
    main()
