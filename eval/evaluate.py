"""Compare the detector against the hand-labeled ground truth.

Usage (from the project root):
    python -m eval.evaluate

Matching: every (prediction, ground-truth) pair with IoU >= 0.5, greedily from
the highest IoU down, each box used at most once. Ground-truth boxes marked
``ambiguous`` are left out of the metrics: a prediction matched to one is
neither a hit nor a false positive, and missing one is not a miss.

Detection is scored with precision / recall / F1; classification with the
accuracy of ``is_checked`` over the correctly detected boxes. Errors are drawn
to eval/output/<name>_errors.png.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.detection import Box, Rect, detect_checkboxes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
GROUND_TRUTH_DIR = ROOT / "eval" / "ground_truth"
OUTPUT_DIR = ROOT / "eval" / "output"
IOU_THRESHOLD = 0.5

FALSE_POSITIVE = (0, 140, 255)  # BGR orange
MISSED = (200, 0, 200)  # magenta
MISCLASSIFIED = (220, 80, 0)  # blue

TABLE_HEADER = "           gt   tp   fp   fn   precis recall     f1   class.acc"


def load_sample(name: str):
    """Read a sample image; the samples are not shipped with the project."""
    path = SAMPLES_DIR / name
    if not path.exists():
        raise SystemExit(f"Missing {path}: see 'Evaluation' in the README.")
    return cv2.imread(str(path))


@dataclass
class Counts:
    """Hits, false positives, misses and correct classifications."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    correct_class: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.correct_class += other.correct_class

    def row(self, name: str) -> str:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = self.correct_class / self.tp if self.tp else 0.0
        return (
            f"{name:<10} {self.tp + self.fn:>4} {self.tp:>4} {self.fp:>4} {self.fn:>4}"
            f"   {precision:6.1%} {recall:6.1%} {f1:6.1%}   {accuracy:6.1%}"
        )


def evaluate_image(predictions: list[Box], ground_truth: list[dict]) -> tuple[Counts, list]:
    """Match predictions to the ground truth (IoU >= 0.5) and count the results."""
    gt_rects = [Rect(*g["bbox"]) for g in ground_truth]
    pairs = sorted(
        (
            (p.bbox.iou(g), i, j)
            for i, p in enumerate(predictions)
            for j, g in enumerate(gt_rects)
            if p.bbox.iou(g) >= IOU_THRESHOLD
        ),
        reverse=True,
    )
    matched_pred: dict[int, int] = {}
    matched_gt: set[int] = set()
    for _, i, j in pairs:
        if i not in matched_pred and j not in matched_gt:
            matched_pred[i] = j
            matched_gt.add(j)

    counts = Counts()
    errors = []  # (rect, color)
    for i, p in enumerate(predictions):
        j = matched_pred.get(i)
        if j is None:
            counts.fp += 1
            errors.append((p.bbox, FALSE_POSITIVE))
        elif not ground_truth[j]["ambiguous"]:
            counts.tp += 1
            if p.is_checked == ground_truth[j]["is_checked"]:
                counts.correct_class += 1
            else:
                errors.append((p.bbox, MISCLASSIFIED))
    for j, g in enumerate(ground_truth):
        if j not in matched_gt and not g["ambiguous"]:
            counts.fn += 1
            errors.append((gt_rects[j], MISSED))
    return counts, errors


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(TABLE_HEADER)
    total = Counts()
    for gt_path in sorted(GROUND_TRUTH_DIR.glob("*.json")):
        ground_truth = json.loads(gt_path.read_text())
        img = load_sample(ground_truth["image"])
        counts, errors = evaluate_image(detect_checkboxes(img), ground_truth["boxes"])
        total.add(counts)
        print(counts.row(gt_path.stem))

        for r, color in errors:
            cv2.rectangle(img, (r.x1 - 4, r.y1 - 4), (r.x2 + 4, r.y2 + 4), color, 3)
        cv2.imwrite(str(OUTPUT_DIR / f"{gt_path.stem}_errors.png"), img)
    print(total.row("total"))


if __name__ == "__main__":
    main()
