"""Run the detector on the sample images and save predictions + overlays.

Usage (from the project root):
    python -m eval.run

Writes, per image, to eval/output/:
    <name>.json  predictions; ``id`` matches the number drawn on the overlay
    <name>.png   the image with every box drawn (green = checked, red = unchecked)
"""

import json
from pathlib import Path

import cv2
import numpy as np

from app.detection import Box, detect_checkboxes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
OUTPUT_DIR = ROOT / "eval" / "output"

GREEN = (40, 150, 40)  # BGR
RED = (40, 40, 215)
WHITE = (255, 255, 255)


def draw_overlay(img: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Draw every box with its id (green = checked, red = unchecked)."""
    out = img.copy()
    for i, box in enumerate(boxes):
        r = box.bbox
        color = GREEN if box.is_checked else RED
        cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, 2)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (r.x1, r.y1 - th - 6), (r.x1 + tw + 4, r.y1), WHITE, -1)
        cv2.putText(out, label, (r.x1 + 2, r.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def main() -> None:
    paths = sorted(SAMPLES_DIR.glob("*.png"))
    if not paths:
        raise SystemExit(f"No images in {SAMPLES_DIR}: see 'Evaluation' in the README.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = cv2.imread(str(path))
        boxes = detect_checkboxes(img)

        predictions = {
            "image": path.name,
            "boxes": [
                {
                    "id": i,
                    "bbox": b.bbox.to_list(),
                    "is_checked": b.is_checked,
                    "ink_ratio": round(b.ink_ratio, 3),
                }
                for i, b in enumerate(boxes)
            ],
        }
        (OUTPUT_DIR / f"{path.stem}.json").write_text(json.dumps(predictions, indent=2))
        cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}.png"), draw_overlay(img, boxes))

        checked = sum(b.is_checked for b in boxes)
        print(f"{path.name}: {len(boxes)} boxes ({checked} checked)")


if __name__ == "__main__":
    main()
