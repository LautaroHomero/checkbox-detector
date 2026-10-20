"""Stress test: score the detector on degraded copies of the samples.

Usage (from the project root):
    python -m eval.robustness

The ground truth was labeled on these same images, so the clean score is
in-sample. Degrading the images (and transforming the ground truth to match)
gives a better idea of how the detector behaves on documents it has not seen.
"""

import json

import cv2
import numpy as np

from app.detection import detect_checkboxes
from eval.evaluate import GROUND_TRUTH_DIR, TABLE_HEADER, Counts, evaluate_image, load_sample


def scale(factor: float):
    def apply(img, gt):
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)
        return resized, [dict(b, bbox=[round(v * factor) for v in b["bbox"]]) for b in gt]

    return apply


def jpeg(quality: int):
    def apply(img, gt):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), gt

    return apply


def noise(sigma: float):
    rng = np.random.default_rng(0)

    def apply(img, gt):
        noisy = img.astype(np.int16) + rng.normal(0, sigma, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8), gt

    return apply


def blur(kernel: int):
    return lambda img, gt: (cv2.GaussianBlur(img, (kernel, kernel), 0), gt)


def rotate(degrees: float):
    def apply(img, gt):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1)
        rotated = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
        boxes = []
        for b in gt:  # new bbox = bounding rect of the rotated corners
            x1, y1, x2, y2 = b["bbox"]
            corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ m.T
            xs, ys = corners[:, 0], corners[:, 1]
            boxes.append(dict(b, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
        return rotated, boxes

    return apply


DEGRADATIONS = {
    "original": lambda img, gt: (img, gt),
    "scale 0.75": scale(0.75),
    "scale 0.5": scale(0.5),
    "scale 1.5": scale(1.5),
    "jpeg q30": jpeg(30),
    "noise 15": noise(15),
    "blur 5x5": blur(5),
    "rotate 1°": rotate(1),
    "rotate 3°": rotate(3),
}


def main() -> None:
    samples = [json.loads(p.read_text()) for p in sorted(GROUND_TRUTH_DIR.glob("*.json"))]
    images = {s["image"]: load_sample(s["image"]) for s in samples}
    print(TABLE_HEADER)
    for name, degrade in DEGRADATIONS.items():
        total = Counts()
        for sample in samples:
            img, gt = degrade(images[sample["image"]], sample["boxes"])
            counts, _ = evaluate_image(detect_checkboxes(img), gt)
            total.add(counts)
        print(total.row(name))


if __name__ == "__main__":
    main()
