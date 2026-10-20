"""Endpoint tests: happy path on a synthetic form, plus the error responses."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app

client = TestClient(app)


def form_png() -> bytes:
    """A small form with one checked and one empty box."""
    img = np.full((200, 400, 3), 255, np.uint8)
    for x in (50, 200):
        cv2.rectangle(img, (x, 80), (x + 30, 110), (0, 0, 0), 2)
    cv2.line(img, (56, 86), (74, 104), (0, 0, 0), 2)
    cv2.line(img, (56, 104), (74, 86), (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


def post_png(data: bytes):
    return client.post("/detect", files={"file": ("form.png", data, "image/png")})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_detect_returns_the_challenge_schema():
    response = post_png(form_png())

    assert response.status_code == 200
    boxes = response.json()["boxes"]
    assert [b["is_checked"] for b in boxes] == [True, False]
    for box in boxes:
        assert set(box) == {"bbox", "is_checked"}
        x1, y1, x2, y2 = box["bbox"]
        assert x1 < x2 and y1 < y2


def test_non_image_is_rejected():
    response = client.post("/detect", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_missing_file_is_rejected():
    assert client.post("/detect").status_code == 422


def test_too_large_file_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    assert post_png(b"x" * 11).status_code == 413


def test_too_many_pixels_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes, "MAX_PIXELS", 100)
    assert post_png(form_png()).status_code == 413


def test_demo_page_and_samples_are_served():
    page = client.get("/")
    assert page.status_code == 200
    assert "Checkbox Detector" in page.text
    assert client.get("/samples/img-002.png").headers["content-type"] == "image/png"
