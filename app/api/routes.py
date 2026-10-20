"""HTTP boundary: validate and decode the upload, call the detector, serialize."""

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import DetectResponse
from app.detection import detect_checkboxes

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000  # a 300 dpi letter page is ~8.4 MP

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


# Plain `def` (not `async def`): detection is CPU-bound, so FastAPI runs it in its
# thread pool instead of blocking the event loop.
@router.post("/detect", response_model=DetectResponse)
def detect(file: UploadFile) -> DetectResponse:
    """Detect and classify the checkboxes in the uploaded image."""
    # Reject on the declared size without reading; the capped read covers clients that omit it.
    too_large = HTTPException(status_code=413, detail="File too large (max 20 MB).")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise too_large
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise too_large

    # Decoding is the real validation: anything OpenCV cannot read as an image is rejected,
    # whatever its declared content type.
    # TODO: guard against decompression bombs by reading the image header's dimensions
    # before decoding (OpenCV's own limit is 2^30 pixels).
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode the file as an image.")
    # Detection time grows with the pixel count: cap it so one request can't hog a worker.
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image too large (max 40 megapixels).")

    return DetectResponse.from_boxes(detect_checkboxes(img))
