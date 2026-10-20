"""FastAPI application. Run with: uvicorn app.main:app"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

STATIC_DIR = Path(__file__).parent / "static"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

app = FastAPI(
    title="Checkbox Detector",
    description="Detects checkboxes in document images and classifies them as checked/unchecked.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Demo page: upload an image or try a sample and see the detected boxes."""
    return FileResponse(STATIC_DIR / "index.html")


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
