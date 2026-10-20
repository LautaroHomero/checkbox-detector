"""HTTP response models. Exactly the schema the challenge asks for."""

from pydantic import BaseModel, Field

from app.detection import Box


class CheckboxOut(BaseModel):
    """One detected checkbox, as returned by the API."""

    bbox: list[int] = Field(description="[x1, y1, x2, y2] in pixels, origin at the top-left")
    is_checked: bool


class DetectResponse(BaseModel):
    """Response of POST /detect."""

    boxes: list[CheckboxOut]

    @classmethod
    def from_boxes(cls, boxes: list[Box]) -> "DetectResponse":
        """Build the response from the detector's boxes (drops the internal score)."""
        return cls(
            boxes=[CheckboxOut(bbox=b.bbox.to_list(), is_checked=b.is_checked) for b in boxes]
        )
