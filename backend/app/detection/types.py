"""Shared geometry and detection types."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_number: int = 0
    timestamp_sec: float = 0.0
    tracking_id: int | None = None


@dataclass
class TrackPoint:
    frame_number: int
    timestamp_sec: float
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    history: list[TrackPoint] = field(default_factory=list)
    class_votes: dict[str, float] = field(default_factory=dict)
    class_ids: dict[str, int] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.history[-1].bbox

    @property
    def confidence(self) -> float:
        return self.history[-1].confidence


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip pixel xyxy coordinates to a frame, rejecting invalid boxes."""
    if width <= 0 or height <= 0 or len(bbox) != 4:
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None
    x1, y1, x2, y2 = bbox
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped
