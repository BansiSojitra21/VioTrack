"""Local license-plate detection and OCR for violation evidence."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from importlib import import_module
from statistics import mean
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_alpr: Any = None
_alpr_init_attempted = False
_alpr_lock = threading.Lock()


def _get_alpr() -> Any:
    global _alpr, _alpr_init_attempted
    if _alpr_init_attempted:
        return _alpr
    with _alpr_lock:
        if _alpr_init_attempted:
            return _alpr
        try:
            _alpr = import_module("fast_alpr").ALPR()
        except Exception:
            logger.exception("Could not initialize local license-plate recognition.")
        finally:
            _alpr_init_attempted = True
    return _alpr


def license_plate_reader_available() -> bool:
    return _get_alpr() is not None


def _readable_plate(result: Any) -> tuple[str, float] | None:
    ocr = result.ocr
    if ocr is None or not ocr.text:
        return None
    text = re.sub(r"[^A-Z0-9]", "", ocr.text.upper())
    if not text:
        return None
    confidence = ocr.confidence
    if isinstance(confidence, list):
        if not confidence:
            return None
        score = mean(confidence)
    else:
        score = float(confidence)
    return (text, score) if score >= 0.5 else None


def _overlap_ratio(
    plate_bbox: tuple[float, float, float, float],
    vehicle_bbox: tuple[float, float, float, float],
) -> float:
    px1, py1, px2, py2 = plate_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    plate_area = max(0, px2 - px1) * max(0, py2 - py1)
    if plate_area == 0:
        return 0
    intersection = max(0, min(vx2, px2) - max(vx1, px1)) * max(
        0, min(vy2, py2) - max(vy1, py1)
    )
    return intersection / plate_area


def select_vehicle_plates(
    results: Sequence[Any],
    vehicles: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> dict[int, tuple[str, float]]:
    """Associate each readable plate with the vehicle box that contains it."""
    best: dict[int, tuple[str, float]] = {}
    for result in results:
        plate = _readable_plate(result)
        if plate is None:
            continue
        box = result.detection.bounding_box
        plate_bbox = (box.x1, box.y1, box.x2, box.y2)
        matches = [
            (tracking_id, _overlap_ratio(plate_bbox, vehicle_bbox))
            for tracking_id, vehicle_bbox in vehicles
        ]
        if not matches:
            continue
        tracking_id, overlap = max(matches, key=lambda match: match[1])
        if overlap < 0.5:
            continue
        if tracking_id not in best or plate[1] > best[tracking_id][1]:
            best[tracking_id] = plate
    return best


def read_vehicle_plates(
    frame: np.ndarray,
    vehicles: Sequence[tuple[int, tuple[float, float, float, float]]],
    *,
    raise_on_error: bool = False,
) -> dict[int, tuple[str, float]]:
    """Detect and read plates for all tracked vehicles in one frame."""
    if not vehicles:
        return {}
    alpr = _get_alpr()
    if alpr is None:
        return {}
    try:
        return select_vehicle_plates(alpr.predict(frame), vehicles)
    except Exception:
        if raise_on_error:
            raise
        logger.exception("Could not read license plates in an analysis frame.")
        return {}
