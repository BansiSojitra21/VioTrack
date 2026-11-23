"""Collapse duplicate boxes on the same physical vehicle."""

from __future__ import annotations

from app.detection.labels import VEHICLE_CLASSES
from app.detection.types import Detection, iou as bbox_iou

# Prefer passenger car over truck/bike when YOLO double-fires on one hatchback.
CLASS_RANK = {
    "car": 4,
    "bus": 3,
    "truck": 2,
    "motorcycle": 1,
    "bicycle": 0,
}


def box_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def box_containment(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    smaller = min(box_area(a), box_area(b))
    return intersection / smaller if smaller > 0 else 0.0


def _center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _box_radius(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0) * 0.5


def boxes_same_vehicle(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    if box_containment(a, b) >= 0.5:
        return True
    overlap = bbox_iou(a, b)
    if overlap >= 0.22:
        return True
    larger = a if box_area(a) >= box_area(b) else b
    return overlap >= 0.08 and _center_distance(a, b) <= _box_radius(larger) * 1.1


def prefer_vehicle_detection(a: Detection, b: Detection) -> Detection:
    """Pick one box for an overlapping pair — prefer car over truck/bike when close in confidence."""
    conf_a = a.confidence if _finite(a.confidence) else 0.0
    conf_b = b.confidence if _finite(b.confidence) else 0.0
    rank_a = CLASS_RANK.get(a.class_name, 0)
    rank_b = CLASS_RANK.get(b.class_name, 0)

    if a.class_name != b.class_name:
        # Hatchbacks often get a second truck/bike fire at lower confidence — keep the car.
        if rank_a != rank_b and min(conf_a, conf_b) >= max(conf_a, conf_b) * 0.55:
            winner = a if rank_a > rank_b else b
        elif conf_b > conf_a:
            winner = b
        elif conf_a > conf_b:
            winner = a
        else:
            winner = a if rank_a >= rank_b else b
    else:
        winner = a if box_area(a.bbox) >= box_area(b.bbox) else b
        if conf_b > conf_a and box_containment(a.bbox, b.bbox) < 0.5:
            winner = b

    winner.confidence = max(conf_a, conf_b)
    return winner


def is_spurious_truck(
    bbox: tuple[float, float, float, float],
    class_name: str,
    frame_w: int = 0,
    frame_h: int = 0,
) -> bool:
    """Drop truck FPs on trees/buildings (common with COCO YOLO on street cams).

    Real trucks sit on the roadway; foliage hallucinations usually float in the
    upper canopy and never reach the road surface.
    """
    if class_name != "truck" or frame_h <= 0:
        return False
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 1 or height < 1:
        return True
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    # Bottom of box never reaches mid-frame → not on the road.
    if y2 < frame_h * 0.48:
        return True
    # Center stuck in the upper canopy band.
    if cy < frame_h * 0.36:
        return True
    # Tall canopy patch hugging the left/right edge (trees beside the street).
    roadside = frame_w > 0 and (cx < frame_w * 0.18 or cx > frame_w * 0.82)
    if roadside and cy < frame_h * 0.5:
        return True
    return False


def suppress_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    """Collapse stacked vehicle boxes. Keep pedestrians separate for helmet heuristics."""
    vehicles = [det for det in detections if det.class_name in VEHICLE_CLASSES]
    others = [det for det in detections if det.class_name not in VEHICLE_CLASSES]
    kept: list[Detection] = []
    for detection in sorted(vehicles, key=lambda item: item.confidence, reverse=True):
        if not _finite(detection.confidence):
            continue
        duplicate_index = -1
        for index, previous in enumerate(kept):
            if boxes_same_vehicle(detection.bbox, previous.bbox):
                duplicate_index = index
                break
        if duplicate_index < 0:
            kept.append(detection)
            continue
        kept[duplicate_index] = prefer_vehicle_detection(kept[duplicate_index], detection)
    return kept + [det for det in others if _finite(det.confidence)]


def _finite(value: float) -> bool:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return False
    return conf == conf and conf not in (float("inf"), float("-inf")) and 0.0 <= conf <= 1.0
