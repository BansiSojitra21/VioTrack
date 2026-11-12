"""COCO traffic classes and optional custom-model labels."""

from __future__ import annotations

# COCO class ids used by official YOLO detection weights.
COCO_TRAFFIC_CLASSES: dict[int, str] = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}

HELMET_POSITIVE = {"helmet", "with_helmet"}
HELMET_NEGATIVE = {"no_helmet", "without_helmet", "no-helmet"}

CATEGORY_COLORS = {
    "car": "#38bdf8",
    "motorcycle": "#f59e0b",
    "bus": "#a78bfa",
    "truck": "#34d399",
    "bicycle": "#fb7185",
    "pedestrian": "#e2e8f0",
    "helmet": "#22c55e",
    "no_helmet": "#ef4444",
}


def normalize_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
