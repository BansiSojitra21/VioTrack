"""Model card assembled from the configured weights and optional evaluation file."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.detection.labels import COCO_TRAFFIC_CLASSES
from app.detection.yolo_engine import vehicle_engine


def _on_disk(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.is_file() or (settings.weights_dir / value).is_file()


def load_evaluation() -> dict | None:
    path = settings.weights_dir / "evaluation.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("example_only"):
        return None
    data["source"] = "evaluation_file"
    data["sample"] = False
    return data


def model_info(conf: float | None = None, iou: float | None = None, probe: bool = False) -> dict:
    engine = vehicle_engine(conf if conf is not None else settings.yolo_conf, iou if iou is not None else settings.yolo_iou)
    on_disk = _on_disk(settings.yolo_weights)
    status = engine.load() if probe and on_disk else engine.status
    return {
        "sample": False,
        "loaded": bool(status.loaded and on_disk),
        "weights_on_disk": on_disk,
        "allow_weight_download": settings.allow_weight_download,
        "name": status.name or Path(settings.yolo_weights).name,
        "version": status.name,
        "architecture": status.architecture,
        "task": "detect",
        "dataset": "COCO traffic subset (person, bicycle, car, motorcycle, bus, truck)" if on_disk or not settings.helmet_weights else status.dataset,
        "dataset_note": "Official YOLO weights are trained on COCO. Helmet classes are not included.",
        "classes": list(COCO_TRAFFIC_CLASSES.values()) if status.classes == [] else status.classes,
        "num_classes": len(COCO_TRAFFIC_CLASSES) if not status.classes else len(status.classes),
        "input_size": settings.yolo_imgsz,
        "confidence_threshold": conf if conf is not None else settings.yolo_conf,
        "iou_threshold": iou if iou is not None else settings.yolo_iou,
        "device": settings.device,
        "weights": settings.yolo_weights,
        "available_models": sorted(path.name for path in settings.weights_dir.glob("*.pt")),
        "error": status.error or (
            None
            if on_disk
            else (
                f"Weights file '{settings.yolo_weights}' is not in backend/app/weights. "
                "Detection will not run until weights are configured."
            )
        ),
        "metrics": load_evaluation(),
        "helmet": {
            "configured": _on_disk(settings.helmet_weights),
            "weights": settings.helmet_weights or None,
            "message": "Set HELMET_WEIGHTS to a custom model with helmet and no_helmet classes."
            if not settings.helmet_weights
            else ("Weights file found." if _on_disk(settings.helmet_weights) else "HELMET_WEIGHTS is set but the file was not found."),
        },
        "custom_weights_help": [
            "Train or export a YOLO .pt file with Ultralytics.",
            "Copy it to backend/app/weights/your-model.pt.",
            "Set YOLO_WEIGHTS or HELMET_WEIGHTS to the file name.",
            "Optional: add backend/app/weights/evaluation.json with precision, recall, f1, map50, map50_95, confusion_matrix, train_loss, and val_loss.",
            "Metrics from that file are shown as real evaluation results. The interface does not invent mAP.",
        ],
    }
