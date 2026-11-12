"""YOLO inference wrapper.

Predictions come only from an Ultralytics model. If weights or the runtime are
missing, callers receive an explicit error and no invented boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import weakref

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.detection.labels import COCO_TRAFFIC_CLASSES, normalize_label
from app.detection.types import Detection, clip_bbox

logger = get_logger(__name__)

# Cache loaded YOLO model instances to prevent reloading weights on every video job
_MODEL_CACHE: dict[str, Any] = {}


class ModelUnavailableError(RuntimeError):
    """Raised when inference cannot run because the model is not configured."""


@dataclass
class ModelStatus:
    loaded: bool
    name: str
    weights: str
    architecture: str
    task: str
    dataset: str
    classes: list[str] = field(default_factory=list)
    class_map: dict[int, str] = field(default_factory=dict)
    input_size: int = 640
    error: str | None = None
    metrics_source: str | None = None


class YoloEngine:
    def __init__(
        self,
        weights: str,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        class_filter: dict[int, str] | None = None,
        task_name: str = "vehicle",
        track_buffer: int | None = None,
        tracker_type: str = "bytetrack",
    ) -> None:
        self.weights_setting = weights
        self.conf = settings.yolo_conf if conf is None else conf
        self.iou = settings.yolo_iou if iou is None else iou
        self.imgsz = settings.yolo_imgsz if imgsz is None else imgsz
        self.device = settings.device if device is None else device
        self.class_filter = class_filter
        self.task_name = task_name
        self.track_buffer = settings.track_buffer if track_buffer is None else track_buffer
        self.tracker_type = tracker_type.lower() if tracker_type else "bytetrack"
        self._model = None
        self._tracker_config: str | None = None
        self._tracker_config_finalizer: weakref.finalize | None = None
        self.invalid_bbox_count = 0
        self.status = self._inspect()

    def _inspect(self) -> ModelStatus:
        resolved = settings.resolve_weights(self.weights_setting)
        name = Path(self.weights_setting).name if self.weights_setting else ""
        architecture = "YOLO11" if "yolo11" in name.lower() or "yolo12" in name.lower() else "YOLO"
        if "yolo12" in name.lower():
            architecture = "YOLO12"
        elif "yolov8" in name.lower() or name.lower().startswith("yolov8"):
            architecture = "YOLOv8"
        elif "yolo11" in name.lower():
            architecture = "YOLO11"
        dataset = "COCO (traffic subset)" if self.class_filter is COCO_TRAFFIC_CLASSES else "Custom"
        if not self.weights_setting:
            return ModelStatus(
                loaded=False,
                name=self.task_name,
                weights="",
                architecture=architecture,
                task=self.task_name,
                dataset=dataset,
                input_size=self.imgsz,
                error=f"{self.task_name} weights are not configured.",
            )
        return ModelStatus(
            loaded=False,
            name=name,
            weights=str(resolved) if resolved else self.weights_setting,
            architecture=architecture,
            task=self.task_name,
            dataset=dataset,
            input_size=self.imgsz,
            error=None,
        )

    def load(self) -> ModelStatus:
        if self._model is not None:
            self.status.loaded = True
            self.status.error = None
            return self.status
        if not self.weights_setting:
            self.status.loaded = False
            self.status.error = f"{self.task_name} weights are not configured."
            return self.status
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on install
            self.status.loaded = False
            self.status.error = (
                "Ultralytics is not installed. Install backend requirements before running inference. "
                f"Details: {exc}"
            )
            logger.error("ultralytics import failed: %s", exc)
            return self.status
        try:
            resolved = settings.resolve_weights(self.weights_setting)
            target = str(resolved) if resolved else self.weights_setting
            if target in _MODEL_CACHE:
                self._model = _MODEL_CACHE[target]
            else:
                self._model = YOLO(target)
                _MODEL_CACHE[target] = self._model
            if self.device == "auto":
                try:
                    import torch

                    if torch.cuda.is_available():
                        self.device = "0"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        self.device = "mps"
                    else:
                        self.device = "cpu"
                except Exception:
                    self.device = "cpu"
            names = getattr(self._model, "names", {}) or {}
            if isinstance(names, dict):
                self.status.class_map = {int(k): normalize_label(str(v)) for k, v in names.items()}
            elif isinstance(names, list):
                self.status.class_map = {i: normalize_label(str(v)) for i, v in enumerate(names)}
            if self.class_filter is not None:
                self.status.classes = sorted(set(self.class_filter.values()))
            else:
                self.status.classes = sorted(set(self.status.class_map.values()))
            self.status.loaded = True
            self.status.error = None
            logger.info("loaded yolo weights %s for %s", target, self.task_name)
        except Exception as exc:
            self._model = None
            self.status.loaded = False
            self.status.error = f"Failed to load YOLO weights '{self.weights_setting}': {exc}"
            logger.error(self.status.error)
        return self.status

    def reset_tracking(self) -> None:
        """Reset internal tracker state so each video begins with a fresh tracking session."""
        if self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = []
        self._tracker_config = None

    def predict(self, frame: np.ndarray, frame_number: int, timestamp_sec: float) -> list[Detection]:
        results = self._infer(frame, use_tracker=False)
        return self._decode(results, frame, frame_number, timestamp_sec, has_track_ids=False)

    def track(self, frame: np.ndarray, frame_number: int, timestamp_sec: float) -> list[Detection]:
        """Run sequential inference with one persistent Ultralytics tracking session."""
        results = self._infer(frame, use_tracker=True, tracker_path=self._tracker_config_path())
        return self._decode(results, frame, frame_number, timestamp_sec, has_track_ids=True)

    def _infer(
        self, frame: np.ndarray, use_tracker: bool, tracker_path: str | None = None
    ) -> list:
        status = self.load()
        if not status.loaded or self._model is None:
            raise ModelUnavailableError(status.error or "YOLO model is unavailable.")
        predict_options = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.class_filter is not None:
            predict_options["classes"] = list(self.class_filter)
        if use_tracker:
            predict_options["persist"] = True
            predict_options["tracker"] = tracker_path or "bytetrack.yaml"
        infer = self._model.track if use_tracker else self._model.predict
        try:
            return infer(frame, device=self.device, **predict_options)
        except Exception:
            if self.device == "cpu":
                raise
            logger.exception("YOLO inference failed on device %s; retrying on CPU", self.device)
            self.device = "cpu"
            return infer(frame, device="cpu", **predict_options)

    def _tracker_config_path(self) -> str:
        if self._tracker_config is not None:
            return self._tracker_config
        directory = settings.resolved_data_dir / "tracker-config"
        directory.mkdir(parents=True, exist_ok=True)
        tracker_type = getattr(self, "tracker_type", "bytetrack") or "bytetrack"
        if tracker_type == "botsort":
            config_lines = [
                "tracker_type: botsort",
                f"track_high_thresh: {self.conf}",
                f"track_low_thresh: {min(self.conf, 0.1)}",
                f"new_track_thresh: {self.conf}",
                f"track_buffer: {self.track_buffer}",
                "match_thresh: 0.8",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.8",
                "with_reid: False",
                "",
            ]
        else:
            config_lines = [
                "tracker_type: bytetrack",
                f"track_high_thresh: {self.conf}",
                f"track_low_thresh: {min(self.conf, 0.1)}",
                f"new_track_thresh: {self.conf}",
                f"track_buffer: {self.track_buffer}",
                "match_thresh: 0.8",
                "fuse_score: True",
                "",
            ]
        config_text = "\n".join(config_lines)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix=f"{tracker_type}-", dir=directory, delete=False
        ) as config_file:
            config_file.write(config_text)
            config_path = Path(config_file.name)
        self._tracker_config = str(config_path)
        self._tracker_config_finalizer = weakref.finalize(self, config_path.unlink, missing_ok=True)
        return self._tracker_config

    def _decode(
        self,
        results: list,
        frame: np.ndarray,
        frame_number: int,
        timestamp_sec: float,
        has_track_ids: bool,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if has_track_ids and boxes.id is not None else None
        for index, (box, score, class_id) in enumerate(zip(xyxy, confs, classes)):
            class_id = int(class_id)
            raw_name = self.status.class_map.get(class_id, str(class_id))
            if self.class_filter is not None:
                if class_id not in self.class_filter:
                    continue
                class_name = self.class_filter[class_id]
            else:
                class_name = normalize_label(raw_name)
            x1, y1, x2, y2 = [float(v) for v in box]
            valid_bbox = clip_bbox((x1, y1, x2, y2), int(frame.shape[1]), int(frame.shape[0]))
            if valid_bbox is None:
                self.invalid_bbox_count += 1
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    bbox=valid_bbox,
                    frame_number=frame_number,
                    timestamp_sec=timestamp_sec,
                    tracking_id=int(track_ids[index]) if track_ids is not None else None,
                )
            )
        return detections


def vehicle_engine(
    conf: float,
    iou: float,
    imgsz: int | None = None,
    track_buffer: int | None = None,
    tracker_type: str | None = None,
) -> YoloEngine:
    return YoloEngine(
        weights=settings.yolo_weights,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        track_buffer=track_buffer,
        tracker_type=tracker_type or settings.yolo_tracker,
        class_filter=COCO_TRAFFIC_CLASSES,
        task_name="vehicle",
    )


def specialized_engine(weights: str, task_name: str, conf: float, iou: float) -> YoloEngine | None:
    if not weights:
        return None
    return YoloEngine(weights=weights, conf=conf, iou=iou, class_filter=None, task_name=task_name)
