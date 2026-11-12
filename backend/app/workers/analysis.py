"""Background video analysis worker."""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.analytics.processor import summarize
from app.core.config import settings
from app.core.logging import get_logger
from app.detection.license_plate import read_vehicle_plates
from app.detection.labels import VEHICLE_CLASSES
from app.detection.types import Detection
from app.detection.yolo_engine import ModelUnavailableError, specialized_engine, vehicle_engine
from app.services import repository
from app.tracking.bytetrack import ByteTrack
from app.violations.engine import _signal_at, evaluate_helmet, evaluate_overspeed, evaluate_red_light, evaluate_wrong_lane
from app.violations.geometry import segments_intersect

logger = get_logger(__name__)
_threads: dict[str, threading.Thread] = {}
_cancel_events: dict[str, threading.Event] = {}
_worker_lock = threading.RLock()


class AnnotatedVideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        import imageio_ffmpeg

        self.path = path
        self._writer = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            codec="libx264",
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=fps,
            quality=None,
            output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"],
            ffmpeg_log_level="error",
        )
        try:
            self._writer.send(None)
        except Exception as exc:
            self._writer.close()
            self.path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not start H.264 encoding for annotated video: {exc}") from exc

    def write(self, frame: np.ndarray) -> None:
        try:
            self._writer.send(frame.tobytes())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise

    def release(self) -> None:
        try:
            self._writer.close()
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        if not self.path.is_file() or self.path.stat().st_size == 0:
            self.path.unlink(missing_ok=True)
            raise RuntimeError("H.264 encoding produced no annotated video.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue(job_id: str) -> None:
    thread = threading.Thread(target=run_job, args=(job_id,), name=f"analysis-{job_id}", daemon=True)
    with _worker_lock:
        _cancel_events[job_id] = threading.Event()
        _threads[job_id] = thread
    thread.start()


def cancel_analysis(job_id: str) -> bool:
    with _worker_lock:
        job = repository.get_job(job_id)
        if job is None or job["status"] not in {"pending", "processing", "cancelling"}:
            return False
        event = _cancel_events.get(job_id)
        if event is None:
            _clear_job_output(job_id, job.get("annotated_path"))
            repository.update_job(
                job_id,
                status="cancelled",
                error=None,
                finished_at=now(),
                annotated_path=None,
            )
            return True
        event.set()
        repository.update_job(job_id, status="cancelling")
        return True


def _cancel_requested(job_id: str) -> bool:
    event = _cancel_events.get(job_id)
    return event is not None and event.is_set()


def _clear_job_output(
    job_id: str,
    annotated_path: str | None = None,
    evidence_paths: list[Path] | None = None,
) -> None:
    violation_ids = repository.delete_job_results(job_id)
    evidence_dir = settings.resolved_data_dir / "evidence"
    for violation_id in violation_ids:
        (evidence_dir / f"{violation_id}.jpg").unlink(missing_ok=True)
    for path in evidence_paths or []:
        path.unlink(missing_ok=True)
    if annotated_path:
        Path(annotated_path).unlink(missing_ok=True)
    shutil.rmtree(settings.resolved_data_dir / "debug" / job_id, ignore_errors=True)


def _finish_cancelled(
    job_id: str,
    annotated_path: Path | None = None,
    evidence_paths: list[Path] | None = None,
) -> bool:
    with _worker_lock:
        if not _cancel_requested(job_id):
            return False
        job = repository.get_job(job_id)
        stored_path = job.get("annotated_path") if job else None
        _clear_job_output(
            job_id,
            str(annotated_path) if annotated_path else stored_path,
            evidence_paths,
        )
        if stored_path and stored_path != str(annotated_path):
            Path(stored_path).unlink(missing_ok=True)
        repository.update_job(
            job_id,
            status="cancelled",
            error=None,
            finished_at=now(),
            annotated_path=None,
        )
        return True


def _scale_point(point: list[float], width: int, height: int) -> list[float]:
    x, y = point
    if 0 <= x <= 1 and 0 <= y <= 1:
        return [x * width, y * height]
    return [x, y]


def _prepare_config(config: dict, width: int, height: int) -> dict:
    prepared = dict(config)
    if config.get("stop_line"):
        prepared["stop_line"] = [_scale_point(point, width, height) for point in config["stop_line"]]
    lanes = []
    for lane in config.get("lanes") or []:
        lanes.append(
            {
                **lane,
                "polygon": [_scale_point(point, width, height) for point in lane.get("polygon", [])],
            }
        )
    prepared["lanes"] = lanes
    return prepared


def _safe_model_name(value: str | None) -> str | None:
    if not value:
        return None
    name = Path(value).name
    if not name.endswith(".pt"):
        raise ModelUnavailableError("Model selection must be a .pt file in the weights directory.")
    candidate = settings.weights_dir / name
    if not candidate.is_file() and not settings.allow_weight_download:
        raise ModelUnavailableError(f"Selected model '{name}' is not in the weights directory.")
    return name


def _weights_on_disk(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if path.is_file():
        return True
    return (settings.weights_dir / value).is_file()


def _should_process_frame(frame_index: int, processed: int, fps: float, sample_fps: float, stride: int) -> bool:
    if sample_fps == fps / stride:
        return frame_index % stride == 0
    next_sample_frame = round(processed * fps / sample_fps)
    return frame_index >= next_sample_frame


def _frame_timestamp(frame_index: int, fps: float) -> float:
    if frame_index < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame index must be non-negative and FPS must be positive.")
    return frame_index / fps


def _associate_helmet_detections(
    detections: list[Detection],
    tracked_objects: list[tuple[int, str, tuple[float, float, float, float]]],
) -> list[Detection]:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        matching_tracks = [
            (
                0 if class_name in {"person", "pedestrian"} else 1,
                (vx2 - vx1) * (vy2 - vy1),
                tracking_id,
            )
            for tracking_id, class_name, (vx1, vy1, vx2, vy2) in tracked_objects
            if vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2
        ]
        if matching_tracks:
            detection.tracking_id = min(matching_tracks)[2]
        elif detection.tracking_id is not None:
            detection.tracking_id = -abs(detection.tracking_id)
    return detections


def run_job(job_id: str) -> None:
    job = repository.get_job(job_id)
    if job is None:
        with _worker_lock:
            _threads.pop(job_id, None)
            _cancel_events.pop(job_id, None)
        return
    try:
        with _worker_lock:
            if _cancel_requested(job_id):
                cancelled = True
            else:
                cancelled = False
                repository.update_job(
                    job_id,
                    status="processing",
                    started_at=now(),
                    progress=0.01,
                    worker_pid=os.getpid(),
                )
        if cancelled:
            _finish_cancelled(job_id)
            return
        video = repository.get_video(job["video_id"])
        if video is None:
            repository.update_job(job_id, status="failed", error="Video record is missing.", finished_at=now())
            return
        config = json.loads(job["config_json"])
        _analyze(job, video, config)
    except ModelUnavailableError as exc:
        if _finish_cancelled(job_id):
            return
        logger.error("job %s model unavailable: %s", job_id, exc)
        repository.update_job(job_id, status="failed", error=str(exc), finished_at=now(), progress=1)
    except Exception as exc:
        if _finish_cancelled(job_id):
            return
        logger.exception("job %s failed", job_id)
        repository.update_job(job_id, status="failed", error=str(exc), finished_at=now(), progress=1)
    finally:
        with _worker_lock:
            _threads.pop(job_id, None)
            _cancel_events.pop(job_id, None)


def _analyze(job: dict, video: dict, config: dict) -> None:
    import cv2

    job_id = job["id"]
    weights_name = _safe_model_name(config.get("model_name")) or settings.yolo_weights
    if not _weights_on_disk(weights_name) and not settings.allow_weight_download:
        raise ModelUnavailableError(
            f"YOLO weights '{weights_name}' are not on disk. "
            "Place the .pt file in backend/app/weights or set ALLOW_WEIGHT_DOWNLOAD=true for official Ultralytics weights. "
            "No detections were invented."
        )
    conf = float(config.get("conf", settings.yolo_conf))
    iou = float(config.get("iou", settings.yolo_iou))
    imgsz = int(config.get("imgsz", settings.yolo_imgsz))
    tracker_type = config.get("tracker_type", settings.yolo_tracker)
    if config.get("model_name"):
        from app.detection.labels import COCO_TRAFFIC_CLASSES
        from app.detection.yolo_engine import YoloEngine

        engine = YoloEngine(
            weights=weights_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            class_filter=COCO_TRAFFIC_CLASSES,
            task_name="vehicle",
            track_buffer=int(config.get("track_buffer", settings.track_buffer)),
            tracker_type=tracker_type,
        )
    else:
        engine = vehicle_engine(
            conf,
            iou,
            imgsz,
            int(config.get("track_buffer", settings.track_buffer)),
            tracker_type=tracker_type,
        )
    status = engine.load()
    if not status.loaded:
        raise ModelUnavailableError(status.error or "YOLO model failed to load.")
    # Reset internal tracker state so every video analysis starts with clean tracking IDs
    engine.reset_tracking()

    helmet_engine = None
    if config.get("enable_helmet", True):
        helmet_engine = specialized_engine(settings.helmet_weights, "helmet", conf, iou)
        if helmet_engine:
            helmet_status = helmet_engine.load()
            if not helmet_status.loaded:
                helmet_engine = None
                helmet_error = helmet_status.error
            else:
                helmet_error = None
        else:
            helmet_error = None
    else:
        helmet_error = "Helmet module disabled for this job."

    capture = cv2.VideoCapture(video["path"])
    if not capture.isOpened():
        raise RuntimeError("Could not open the stored video for analysis.")
    capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    metadata_fps = float(video.get("fps") or 0)
    fps = capture_fps if math.isfinite(capture_fps) and capture_fps > 0 else metadata_fps
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or video.get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or video.get("height") or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not determine the source video frame dimensions.")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(config.get("frame_stride", settings.frame_stride)))
    target_fps = config.get("target_fps")
    sample_fps = min(float(target_fps), fps) if target_fps is not None else fps / stride
    max_frames = max(1, int(config.get("max_frames", settings.max_frames)))
    planned_frames = (
        max(1, int(np.ceil(frame_count * sample_fps / fps)))
        if frame_count
        else max_frames
    )
    planned = min(max_frames, planned_frames)
    repository.update_job(job_id, frames_total=planned)

    writer = None
    annotated_path = settings.resolved_data_dir / "annotated" / f"{job_id}.mp4"
    if config.get("save_annotated", True):
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        writer = AnnotatedVideoWriter(
            annotated_path,
            max(sample_fps, 1),
            width,
            height,
        )

    track_buffer = int(config.get("track_buffer", settings.track_buffer))
    tracker = ByteTrack(
        high_thresh=conf,
        low_thresh=min(conf, 0.1),
        match_iou=0.3,
        max_age=max(track_buffer, stride * 3),
    )
    tracker.external_ids_enabled = callable(getattr(engine, "track", None))
    helmet_tracker = ByteTrack(
        high_thresh=conf,
        low_thresh=min(conf, 0.1),
        match_iou=0.1,
        max_age=max(track_buffer, stride * 3),
    )
    prepared = _prepare_config(config, width, height)
    stop_line = prepared.get("stop_line")
    signal_timeline = prepared.get("signal_timeline")
    violating_track_ids: set[int] = set()

    detection_rows: list[dict] = []
    plate_by_track: dict[int, tuple[str, float]] = {}
    helmet_detections = []
    debug_mode = bool(config.get("debug_mode", False))
    debug_dir = settings.resolved_data_dir / "debug" / job_id if debug_mode else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    frame_index = 0
    started = datetime.now(timezone.utc)
    processing_started = time.perf_counter()
    previous_invalid_boxes = getattr(engine, "invalid_bbox_count", 0)

    while processed < max_frames:
        if _cancel_requested(job_id):
            capture.release()
            if writer is not None:
                writer.release()
            _finish_cancelled(job_id, annotated_path)
            return
        ok, frame = capture.read()
        if not ok:
            break
        if not _should_process_frame(frame_index, processed, fps, sample_fps, stride):
            frame_index += 1
            continue
        timestamp = _frame_timestamp(frame_index, fps)
        debug_input = frame.copy() if debug_dir is not None and processed < 10 else None
        inference_started = time.perf_counter()
        detections = (
            engine.track(frame, frame_index, timestamp)
            if tracker.external_ids_enabled
            else engine.predict(frame, frame_index, timestamp)
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000
        tracked = tracker.update(detections, frame_index, timestamp)
        if processed == 0 or (processed + 1) % 10 == 0:
            active_ids = sorted(
                detection.tracking_id
                for detection in tracked
                if detection.tracking_id is not None
            )
            lost_ids = sorted(
                track.track_id for track in tracker.tracks if track.time_since_update > 0
            )
            current_invalid_boxes = getattr(engine, "invalid_bbox_count", previous_invalid_boxes)
            invalid_bbox_count = current_invalid_boxes - previous_invalid_boxes
            previous_invalid_boxes = current_invalid_boxes
            logger.info(
                "analysis frame processed",
                extra={
                    "context": {
                        "job_id": job_id,
                        "frame_index": frame_index,
                        "timestamp_sec": round(timestamp, 6),
                        "inference_ms": round(inference_ms, 2),
                        "detection_count": len(detections),
                        "active_track_ids": active_ids,
                        "lost_track_ids": lost_ids,
                        "processing_fps": round(
                            (processed + 1) / max(0.001, time.perf_counter() - processing_started), 2
                        ),
                        "queue_size": 0,
                        "invalid_bbox_count": invalid_bbox_count,
                    },
                },
            )

        # Check real-time red light violations across the stop line
        if stop_line and len(stop_line) == 2 and signal_timeline:
            line_seg = (tuple(stop_line[0]), tuple(stop_line[1]))
            for t in tracker.tracks:
                if t.track_id in violating_track_ids or len(t.history) < 2:
                    continue
                prev_pt, curr_pt = t.history[-2], t.history[-1]
                if _signal_at(signal_timeline, curr_pt.timestamp_sec) == "red":
                    if segments_intersect(prev_pt.centroid, curr_pt.centroid, line_seg[0], line_seg[1]):
                        violating_track_ids.add(t.track_id)

        tracked_vehicles = [
            (det.tracking_id, det.bbox)
            for det in tracked
            if det.tracking_id is not None and det.class_name in VEHICLE_CLASSES
        ]
        tracked_persons_and_motorcycles = [
            (det.tracking_id, det.class_name, det.bbox)
            for det in tracked
            if det.tracking_id is not None and det.class_name in {"person", "pedestrian", "motorcycle"}
        ]
        for tracking_id, plate in read_vehicle_plates(frame, tracked_vehicles).items():
            if tracking_id not in plate_by_track or plate[1] > plate_by_track[tracking_id][1]:
                plate_by_track[tracking_id] = plate
        for det in tracked:
            x1, y1, x2, y2 = det.bbox
            detection_rows.append(
                {
                    "id": uuid.uuid4().hex,
                    "job_id": job_id,
                    "video_id": video["id"],
                    "frame_number": frame_index,
                    "timestamp_sec": timestamp,
                    "class_name": det.class_name,
                    "class_id": det.class_id,
                    "confidence": det.confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "tracking_id": det.tracking_id,
                    "plate_number": None,
                }
            )
            if writer is not None and det.class_name in VEHICLE_CLASSES:
                # Green for compliant vehicles, Red for signal-breaking violators
                is_violator = det.tracking_id in violating_track_ids
                box_color = (40, 40, 235) if is_violator else (46, 204, 113)  # BGR
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3 if is_violator else 2)
                plate_number = (
                    plate_by_track.get(det.tracking_id, ("", 0.0))[0]
                    if det.tracking_id is not None
                    else ""
                )
                tag = f"{det.class_name.upper()} #{det.tracking_id} CONF {det.confidence:.0%}"
                if is_violator:
                    tag += " [RED LIGHT]"
                if plate_number:
                    tag += f" {plate_number}"
                cv2.putText(
                    frame,
                    tag,
                    (int(x1), max(16, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

        if writer is not None and stop_line and len(stop_line) == 2:
            current_sig = _signal_at(signal_timeline, timestamp) if signal_timeline else None
            sl_color = (40, 40, 235) if current_sig == "red" else (46, 204, 113)
            p1 = (int(stop_line[0][0]), int(stop_line[0][1]))
            p2 = (int(stop_line[1][0]), int(stop_line[1][1]))
            cv2.line(frame, p1, p2, sl_color, 2)
            cv2.putText(
                frame,
                f"STOP LINE ({current_sig.upper() if current_sig else 'GREEN'})",
                (p1[0] + 10, max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                sl_color,
                1,
                cv2.LINE_AA,
            )

        if helmet_engine is not None:
            frame_helmet_detections = helmet_engine.predict(frame, frame_index, timestamp)
            tracked_helmet_detections = helmet_tracker.update(
                frame_helmet_detections, frame_index, timestamp
            )
            helmet_detections.extend(
                _associate_helmet_detections(
                    tracked_helmet_detections, tracked_persons_and_motorcycles
                )
            )
        if debug_input is not None and debug_dir is not None:
            stem = f"frame_{frame_index:06d}"
            # 1. Original frame
            cv2.imwrite(str(debug_dir / f"{stem}_1_original.jpg"), debug_input)
            cv2.imwrite(str(debug_dir / f"{stem}_input.jpg"), debug_input)

            # 2. YOLO input frame (scaled to imgsz)
            yolo_input_preview = cv2.resize(debug_input, (imgsz, int(imgsz * height / max(1, width))))
            cv2.imwrite(str(debug_dir / f"{stem}_2_yolo_input.jpg"), yolo_input_preview)

            # 3. Detection output (raw detections before tracking)
            raw_output = debug_input.copy()
            for det in detections:
                if det.class_name not in VEHICLE_CLASSES:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in det.bbox]
                cv2.rectangle(raw_output, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
                cv2.putText(
                    raw_output,
                    f"{det.class_name} {det.confidence:.0%}",
                    (rx1, max(16, ry1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 165, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(debug_dir / f"{stem}_3_detections.jpg"), raw_output)

            # 4. Tracking output (active tracked vehicles with persistent IDs)
            tracked_output = debug_input.copy()
            for det in tracked:
                if det.class_name not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(value) for value in det.bbox]
                cv2.rectangle(tracked_output, (x1, y1), (x2, y2), (46, 204, 113), 2)
                label = f"{det.class_name} #{det.tracking_id} {det.confidence:.0%}"
                cv2.putText(
                    tracked_output,
                    label,
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (46, 204, 113),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(debug_dir / f"{stem}_4_tracked.jpg"), tracked_output)
            cv2.imwrite(str(debug_dir / f"{stem}_tracked.jpg"), tracked_output)

            # 5. Final rendered frame
            cv2.imwrite(str(debug_dir / f"{stem}_5_final_rendered.jpg"), frame)

            active_tids = [d.tracking_id for d in tracked if d.tracking_id is not None]
            (debug_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp,
                        "inference_ms": round(inference_ms, 2),
                        "source_width": int(frame.shape[1]),
                        "source_height": int(frame.shape[0]),
                        "active_track_ids": active_tids,
                        "raw_detections_count": len(detections),
                        "tracked_count": len(tracked),
                        "detections": [
                            {
                                "class_name": item.class_name,
                                "class_id": item.class_id,
                                "confidence": item.confidence,
                                "bbox_xyxy": list(item.bbox),
                                "track_id": item.tracking_id,
                            }
                            for item in tracked
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if writer is not None:
            writer.write(frame)
        processed += 1
        frame_index += 1
        if processed % 10 == 0 or processed == planned:
            if _cancel_requested(job_id):
                capture.release()
                if writer is not None:
                    writer.release()
                _finish_cancelled(job_id, annotated_path)
                return
            if detection_rows:
                for row in detection_rows:
                    if row["tracking_id"] in plate_by_track:
                        row["plate_number"] = plate_by_track[row["tracking_id"]][0]
                repository.insert_detections(detection_rows)
                detection_rows.clear()
            repository.update_job(
                job_id,
                frames_processed=processed,
                progress=min(0.95, processed / planned),
            )

    capture.release()
    if writer is not None:
        writer.release()
        repository.update_job(job_id, annotated_path=str(annotated_path))
    if _finish_cancelled(job_id, annotated_path):
        return
    if detection_rows:
        for row in detection_rows:
            if row["tracking_id"] in plate_by_track:
                row["plate_number"] = plate_by_track[row["tracking_id"]][0]
        repository.insert_detections(detection_rows)
        detection_rows.clear()

    modules = []
    if config.get("enable_red_light", True):
        modules.append(
            evaluate_red_light(
                tracker.all_tracks,
                prepared.get("stop_line"),
                prepared.get("signal_timeline"),
                conf,
            )
        )
    if config.get("enable_wrong_lane", True):
        modules.append(evaluate_wrong_lane(tracker.all_tracks, prepared.get("lanes"), conf))
    if config.get("enable_overspeed", True):
        modules.append(
            evaluate_overspeed(tracker.all_tracks, prepared.get("calibration"), conf)
        )
    helmet_report = evaluate_helmet(helmet_detections, helmet_engine is not None)
    if helmet_error and helmet_report.status == "not_configured":
        helmet_report.message = helmet_error or helmet_report.message
    if config.get("enable_helmet", True):
        modules.append(helmet_report)

    evidence_dir = settings.resolved_data_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    violation_rows = []
    evidence_paths: list[Path] = []
    created = now()
    capture = cv2.VideoCapture(video["path"])
    for module in modules:
        for event in module.events:
            if _cancel_requested(job_id):
                break
            violation_id = uuid.uuid4().hex
            evidence_file = evidence_dir / f"{violation_id}.jpg"
            evidence_paths.append(evidence_file)
            _write_evidence(capture, event.frame_number, event.bbox, evidence_file)
            plate_number = (
                plate_by_track.get(event.tracking_id, ("", 0.0))[0]
                if event.tracking_id is not None
                else ""
            )
            violation_rows.append(
                {
                    "id": violation_id,
                    "job_id": job_id,
                    "video_id": video["id"],
                    "tracking_id": event.tracking_id,
                    "vehicle_category": event.vehicle_category,
                    "violation_type": event.violation_type,
                    "timestamp_sec": event.timestamp_sec,
                    "frame_number": event.frame_number,
                    "confidence": event.confidence,
                    "review_status": "needs_review" if event.needs_review else "pending",
                    "created_at": created,
                    "evidence": {
                        "explanation": event.explanation,
                        "rule": event.rule,
                        "needs_review": event.needs_review,
                        "bbox": list(event.bbox),
                        "details": event.evidence,
                        "plate_number": plate_number,
                        "image": f"/api/evidence/{violation_id}",
                        "model": status.name,
                    },
                }
            )
    capture.release()
    if _finish_cancelled(job_id, annotated_path, evidence_paths):
        return
    repository.insert_violations(violation_rows)

    finished = datetime.now(timezone.utc)
    vehicle_metrics = repository.detection_metrics(job_id=job_id, latest_job_id=job_id)
    violations = repository.list_violations(video_id=video["id"])
    job_violations = [row for row in violations if row["job_id"] == job_id]
    summary = summarize(
        [],
        job_violations,
        [video],
        [{**job, "status": "completed"}],
        vehicle_metrics,
    )
    summary.update(
        {
            "processing_seconds": round((finished - started).total_seconds(), 2),
            "frames_processed": processed,
            "frame_stride": stride,
            "target_fps": target_fps,
            "model": {
                "name": status.name,
                "architecture": status.architecture,
                "weights": status.weights,
                "classes": status.classes,
                "conf": conf,
                "iou": iou,
                "input_size": status.input_size,
                "track_buffer": track_buffer,
            "debug_mode": debug_mode,
            },
            "modules": [
                {"name": module.name, "status": module.status, "message": module.message, "count": len(module.events)}
                for module in modules
            ],
            "annotated_video": bool(writer is None and annotated_path.exists()) or annotated_path.exists(),
            "sample": False,
        }
    )
    # writer variable may already be released; existence of file is the source of truth.
    summary["annotated_video"] = annotated_path.exists() and annotated_path.stat().st_size > 0
    with _worker_lock:
        if _cancel_requested(job_id):
            cancelled = True
        else:
            cancelled = False
            repository.update_job(
                job_id,
                status="completed",
                progress=1,
                frames_processed=processed,
                summary_json=json.dumps(summary),
                finished_at=finished.isoformat(),
                error=None,
            )
    if cancelled:
        _finish_cancelled(job_id, annotated_path)
        return
    logger.info("job %s completed frames=%s violations=%s", job_id, processed, len(violation_rows))


def _write_evidence(
    capture, frame_number: int, bbox: tuple[float, float, float, float], path: Path
) -> None:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    ok, frame = capture.read()
    if not ok:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 70, 255), 2)
    cv2.imwrite(str(path), frame)
