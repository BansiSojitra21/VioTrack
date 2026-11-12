"""Backfill plate numbers on detections from completed video analyses."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.detection.labels import VEHICLE_CLASSES
from app.detection.license_plate import license_plate_reader_available, read_vehicle_plates
from app.services import repository

logger = logging.getLogger(__name__)
_start_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "status": task["status"],
        "frames_processed": task["frames_processed"],
        "frames_total": task["frames_total"],
        "detections_updated": task["detections_updated"],
        "plates_found": task["plates_found"],
        "error": task["error"],
        "created_at": task["created_at"],
        "finished_at": task["finished_at"],
    }


def start_plate_reanalysis() -> dict[str, Any]:
    with _start_lock:
        task, created = repository.create_plate_reanalysis_task(uuid.uuid4().hex, _now())
        if created:
            threading.Thread(
                target=run_plate_reanalysis,
                args=(task["id"],),
                name=f"plate-reanalysis-{task['id']}",
                daemon=True,
            ).start()
    return _task_payload(task)


def get_plate_reanalysis(task_id: str) -> dict[str, Any] | None:
    task = repository.get_plate_reanalysis_task(task_id)
    return _task_payload(task) if task else None


def run_plate_reanalysis(task_id: str) -> None:
    try:
        repository.update_plate_reanalysis_task(task_id, status="processing", error=None)
        if not license_plate_reader_available():
            raise RuntimeError(
                "Local plate recognition is unavailable. Install the backend requirements, "
                "including fast-alpr[onnx], and retry."
            )

        detections = repository.plate_reanalysis_detections()
        by_video: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        video_paths: dict[str, str] = {}
        for detection in detections:
            video_id = detection["video_id"]
            video_paths[video_id] = detection["video_path"]
            by_video[video_id][detection["frame_number"]][detection["job_id"]].append(detection)

        total_frames = sum(len(frames) for frames in by_video.values())
        repository.update_plate_reanalysis_task(task_id, frames_total=total_frames)
        if not total_frames:
            repository.update_plate_reanalysis_task(
                task_id,
                status="completed",
                error=None,
                finished_at=_now(),
            )
            return

        import cv2

        best_by_track: dict[tuple[str, int], tuple[str, float]] = {}
        frames_processed = 0
        for video_id, frames in by_video.items():
            video_path = video_paths[video_id]
            if not video_path or not Path(video_path).is_file():
                raise RuntimeError(f"Stored video file is missing for video {video_id}.")
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open stored video {video_id} for plate recognition.")
            try:
                last_frame = max(frames)
                frame_number = 0
                while frame_number <= last_frame:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Could not read frame {frame_number} from stored video {video_id}."
                        )
                    frame_jobs = frames.get(frame_number)
                    if frame_jobs:
                        for job_id, frame_detections in frame_jobs.items():
                            vehicles = [
                                (
                                    row["tracking_id"],
                                    (row["x1"], row["y1"], row["x2"], row["y2"]),
                                )
                                for row in frame_detections
                                if row["class_name"] in VEHICLE_CLASSES
                            ]
                            recognized = read_vehicle_plates(frame, vehicles, raise_on_error=True)
                            for tracking_id, plate in recognized.items():
                                key = (job_id, tracking_id)
                                if key not in best_by_track or plate[1] > best_by_track[key][1]:
                                    best_by_track[key] = plate
                        frames_processed += 1
                        if frames_processed % 10 == 0:
                            repository.update_plate_reanalysis_task(
                                task_id,
                                frames_processed=frames_processed,
                            )
                    frame_number += 1
            finally:
                capture.release()

        track_plates = {
            key: plate[0]
            for key, plate in best_by_track.items()
        }
        detection_plates = {
            row["id"]: track_plates[(row["job_id"], row["tracking_id"])]
            for video_frames in by_video.values()
            for frame_jobs in video_frames.values()
            for job_id, frame_detections in frame_jobs.items()
            for row in frame_detections
            if (job_id, row["tracking_id"]) in track_plates
        }
        frame_plates: dict[
            tuple[str, int],
            list[tuple[str, tuple[float, float, float, float], str]],
        ] = defaultdict(list)
        for video_frames in by_video.values():
            for frame_number, frame_jobs in video_frames.items():
                for job_id, frame_detections in frame_jobs.items():
                    for row in frame_detections:
                        plate_number = track_plates.get((job_id, row["tracking_id"]))
                        if plate_number is None:
                            continue
                        frame_plates[(job_id, frame_number)].append(
                            (
                                row["class_name"],
                                (row["x1"], row["y1"], row["x2"], row["y2"]),
                                plate_number,
                            )
                        )
        repository.update_detection_plates(detection_plates)
        repository.update_violation_plates(track_plates, frame_plates)
        repository.update_plate_reanalysis_task(
            task_id,
            status="completed",
            frames_processed=frames_processed,
            detections_updated=len(detection_plates),
            plates_found=len(track_plates),
            error=None,
            finished_at=_now(),
        )
    except Exception as exc:
        logger.exception("Plate reanalysis task %s failed", task_id)
        repository.update_plate_reanalysis_task(
            task_id,
            status="failed",
            error=str(exc),
            finished_at=_now(),
        )
