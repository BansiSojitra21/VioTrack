"""HTTP routes for the inference service."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from app.analytics.processor import most_common_violation, summarize, traffic_series
from app.core.config import settings
from app.detection.labels import VEHICLE_CLASSES
from app.detection.types import Detection, iou as bbox_iou
from app.schemas.models import ReviewUpdate, StartAnalysis
from app.services import repository, storage
from app.services.model_card import model_info
from app.services.reports import build_pdf, detections_csv, sanitize_summary, violations_csv
from app.tracking.bytetrack import ByteTrack
from app.utils.video import VideoRejected, probe_video, validate_upload
from app.violations.stop_line import suggest_stop_line
from app.workers.analysis import cancel_analysis, enqueue, now
from app.workers.plate_reanalysis import get_plate_reanalysis, start_plate_reanalysis

router = APIRouter()
_frame_engines: dict[tuple[float, float, int], Any] = {}
_playback_trackers: dict[str, ByteTrack] = {}


def _job_payload(job: dict | None) -> dict | None:
    if job is None:
        return None
    summary = sanitize_summary(json.loads(job["summary_json"])) if job.get("summary_json") else None
    config = json.loads(job["config_json"])
    config.pop("enable_seatbelt", None)
    return {
        "id": job["id"],
        "video_id": job["video_id"],
        "status": job["status"],
        "progress": job["progress"],
        "frames_processed": job["frames_processed"],
        "frames_total": job["frames_total"],
        "error": job["error"],
        "config": config,
        "summary": summary,
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "annotated_ready": bool(job.get("annotated_path") and Path(job["annotated_path"]).is_file()),
        "sample": False,
    }


def _video_payload(video: dict, jobs: list[dict]) -> dict:
    related = [job for job in jobs if job["video_id"] == video["id"]]
    latest = related[0] if related else None
    return {
        "id": video["id"],
        "original_name": video["original_name"],
        "content_type": video["content_type"],
        "size_bytes": video["size_bytes"],
        "duration_sec": video["duration_sec"],
        "width": video["width"],
        "height": video["height"],
        "fps": video["fps"],
        "storage": video["storage"],
        "url": video["url"] or f"/api/videos/{video['id']}/media",
        "media_path": f"/api/videos/{video['id']}/media",
        "created_at": video["created_at"],
        "status": latest["status"] if latest else "pending",
        "latest_job": _job_payload(latest),
        "sample": False,
    }


def _live_model_info() -> dict:
    info = model_info()
    engine = _frame_engines.get((settings.yolo_conf, settings.yolo_iou, settings.yolo_imgsz))
    if engine is not None:
        status = engine.status
        info["loaded"] = bool(status.loaded and info["weights_on_disk"])
        if status.error:
            info["error"] = status.error
    return info


@router.get("/health")
def health() -> dict:
    info = _live_model_info()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_loaded": info["loaded"],
        "weights_on_disk": info["weights_on_disk"],
        "model_error": info["error"],
        "storage": "sqlite",
        "database": str(settings.resolved_database_path),
        "max_upload_mb": settings.max_upload_mb,
        "max_duration_sec": settings.max_duration_sec,
    }


@router.get("/model/info")
def get_model_info() -> dict:
    return _live_model_info()


@router.post("/model/probe")
def probe_model() -> dict:
    info = model_info(probe=True)
    if not info["loaded"]:
        raise HTTPException(status_code=503, detail=info["error"] or "YOLO model is not available.")
    return info


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        suffix = validate_upload(file, len(data))
    except VideoRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    video_id = uuid.uuid4().hex
    destination = settings.resolved_data_dir / "uploads" / f"{video_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    try:
        meta = probe_video(destination)
    except VideoRejected as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    remote = storage.upload_file(destination, public_id=f"trafficai/{video_id}")
    record = {
        "id": video_id,
        "original_name": Path(file.filename or "video").name,
        "stored_name": destination.name,
        "content_type": file.content_type,
        "size_bytes": len(data),
        "duration_sec": meta["duration_sec"],
        "width": meta["width"],
        "height": meta["height"],
        "fps": meta["fps"],
        "storage": "cloudinary" if remote else "local",
        "path": str(destination),
        "url": remote,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    repository.insert_video(record)
    return _video_payload(record, [])


@router.get("/videos")
def get_videos() -> dict:
    videos = repository.list_videos()
    jobs = repository.list_jobs()
    return {"sample": False, "items": [_video_payload(video, jobs) for video in videos]}


@router.get("/videos/{video_id}")
def get_video(video_id: str) -> dict:
    video = repository.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    return _video_payload(video, repository.list_jobs(video_id))


@router.get("/videos/{video_id}/media")
def video_media(video_id: str) -> FileResponse:
    video = repository.get_video(video_id)
    if video is None or not video.get("path") or not Path(video["path"]).is_file():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(video["path"], filename=video["original_name"])


@router.post("/videos/{video_id}/stop-line/suggest")
def suggest_video_stop_line(video_id: str) -> dict:
    video = repository.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    if not video.get("path") or not Path(video["path"]).is_file():
        raise HTTPException(status_code=404, detail="Stored video file not found.")
    try:
        return suggest_stop_line(video["path"])
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    video = repository.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    jobs = repository.list_jobs(video_id)
    if any(job["status"] in {"pending", "processing", "cancelling"} for job in jobs):
        raise HTTPException(
            status_code=409,
            detail="Cancel the active analysis before deleting this video.",
        )
    repository.delete_video(video_id)
    if video.get("path"):
        Path(video["path"]).unlink(missing_ok=True)
    for job in jobs:
        if job.get("annotated_path"):
            Path(job["annotated_path"]).unlink(missing_ok=True)
    return {"deleted": video_id}


@router.post("/analysis/start")
def start_analysis(body: StartAnalysis) -> dict:
    video = repository.get_video(body.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    active = [
        job
        for job in repository.list_jobs(body.video_id)
        if job["status"] in {"pending", "processing", "cancelling"}
    ]
    if active:
        raise HTTPException(status_code=409, detail="This video already has an analysis job in progress.")
    config = body.config.model_dump()
    job_id = uuid.uuid4().hex
    record = {
        "id": job_id,
        "video_id": body.video_id,
        "status": "pending",
        "progress": 0,
        "frames_processed": 0,
        "frames_total": 0,
        "error": None,
        "config_json": json.dumps(config),
        "summary_json": None,
        "annotated_path": None,
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
    }
    repository.insert_job(record)
    enqueue(job_id)
    stored = repository.get_job(job_id)
    return _job_payload(stored)


@router.get("/analysis/{job_id}/status")
def analysis_status(job_id: str) -> dict:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    payload = _job_payload(job)
    video = repository.get_video(job["video_id"])
    if video:
        payload["video"] = _video_payload(video, [job])
    return payload


@router.post("/analysis/{job_id}/cancel")
def cancel_analysis_job(job_id: str) -> dict:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    if not cancel_analysis(job_id):
        raise HTTPException(status_code=409, detail=f"Analysis cannot be cancelled while it is {job['status']}.")
    cancelled_job = repository.get_job(job_id)
    return _job_payload(cancelled_job)


@router.get("/analysis/{job_id}/results")
def analysis_results(job_id: str) -> dict:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=job["error"] or f"Job is {job['status']}.")
    video = repository.get_video(job["video_id"])
    detections = repository.query_detections(job_id=job_id, limit=20000, offset=0)
    violations = [row for row in repository.list_violations() if row["job_id"] == job_id]
    return {
        "sample": False,
        "job": _job_payload(job),
        "video": _video_payload(video, [job]) if video else None,
        "detections": [_detection_payload(row) for row in detections],
        "violations": [_violation_payload(row) for row in violations],
    }


@router.get("/analysis/{job_id}/annotated")
def annotated_video(job_id: str) -> FileResponse:
    job = repository.get_job(job_id)
    if job is None or not job.get("annotated_path") or not Path(job["annotated_path"]).is_file():
        raise HTTPException(status_code=404, detail="Annotated video is not available for this job.")
    return FileResponse(job["annotated_path"], media_type="video/mp4", filename=f"{job_id}-annotated.mp4")


@router.get("/analysis/{job_id}/debug")
def analysis_debug_artifacts(job_id: str) -> dict:
    if repository.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    directory = settings.resolved_data_dir / "debug" / job_id
    if not directory.is_dir():
        return {"items": []}
    items = [
        {"name": path.name, "url": f"/api/analysis/{job_id}/debug/{path.name}"}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".jpg", ".json"}
    ]
    return {"items": items}


@router.get("/analysis/{job_id}/debug/{artifact_name}")
def analysis_debug_artifact(job_id: str, artifact_name: str) -> FileResponse:
    if repository.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    directory = (settings.resolved_data_dir / "debug" / job_id).resolve()
    artifact = (directory / artifact_name).resolve()
    if (
        Path(artifact_name).name != artifact_name
        or artifact.parent != directory
        or artifact.suffix not in {".jpg", ".json"}
        or not artifact.is_file()
    ):
        raise HTTPException(status_code=404, detail="Debug artifact not found.")
    media_type = "image/jpeg" if artifact.suffix == ".jpg" else "application/json"
    return FileResponse(artifact, media_type=media_type, filename=artifact.name)


@router.websocket("/analysis/{job_id}/ws")
async def analysis_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            job = repository.get_job(job_id)
            if job is None:
                await websocket.send_json({"error": "Analysis job not found."})
                break
            await websocket.send_json(_job_payload(job))
            if job["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        return


@router.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    conf: float = Form(default=settings.yolo_conf),
    iou: float = Form(default=settings.yolo_iou),
    session_id: str = Form(default=""),
    frame_number: int = Form(default=0),
    timestamp_sec: float = Form(default=0.0),
) -> dict:
    import numpy as np

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"OpenCV is not installed. {exc}") from exc
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image exceeds the 15 MB frame limit.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image. Use JPEG or PNG.")
    if not _weights_ready():
        info = model_info()
        raise HTTPException(status_code=503, detail=info["error"] or "YOLO weights are not configured.")
    from app.detection.yolo_engine import ModelUnavailableError

    # Reuse the model between playback-frame requests; loading weights on every
    # frame would add seconds of latency and make live overlays unusable.
    engine = _frame_engine(float(conf), float(iou))
    try:
        detections = engine.predict(image, 0, 0.0)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    detections = _suppress_duplicate_frame_detections(
        [det for det in detections if det.class_name in VEHICLE_CLASSES]
    )
    if session_id:
        tracker = _playback_trackers.get(session_id)
        if tracker is None:
            if len(_playback_trackers) >= 32:
                _playback_trackers.pop(next(iter(_playback_trackers)))
            tracker = ByteTrack(
                high_thresh=float(conf),
                low_thresh=min(float(conf), 0.1),
                match_iou=0.12,
                max_age=12,
            )
            _playback_trackers[session_id] = tracker
        detections = tracker.update(
            detections,
            max(0, int(frame_number)),
            max(0.0, float(timestamp_sec)),
        )
    return {
        "sample": False,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "items": [
            {
                "class_name": det.class_name,
                "class_id": det.class_id,
                "confidence": det.confidence,
                "bbox": list(det.bbox),
                "frame_number": max(0, int(frame_number)),
                "frame_index": max(0, int(frame_number)),
                "timestamp_sec": max(0.0, float(timestamp_sec)),
                "timestamp": max(0.0, float(timestamp_sec)),
                "tracking_id": det.tracking_id,
                "track_id": det.tracking_id,
                "sample": False,
            }
            for det in detections
        ],
    }


def _suppress_duplicate_frame_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            detection.class_name != previous.class_name
            and bbox_iou(detection.bbox, previous.bbox) >= 0.85
            for previous in kept
        ):
            continue
        kept.append(detection)
    return kept


def _weights_ready() -> bool:
    from pathlib import Path

    value = settings.yolo_weights
    return Path(value).is_file() or (settings.weights_dir / value).is_file() or settings.allow_weight_download


def _frame_engine(conf: float, iou: float) -> Any:
    from app.detection.yolo_engine import vehicle_engine

    engine_key = (conf, iou, settings.yolo_imgsz)
    engine = _frame_engines.get(engine_key)
    if engine is None:
        _frame_engines.clear()
        engine = vehicle_engine(conf, iou, settings.yolo_imgsz)
        _frame_engines[engine_key] = engine
    return engine


def preload_frame_engine() -> str | None:
    if not _weights_ready():
        return None
    engine = _frame_engine(settings.yolo_conf, settings.yolo_iou)
    status = engine.load()
    if not status.loaded:
        return status.error or "The live vehicle model could not be loaded."
    import numpy as np

    try:
        engine.predict(
            np.zeros((settings.yolo_imgsz, settings.yolo_imgsz, 3), dtype=np.uint8),
            0,
            0.0,
        )
    except RuntimeError as exc:
        return f"The live vehicle model warm-up failed: {exc}"
    return None


@router.get("/detections")
def get_detections(
    job_id: str | None = None,
    video_id: str | None = None,
    from_sec: float | None = None,
    to_sec: float | None = None,
    limit: int = Query(default=500, ge=1, le=20000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    rows = repository.query_detections(job_id, video_id, from_sec, to_sec, limit, offset)
    return {"sample": False, "items": [_detection_payload(row) for row in rows]}


@router.post("/detections/plates/reanalyze")
def reanalyze_detection_plates() -> dict:
    active_jobs = [
        job
        for job in repository.list_jobs()
        if job["status"] in {"pending", "processing", "cancelling"}
    ]
    if active_jobs:
        raise HTTPException(
            status_code=409,
            detail="Wait for active video analysis jobs to finish before re-reading plates.",
        )
    return start_plate_reanalysis()


@router.get("/detections/plates/reanalyze/{task_id}")
def plate_reanalysis_status(task_id: str) -> dict:
    task = get_plate_reanalysis(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Plate reanalysis task not found.")
    return task


@router.get("/violations")
def get_violations(
    violation_type: str | None = None,
    review_status: str | None = None,
    video_id: str | None = None,
    min_confidence: float | None = Query(default=None, ge=0, le=1),
) -> dict:
    rows = repository.list_violations(violation_type, review_status, video_id, min_confidence)
    videos = {video["id"]: video for video in repository.list_videos()}
    return {"sample": False, "items": [_violation_payload(row, videos.get(row["video_id"])) for row in rows]}


@router.post("/violations/{violation_id}/review")
def review_violation(violation_id: str, body: ReviewUpdate) -> dict:
    existing = repository.get_violation(violation_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Violation not found.")
    updated = repository.update_violation_status(violation_id, body.status)
    video = repository.get_video(updated["video_id"]) if updated else None
    return _violation_payload(updated, video)


@router.get("/evidence/{violation_id}")
def evidence_image(violation_id: str) -> FileResponse:
    if not violation_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid evidence id.")
    path = settings.resolved_data_dir / "evidence" / f"{violation_id}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence frame was not saved for this record.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/analytics/summary")
def analytics_summary() -> dict:
    violations = repository.list_violations()
    videos = repository.list_videos()
    jobs = repository.list_jobs()
    latest_job_id = jobs[0]["id"] if jobs else None
    summary = summarize(
        [],
        violations,
        videos,
        jobs,
        repository.detection_metrics(latest_job_id=latest_job_id),
    )
    summary["most_common_violation"] = most_common_violation(summary["violation_counts"])
    reviewed = [row for row in violations if row["review_status"] in {"reviewed", "false_positive"}]
    confirmed = [row for row in reviewed if row["review_status"] == "reviewed"]
    summary["detection_accuracy"] = round(len(confirmed) / len(reviewed), 4) if reviewed else None
    if summary["detection_accuracy"] is not None:
        summary["detection_accuracy_note"] = "Share of reviewed records that were kept (not marked false positive)."
    return summary


@router.get("/analytics/traffic")
def analytics_traffic() -> dict:
    return traffic_series(
        [],
        repository.list_violations(),
        repository.list_videos(),
        repository.detection_metrics(),
    )


@router.get("/reports/history")
def report_history() -> dict:
    jobs = repository.list_jobs()
    videos = {video["id"]: video for video in repository.list_videos()}
    items = []
    for job in jobs:
        if job["status"] != "completed":
            continue
        video = videos.get(job["video_id"])
        items.append(
            {
                "job_id": job["id"],
                "video_id": job["video_id"],
                "video_name": video["original_name"] if video else job["video_id"],
                "finished_at": job["finished_at"],
                "summary": sanitize_summary(json.loads(job["summary_json"])) if job.get("summary_json") else None,
            }
        )
    return {"sample": False, "items": items}


@router.get("/reports/export/csv")
def export_csv(kind: str = Query(default="violations"), job_id: str | None = None) -> Response:
    if kind == "detections":
        content = detections_csv(job_id)
        filename = "detections.csv"
    elif kind == "violations":
        rows = repository.list_violations()
        if job_id:
            rows = [row for row in rows if row["job_id"] == job_id]
        content = violations_csv(rows)
        filename = "violations.csv"
    else:
        raise HTTPException(status_code=400, detail="kind must be violations or detections.")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reports/export/pdf")
def export_pdf(job_id: str | None = None) -> Response:
    if job_id and repository.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    payload = build_pdf(job_id)
    name = f"{job_id or 'trafficai'}-report.pdf"
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def _detection_payload(row: dict) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "video_id": row["video_id"],
        "frame_number": row["frame_number"],
        "frame_index": row["frame_number"],
        "timestamp_sec": row["timestamp_sec"],
        "timestamp": row["timestamp_sec"],
        "class_name": row["class_name"],
        "class_id": row["class_id"],
        "confidence": row["confidence"],
        "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
        "tracking_id": row["tracking_id"],
        "track_id": row["tracking_id"],
        "plate_number": row.get("plate_number"),
        "sample": False,
    }


def _violation_payload(row: dict, video: dict | None = None) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "video_id": row["video_id"],
        "video_name": video["original_name"] if video else row["video_id"],
        "tracking_id": row["tracking_id"],
        "vehicle_category": row["vehicle_category"],
        "violation_type": row["violation_type"],
        "timestamp_sec": row["timestamp_sec"],
        "frame_number": row["frame_number"],
        "confidence": row["confidence"],
        "review_status": row["review_status"],
        "evidence": row["evidence"],
        "created_at": row["created_at"],
        "sample": False,
    }
