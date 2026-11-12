from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.analytics.processor import summarize, traffic_series
from app.detection.types import Detection, clip_bbox
from app.detection.yolo_engine import YoloEngine
from app.tracking.bytetrack import ByteTrack
from app.workers.analysis import _frame_timestamp


def test_clip_bbox_clips_to_source_frame_and_rejects_invalid_boxes() -> None:
    assert clip_bbox((-10, 10, 110, 90), 100, 80) == (0.0, 10, 100.0, 80.0)
    assert clip_bbox((5, 5, 5, 20), 100, 80) is None
    assert clip_bbox((0, 0, math.inf, 10), 100, 80) is None


def test_frame_timestamp_uses_original_frame_index_and_source_fps() -> None:
    assert _frame_timestamp(100, 25) == 4.0
    assert _frame_timestamp(150, 25) == 6.0
    with pytest.raises(ValueError):
        _frame_timestamp(0, 0)


def test_tracker_keeps_id_after_a_short_missed_frame() -> None:
    tracker = ByteTrack(high_thresh=0.25, low_thresh=0.1, match_iou=0.3, max_age=10)
    first = Detection(2, "car", 0.9, (0, 0, 20, 20), 0, 0.0)
    second = Detection(2, "car", 0.9, (20, 0, 40, 20), 1, 0.04)
    recovered = Detection(2, "car", 0.9, (60, 0, 80, 20), 3, 0.12)

    first_track = tracker.update([first], 0, 0.0)[0].tracking_id
    tracker.update([second], 1, 0.04)
    assert tracker.update([], 2, 0.08) == []
    recovered_track = tracker.update([recovered], 3, 0.12)[0].tracking_id

    assert recovered_track == first_track
    assert len(tracker.all_tracks) == 1


def test_new_tracker_session_starts_with_clean_track_ids() -> None:
    detection = Detection(2, "car", 0.9, (0, 0, 20, 20))
    first = ByteTrack().update([detection], 0, 0.0)[0].tracking_id
    second = ByteTrack().update([detection], 0, 0.0)[0].tracking_id
    assert first == second == 1


def test_ultralytics_tracker_config_uses_the_job_buffer(tmp_path, monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    engine = YoloEngine("unused.pt", conf=0.25, iou=0.5, imgsz=960, track_buffer=37)

    tracker_config = Path(engine._tracker_config_path())

    assert tracker_config.is_file()
    assert "tracker_type: bytetrack" in tracker_config.read_text(encoding="utf-8")
    assert "track_buffer: 37" in tracker_config.read_text(encoding="utf-8")


def test_debug_artifacts_are_listed_and_path_traversal_is_rejected(tmp_path, monkeypatch) -> None:
    from fastapi import HTTPException

    from app.api import routes
    from app.core.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(routes.repository, "get_job", lambda job_id: {"id": job_id})
    artifact_dir = tmp_path / "debug" / "job-1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "frame_000000.json").write_text("{}", encoding="utf-8")

    response = routes.analysis_debug_artifacts("job-1")
    assert response["items"] == [
        {
            "name": "frame_000000.json",
            "url": "/api/analysis/job-1/debug/frame_000000.json",
        }
    ]
    assert Path(routes.analysis_debug_artifact("job-1", "frame_000000.json").path).is_file()
    with pytest.raises(HTTPException) as error:
        routes.analysis_debug_artifact("job-1", "../frame_000000.json")
    assert error.value.status_code == 404


def test_sqlite_detection_metrics_aggregate_tracks_and_latest_frame(tmp_path, monkeypatch) -> None:
    from app.core.config import settings
    from app.core.database import init_db
    from app.services import repository

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "metrics.sqlite"))
    init_db()
    for video_id, job_id, created_at in (
        ("video-a", "job-a", "2026-01-01"),
        ("video-b", "job-b", "2026-01-02"),
    ):
        repository.insert_video(
            {
                "id": video_id,
                "original_name": f"{video_id}.mp4",
                "stored_name": f"{video_id}.mp4",
                "content_type": "video/mp4",
                "size_bytes": 1,
                "duration_sec": 2,
                "width": 100,
                "height": 100,
                "fps": 25,
                "storage": "local",
                "path": str(tmp_path / f"{video_id}.mp4"),
                "url": None,
                "created_at": created_at,
            }
        )
        repository.insert_job(
            {
                "id": job_id,
                "video_id": video_id,
                "status": "completed",
                "progress": 1,
                "frames_processed": 2,
                "frames_total": 2,
                "error": None,
                "config_json": "{}",
                "summary_json": None,
                "annotated_path": None,
                "created_at": created_at,
                "started_at": created_at,
                "finished_at": created_at,
            }
        )
    rows = []
    for row_id, video_id, job_id, frame, timestamp, class_name, tracking_id in (
        ("d1", "video-a", "job-a", 0, 0.0, "car", 1),
        ("d2", "video-a", "job-a", 1, 0.04, "car", 1),
        ("d3", "video-b", "job-b", 0, 0.0, "truck", 1),
        ("d4", "video-b", "job-b", 1, 0.04, "pedestrian", 9),
        ("d5", "video-b", "job-b", 1, 0.04, "truck", 1),
    ):
        rows.append(
            {
                "id": row_id,
                "video_id": video_id,
                "job_id": job_id,
                "frame_number": frame,
                "timestamp_sec": timestamp,
                "class_name": class_name,
                "class_id": 2,
                "confidence": 0.9,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
                "tracking_id": tracking_id,
                "plate_number": None,
            }
        )
    repository.insert_detections(rows)

    metrics = repository.detection_metrics(latest_job_id="job-b")

    assert metrics["vehicles_detected"] == 2
    assert metrics["vehicles_visible_now"] == 1
    assert metrics["vehicle_detections_total"] == 4
    assert metrics["vehicle_counts"] == {"car": 1, "truck": 1}
    assert metrics["vehicles_by_minute"] == [{"minute": 0, "vehicles": 2}]


def test_summary_scopes_unique_tracks_to_video_and_job_and_separates_events() -> None:
    detections = [
        {"video_id": "video-a", "job_id": "job-a", "tracking_id": 1, "class_name": "car", "timestamp_sec": 1.0},
        {"video_id": "video-a", "job_id": "job-a", "tracking_id": 1, "class_name": "car", "timestamp_sec": 2.0},
        {"video_id": "video-b", "job_id": "job-b", "tracking_id": 1, "class_name": "truck", "timestamp_sec": 1.0},
        {"video_id": "video-b", "job_id": "job-b", "tracking_id": 2, "class_name": "pedestrian", "timestamp_sec": 2.0},
    ]
    violations = [
        {"violation_type": "red_light", "review_status": "pending"},
        {"violation_type": "wrong_lane", "review_status": "reviewed"},
        {"violation_type": "helmet", "review_status": "false_positive"},
    ]

    result = summarize(
        detections,
        violations,
        [],
        [
            {"id": "job-a", "created_at": "2026-01-01", "status": "completed"},
            {"id": "job-b", "created_at": "2026-01-02", "status": "completed"},
        ],
    )

    assert result["vehicles_detected"] == 2
    assert result["vehicles_visible_now"] == 1
    assert result["vehicle_detections_total"] == 3
    assert result["vehicle_counts"] == {"car": 1, "truck": 1}
    assert result["violations_total"] == 1
    assert result["violations_confirmed"] == 1


def test_traffic_series_counts_track_ids_with_video_and_job_scope() -> None:
    rows = [
        {"video_id": "video-a", "job_id": "job-a", "tracking_id": 1, "class_name": "car", "timestamp_sec": 2},
        {"video_id": "video-a", "job_id": "job-a", "tracking_id": 1, "class_name": "car", "timestamp_sec": 3},
        {"video_id": "video-b", "job_id": "job-b", "tracking_id": 1, "class_name": "car", "timestamp_sec": 4},
        {"video_id": "video-a", "job_id": "job-a", "tracking_id": 9, "class_name": "pedestrian", "timestamp_sec": 5},
    ]

    result = traffic_series(rows, [], [])

    assert result["vehicles_by_minute"] == [{"minute": 0, "vehicles": 2}]
