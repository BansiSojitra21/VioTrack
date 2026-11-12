"""Tracking, violation rules, analytics, and API behavior."""

from __future__ import annotations

import time
import sys
import os
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.analytics.processor import summarize, traffic_series
from app.api.routes import _suppress_duplicate_frame_detections
from app.core.database import get_connection, init_db
from app.detection.license_plate import select_vehicle_plates
from app.detection.types import Detection, Track, TrackPoint
from app.schemas.models import AnalysisConfig
from app.services import repository
from app.tracking.bytetrack import ByteTrack
from app.services.reports import sanitize_summary, violations_csv
from app.workers.analysis import AnnotatedVideoWriter, _should_process_frame
from app.violations.engine import (
    _signal_at,
    evaluate_helmet,
    evaluate_overspeed,
    evaluate_red_light,
    evaluate_wrong_lane,
)
from app.violations.geometry import segments_intersect
from app.violations.stop_line import detect_stop_line_candidate


def _track(track_id: int, points: list[tuple[int, float, tuple[float, float, float, float]]], class_name: str = "car") -> Track:
    track = Track(track_id=track_id, class_id=2, class_name=class_name)
    for frame, timestamp, bbox in points:
        track.history.append(TrackPoint(frame, timestamp, bbox, 0.9))
        track.hits += 1
    return track


def test_all_violation_checks_are_enabled_by_default() -> None:
    config = AnalysisConfig()

    assert config.enable_red_light is True
    assert config.enable_wrong_lane is True
    assert config.enable_overspeed is True
    assert config.enable_helmet is True


def test_bytetrack_starts_tracks_at_configured_low_confidence_threshold() -> None:
    tracker = ByteTrack(high_thresh=0.1, low_thresh=0.1)
    detections = [
        Detection(2, "car", 0.12, (10, 10, 30, 30), 0, 0.0),
        Detection(2, "car", 0.2, (50, 10, 70, 30), 0, 0.0),
    ]

    tracked = tracker.update(detections, 0, 0.0)

    assert len(tracked) == 2
    assert all(item.tracking_id is not None for item in tracked)


def test_live_frame_drops_near_identical_cross_class_boxes() -> None:
    detections = [
        Detection(2, "car", 0.52, (295, 111, 727, 382)),
        Detection(7, "truck", 0.48, (299, 112, 729, 384)),
        Detection(3, "motorcycle", 0.64, (740, 100, 800, 180)),
    ]

    filtered = _suppress_duplicate_frame_detections(detections)

    assert filtered == [detections[2], detections[0]]


def test_target_fps_sampling_averages_frames_at_requested_rate() -> None:
    sampled = []
    for frame_index in range(882):
        if _should_process_frame(frame_index, len(sampled), 29.97, 20, 1):
            sampled.append(frame_index)

    assert len(sampled) == 589
    assert sampled[0] == 0
    assert sampled[-1] == 881


def test_frame_stride_sampling_is_unchanged_without_target_fps() -> None:
    assert _should_process_frame(4, 2, 30, 15, 2)
    assert not _should_process_frame(3, 2, 30, 15, 2)


def test_segments_intersect() -> None:
    assert segments_intersect((50, 40), (50, 80), (0, 60), (100, 60))
    assert not segments_intersect((10, 10), (20, 10), (0, 60), (100, 60))


def test_bytetrack_keeps_id_and_uses_low_score_recovery() -> None:
    tracker = ByteTrack(high_thresh=0.5, low_thresh=0.1, match_iou=0.2, max_age=5)
    first = tracker.update(
        [Detection(2, "car", 0.92, (0, 0, 40, 40))],
        0,
        0.0,
    )
    assert first[0].tracking_id == 1
    second = tracker.update(
        [Detection(2, "car", 0.2, (4, 2, 44, 42))],
        1,
        0.1,
    )
    assert second[0].tracking_id == 1
    third = tracker.update(
        [Detection(2, "car", 0.95, (200, 200, 240, 240))],
        2,
        0.2,
    )
    assert [item.tracking_id for item in third] == [2]
    assert {track.track_id for track in tracker.tracks} == {1, 2}


def test_annotated_video_writer_creates_browser_playable_h264(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    path = tmp_path / "annotated.mp4"
    writer = AnnotatedVideoWriter(path, 24, 64, 64)
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).lower()
    ok, frame = capture.read()
    capture.release()

    assert codec in {"avc1", "h264"}
    assert ok
    assert frame is not None


def test_database_uses_wal_and_busy_timeout(client: TestClient) -> None:
    with get_connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_red_light_evaluates_tracks_after_they_expire_from_active_tracker() -> None:
    tracker = ByteTrack(high_thresh=0.5, low_thresh=0.1, match_iou=0.2, max_age=1)
    tracker.update([Detection(2, "car", 0.9, (40, 20, 80, 60))], 1, 0.2)
    tracker.update([Detection(2, "car", 0.9, (40, 40, 80, 80))], 2, 0.4)
    tracker.update([], 3, 0.6)
    tracker.update([], 4, 0.8)

    assert tracker.tracks == []
    report = evaluate_red_light(
        tracker.all_tracks,
        [[0, 50], [200, 50]],
        [{"start": 0, "end": 2, "state": "red"}],
        0.3,
    )
    assert [event.tracking_id for event in report.events] == [1]


def test_red_light_requires_red_state() -> None:
    track = _track(
        7,
        [
            (1, 0.2, (40, 20, 80, 60)),
            (2, 0.4, (40, 70, 80, 110)),
        ],
    )
    line = [[0, 65], [200, 65]]
    red = evaluate_red_light([track], line, [{"start": 0, "end": 2, "state": "red"}], 0.3)
    green = evaluate_red_light([track], line, [{"start": 0, "end": 2, "state": "green"}], 0.3)
    missing = evaluate_red_light([track], line, None, 0.3)
    assert red.status == "active"
    assert len(red.events) == 1
    assert red.events[0].tracking_id == 7
    assert green.events == []
    assert missing.status == "inactive"
    assert missing.events == []


def test_signal_timeline_uses_start_inclusive_end_exclusive_boundaries() -> None:
    timeline = [
        {"start": 0, "end": 10, "state": "red"},
        {"start": 10, "end": 25, "state": "green"},
        {"start": 25, "end": 35, "state": "red"},
        {"start": 35, "end": 50, "state": "green"},
    ]

    assert _signal_at(timeline, 0) == "red"
    assert _signal_at(timeline, 9.999) == "red"
    assert _signal_at(timeline, 10) == "green"
    assert _signal_at(timeline, 25) == "red"
    assert _signal_at(timeline, 50) == "green"


def test_wrong_lane_only_when_heading_opposes_lane() -> None:
    # Image y grows downward. Moving left is about 180 degrees.
    wrong = _track(
        3,
        [
            (1, 0.0, (180, 40, 220, 80)),
            (2, 0.2, (80, 40, 120, 80)),
        ],
    )
    aligned = _track(
        4,
        [
            (1, 0.0, (20, 40, 60, 80)),
            (2, 0.2, (120, 40, 160, 80)),
        ],
    )
    lanes = [{"id": "eastbound", "polygon": [[0, 0], [300, 0], [300, 200], [0, 200]], "direction_deg": 0}]
    report = evaluate_wrong_lane([wrong, aligned], lanes, 0.3, min_displacement_px=20)
    assert [event.tracking_id for event in report.events] == [3]
    idle = evaluate_wrong_lane([wrong], None, 0.3)
    assert idle.status == "inactive"


def test_overspeed_is_unavailable_without_validated_calibration() -> None:
    track = _track(
        9,
        [
            (0, 0.0, (0, 0, 20, 20)),
            (10, 1.0, (100, 0, 120, 20)),
        ],
    )
    missing = evaluate_overspeed([track], None, 0.3)
    assert missing.status == "unavailable"
    assert missing.events == []
    unvalidated = evaluate_overspeed([track], {"meters_per_pixel": 0.5, "validated": False}, 0.3)
    assert unvalidated.events == []
    # 100 px * 0.5 m = 50 m in 1 s = 180 km/h.
    flagged = evaluate_overspeed([track], {"meters_per_pixel": 0.5, "validated": True}, 0.3)
    assert len(flagged.events) == 1
    assert flagged.events[0].evidence["speed_kmh"] == pytest.approx(180.0, rel=0.01)


@pytest.mark.parametrize(
    ("class_name", "limit_kmh"),
    [("car", 70), ("bus", 60), ("truck", 60), ("motorcycle", 60)],
)
def test_overspeed_uses_default_vehicle_class_limits(class_name: str, limit_kmh: int) -> None:
    calibration = {"meters_per_pixel": 0.5, "validated": True}

    def track_at_speed(speed_kmh: float) -> Track:
        distance_px = speed_kmh / 1.8
        return _track(
            10,
            [(0, 0.0, (0, 0, 20, 20)), (1, 1.0, (distance_px, 0, distance_px + 20, 20))],
            class_name,
        )

    at_limit = evaluate_overspeed([track_at_speed(limit_kmh)], calibration, 0.3)
    assert at_limit.events == []

    above_limit = evaluate_overspeed([track_at_speed(limit_kmh + 1)], calibration, 0.3)
    assert len(above_limit.events) == 1
    assert above_limit.events[0].evidence["speed_limit_kmh"] == limit_kmh
    assert f"exceeds {limit_kmh} km/h" in above_limit.events[0].rule


def test_overspeed_ignores_vehicle_classes_without_a_default_limit() -> None:
    bicycle = _track(
        11,
        [(0, 0.0, (0, 0, 20, 20)), (1, 1.0, (100, 0, 120, 20))],
        "bicycle",
    )

    report = evaluate_overspeed([bicycle], {"meters_per_pixel": 0.5, "validated": True}, 0.3)

    assert report.events == []


def test_helmet_does_not_invent_labels() -> None:
    helmet = evaluate_helmet([], model_configured=False)
    assert helmet.status == "not_configured"
    assert helmet.events == []
    detected = evaluate_helmet(
        [Detection(1, "no_helmet", 0.42, (1, 1, 8, 8), frame_number=4, timestamp_sec=0.4)],
        model_configured=True,
    )
    assert len(detected.events) == 1
    assert detected.events[0].needs_review is True
    ignored = evaluate_helmet(
        [Detection(1, "car", 0.99, (1, 1, 8, 8))],
        model_configured=True,
    )
    assert ignored.events == []


def test_helmet_reports_one_best_detection_per_track() -> None:
    report = evaluate_helmet(
        [
            Detection(0, "no_helmet", 0.72, (20, 20, 40, 40), 1, 0.1, 7),
            Detection(0, "no_helmet", 0.94, (22, 21, 42, 41), 2, 0.2, 7),
            Detection(0, "no_helmet", 0.88, (80, 20, 100, 40), 2, 0.2, 8),
        ],
        model_configured=True,
    )

    assert len(report.events) == 2
    assert {event.tracking_id for event in report.events} == {7, 8}
    assert next(event for event in report.events if event.tracking_id == 7).confidence == 0.94


def test_helmet_detections_are_linked_to_person_or_motorcycle_tracks() -> None:
    from app.workers.analysis import _associate_helmet_detections

    tracked = _associate_helmet_detections(
        [
            Detection(0, "no_helmet", 0.9, (50, 20, 60, 30), tracking_id=3),
            Detection(0, "no_helmet", 0.8, (150, 20, 160, 30), tracking_id=4),
        ],
        [
            (12, "pedestrian", (10, 10, 100, 100)),
            (13, "motorcycle", (10, 10, 170, 110)),
        ],
    )

    assert [item.tracking_id for item in tracked] == [12, 13]


def test_plate_reader_selects_confident_plate_inside_violating_vehicle() -> None:
    def result(text: str, confidence: float, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
        x1, y1, x2, y2 = bbox
        return SimpleNamespace(
            ocr=SimpleNamespace(text=text, confidence=confidence),
            detection=SimpleNamespace(
                bounding_box=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)
            ),
        )

    plates = [
        result("outside-plate", 0.99, (210, 210, 260, 230)),
        result("ab 12 cd 3456", 0.82, (40, 70, 90, 90)),
        result("low-confidence", 0.3, (50, 100, 100, 120)),
    ]

    recognized = select_vehicle_plates(plates, [(7, (20, 40, 180, 160))])

    assert recognized == {7: ("AB12CD3456", 0.82)}


def test_plate_reader_matches_each_plate_to_its_vehicle() -> None:
    def result(text: str, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
        x1, y1, x2, y2 = bbox
        return SimpleNamespace(
            ocr=SimpleNamespace(text=text, confidence=0.9),
            detection=SimpleNamespace(
                bounding_box=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)
            ),
        )

    recognized = select_vehicle_plates(
        [
            result("CAR111", (30, 80, 65, 95)),
            result("TRK222", (230, 80, 265, 95)),
        ],
        [
            (1, (10, 30, 150, 150)),
            (2, (210, 30, 350, 150)),
        ],
    )

    assert recognized == {1: ("CAR111", 0.9), 2: ("TRK222", 0.9)}


def test_plate_reanalysis_backfills_existing_detections_and_violations(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.workers.plate_reanalysis as plate_reanalysis

    created = "2026-01-01T00:00:00+00:00"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    repository.insert_video(
        {
            "id": "backfill-video",
            "original_name": "clip.mp4",
            "stored_name": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": 5,
            "duration_sec": 1,
            "width": 640,
            "height": 480,
            "fps": 25,
            "storage": "local",
            "path": str(video_path),
            "url": None,
            "created_at": created,
        }
    )
    repository.insert_job(
        {
            "id": "backfill-job",
            "video_id": "backfill-video",
            "status": "completed",
            "progress": 1,
            "frames_processed": 2,
            "frames_total": 2,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "created_at": created,
            "started_at": created,
            "finished_at": created,
        }
    )
    repository.insert_detections(
        [
            {
                "id": f"backfill-detection-{frame}",
                "job_id": "backfill-job",
                "video_id": "backfill-video",
                "frame_number": frame,
                "timestamp_sec": frame / 25,
                "class_name": "car",
                "class_id": 2,
                "confidence": 0.9,
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "tracking_id": 7,
            }
            for frame in (0, 1)
        ]
    )
    repository.insert_violations(
        [
            {
                "id": "backfill-violation",
                "job_id": "backfill-job",
                "video_id": "backfill-video",
                "tracking_id": 7,
                "vehicle_category": "car",
                "violation_type": "red_light",
                "timestamp_sec": 0,
                "frame_number": 0,
                "confidence": 0.9,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {"plate_number": None},
            }
        ]
    )

    class Capture:
        def __init__(self) -> None:
            self.frame_number = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object | None]:
            if self.frame_number > 1:
                return False, None
            self.frame_number += 1
            return True, object()

        def release(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=lambda _: Capture()),
    )
    monkeypatch.setattr(plate_reanalysis, "license_plate_reader_available", lambda: True)
    monkeypatch.setattr(
        plate_reanalysis,
        "read_vehicle_plates",
        lambda _frame, vehicles, **_kwargs: {
            tracking_id: ("AB12CD3456", 0.9) for tracking_id, _ in vehicles
        },
    )
    task, created_task = repository.create_plate_reanalysis_task("backfill-task", created)
    assert created_task is True

    plate_reanalysis.run_plate_reanalysis(task["id"])

    result = repository.get_plate_reanalysis_task(task["id"])
    assert result is not None
    assert result["status"] == "completed"
    assert result["frames_processed"] == 2
    assert result["detections_updated"] == 2
    assert result["plates_found"] == 1
    detections = repository.query_detections(job_id="backfill-job")
    assert [row["plate_number"] for row in detections] == ["AB12CD3456", "AB12CD3456"]
    assert repository.get_violation("backfill-violation")["evidence"]["plate_number"] == "AB12CD3456"


def test_violations_csv_includes_plate_number() -> None:
    output = violations_csv(
        [
            {
                "id": "violation",
                "video_id": "video",
                "job_id": "job",
                "tracking_id": 7,
                "vehicle_category": "car",
                "violation_type": "red_light",
                "timestamp_sec": 1.25,
                "frame_number": 30,
                "confidence": 0.9,
                "review_status": "needs_review",
                "evidence": {"plate_number": "AB12CD3456"},
            }
        ]
    )

    assert "plate_number" in output.splitlines()[0]
    assert "AB12CD3456" in output.splitlines()[1]


def test_plate_number_is_stored_and_returned_with_violation(client: TestClient) -> None:
    created = "2026-01-01T00:00:00+00:00"
    repository.insert_video(
        {
            "id": "plate-video",
            "original_name": "clip.avi",
            "stored_name": "clip.avi",
            "content_type": "video/avi",
            "size_bytes": 1,
            "duration_sec": 1,
            "width": 1,
            "height": 1,
            "fps": 25,
            "storage": "local",
            "path": None,
            "url": None,
            "created_at": created,
        }
    )
    repository.insert_job(
        {
            "id": "plate-job",
            "video_id": "plate-video",
            "status": "completed",
            "progress": 1,
            "frames_processed": 1,
            "frames_total": 1,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "created_at": created,
            "started_at": created,
            "finished_at": created,
        }
    )
    repository.insert_violations(
        [
            {
                "id": "plate-violation",
                "job_id": "plate-job",
                "video_id": "plate-video",
                "tracking_id": 59,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 1,
                "frame_number": 25,
                "confidence": 0.9,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {
                    "plate_number": None,
                    "rule": "helmet",
                    "bbox": [40, 40, 60, 60],
                },
            },
            {
                "id": "plate-violation-second",
                "job_id": "plate-job",
                "video_id": "plate-video",
                "tracking_id": None,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 2,
                "frame_number": 50,
                "confidence": 0.9,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {
                    "plate_number": None,
                    "rule": "helmet",
                    "bbox": [140, 40, 160, 60],
                },
            },
        ]
    )
    repository.update_violation_plates(
        {},
        {
            ("plate-job", 25): [
                ("motorcycle", (10, 10, 100, 100), "AB12CD3456"),
            ],
            ("plate-job", 50): [
                ("motorcycle", (110, 10, 200, 100), "XY98ZZ7654"),
            ],
        },
    )

    response = client.get("/api/violations")
    assert response.status_code == 200
    evidence_by_id = {item["id"]: item["evidence"] for item in response.json()["items"]}
    assert evidence_by_id["plate-violation"]["plate_number"] == "AB12CD3456"
    assert evidence_by_id["plate-violation-second"]["plate_number"] == "XY98ZZ7654"
    csv_response = client.get("/api/reports/export/csv", params={"kind": "violations"})
    assert "AB12CD3456" in csv_response.text
    assert "XY98ZZ7654" in csv_response.text


def test_init_db_removes_existing_helmet_duplicates(client: TestClient) -> None:
    created = "2026-01-01T00:00:00+00:00"
    repository.insert_video(
        {
            "id": "duplicate-video",
            "original_name": "clip.avi",
            "stored_name": "clip.avi",
            "content_type": "video/avi",
            "size_bytes": 1,
            "duration_sec": 1,
            "width": 200,
            "height": 150,
            "fps": 25,
            "storage": "local",
            "path": None,
            "url": None,
            "created_at": created,
        }
    )
    repository.insert_job(
        {
            "id": "duplicate-job",
            "video_id": "duplicate-video",
            "status": "completed",
            "progress": 1,
            "frames_processed": 2,
            "frames_total": 2,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "created_at": created,
            "started_at": created,
            "finished_at": created,
        }
    )
    repository.insert_detections(
        [
            {
                "id": "motorcycle-detection",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "frame_number": 10,
                "timestamp_sec": 0.4,
                "class_name": "pedestrian",
                "class_id": 3,
                "confidence": 0.9,
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "tracking_id": 12,
            },
            {
                "id": "motorcycle-detection-later",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "frame_number": 20,
                "timestamp_sec": 0.8,
                "class_name": "pedestrian",
                "class_id": 3,
                "confidence": 0.91,
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "tracking_id": 12,
            },
        ]
    )
    repository.insert_violations(
        [
            {
                "id": "helmet-repeat-1",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "tracking_id": None,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 0.4,
                "frame_number": 10,
                "confidence": 0.72,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {"bbox": [40, 20, 55, 35]},
            },
            {
                "id": "helmet-repeat-2",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "tracking_id": None,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 0.8,
                "frame_number": 20,
                "confidence": 0.94,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {"bbox": [45, 22, 60, 37]},
            },
            {
                "id": "legacy-repeat-1",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "tracking_id": None,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 1.2,
                "frame_number": 30,
                "confidence": 0.61,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {"bbox": [350, 350, 365, 365]},
            },
            {
                "id": "legacy-repeat-2",
                "job_id": "duplicate-job",
                "video_id": "duplicate-video",
                "tracking_id": None,
                "vehicle_category": "motorcycle",
                "violation_type": "helmet",
                "timestamp_sec": 1.24,
                "frame_number": 31,
                "confidence": 0.81,
                "review_status": "needs_review",
                "created_at": created,
                "evidence": {"bbox": [352, 351, 367, 366]},
            },
        ]
    )

    init_db()

    rows = [row for row in repository.list_violations() if row["job_id"] == "duplicate-job"]
    assert len(rows) == 2
    assert {row["id"] for row in rows} == {"helmet-repeat-2", "legacy-repeat-2"}
    assert next(row for row in rows if row["id"] == "helmet-repeat-2")["tracking_id"] == 12


def test_analytics_summary_leaves_speed_empty_without_evidence() -> None:
    summary = summarize(
        [{"class_name": "car", "tracking_id": 1, "timestamp_sec": 1}],
        [{"violation_type": "red_light", "video_id": "v1", "evidence": {}}],
        [{"id": "v1"}],
        [{"status": "completed"}],
    )
    assert summary["sample"] is False
    assert summary["vehicles_detected"] == 1
    assert summary["red_light"] == 1
    assert summary["average_speed_kmh"] is None
    series = traffic_series([], [], [])
    assert len(series["hourly"]) == 24
    assert series["daily"] == []


def test_removed_violation_is_hidden_from_existing_data(client: TestClient) -> None:
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO violations (
                id, job_id, video_id, tracking_id, vehicle_category, violation_type,
                timestamp_sec, frame_number, confidence, review_status, evidence_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("legacy-seatbelt", "job", "video", None, "car", "seatbelt", 1, 25, 0.9, "pending", "{}", "2026-01-01"),
        )

    assert client.get("/api/violations").json()["items"] == []
    assert client.get("/api/violations", params={"violation_type": "seatbelt"}).json()["items"] == []
    assert client.post("/api/violations/legacy-seatbelt/review", json={"status": "reviewed"}).status_code == 404
    summary = sanitize_summary(
        {
            "violations_total": 2,
            "violation_counts": {"seatbelt": 1, "helmet": 1},
            "modules": [{"name": "seatbelt"}, {"name": "helmet"}],
        }
    )
    assert summary == {
        "violations_total": 1,
        "violation_counts": {"helmet": 1},
        "modules": [{"name": "helmet"}],
    }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.core.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.sqlite"))
    monkeypatch.setattr(settings, "yolo_weights", "missing-weights.pt")
    monkeypatch.setattr(settings, "helmet_weights", "")
    monkeypatch.setattr(settings, "allow_weight_download", False)
    monkeypatch.setattr(settings, "max_duration_sec", 30)
    from main import app

    with TestClient(app) as test_client:
        yield test_client


def test_model_info_does_not_invent_metrics_or_loaded_state(client: TestClient) -> None:
    response = client.get("/api/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample"] is False
    assert payload["loaded"] is False
    assert payload["weights_on_disk"] is False
    assert payload["metrics"] is None
    assert payload["error"]
    assert "seatbelt" not in payload
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False


def test_api_startup_marks_interrupted_jobs_failed_and_keeps_partial_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "restart.sqlite"))
    def fake_kill(pid: int, signal: int) -> None:
        if pid != os.getpid():
            raise ProcessLookupError

    monkeypatch.setattr(repository.os, "kill", fake_kill)
    from app.core.database import init_db

    init_db()
    repository.insert_video(
        {
            "id": "restart-video",
            "original_name": "traffic.mp4",
            "stored_name": "traffic.mp4",
            "content_type": "video/mp4",
            "size_bytes": 1,
            "duration_sec": 1,
            "width": 1,
            "height": 1,
            "fps": 25,
            "storage": "local",
            "path": str(tmp_path / "traffic.mp4"),
            "url": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    repository.insert_job(
        {
            "id": "restart-job",
            "video_id": "restart-video",
            "status": "processing",
            "progress": 0.1,
            "frames_processed": 1,
            "frames_total": 10,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "worker_pid": 999999,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
        }
    )
    repository.insert_detections(
        [
            {
                "id": "restart-detection",
                "job_id": "restart-job",
                "video_id": "restart-video",
                "frame_number": 0,
                "timestamp_sec": 0,
                "class_name": "car",
                "class_id": 2,
                "confidence": 0.9,
                "x1": 0,
                "y1": 0,
                "x2": 1,
                "y2": 1,
                "tracking_id": 1,
                "plate_number": None,
            }
        ]
    )

    from main import app

    with TestClient(app) as test_client:
        assert test_client.get("/api/health").status_code == 200

    job = repository.get_job("restart-job")
    assert job["status"] == "failed"
    assert "API restarted" in job["error"]
    assert job["finished_at"] is not None
    assert repository.query_detections(job_id="restart-job")

    repository.insert_job(
        {
            "id": "live-worker-job",
            "video_id": "restart-video",
            "status": "processing",
            "progress": 0.1,
            "frames_processed": 1,
            "frames_total": 10,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "worker_pid": os.getpid(),
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
        }
    )
    assert repository.fail_interrupted_jobs() == 0
    assert repository.get_job("live-worker-job")["status"] == "processing"


def test_active_analysis_cannot_delete_its_video(client: TestClient, tmp_path: Path) -> None:
    video_id = "active-video"
    repository.insert_video(
        {
            "id": video_id,
            "original_name": "traffic.mp4",
            "stored_name": "traffic.mp4",
            "content_type": "video/mp4",
            "size_bytes": 1,
            "duration_sec": 1,
            "width": 1,
            "height": 1,
            "fps": 25,
            "storage": "local",
            "path": str(tmp_path / "traffic.mp4"),
            "url": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    repository.insert_job(
        {
            "id": "active-job",
            "video_id": video_id,
            "status": "processing",
            "progress": 0.1,
            "frames_processed": 1,
            "frames_total": 10,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
        }
    )

    response = client.delete(f"/api/videos/{video_id}")

    assert response.status_code == 409
    assert repository.get_video(video_id) is not None
    assert repository.get_job("active-job")["status"] == "processing"


def test_cancel_active_analysis_keeps_video(client: TestClient, tmp_path: Path) -> None:
    video_id = "cancel-video"
    repository.insert_video(
        {
            "id": video_id,
            "original_name": "traffic.mp4",
            "stored_name": "traffic.mp4",
            "content_type": "video/mp4",
            "size_bytes": 1,
            "duration_sec": 1,
            "width": 1,
            "height": 1,
            "fps": 25,
            "storage": "local",
            "path": str(tmp_path / "traffic.mp4"),
            "url": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    repository.insert_job(
        {
            "id": "cancel-job",
            "video_id": video_id,
            "status": "processing",
            "progress": 0.1,
            "frames_processed": 1,
            "frames_total": 10,
            "error": None,
            "config_json": "{}",
            "summary_json": None,
            "annotated_path": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
        }
    )

    response = client.post("/api/analysis/cancel-job/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert repository.get_video(video_id) is not None
    assert repository.get_job("cancel-job")["status"] == "cancelled"


def test_model_info_does_not_report_existing_weights_as_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import settings
    from app.services.model_card import model_info

    weights = tmp_path / "model.pt"
    weights.touch()
    monkeypatch.setattr(settings, "yolo_weights", str(weights))

    payload = model_info()

    assert payload["weights_on_disk"] is True
    assert payload["loaded"] is False
    assert payload["error"] is None


def test_upload_rejects_non_video(client: TestClient) -> None:
    response = client.post(
        "/api/videos/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def _wait_job(client: TestClient, job_id: str) -> dict:
    payload = {}
    for _ in range(80):
        payload = client.get(f"/api/analysis/{job_id}/status").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.1)
    return payload


def _write_clip(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("codec")
    for index in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        y = 10 + index * 12
        cv2.rectangle(frame, (30, y), (70, y + 28), (40, 40, 220), -1)
        writer.write(frame)
    writer.release()


def test_stop_line_suggestion_endpoint_does_not_invent_a_line(client: TestClient, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    clip = tmp_path / "clip.avi"
    _write_clip(clip)
    upload = client.post("/api/videos/upload", files={"file": ("clip.avi", clip.read_bytes(), "video/avi")})
    assert upload.status_code == 200, upload.text

    response = client.post(f"/api/videos/{upload.json()['id']}/stop-line/suggest")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["found"] is False
    assert result["line"] is None
    assert result["preview_jpeg"]


def test_analysis_fails_explicitly_without_weights(client: TestClient, tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    assert cv2 is not None
    clip = tmp_path / "clip.avi"
    _write_clip(clip)
    upload = client.post("/api/videos/upload", files={"file": ("clip.avi", clip.read_bytes(), "video/avi")})
    assert upload.status_code == 200, upload.text
    video_id = upload.json()["id"]
    started = client.post("/api/analysis/start", json={"video_id": video_id, "config": {"save_annotated": False, "max_frames": 4}})
    assert started.status_code == 200, started.text
    job = _wait_job(client, started.json()["id"])
    assert job["status"] == "failed"
    assert "weights" in (job["error"] or "").lower()
    assert job["config"]["stop_line"] is None
    assert job["config"]["signal_timeline"] is None
    detections = client.get("/api/detections", params={"job_id": job["id"]})
    assert detections.json()["items"] == []


def test_stop_line_suggestion_detects_broad_white_marking() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.rectangle(frame, (40, 110), (280, 122), (245, 245, 245), -1)

    candidate = detect_stop_line_candidate(frame)

    assert candidate is not None
    line, _score = candidate
    assert line[0][1] == pytest.approx(line[1][1])
    assert line[0][0] == 0
    assert line[1][0] == 1
    assert 0 < line[0][1] < 1


def test_stop_line_suggestion_ignores_thin_lane_marking() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.line(frame, (40, 120), (280, 120), (245, 245, 245), 2)

    assert detect_stop_line_candidate(frame) is None


def test_stop_line_suggestion_ignores_short_and_vertical_markings() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    short_marking = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.rectangle(short_marking, (100, 110), (220, 122), (245, 245, 245), -1)
    vertical_marking = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.rectangle(vertical_marking, (150, 40), (162, 200), (245, 245, 245), -1)

    assert detect_stop_line_candidate(short_marking) is None
    assert detect_stop_line_candidate(vertical_marking) is None


def test_pipeline_flags_red_light_from_tracked_motion(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    from app.detection.yolo_engine import ModelStatus
    from app.workers import analysis as worker

    class FakeEngine:
        status = ModelStatus(
            loaded=True,
            name="unit-test",
            weights="unit-test.pt",
            architecture="YOLO11",
            task="vehicle",
            dataset="synthetic",
            classes=["car"],
        )

        def load(self) -> ModelStatus:
            return self.status

        def predict(self, frame, frame_number: int, timestamp_sec: float) -> list[Detection]:
            y = 10 + frame_number * 12
            return [Detection(2, "car", 0.93, (30, y, 70, y + 28), frame_number, timestamp_sec)]

    monkeypatch.setattr(worker, "_weights_on_disk", lambda value: True)
    monkeypatch.setattr(worker, "vehicle_engine", lambda conf, iou, imgsz=None, track_buffer=None: FakeEngine())
    plate_read_calls = 0

    def fake_read_vehicle_plates(frame, vehicles):
        nonlocal plate_read_calls
        plate_read_calls += 1
        plate = ("PLATELOW", 0.7) if plate_read_calls == 1 else ("PLATEBEST", 0.95)
        return {tracking_id: plate for tracking_id, _ in vehicles}

    monkeypatch.setattr(worker, "read_vehicle_plates", fake_read_vehicle_plates)
    clip = tmp_path / "crossing.avi"
    _write_clip(clip)
    upload = client.post("/api/videos/upload", files={"file": ("crossing.avi", clip.read_bytes(), "video/avi")})
    assert upload.status_code == 200, upload.text
    started = client.post(
        "/api/analysis/start",
        json={
            "video_id": upload.json()["id"],
            "config": {
                "save_annotated": False,
                "max_frames": 8,
                "enable_helmet": False,
                "enable_wrong_lane": False,
                "enable_overspeed": True,
                "stop_line": [[0, 0.55], [1, 0.55]],
                "signal_timeline": [{"start": 0, "end": 30, "state": "red"}],
            },
        },
    )
    assert started.status_code == 200, started.text
    job = _wait_job(client, started.json()["id"])
    assert job["status"] == "completed", job
    results = client.get(f"/api/analysis/{job['id']}/results")
    assert results.status_code == 200
    body = results.json()
    assert body["sample"] is False
    assert body["detections"]
    assert all(
        item["timestamp_sec"] == pytest.approx(item["frame_number"] / 10)
        for item in body["detections"]
    )
    assert plate_read_calls == job["frames_processed"]
    assert all(item["plate_number"] == "PLATEBEST" for item in body["detections"])
    assert any(item["violation_type"] == "red_light" for item in body["violations"])
    assert body["violations"][0]["evidence"]["plate_number"] == "PLATEBEST"
    assert all(item["violation_type"] != "overspeed" for item in body["violations"])
    modules = {item["name"]: item for item in job["summary"]["modules"]}
    assert modules["overspeed"]["status"] == "unavailable"
    assert "helmet" not in modules
    assert "seatbelt" not in modules
    assert "seatbelt" not in body["job"]["summary"]
    csv_response = client.get("/api/reports/export/csv", params={"job_id": job["id"]})
    assert csv_response.status_code == 200
    assert "red_light" in csv_response.text
    assert "PLATEBEST" in csv_response.text
    pdf_response = client.get("/api/reports/export/pdf", params={"job_id": job["id"]})
    assert pdf_response.status_code == 200
    assert pdf_response.content.startswith(b"%PDF")
