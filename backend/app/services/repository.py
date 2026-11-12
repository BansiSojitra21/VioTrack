"""SQLite repository for videos, jobs, detections, and violations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from app.core.database import get_connection


def _row(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def insert_video(record: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO videos (
                id, original_name, stored_name, content_type, size_bytes, duration_sec,
                width, height, fps, storage, path, url, created_at
            ) VALUES (
                :id, :original_name, :stored_name, :content_type, :size_bytes, :duration_sec,
                :width, :height, :fps, :storage, :path, :url, :created_at
            )
            """,
            record,
        )


def list_videos() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_video(video_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _row(conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone())


def delete_video(video_id: str) -> None:
    with get_connection() as conn:
        job_ids = [row["id"] for row in conn.execute("SELECT id FROM jobs WHERE video_id = ?", (video_id,))]
        for job_id in job_ids:
            conn.execute("DELETE FROM detections WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM violations WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


def delete_job_results(job_id: str) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id FROM violations WHERE job_id = ?", (job_id,)).fetchall()
        conn.execute("DELETE FROM detections WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM violations WHERE job_id = ?", (job_id,))
    return [row["id"] for row in rows]


def insert_job(record: dict[str, Any]) -> None:
    record = {**record, "worker_pid": record.get("worker_pid")}
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, video_id, status, progress, frames_processed, frames_total, error,
                config_json, summary_json, annotated_path, worker_pid, created_at, started_at, finished_at
            ) VALUES (
                :id, :video_id, :status, :progress, :frames_processed, :frames_total, :error,
                :config_json, :summary_json, :annotated_path, :worker_pid, :created_at, :started_at, :finished_at
            )
            """,
            record,
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    with get_connection() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*fields.values(), job_id))


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _row(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def list_jobs(video_id: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if video_id:
            rows = conn.execute("SELECT * FROM jobs WHERE video_id = ? ORDER BY created_at DESC", (video_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def fail_interrupted_jobs() -> int:
    with get_connection() as conn:
        active_jobs = conn.execute(
            """
            SELECT id, worker_pid
            FROM jobs
            WHERE status IN ('pending', 'processing', 'cancelling')
            """
        ).fetchall()

    interrupted_ids = [
        row["id"]
        for row in active_jobs
        if row["worker_pid"] is None or not _process_is_running(row["worker_pid"])
    ]
    if not interrupted_ids:
        return 0

    error = (
        "Analysis was interrupted because the inference API restarted. "
        "Partial results are preserved; restart analysis to process the full video."
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.executemany(
            """
            UPDATE jobs
            SET status = 'failed', error = ?, finished_at = ?
            WHERE id = ? AND status IN ('pending', 'processing', 'cancelling')
            """,
            [(error, finished_at, job_id) for job_id in interrupted_ids],
        )
    return len(interrupted_ids)


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 87:
            return False
        raise
    return True


def insert_detections(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    records = [dict(row, plate_number=row.get("plate_number")) for row in rows]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO detections (
                id, job_id, video_id, frame_number, timestamp_sec, class_name, class_id,
                confidence, x1, y1, x2, y2, tracking_id, plate_number
            ) VALUES (
                :id, :job_id, :video_id, :frame_number, :timestamp_sec, :class_name, :class_id,
                :confidence, :x1, :y1, :x2, :y2, :tracking_id, :plate_number
            )
            """,
            records,
        )


def query_detections(
    job_id: str | None = None,
    video_id: str | None = None,
    from_sec: float | None = None,
    to_sec: float | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if job_id:
        clauses.append("job_id = ?")
        params.append(job_id)
    if video_id:
        clauses.append("video_id = ?")
        params.append(video_id)
    if from_sec is not None:
        clauses.append("timestamp_sec >= ?")
        params.append(from_sec)
    if to_sec is not None:
        clauses.append("timestamp_sec <= ?")
        params.append(to_sec)
    params.extend([limit, offset])
    sql = f"""
        SELECT * FROM detections
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp_sec, tracking_id
        LIMIT ? OFFSET ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def all_detections() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM detections").fetchall()
    return [dict(row) for row in rows]


def detection_metrics(job_id: str | None = None, latest_job_id: str | None = None) -> dict[str, Any]:
    """Aggregate dashboard vehicle metrics in SQLite without loading every row."""
    from app.detection.labels import VEHICLE_CLASSES

    classes = sorted(VEHICLE_CLASSES)
    class_marks = ", ".join("?" for _ in classes)
    scope = " AND job_id = ?" if job_id else ""
    scope_params = [job_id] if job_id else []
    with get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE class_name IN ({class_marks}){scope}",
            [*classes, *scope_params],
        ).fetchone()[0]
        grouped_tracks = conn.execute(
            f"""
            SELECT video_id, job_id, tracking_id, class_name, COUNT(*) AS hits
            FROM detections
            WHERE class_name IN ({class_marks}) AND tracking_id IS NOT NULL{scope}
            GROUP BY video_id, job_id, tracking_id, class_name
            """,
            [*classes, *scope_params],
        ).fetchall()
        best_class: dict[tuple[str, str, int], tuple[str, int]] = {}
        for row in grouped_tracks:
            identity = (row["video_id"], row["job_id"], row["tracking_id"])
            candidate = (row["class_name"], row["hits"])
            if identity not in best_class or candidate[1] > best_class[identity][1]:
                best_class[identity] = candidate
        vehicle_counts: dict[str, int] = {}
        for class_name, _hits in best_class.values():
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

        visible_now = 0
        if latest_job_id:
            visible_now = conn.execute(
                f"""
                SELECT COUNT(DISTINCT tracking_id) FROM detections
                WHERE job_id = ? AND frame_number = (
                    SELECT MAX(frame_number) FROM detections WHERE job_id = ?
                ) AND class_name IN ({class_marks}) AND tracking_id IS NOT NULL
                """,
                [latest_job_id, latest_job_id, *classes],
            ).fetchone()[0]
        by_minute = conn.execute(
            f"""
            SELECT minute, COUNT(*) AS vehicles FROM (
                SELECT CAST(timestamp_sec / 60 AS INTEGER) AS minute,
                       video_id, job_id, tracking_id
                FROM detections
                WHERE class_name IN ({class_marks}) AND tracking_id IS NOT NULL{scope}
                GROUP BY minute, video_id, job_id, tracking_id
            ) GROUP BY minute ORDER BY minute
            """,
            [*classes, *scope_params],
        ).fetchall()
        density = conn.execute(
            f"""
            SELECT CAST(timestamp_sec / 10 AS INTEGER) * 10 AS bucket, COUNT(*) AS detections
            FROM detections
            WHERE class_name IN ({class_marks}){scope}
            GROUP BY bucket ORDER BY bucket
            """,
            [*classes, *scope_params],
        ).fetchall()
    return {
        "vehicles_detected": len(best_class),
        "vehicles_visible_now": int(visible_now),
        "vehicle_detections_total": int(total),
        "vehicle_counts": vehicle_counts,
        "vehicles_by_minute": [
            {"minute": int(row["minute"]), "vehicles": int(row["vehicles"])} for row in by_minute
        ],
        "density": [
            {"t": int(row["bucket"]), "detections": int(row["detections"])} for row in density
        ],
    }


def plate_reanalysis_detections() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.job_id, d.video_id, d.frame_number, d.class_name, d.tracking_id,
                   d.x1, d.y1, d.x2, d.y2, v.path AS video_path
            FROM detections AS d
            JOIN jobs AS j ON j.id = d.job_id
            JOIN videos AS v ON v.id = d.video_id
            WHERE j.status = 'completed'
              AND d.class_name IN ('car', 'motorcycle', 'bus', 'truck', 'bicycle')
              AND d.tracking_id IS NOT NULL
            ORDER BY d.video_id, d.job_id, d.frame_number
            """
        ).fetchall()
    return [dict(row) for row in rows]


def update_detection_plates(plates: dict[str, str]) -> None:
    if not plates:
        return
    with get_connection() as conn:
        conn.executemany(
            "UPDATE detections SET plate_number = ? WHERE id = ?",
            [(plate_number, detection_id) for detection_id, plate_number in plates.items()],
        )


def update_violation_plates(
    plates: dict[tuple[str, int], str],
    frame_plates: dict[
        tuple[str, int],
        list[tuple[str, tuple[float, float, float, float], str]],
    ] | None = None,
) -> None:
    if not plates and not frame_plates:
        return
    frame_plates = frame_plates or {}
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, tracking_id, frame_number, vehicle_category, evidence_json
            FROM violations
            """
        ).fetchall()
        updates = []
        for row in rows:
            plate_number = (
                plates.get((row["job_id"], row["tracking_id"]))
                if row["tracking_id"] is not None
                else None
            )
            if plate_number is None:
                evidence = json.loads(row["evidence_json"])
                bbox = evidence.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                center_x = (float(bbox[0]) + float(bbox[2])) / 2
                center_y = (float(bbox[1]) + float(bbox[3])) / 2
                matches = []
                for class_name, vehicle_bbox, candidate_plate in frame_plates.get(
                    (row["job_id"], row["frame_number"]),
                    [],
                ):
                    if class_name != row["vehicle_category"]:
                        continue
                    x1, y1, x2, y2 = vehicle_bbox
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        matches.append((area, candidate_plate))
                if matches:
                    plate_number = min(matches, key=lambda match: match[0])[1]
            if plate_number is None:
                continue
            evidence = json.loads(row["evidence_json"])
            if evidence.get("plate_number") != plate_number:
                evidence["plate_number"] = plate_number
                updates.append((json.dumps(evidence), row["id"]))
        if updates:
            conn.executemany(
                "UPDATE violations SET evidence_json = ? WHERE id = ?",
                updates,
            )


def create_plate_reanalysis_task(task_id: str, created_at: str) -> tuple[dict[str, Any], bool]:
    with get_connection() as conn:
        active = conn.execute(
            """
            SELECT * FROM plate_reanalysis_tasks
            WHERE status IN ('pending', 'processing')
            LIMIT 1
            """
        ).fetchone()
        if active is not None:
            return dict(active), False
        conn.execute(
            """
            INSERT INTO plate_reanalysis_tasks (id, status, created_at)
            VALUES (?, 'pending', ?)
            """,
            (task_id, created_at),
        )
        row = conn.execute(
            "SELECT * FROM plate_reanalysis_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return dict(row), True


def update_plate_reanalysis_task(task_id: str, **fields: Any) -> None:
    allowed = {
        "status",
        "frames_processed",
        "frames_total",
        "detections_updated",
        "plates_found",
        "error",
        "finished_at",
    }
    if not fields:
        return
    unknown = fields.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported plate reanalysis task fields: {', '.join(sorted(unknown))}")
    columns = ", ".join(f"{key} = ?" for key in fields)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE plate_reanalysis_tasks SET {columns} WHERE id = ?",
            (*fields.values(), task_id),
        )


def get_plate_reanalysis_task(task_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM plate_reanalysis_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return _row(row)


def insert_violations(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    payload = []
    for row in rows:
        item = dict(row)
        item["evidence_json"] = json.dumps(item.pop("evidence"))
        payload.append(item)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO violations (
                id, job_id, video_id, tracking_id, vehicle_category, violation_type,
                timestamp_sec, frame_number, confidence, review_status, evidence_json, created_at
            ) VALUES (
                :id, :job_id, :video_id, :tracking_id, :vehicle_category, :violation_type,
                :timestamp_sec, :frame_number, :confidence, :review_status, :evidence_json, :created_at
            )
            """,
            payload,
        )


def list_violations(
    violation_type: str | None = None,
    review_status: str | None = None,
    video_id: str | None = None,
    min_confidence: float | None = None,
) -> list[dict[str, Any]]:
    clauses = ["violation_type != ?"]
    params: list[Any] = ["seatbelt"]
    if violation_type:
        clauses.append("violation_type = ?")
        params.append(violation_type)
    if review_status:
        clauses.append("review_status = ?")
        params.append(review_status)
    if video_id:
        clauses.append("video_id = ?")
        params.append(video_id)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    sql = f"SELECT * FROM violations WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json"))
        results.append(item)
    return results


def get_violation(violation_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM violations WHERE id = ? AND violation_type != ?",
            (violation_id, "seatbelt"),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["evidence"] = json.loads(item.pop("evidence_json"))
    return item


def update_violation_status(violation_id: str, review_status: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE violations SET review_status = ? WHERE id = ? AND violation_type != ?",
            (review_status, violation_id, "seatbelt"),
        )
    return get_violation(violation_id)
