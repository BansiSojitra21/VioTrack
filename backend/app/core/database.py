"""SQLite metadata store.

SQLite keeps local development simple. On Render or Railway, point DATABASE_PATH
at a persistent disk. The file is the deployment-compatible store; no account
system is required.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings


def _remove_duplicate_helmet_violations(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT id, job_id, tracking_id, timestamp_sec, confidence, review_status, evidence_json
        FROM violations
        WHERE violation_type = 'helmet'
        ORDER BY job_id, timestamp_sec, frame_number
        """
    ).fetchall()
    clusters: list[list[tuple[sqlite3.Row, tuple[float, float, float, float]]]] = []
    for row in rows:
        try:
            bbox = tuple(float(value) for value in json.loads(row["evidence_json"])["bbox"])
            if len(bbox) != 4:
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        candidates: list[tuple[float, list[tuple[sqlite3.Row, tuple[float, float, float, float]]]]] = []
        for cluster in clusters:
            previous, previous_bbox = cluster[-1]
            time_delta = row["timestamp_sec"] - previous["timestamp_sec"]
            if previous["job_id"] != row["job_id"] or time_delta > 1:
                continue
            px1, py1, px2, py2 = previous_bbox
            previous_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            max_dimension = max(x2 - x1, y2 - y1, px2 - px1, py2 - py1)
            distance = math.dist(center, previous_center)
            same_track = row["tracking_id"] is not None and row["tracking_id"] == previous["tracking_id"]
            if same_track or distance <= max(80, max_dimension * 6):
                candidates.append((distance, cluster))
        if candidates:
            min(candidates, key=lambda candidate: candidate[0])[1].append((row, bbox))
        else:
            clusters.append([(row, bbox)])

    duplicate_ids = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        keep = min(
            (row for row, _ in cluster),
            key=lambda row: (
                0 if row["review_status"] in {"reviewed", "false_positive"} else 1,
                -row["confidence"],
                row["timestamp_sec"],
                row["id"],
            ),
        )
        duplicate_ids.extend(row["id"] for row, _ in cluster if row["id"] != keep["id"])
    if duplicate_ids:
        connection.executemany("DELETE FROM violations WHERE id = ?", [(item_id,) for item_id in duplicate_ids])


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    content_type TEXT,
    size_bytes INTEGER NOT NULL,
    duration_sec REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    storage TEXT NOT NULL,
    path TEXT,
    url TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    frames_processed INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    config_json TEXT NOT NULL,
    summary_json TEXT,
    annotated_path TEXT,
    worker_pid INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    class_name TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    tracking_id INTEGER,
    plate_number TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS violations (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    tracking_id INTEGER,
    vehicle_category TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    timestamp_sec REAL NOT NULL,
    frame_number INTEGER NOT NULL,
    confidence REAL NOT NULL,
    review_status TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS plate_reanalysis_tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    frames_processed INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER NOT NULL DEFAULT 0,
    detections_updated INTEGER NOT NULL DEFAULT 0,
    plates_found INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_job_time ON detections(job_id, timestamp_sec);
CREATE INDEX IF NOT EXISTS idx_violations_job ON violations(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_plate_reanalysis
ON plate_reanalysis_tasks ((1))
WHERE status IN ('pending', 'processing');
"""


def connect() -> sqlite3.Connection:
    settings.resolved_data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.resolved_database_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        detection_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(detections)").fetchall()
        }
        if "plate_number" not in detection_columns:
            connection.execute("ALTER TABLE detections ADD COLUMN plate_number TEXT")
        job_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "worker_pid" not in job_columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN worker_pid INTEGER")
        connection.execute(
            """
            UPDATE violations AS v
            SET tracking_id = (
                SELECT d.tracking_id
                FROM detections AS d
                WHERE d.job_id = v.job_id
                  AND d.frame_number = v.frame_number
                  AND d.class_name IN ('person', 'pedestrian', 'motorcycle')
                  AND d.tracking_id IS NOT NULL
                  AND json_valid(v.evidence_json)
                  AND (CAST(json_extract(v.evidence_json, '$.bbox[0]') AS REAL)
                       + CAST(json_extract(v.evidence_json, '$.bbox[2]') AS REAL)) / 2 BETWEEN d.x1 AND d.x2
                  AND (CAST(json_extract(v.evidence_json, '$.bbox[1]') AS REAL)
                       + CAST(json_extract(v.evidence_json, '$.bbox[3]') AS REAL)) / 2 BETWEEN d.y1 AND d.y2
                ORDER BY
                    CASE WHEN d.class_name IN ('person', 'pedestrian') THEN 0 ELSE 1 END,
                    d.confidence DESC
                LIMIT 1
            )
            WHERE v.violation_type = 'helmet'
              AND v.tracking_id IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM detections AS d
                  WHERE d.job_id = v.job_id
                    AND d.frame_number = v.frame_number
                    AND d.class_name IN ('person', 'pedestrian', 'motorcycle')
                    AND d.tracking_id IS NOT NULL
                    AND json_valid(v.evidence_json)
                    AND (CAST(json_extract(v.evidence_json, '$.bbox[0]') AS REAL)
                         + CAST(json_extract(v.evidence_json, '$.bbox[2]') AS REAL)) / 2 BETWEEN d.x1 AND d.x2
                    AND (CAST(json_extract(v.evidence_json, '$.bbox[1]') AS REAL)
                         + CAST(json_extract(v.evidence_json, '$.bbox[3]') AS REAL)) / 2 BETWEEN d.y1 AND d.y2
              )
            """
        )
        connection.execute(
            """
            UPDATE plate_reanalysis_tasks
            SET status = 'failed',
                error = 'Service restarted before plate reanalysis finished.',
                finished_at = CURRENT_TIMESTAMP
            WHERE status IN ('pending', 'processing')
            """
        )
        _remove_duplicate_helmet_violations(connection)
        connection.execute(
            """
            DELETE FROM violations
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY job_id, violation_type, tracking_id
                            ORDER BY
                                CASE WHEN review_status IN ('reviewed', 'false_positive') THEN 0 ELSE 1 END,
                                confidence DESC,
                                timestamp_sec,
                                id
                        ) AS duplicate_rank
                    FROM violations
                    WHERE tracking_id IS NOT NULL
                )
                WHERE duplicate_rank > 1
            )
            """
        )
        connection.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
