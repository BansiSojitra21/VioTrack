"""Pandas aggregations for dashboard analytics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from app.detection.labels import VEHICLE_CLASSES


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize(
    detections: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    vehicle_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    det = _frame(detections)
    vio = _frame(violations)
    completed = [job for job in jobs if job.get("status") == "completed"]

    vehicle_counts: dict[str, int] = {}
    unique_tracks = 0
    total_vehicle_detections = 0
    visible_now = 0
    if not det.empty:
        vehicles = det[det["class_name"].isin(VEHICLE_CLASSES)].copy()
        total_vehicle_detections = int(len(vehicles))
        identity_columns = [column for column in ("video_id", "job_id", "tracking_id") if column in vehicles]
        tracked = vehicles[vehicles["tracking_id"].notna()] if "tracking_id" in vehicles else vehicles.iloc[0:0]
        if identity_columns and not tracked.empty:
            identities = tracked.drop_duplicates(identity_columns)
            unique_tracks = int(len(identities))
            vehicle_counts = {
                str(name): int(count)
                for name, count in identities.groupby("class_name").size().items()
            }
            ordered_jobs = sorted(
                (item for item in jobs if item.get("id")),
                key=lambda item: item.get("created_at") or "",
            )
            if ordered_jobs and "job_id" in vehicles:
                latest = vehicles[vehicles["job_id"] == ordered_jobs[-1]["id"]]
                if not latest.empty:
                    latest_time = latest["timestamp_sec"].max()
                    frame = latest[latest["timestamp_sec"] == latest_time]
                    if not frame.empty and "tracking_id" in frame:
                        visible_now = int(frame["tracking_id"].dropna().nunique())

    violation_counts: dict[str, int] = {}
    if not vio.empty:
        if "review_status" in vio:
            candidates = vio[~vio["review_status"].isin(["reviewed", "false_positive"])]
            confirmed_violations = int((vio["review_status"] == "reviewed").sum())
        else:
            candidates = vio
            confirmed_violations = 0
        violation_counts = {str(k): int(v) for k, v in candidates["violation_type"].value_counts().to_dict().items()}
    else:
        candidates = vio
        confirmed_violations = 0

    speeds = []
    for row in violations:
        evidence = row.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("speed_kmh") is not None:
            speeds.append(float(evidence["speed_kmh"]))
    avg_speed = float(pd.Series(speeds).mean()) if speeds else None

    result = {
        "sample": False,
        "videos_analyzed": len(completed),
        "videos_uploaded": len(videos),
        "vehicles_detected": unique_tracks,
        "vehicles_visible_now": visible_now,
        "vehicle_detections_total": total_vehicle_detections,
        "vehicle_counts": vehicle_counts,
        "violations_total": int(len(candidates)),
        "violations_confirmed": confirmed_violations,
        "violation_counts": violation_counts,
        "red_light": int(violation_counts.get("red_light", 0)),
        "wrong_lane": int(violation_counts.get("wrong_lane", 0)),
        "helmet": int(violation_counts.get("helmet", 0)),
        "overspeed": int(violation_counts.get("overspeed", 0)),
        "average_speed_kmh": round(avg_speed, 2) if avg_speed is not None else None,
        "average_speed_status": "calibrated" if avg_speed is not None else "unavailable",
        "detection_accuracy": None,
        "detection_accuracy_note": "Accuracy is shown only when an evaluation file or ground-truth review set is provided.",
    }
    if vehicle_metrics is not None:
        for key in (
            "vehicles_detected",
            "vehicles_visible_now",
            "vehicle_detections_total",
            "vehicle_counts",
        ):
            result[key] = vehicle_metrics[key]
    return result


def traffic_series(
    detections: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    videos: list[dict[str, Any]] | None = None,
    detection_series: dict[str, Any] | None = None,
) -> dict[str, Any]:
    det = _frame(detections)
    vio = _frame(violations)
    by_minute: list[dict[str, Any]] = []
    hourly: list[dict[str, Any]] = []
    if not det.empty:
        det = det.copy()
        det = det[det["class_name"].isin(VEHICLE_CLASSES)]
        det["minute"] = (det["timestamp_sec"] // 60).astype(int)
        track_columns = [column for column in ("video_id", "job_id", "tracking_id") if column in det]
        tracked = det[det["tracking_id"].notna()] if "tracking_id" in det else det.iloc[0:0]
        if track_columns and not tracked.empty:
            per_minute = tracked.drop_duplicates(["minute", *track_columns]).groupby("minute").size()
            by_minute = [{"minute": int(i), "vehicles": int(v)} for i, v in per_minute.items()]
    violation_by_minute: list[dict[str, Any]] = []
    if not vio.empty:
        vio = vio.copy()
        vio["minute"] = (vio["timestamp_sec"] // 60).astype(int)
        grouped = vio.groupby(["minute", "violation_type"]).size().reset_index(name="count")
        violation_by_minute = grouped.to_dict(orient="records")
        # Absolute hour is unknown without a capture clock, so this uses video-relative minutes.
        hourly_counter: Counter[int] = Counter()
        for minute, count in vio.groupby("minute").size().items():
            hourly_counter[int(minute) % 60] += int(count)
        hourly = [{"bucket": hour, "count": hourly_counter[hour]} for hour in range(60) if hourly_counter[hour]]
    density = []
    if not det.empty:
        det = det.copy()
        det = det[det["class_name"].isin(VEHICLE_CLASSES)]
        det["bucket"] = (det["timestamp_sec"] // 10).astype(int) * 10
        density_group = det.groupby("bucket").size()
        density = [{"t": int(i), "detections": int(v)} for i, v in density_group.items()]
    if detection_series is not None:
        by_minute = detection_series["vehicles_by_minute"]
        density = detection_series["density"]

    from datetime import datetime

    clock = [0] * 24
    daily: dict[str, int] = {}
    for row in violations:
        created = row.get("created_at")
        if not created:
            continue
        stamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        clock[stamp.hour] += 1
        day = stamp.date().isoformat()
        daily[day] = daily.get(day, 0) + 1
    names = {video.get("id"): video.get("original_name", video.get("id")) for video in videos or []}
    per_video: dict[str, int] = {}
    for row in violations:
        per_video[row["video_id"]] = per_video.get(row["video_id"], 0) + 1
    comparison = [
        {"video_id": video_id, "video_name": names.get(video_id, video_id), "violations": count}
        for video_id, count in per_video.items()
    ]
    return {
        "sample": False,
        "vehicles_by_minute": by_minute,
        "violations_by_minute": violation_by_minute,
        "activity_buckets": hourly,
        "density": density,
        "hourly": [{"hour": hour, "violations": clock[hour]} for hour in range(24)],
        "daily": [{"date": day, "violations": count} for day, count in sorted(daily.items())],
        "by_video": comparison,
    }


def most_common_violation(violation_counts: dict[str, int]) -> str | None:
    if not violation_counts:
        return None
    return max(violation_counts.items(), key=lambda item: item[1])[0]
