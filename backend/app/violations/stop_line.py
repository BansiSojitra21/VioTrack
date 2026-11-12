"""Conservative stop-line suggestions from visible road markings."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2


def detect_stop_line_candidate(frame: Any) -> tuple[list[list[float]], float] | None:
    height, width = frame.shape[:2]
    if width < 100 or height < 100:
        return None

    scale = min(1.0, 960 / width)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    white_markings = cv2.inRange(gray, 175, 255)
    white_markings[saturation > 90] = 0

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.08)
    road = white_markings[margin_y : height - margin_y, margin_x : width - margin_x]
    road_height, road_width = road.shape
    candidates: list[tuple[float, list[list[float]]]] = []

    length = max(80, round(road_width * 0.55))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    opened = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            box_width < road_width * 0.55
            or box_height < max(5, road_height * 0.014)
            or box_height > road_height * 0.08
        ):
            continue

        center_y = (margin_y + y + box_height / 2) / height
        line = [[0.0, round(center_y, 4)], [1.0, round(center_y, 4)]]
        score = (box_width / road_width) * 0.8 + (box_height / road_height) * 0.2
        candidates.append((score, line))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1], round(max(item[0] for item in candidates), 3)


def suggest_stop_line(video_path: str | Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open the stored video to inspect road markings.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_positions = sorted({0, frame_count // 4, frame_count // 2, (frame_count * 3) // 4})
    candidates: list[tuple[float, list[list[float]], Any]] = []
    preview_frame = None
    try:
        for frame_number in sample_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if preview_frame is None:
                preview_frame = frame
            candidate = detect_stop_line_candidate(frame)
            if candidate is not None:
                candidates.append((candidate[1], candidate[0], frame))
    finally:
        capture.release()

    clusters: list[list[tuple[float, list[list[float]], Any]]] = []
    for candidate in candidates:
        line = candidate[1]
        center = ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)
        matching = None
        for cluster in clusters:
            representative = cluster[0][1]
            representative_center = (
                (representative[0][0] + representative[1][0]) / 2,
                (representative[0][1] + representative[1][1]) / 2,
            )
            if abs(center[1] - representative_center[1]) <= 0.04:
                matching = cluster
                break
        if matching is None:
            clusters.append([candidate])
        else:
            matching.append(candidate)

    required_samples = min(3, len(sample_positions))
    consistent = [cluster for cluster in clusters if len(cluster) >= required_samples]
    if consistent:
        best = max(consistent, key=lambda cluster: (len(cluster), sum(item[0] for item in cluster)))
        score, line, preview_frame = max(best, key=lambda item: item[0])
    else:
        score, line = 0.0, None
    if preview_frame is None:
        raise RuntimeError("Could not read any frames from the stored video.")

    preview_scale = min(1.0, 800 / preview_frame.shape[1])
    if preview_scale < 1:
        preview_frame = cv2.resize(
            preview_frame,
            (
                round(preview_frame.shape[1] * preview_scale),
                round(preview_frame.shape[0] * preview_scale),
            ),
        )
    encoded, buffer = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not encoded:
        raise RuntimeError("Could not prepare the video frame preview.")

    return {
        "found": line is not None,
        "line": line,
        "score": score,
        "preview_width": int(preview_frame.shape[1]),
        "preview_height": int(preview_frame.shape[0]),
        "preview_jpeg": base64.b64encode(buffer).decode("ascii"),
    }
