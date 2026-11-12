"""Traffic-rule evaluation.

These modules flag potential violations for human review. They do not issue fines.
A rule runs only when its required evidence is present: a verified signal state,
configured lane geometry, calibration, or a model that actually contains the class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.detection.labels import HELMET_NEGATIVE, VEHICLE_CLASSES
from app.detection.types import Detection, Track, centroid
from app.violations.geometry import angle_diff_deg, movement_angle_deg, point_in_polygon, segments_intersect

OVERSPEED_LIMITS_KMH = {
    "car": 70,
    "bus": 60,
    "truck": 60,
    "motorcycle": 60,
}


@dataclass
class ViolationEvent:
    violation_type: str
    tracking_id: int | None
    vehicle_category: str
    timestamp_sec: float
    frame_number: int
    confidence: float
    bbox: tuple[float, float, float, float]
    needs_review: bool
    explanation: str
    rule: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleReport:
    name: str
    status: str
    message: str
    events: list[ViolationEvent] = field(default_factory=list)


def _signal_at(timeline: list[dict[str, Any]], timestamp: float) -> str | None:
    for index, segment in enumerate(timeline):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        is_last = index == len(timeline) - 1
        if start <= timestamp < end or (is_last and timestamp == end):
            state = str(segment.get("state", "")).lower()
            if state in {"red", "green", "yellow"}:
                return state
    return None


def evaluate_red_light(
    tracks: list[Track],
    stop_line: list[list[float]] | None,
    signal_timeline: list[dict[str, Any]] | None,
    min_confidence: float,
) -> ModuleReport:
    if not stop_line or len(stop_line) != 2:
        return ModuleReport("red_light", "inactive", "Configure a stop line with two points before red-light rules can run.")
    if not signal_timeline:
        return ModuleReport(
            "red_light",
            "inactive",
            "A verified signal timeline is required. Vehicles are not flagged from motion alone.",
        )
    line = (tuple(stop_line[0]), tuple(stop_line[1]))
    events: list[ViolationEvent] = []
    seen: set[int] = set()
    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES or track.track_id in seen:
            continue
        points = track.history
        for prev, curr in zip(points, points[1:]):
            if curr.confidence < min_confidence:
                continue
            state = _signal_at(signal_timeline, curr.timestamp_sec)
            if state != "red":
                continue
            if segments_intersect(prev.centroid, curr.centroid, line[0], line[1]):
                seen.add(track.track_id)
                events.append(
                    ViolationEvent(
                        violation_type="red_light",
                        tracking_id=track.track_id,
                        vehicle_category=track.class_name,
                        timestamp_sec=curr.timestamp_sec,
                        frame_number=curr.frame_number,
                        confidence=round(curr.confidence, 4),
                        bbox=curr.bbox,
                        needs_review=True,
                        rule="Stop-line crossing while the supplied signal state is red.",
                        explanation=(
                            f"Track {track.track_id} ({track.class_name}) crossed the configured stop line "
                            f"at {curr.timestamp_sec:.2f}s during a verified red interval."
                        ),
                        evidence={"signal_state": "red", "stop_line": stop_line},
                    )
                )
                break
    return ModuleReport("red_light", "active", "Evaluated configured stop line against the signal timeline.", events)


def evaluate_wrong_lane(
    tracks: list[Track],
    lanes: list[dict[str, Any]] | None,
    min_confidence: float,
    min_displacement_px: float = 30,
    opposite_angle_deg: float = 120,
) -> ModuleReport:
    if not lanes:
        return ModuleReport("wrong_lane", "inactive", "Define at least one lane polygon and travel direction.")
    events: list[ViolationEvent] = []
    seen: set[tuple[int, str]] = set()
    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES:
            continue
        for lane in lanes:
            polygon = [tuple(p) for p in lane.get("polygon", [])]
            direction = lane.get("direction_deg")
            lane_id = str(lane.get("id", "lane"))
            if direction is None or len(polygon) < 3:
                continue
            key = (track.track_id, lane_id)
            if key in seen:
                continue
            for prev, curr in zip(track.history, track.history[1:]):
                if curr.confidence < min_confidence:
                    continue
                if not point_in_polygon(curr.centroid, polygon):
                    continue
                moved = math.dist(prev.centroid, curr.centroid)
                if moved < min_displacement_px:
                    continue
                heading = movement_angle_deg(prev.centroid, curr.centroid)
                if heading is None:
                    continue
                delta = angle_diff_deg(heading, float(direction))
                if delta >= opposite_angle_deg:
                    seen.add(key)
                    events.append(
                        ViolationEvent(
                            violation_type="wrong_lane",
                            tracking_id=track.track_id,
                            vehicle_category=track.class_name,
                            timestamp_sec=curr.timestamp_sec,
                            frame_number=curr.frame_number,
                            confidence=round(min(curr.confidence, max(0.0, delta / 180)), 4),
                            bbox=curr.bbox,
                            needs_review=True,
                            rule=f"Travel heading differs from lane {lane_id} direction by at least {opposite_angle_deg:.0f} degrees.",
                            explanation=(
                                f"Track {track.track_id} moved at {heading:.0f}° inside {lane_id} "
                                f"(configured direction {float(direction):.0f}°, difference {delta:.0f}°)."
                            ),
                            evidence={"lane_id": lane_id, "heading_deg": round(heading, 2), "difference_deg": round(delta, 2)},
                        )
                    )
                    break
    return ModuleReport("wrong_lane", "active", "Compared track headings with configured lane directions.", events)


def evaluate_overspeed(
    tracks: list[Track],
    calibration: dict[str, Any] | None,
    min_confidence: float,
) -> ModuleReport:
    if not calibration or not calibration.get("validated"):
        return ModuleReport(
            "overspeed",
            "unavailable",
            "Speed is unavailable until camera calibration is marked validated. Uncalibrated pixel motion is not reported as speed.",
        )
    meters_per_pixel = calibration.get("meters_per_pixel")
    if not meters_per_pixel or float(meters_per_pixel) <= 0:
        return ModuleReport("overspeed", "unavailable", "meters_per_pixel must be a positive calibrated value.")
    mpp = float(meters_per_pixel)
    events: list[ViolationEvent] = []
    seen: set[int] = set()
    for track in tracks:
        speed_limit_kmh = OVERSPEED_LIMITS_KMH.get(track.class_name)
        if speed_limit_kmh is None or track.track_id in seen:
            continue
        peak = 0.0
        peak_point = None
        for prev, curr in zip(track.history, track.history[1:]):
            dt = curr.timestamp_sec - prev.timestamp_sec
            if dt <= 0 or curr.confidence < min_confidence:
                continue
            pixels = math.dist(centroid(prev.bbox), centroid(curr.bbox))
            speed = (pixels * mpp / dt) * 3.6
            if speed > peak:
                peak = speed
                peak_point = curr
        if peak_point is not None and peak > speed_limit_kmh and not math.isclose(
            peak, speed_limit_kmh, rel_tol=1e-9
        ):
            seen.add(track.track_id)
            events.append(
                ViolationEvent(
                    violation_type="overspeed",
                    tracking_id=track.track_id,
                    vehicle_category=track.class_name,
                    timestamp_sec=peak_point.timestamp_sec,
                    frame_number=peak_point.frame_number,
                    confidence=round(peak_point.confidence, 4),
                    bbox=peak_point.bbox,
                    needs_review=True,
                    rule=f"Calibrated speed exceeds {speed_limit_kmh:.0f} km/h.",
                    explanation=(
                        f"Track {track.track_id} reached {peak:.1f} km/h using validated calibration "
                        f"({mpp} m/px). Limit is {speed_limit_kmh:.0f} km/h."
                    ),
                    evidence={"speed_kmh": round(peak, 2), "speed_limit_kmh": speed_limit_kmh, "meters_per_pixel": mpp},
                )
            )
    return ModuleReport(
        "overspeed",
        "active",
        "Compared calibrated vehicle speeds with the default class-specific limits.",
        events,
    )


def evaluate_class_violation(
    detections: list[Detection],
    violation_type: str,
    negative_labels: set[str],
    review_below: float,
    model_configured: bool,
    missing_message: str,
) -> ModuleReport:
    if not model_configured:
        return ModuleReport(violation_type, "not_configured", missing_message)
    events: list[ViolationEvent] = []
    best_by_track: dict[int, Detection] = {}
    untracked: list[Detection] = []
    for detection in detections:
        if detection.class_name not in negative_labels:
            continue
        if detection.tracking_id is None:
            untracked.append(detection)
            continue
        current = best_by_track.get(detection.tracking_id)
        if current is None or detection.confidence > current.confidence:
            best_by_track[detection.tracking_id] = detection
    for detection in [*best_by_track.values(), *untracked]:
        uncertain = detection.confidence < review_below
        events.append(
            ViolationEvent(
                violation_type=violation_type,
                tracking_id=detection.tracking_id,
                vehicle_category="motorcycle" if violation_type == "helmet" else "car",
                timestamp_sec=detection.timestamp_sec,
                frame_number=detection.frame_number,
                confidence=round(detection.confidence, 4),
                bbox=detection.bbox,
                needs_review=uncertain or detection.confidence < 0.75,
                rule=f"Custom {violation_type} model predicted '{detection.class_name}'.",
                explanation=(
                    f"Model predicted {detection.class_name} with confidence {detection.confidence:.2f} "
                    f"at frame {detection.frame_number}. This is a potential violation for manual review."
                ),
                evidence={"predicted_class": detection.class_name, "uncertain": uncertain},
            )
        )
    return ModuleReport(violation_type, "active", "Evaluated custom-model class predictions.", events)


def evaluate_helmet(detections: list[Detection], model_configured: bool, review_below: float = 0.6) -> ModuleReport:
    return evaluate_class_violation(
        detections,
        "helmet",
        HELMET_NEGATIVE,
        review_below,
        model_configured,
        "Helmet detection needs a custom YOLO model with no_helmet (and helmet) classes. Set HELMET_WEIGHTS. Official COCO weights do not include helmets.",
    )
