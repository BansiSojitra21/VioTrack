"""ByteTrack-style multi-object tracker.

Association follows ByteTrack: high-confidence detections are matched first,
then remaining tracks are matched to low-confidence detections. Matching uses
the Hungarian algorithm on IoU cost when SciPy is available.
"""

from __future__ import annotations

from app.detection.types import Detection, Track, TrackPoint, iou


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.array(cost, dtype=float)
        rows, cols = linear_sum_assignment(matrix)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        return _greedy(cost)


def _greedy(cost: list[list[float]]) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, row in enumerate(cost):
        for j, value in enumerate(row):
            pairs.append((value, i, j))
    pairs.sort()
    used_r: set[int] = set()
    used_c: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matches.append((i, j))
    return matches


class ByteTrack:
    MAX_HISTORY_POINTS = 512

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_iou: float = 0.3,
        max_age: int = 30,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.all_tracks: list[Track] = []
        self._next_id = 1
        self.external_ids_enabled = False

    def update(self, detections: list[Detection], frame_number: int, timestamp_sec: float) -> list[Detection]:
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        if self.external_ids_enabled:
            matched_ids: set[int] = set()
            tracks_by_id = {track.track_id: track for track in self.tracks}
            unassigned: list[Detection] = []
            for detection in detections:
                if detection.tracking_id is None:
                    unassigned.append(detection)
                    continue
                track = tracks_by_id.get(detection.tracking_id)
                if track is None:
                    track = Track(
                        track_id=detection.tracking_id,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                    )
                    self.tracks.append(track)
                    self.all_tracks.append(track)
                    tracks_by_id[track.track_id] = track
                self._append(track, detection, frame_number, timestamp_sec)
                matched_ids.add(track.track_id)

            if unassigned:
                unmatched_tracks = [t for t in self.tracks if t.track_id not in matched_ids]
                self._associate(unmatched_tracks, unassigned, frame_number, timestamp_sec)
                max_id = max([t.track_id for t in self.all_tracks] or [0])
                for det in unassigned:
                    if det.tracking_id is not None:
                        matched_ids.add(det.tracking_id)
                    else:
                        max_id += 1
                        det.tracking_id = max_id
                        track = Track(
                            track_id=max_id,
                            class_id=det.class_id,
                            class_name=det.class_name,
                        )
                        self._append(track, det, frame_number, timestamp_sec)
                        self.tracks.append(track)
                        self.all_tracks.append(track)
                        matched_ids.add(max_id)

            self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
            return [
                Detection(
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    bbox=track.bbox,
                    frame_number=frame_number,
                    timestamp_sec=timestamp_sec,
                    tracking_id=track.track_id,
                )
                for track in self.tracks
                if track.track_id in matched_ids
            ]

        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        unmatched_tracks = self._associate(self.tracks, high, frame_number, timestamp_sec)
        still_open = self._associate(unmatched_tracks, low, frame_number, timestamp_sec)

        matched_ids = {t.track_id for t in self.tracks if t.time_since_update == 0}
        for detection in high:
            if detection.tracking_id is None:
                track = Track(
                    track_id=self._next_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
                self._next_id += 1
                self._append(track, detection, frame_number, timestamp_sec)
                self.tracks.append(track)
                self.all_tracks.append(track)
                matched_ids.add(track.track_id)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        output: list[Detection] = []
        for track in self.tracks:
            if track.track_id not in matched_ids:
                continue
            point = track.history[-1]
            output.append(
                Detection(
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=point.confidence,
                    bbox=point.bbox,
                    frame_number=frame_number,
                    timestamp_sec=timestamp_sec,
                    tracking_id=track.track_id,
                )
            )
        return output

    def _associate(
        self,
        tracks: list[Track],
        detections: list[Detection],
        frame_number: int,
        timestamp_sec: float,
    ) -> list[Track]:
        if not tracks or not detections:
            return list(tracks)
        cost: list[list[float]] = []
        for track in tracks:
            row = []
            for detection in detections:
                if detection.tracking_id is not None:
                    row.append(1.0)
                else:
                    predicted = self._predict_bbox(track, frame_number)
                    overlap = iou(predicted, detection.bbox)
                    distance = self._center_distance(predicted, detection.bbox)
                    size = max(
                        predicted[2] - predicted[0],
                        predicted[3] - predicted[1],
                        detection.bbox[2] - detection.bbox[0],
                        detection.bbox[3] - detection.bbox[1],
                        1.0,
                    )
                    elapsed = max(1, frame_number - track.history[-1].frame_number)
                    distance_gate = size * min(6.0, 1.5 + 0.75 * elapsed)
                    proximity = max(0.0, 1.0 - distance / distance_gate)
                    score = max(overlap, proximity)
                    class_penalty = 0.2 if detection.class_name != track.class_name else 0.0
                    row.append(1.0 - score + class_penalty)
            cost.append(row)
        unmatched = set(range(len(tracks)))
        for row, col in _hungarian(cost):
            if cost[row][col] > 1.0 - self.match_iou:
                continue
            detection = detections[col]
            if detection.tracking_id is not None:
                continue
            track = tracks[row]
            self._append(track, detection, frame_number, timestamp_sec)
            detection.tracking_id = track.track_id
            unmatched.discard(row)
        return [tracks[i] for i in sorted(unmatched)]

    @staticmethod
    def _predict_bbox(track: Track, frame_number: int) -> tuple[float, float, float, float]:
        latest = track.history[-1]
        if len(track.history) < 2:
            return latest.bbox
        previous = track.history[-2]
        elapsed = latest.frame_number - previous.frame_number
        if elapsed <= 0:
            return latest.bbox
        ahead = min(max(0, frame_number - latest.frame_number), elapsed * 4)
        return tuple(
            current + (current - prior) * ahead / elapsed
            for prior, current in zip(previous.bbox, latest.bbox)
        )

    @staticmethod
    def _center_distance(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    @staticmethod
    def _append(track: Track, detection: Detection, frame_number: int, timestamp_sec: float) -> None:
        track.history.append(
            TrackPoint(
                frame_number=frame_number,
                timestamp_sec=timestamp_sec,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        track.hits += 1
        track.time_since_update = 0
        track.class_votes[detection.class_name] = (
            track.class_votes.get(detection.class_name, 0.0) + detection.confidence
        )
        track.class_ids[detection.class_name] = detection.class_id
        track.class_name = max(track.class_votes, key=track.class_votes.get)
        track.class_id = track.class_ids[track.class_name]
        if len(track.history) > ByteTrack.MAX_HISTORY_POINTS:
            history = track.history
            recent_start = max(0, len(history) - 64)
            track.history = history[:recent_start:2] + history[recent_start:]
