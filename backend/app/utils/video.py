"""Video file validation and metadata probing."""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}


class VideoRejected(ValueError):
    pass


def validate_upload(upload: UploadFile, size_bytes: int) -> str:
    name = upload.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise VideoRejected(f"Unsupported file type '{suffix or 'unknown'}'. Use MP4, AVI, MOV, or WebM.")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes <= 0:
        raise VideoRejected("The uploaded file is empty.")
    if size_bytes > max_bytes:
        raise VideoRejected(f"File is larger than the {settings.max_upload_mb} MB limit.")
    return suffix


def probe_video(path: Path) -> dict[str, float | int | None]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise VideoRejected(f"OpenCV is not installed, so the video cannot be read. {exc}") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoRejected(
            "OpenCV could not open this video. MP4, MOV, and WebM need the OpenCV FFmpeg build. "
            "Try an AVI file or install a build of OpenCV with FFmpeg."
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = None
    if fps and frame_count > 0:
        duration = frame_count / fps
    if duration and duration > settings.max_duration_sec:
        raise VideoRejected(
            f"Video duration {duration:.1f}s exceeds the {settings.max_duration_sec:.0f}s limit."
        )
    return {"fps": fps, "width": width, "height": height, "duration_sec": duration, "frame_count": frame_count}
