"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "TrafficAI"
    app_version: str = "1.0.0"
    data_dir: str = str(BACKEND_ROOT / "data")
    database_path: str = ""
    cors_origins: str = "http://localhost:3000"
    max_upload_mb: int = 200
    max_duration_sec: float = 180
    max_frames: int = 2400
    frame_stride: int = 1
    yolo_weights: str = "yolo11n.pt"
    helmet_weights: str = ""
    yolo_conf: float = 0.25
    yolo_iou: float = 0.50
    yolo_imgsz: int = 960
    track_buffer: int = 30
    yolo_tracker: str = "bytetrack"
    device: str = "auto"
    allow_weight_download: bool = False
    cloudinary_url: str = ""
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_data_dir(self) -> Path:
        path = Path(self.data_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def resolved_database_path(self) -> Path:
        if self.database_path:
            path = Path(self.database_path)
            if not path.is_absolute():
                path = BACKEND_ROOT / path
            return path
        return self.resolved_data_dir / "trafficai.sqlite"

    @property
    def weights_dir(self) -> Path:
        return BACKEND_ROOT / "app" / "weights"

    def resolve_weights(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_file():
            return path
        candidate = self.weights_dir / value
        if candidate.is_file():
            return candidate
        # Ultralytics can download official weights by filename (for example yolo11n.pt).
        if path.name == value and not path.is_absolute():
            return path
        return path


settings = Settings()
