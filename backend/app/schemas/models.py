"""Request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SignalSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    state: Literal["red", "green", "yellow"]


class LaneConfig(BaseModel):
    id: str = "lane-1"
    polygon: list[list[float]] = Field(min_length=3)
    direction_deg: float


class Calibration(BaseModel):
    meters_per_pixel: float | None = Field(default=None, gt=0)
    validated: bool = False


class AnalysisConfig(BaseModel):
    conf: float = Field(default=0.25, ge=0.05, le=0.95)
    iou: float = Field(default=0.50, ge=0.05, le=0.95)
    imgsz: int = Field(default=960, ge=320, le=1920)
    track_buffer: int = Field(default=30, ge=1, le=300)
    tracker_type: Literal["bytetrack", "botsort"] = "bytetrack"
    debug_mode: bool = False
    frame_stride: int = Field(default=1, ge=1, le=15)
    target_fps: float | None = Field(default=None, gt=0, le=60)
    max_frames: int = Field(default=2400, ge=1, le=20000)
    stop_line: list[list[float]] | None = None
    signal_timeline: list[SignalSegment] | None = None
    lanes: list[LaneConfig] | None = None
    calibration: Calibration | None = None
    enable_red_light: bool = True
    enable_wrong_lane: bool = True
    enable_overspeed: bool = True
    enable_helmet: bool = True
    save_annotated: bool = True
    model_name: str | None = None


class ReviewUpdate(BaseModel):
    status: Literal["pending", "needs_review", "reviewed", "false_positive"]


class StartAnalysis(BaseModel):
    video_id: str = Field(min_length=8)
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)
