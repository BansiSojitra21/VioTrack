"""Train a helmet / no-helmet YOLO model from the Kaggle YOLO-format dataset.

Dataset: mobeenfatimah/helmet-detection-dataset-yolo-format
Classes written into weights: without_helmet, with_helmet
(compatible with HELMET_NEGATIVE / HELMET_POSITIVE in app.detection.labels)

Usage (from backend/):
  .venv\\Scripts\\python.exe scripts\\train_helmet.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = BACKEND_ROOT / "app" / "datasets" / "helmet" / "data.yaml"
WEIGHTS_OUT = BACKEND_ROOT / "app" / "weights" / "helmet.pt"
RUN_DIR = BACKEND_ROOT / "runs" / "helmet"


def main() -> int:
    if not DATA_YAML.is_file():
        print(f"Missing data.yaml at {DATA_YAML}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Small nano model + modest imgsz/batch for CPU (batch=8 can OOM on 16GB hosts).
    model = YOLO("yolo11n.pt")
    try:
        model.train(
            data=str(DATA_YAML),
            epochs=20,
            imgsz=416,
            batch=4,
            device="cpu",
            workers=0,
            project=str(RUN_DIR),
            name="yolo11n",
            exist_ok=True,
            patience=6,
            seed=0,
            deterministic=True,
            plots=False,
            verbose=True,
            close_mosaic=5,
        )
    except Exception as exc:
        print(f"Training stopped early: {exc}", file=sys.stderr)

    best = RUN_DIR / "yolo11n" / "weights" / "best.pt"
    if not best.is_file():
        candidates = sorted(RUN_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No best.pt found — training did not produce weights.", file=sys.stderr)
            return 1
        best = candidates[0]

    shutil.copy2(best, WEIGHTS_OUT)
    print(f"Copied {best} -> {WEIGHTS_OUT}")
    print("Set HELMET_WEIGHTS=helmet.pt in backend/.env and restart the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
