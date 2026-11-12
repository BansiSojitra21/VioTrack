"""Download official Ultralytics weights into app/weights.

Usage (from the backend directory, with dependencies installed):

    python -m app.scripts.download_weights yolo11n.pt
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from app.core.config import settings


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n.pt"
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Ultralytics is not installed: {exc}") from exc
    model = YOLO(name)
    source = Path(getattr(model, "ckpt_path", name))
    if not source.is_file():
        source = Path(name)
    if not source.is_file():
        raise SystemExit(f"Could not locate downloaded weights for {name}.")
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.weights_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    print(f"Weights ready at {destination}")


if __name__ == "__main__":
    main()
