#!/usr/bin/env python3
"""Download MediaPipe hand landmarker model."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "gesture_car" / "models" / "hand_landmarker.task"


def main() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        print(f"Model already exists: {MODEL_PATH}")
        return 0

    print(f"Downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
