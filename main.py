#!/usr/bin/env python3
"""Gesture-controlled keyboard driver for online car games.

Setup
-----
    pip install -r requirements.txt
    python download_model.py
    python main.py

Usage
-----
1. Run this app (webcam preview opens).
2. Press TAB to arm keyboard output.
3. Click your browser car game so it has focus.
4. Drive with hand gestures (see on-screen help).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_car.app import GestureCarApp


def main() -> int:
    return GestureCarApp(camera_index=0).run()


if __name__ == "__main__":
    raise SystemExit(main())
