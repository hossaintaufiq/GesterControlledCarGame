"""Performance benchmark: capture rate, inference cost, and full loop FPS.

Run with the app closed so the webcam is free:
    python tools/bench.py

Each phase runs in a fresh subprocess. MediaPipe keeps worker threads alive
after close(), so several graphs in one process skew later measurements.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_car.camera import CameraStream  # noqa: E402
from gesture_car.driver import GestureDriver  # noqa: E402
from gesture_car.filters import clamp_dt  # noqa: E402
from gesture_car.keyboard_out import KeyboardDriver  # noqa: E402
from gesture_car.overlay import (  # noqa: E402
    draw_hands,
    draw_hud,
    draw_pedals,
    draw_steering_wheel,
)
from gesture_car.tracker import HandTracker  # noqa: E402

MODEL = ROOT / "gesture_car" / "models" / "hand_landmarker.task"


def grab_frames(count: int = 25, size: tuple[int, int] = (1280, 720)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return []
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.flip(frame, 1))
        return frames
    finally:
        cap.release()


def bench_capture(seconds: float = 4.0) -> tuple[float, str]:
    stream = CameraStream()
    if not stream.start():
        return 0.0, "unavailable"
    try:
        last_seq, frames, size = -1, 0, ""
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            frame, last_seq = stream.read(last_seq)
            if frame is None:
                time.sleep(0.001)
                continue
            size = f"{frame.shape[1]}x{frame.shape[0]}"
            frames += 1
        return frames / (time.perf_counter() - start), size
    finally:
        stream.release()


def bench_inference(frames: list[np.ndarray], width: int) -> float:
    tracker = HandTracker(MODEL, max_hands=2)
    try:
        tracker.process(frames[0], infer_max_width=width)
        start = time.perf_counter()
        for frame in frames:
            tracker.process(frame, infer_max_width=width)
        return (time.perf_counter() - start) / len(frames) * 1000.0
    finally:
        tracker.close()


def bench_loop(infer_width: int, seconds: float = 5.0) -> float:
    tracker = HandTracker(MODEL, max_hands=2)
    driver = GestureDriver()
    keyboard = KeyboardDriver()
    keyboard._press = lambda key: None  # never emit real keystrokes
    keyboard._release = lambda key: None

    stream = CameraStream()
    if not stream.start():
        tracker.close()
        return 0.0

    frames, last_seq = 0, -1
    last = start = time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            frame, last_seq = stream.read(last_seq)
            if frame is None:
                time.sleep(0.001)
                continue
            now = time.perf_counter()
            dt = clamp_dt(now - last)
            last = now

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, infer_max_width=infer_width, dt=dt)
            state = driver.update(hands, enabled=True, dt=dt)
            keyboard.apply(state, enabled=True)
            draw_hands(frame, hands)
            draw_steering_wheel(frame, state)
            draw_pedals(frame, state)
            draw_hud(frame, state, enabled=True, fps=30.0)
            frames += 1
        return frames / (time.perf_counter() - start)
    finally:
        stream.release()
        tracker.close()


def _phase(name: str, arg: str = "") -> int:
    """Run one phase in a clean interpreter."""
    cmd = [sys.executable, str(Path(__file__).resolve()), name]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, stderr=subprocess.DEVNULL).returncode


def run_phase(name: str, arg: str) -> int:
    if name == "capture":
        fps, size = bench_capture()
        print(f"CAPTURE (threaded)        {fps:5.1f} fps at {size}")
        return 0

    if name == "inference":
        frames = grab_frames()
        if not frames:
            print("  camera unavailable — is the app still running?")
            return 1
        ms = bench_inference(frames, int(arg))
        print(f"  width {int(arg):>4} -> {ms:5.1f} ms ({1000 / ms:4.1f} fps ceiling)")
        return 0

    if name == "loop":
        print(f"  infer width {int(arg):>4} -> {bench_loop(int(arg)):5.1f} fps")
        return 0

    print(f"unknown phase: {name}")
    return 2


def main() -> int:
    if len(sys.argv) > 1:
        return run_phase(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")

    _phase("capture")
    print("\nINFERENCE cost by input width (real frames)", flush=True)
    for width in (960, 768, 640, 512):
        _phase("inference", str(width))
    print("\nFULL LOOP (capture + inference + overlay)", flush=True)
    for width in (960, 640):
        _phase("loop", str(width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
