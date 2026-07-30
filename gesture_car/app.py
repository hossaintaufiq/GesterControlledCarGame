"""Main gesture car control application."""

from __future__ import annotations

import time
import webbrowser
from pathlib import Path

import cv2

from gesture_car.camera import CameraStream
from gesture_car.driver import ControlMode, GestureDriver
from gesture_car.filters import clamp_dt
from gesture_car.keyboard_out import KeyScheme, KeyboardDriver
from gesture_car.overlay import draw_hands, draw_help, draw_hud, draw_pedals, draw_steering_wheel
from gesture_car.tracker import HandTracker
from gesture_car.window import PictureInPictureWindow


class GestureCarApp:
    # MediaPipe rescales internally, so 640 costs ~20% less than 960 for the
    # same landmark quality.
    INFER_WIDTH = 640
    CAPTURE_SIZE = (1280, 720)
    PIP_SIZE = (480, 270)
    GAME_URL = "https://slowroads.io"

    def __init__(self, camera_index: int = 0) -> None:
        root = Path(__file__).resolve().parent
        model = root / "models" / "hand_landmarker.task"
        self.tracker = HandTracker(model, max_hands=2)
        self.driver = GestureDriver()
        self.keyboard = KeyboardDriver()
        self.camera_index = camera_index

        self.enabled = False
        self.show_help = False
        self._fps = 0.0
        self._frames = 0
        self._fps_time = time.perf_counter()

    def run(self) -> int:
        width, height = self.CAPTURE_SIZE
        stream = CameraStream(self.camera_index, width=width, height=height)
        if not stream.start():
            print("ERROR: Could not open webcam.")
            return 1

        win = "Gesture Car Control"
        pip = PictureInPictureWindow(
            win,
            width=self.PIP_SIZE[0],
            height=self.PIP_SIZE[1],
        )
        pip.create()

        print("Gesture Car Control")
        print("  TAB  = arm / pause keyboard output")
        print("  M    = toggle wheel / pointer mode")
        print("  K    = toggle arrow keys / WASD")
        print("  G    = open Slow Roads")
        print("  [ ]  = steering sensitivity down / up")
        print("  H    = help overlay")
        print("  Q    = quit")
        print()
        print("Tip: arm controls (TAB), click your browser game, then drive with gestures.")

        last_seq = -1
        last_time = time.perf_counter()
        try:
            while True:
                frame, last_seq = stream.read(last_seq)
                if frame is None:
                    # No new camera frame yet; stay responsive to key presses.
                    if not self._handle_key(cv2.waitKey(1) & 0xFF):
                        break
                    continue

                now = time.perf_counter()
                dt = clamp_dt(now - last_time)
                last_time = now

                frame = cv2.flip(frame, 1)
                hands = self.tracker.process(
                    frame,
                    mirrored=True,
                    infer_max_width=self.INFER_WIDTH,
                    dt=dt,
                )
                state = self.driver.update(hands, enabled=self.enabled, dt=dt)
                self.keyboard.apply(state, enabled=self.enabled)

                display = cv2.resize(
                    frame,
                    self.PIP_SIZE,
                    interpolation=cv2.INTER_AREA,
                )
                draw_hands(
                    display,
                    hands,
                    palm_labels={
                        "Left": state.left_palm,
                        "Right": state.right_palm,
                    },
                    openness={
                        "Left": state.left_openness,
                        "Right": state.right_openness,
                    },
                )
                draw_steering_wheel(display, state)
                draw_pedals(display, state)
                draw_hud(
                    display,
                    state,
                    enabled=self.enabled,
                    fps=self._fps,
                    compact=True,
                )
                if self.show_help:
                    draw_help(display, self._help_lines())

                self._tick_fps(now)
                pip.show(display)

                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.keyboard.release_all()
            self.tracker.close()
            stream.release()
            cv2.destroyAllWindows()
        return 0

    def _handle_key(self, key: int) -> bool:
        if key == 255:
            return True
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key == 9:  # TAB
            self.enabled = not self.enabled
            if not self.enabled:
                self.keyboard.release_all()
            print("Controls", "ARMED" if self.enabled else "PAUSED")
        if key in (ord("m"), ord("M")):
            mode = (
                ControlMode.POINTER
                if self.driver.mode == ControlMode.WHEEL
                else ControlMode.WHEEL
            )
            self.driver.set_mode(mode)
            print("Mode:", mode.name)
        if key in (ord("k"), ord("K")):
            scheme = (
                KeyScheme.WASD
                if self.keyboard.scheme == KeyScheme.ARROWS
                else KeyScheme.ARROWS
            )
            self.keyboard.scheme = scheme
            self.keyboard.release_all()
            print("Keys:", scheme.name)
        if key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        if key in (ord("g"), ord("G")):
            webbrowser.open(self.GAME_URL)
        if key in (ord("["), ord("-")):
            print("Steering sensitivity:", self.driver.adjust_sensitivity(-0.1))
        if key in (ord("]"), ord("=")):
            print("Steering sensitivity:", self.driver.adjust_sensitivity(+0.1))
        return True

    def _help_lines(self) -> list[str]:
        return [
            "Tilt hands = steer   [ ] = sensitivity",
            "Both closed = gas | both open = brake",
            "TAB start | G game | H hide | Q quit",
        ]

    def _tick_fps(self, now: float) -> None:
        self._frames += 1
        span = now - self._fps_time
        if span >= 0.5:
            self._fps = self._frames / span
            self._frames = 0
            self._fps_time = now
