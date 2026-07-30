"""Main gesture car control application."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

import cv2

from gesture_car.driver import ControlMode, GestureDriver
from gesture_car.keyboard_out import KeyScheme, KeyboardDriver
from gesture_car.overlay import draw_hands, draw_help, draw_hud, draw_pedals, draw_steering_wheel
from gesture_car.tracker import HandTracker


class GestureCarApp:
    INFER_WIDTH = 960
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

    def run(self) -> int:
        cap = self._open_camera()
        if cap is None:
            print("ERROR: Could not open webcam.")
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        win = "Gesture Car Control"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

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

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                hands = self.tracker.process(
                    frame, mirrored=True, infer_max_width=self.INFER_WIDTH,
                )
                state = self.driver.update(hands, enabled=self.enabled)
                self.keyboard.apply(state, enabled=self.enabled)

                draw_hands(
                    frame,
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
                draw_steering_wheel(frame, state)
                draw_pedals(frame, state)
                draw_hud(
                    frame,
                    state,
                    enabled=self.enabled,
                )
                if self.show_help:
                    draw_help(frame, self._help_lines())

                cv2.imshow(win, frame)

                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.keyboard.release_all()
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        cap = cv2.VideoCapture(self.camera_index)
        return cap if cap.isOpened() else None

    def _handle_key(self, key: int) -> bool:
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
