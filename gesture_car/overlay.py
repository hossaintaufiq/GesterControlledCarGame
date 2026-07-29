"""On-screen feedback for gesture car control."""

from __future__ import annotations

import math

import cv2
import numpy as np

from gesture_car.driver import CarControlState, ControlMode
from gesture_car import ui
from gesture_car.tracker import HandResult

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hands(
    frame: np.ndarray,
    hands: list[HandResult],
    *,
    palm_labels: dict[str, str] | None = None,
) -> None:
    h, w = frame.shape[:2]
    labels = palm_labels or {}
    for hand in hands:
        pts = []
        for lm in hand.landmarks:
            pts.append((
                int(np.clip(lm[0], 0, 1) * (w - 1)),
                int(np.clip(lm[1], 0, 1) * (h - 1)),
            ))
        label = labels.get(hand.handedness, "")
        if label == "CLOSED":
            color = ui.SUCCESS
        elif label == "OPEN":
            color = ui.DANGER
        else:
            color = ui.ACCENT if hand.handedness == "Left" else ui.WARN

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)

        if label:
            wx = int(hand.landmarks[0][0] * (w - 1))
            wy = int(hand.landmarks[0][1] * (h - 1)) - 12
            ui.put_text(frame, label, (wx - 28, wy), scale=0.42, color=color, weight=2)


def draw_steering_wheel(frame: np.ndarray, state: CarControlState) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.72)
    radius = min(w, h) // 5
    angle = state.steer * 0.9
    cv2.circle(frame, (cx, cy), radius, ui.MUTED, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, ui.ACCENT, -1, cv2.LINE_AA)

    knob_x = int(cx + math.sin(angle) * radius * 0.85)
    knob_y = int(cy - math.cos(angle) * radius * 0.85)
    cv2.line(frame, (cx, cy), (knob_x, knob_y), ui.ACCENT, 3, cv2.LINE_AA)
    cv2.circle(frame, (knob_x, knob_y), 10, ui.WARN, -1, cv2.LINE_AA)


def draw_pedals(frame: np.ndarray, state: CarControlState) -> None:
    h, w = frame.shape[:2]
    base_y = h - 36
    gas_w = int(90 * state.throttle)
    brake_w = int(90 * state.brake)
    cv2.rectangle(frame, (w - 210, base_y - 18), (w - 120, base_y + 18), ui.MUTED, 1)
    cv2.rectangle(frame, (w - 100, base_y - 18), (w - 10, base_y + 18), ui.MUTED, 1)
    if gas_w > 0:
        cv2.rectangle(frame, (w - 208, base_y - 16), (w - 208 + gas_w, base_y + 16), ui.SUCCESS, -1)
    if brake_w > 0:
        cv2.rectangle(frame, (w - 98, base_y - 16), (w - 98 + brake_w, base_y + 16), ui.DANGER, -1)
    ui.put_text(frame, "GAS", (w - 205, base_y - 28), scale=0.42, color=ui.MUTED)
    ui.put_text(frame, "BRK", (w - 95, base_y - 28), scale=0.42, color=ui.MUTED)


def draw_hud(
    frame: np.ndarray,
    state: CarControlState,
    *,
    enabled: bool,
    keys: set[str],
    fps: float,
    scheme: str,
) -> None:
    h, w = frame.shape[:2]
    ui.panel(frame, 0, 0, w, 108, alpha=0.72)

    status = "LIVE" if enabled else "PAUSED"
    status_color = ui.SUCCESS if enabled else ui.WARN
    ui.put_text(frame, "GESTURE CAR", (18, 30), scale=0.7, color=ui.ACCENT, weight=2)
    ui.put_text(frame, status, (220, 30), scale=0.62, color=status_color, weight=2)

    mode = "WHEEL" if state.mode == ControlMode.WHEEL else "POINTER"
    ui.put_text(frame, f"Mode {mode}  |  Keys {scheme}  |  FPS {fps:.0f}", (18, 58), scale=0.45, color=ui.MUTED)
    ui.put_text(
        frame,
        f"Steer {state.steer_label}  |  {state.pedal_label}  |  Hands {state.hands_visible}",
        (18, 78),
        scale=0.45,
        color=ui.TEXT,
    )
    ui.put_text(
        frame,
        f"L: {state.left_palm}   R: {state.right_palm}   (green=closed  red=open)",
        (18, 98),
        scale=0.42,
        color=ui.MUTED,
    )

    if keys:
        labels = " ".join(k.upper() for k in sorted(keys))
        ui.put_text(frame, f"Sending: {labels}", (w - 260, 30), scale=0.5, color=ui.SUCCESS)

    if not enabled:
        ui.put_text(frame, "Press TAB to arm controls", (w // 2 - 150, h // 2), scale=0.7, color=ui.WARN, weight=2)


def draw_help(frame: np.ndarray, lines: list[str]) -> None:
    h, w = frame.shape[:2]
    panel_h = 24 * len(lines) + 24
    y1 = h - panel_h - 16
    ui.panel(frame, 16, y1, 460, h - 16, alpha=0.75)
    for i, line in enumerate(lines):
        ui.put_text(frame, line, (28, y1 + 28 + i * 24), scale=0.45, color=ui.TEXT)
