"""Map hand poses to car control signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from gesture_car.filters import REF_DT, clamp_dt, lerp
from gesture_car.gestures import PalmAnalyzer, PalmState
from gesture_car.steer import SteerSmoother
from gesture_car.tracker import HandResult

WRIST = 0
PALM_LANDMARKS = (0, 5, 9, 13, 17)


class ControlMode(Enum):
    WHEEL = auto()
    POINTER = auto()


@dataclass
class CarControlState:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    mode: ControlMode = ControlMode.WHEEL
    hands_visible: int = 0
    active: bool = False
    steer_label: str = "CENTER"
    pedal_label: str = "COAST"
    left_palm: str = "—"
    right_palm: str = "—"
    left_openness: float = 0.0
    right_openness: float = 0.0


class GestureDriver:
    STEER_GAIN = 1.15
    WHEEL_MAX_ANGLE = 0.72
    MIN_HAND_SCORE = 0.30
    PEDAL_CONFIRM_SECONDS = 0.10
    TRACKING_GRACE_SECONDS = 0.14
    SENSITIVITY_RANGE = (0.4, 1.6)

    def __init__(self) -> None:
        self.mode = ControlMode.WHEEL
        self.palms = PalmAnalyzer(min_hand_score=self.MIN_HAND_SCORE)
        self.steer = SteerSmoother()
        self._smooth_throttle = 0.0
        self._smooth_brake = 0.0
        self._gas_time = 0.0
        self._brake_time = 0.0
        self._last_reliable: list[HandResult] = []
        self._tracking_gap = 0.0

    def set_mode(self, mode: ControlMode) -> None:
        self.mode = mode
        self.palms.reset()
        self.steer.reset()
        self._gas_time = 0.0
        self._brake_time = 0.0

    @property
    def sensitivity(self) -> float:
        return self.steer.sensitivity

    def adjust_sensitivity(self, delta: float) -> float:
        low, high = self.SENSITIVITY_RANGE
        self.steer.sensitivity = round(
            min(high, max(low, self.steer.sensitivity + delta)), 2
        )
        return self.steer.sensitivity

    def update(
        self,
        hands: list[HandResult],
        *,
        enabled: bool,
        dt: float = REF_DT,
    ) -> CarControlState:
        dt = clamp_dt(dt)
        reliable = [h for h in hands if h.score >= self.MIN_HAND_SCORE]
        required = 2 if self.mode == ControlMode.WHEEL else 1

        if enabled and len(reliable) >= required:
            self._last_reliable = reliable
            self._tracking_gap = 0.0
        elif enabled and self._last_reliable:
            self._tracking_gap += dt
            if self._tracking_gap <= self.TRACKING_GRACE_SECONDS:
                reliable = self._last_reliable
            else:
                self._last_reliable = []
        else:
            self._last_reliable = []
            self._tracking_gap = 0.0

        if not enabled or len(reliable) < required:
            smooth_steer = self.steer.decay(dt)
            self._decay_pedals(dt)
            return self._state(
                smooth_steer,
                len(hands),
                active=False,
            )

        if self.mode == ControlMode.WHEEL:
            raw = self._wheel_controls(reliable, dt)
        else:
            raw = self._pointer_controls(reliable[0], dt)

        self._smooth_throttle = lerp(self._smooth_throttle, raw["throttle"], 0.28, dt)
        self._smooth_brake = lerp(self._smooth_brake, raw["brake"], 0.32, dt)

        return self._state(
            raw["steer"],
            len(hands),
            active=True,
            left_palm=raw.get("left_palm", "—"),
            right_palm=raw.get("right_palm", "—"),
            left_openness=raw.get("left_openness", 0.0),
            right_openness=raw.get("right_openness", 0.0),
        )

    def _state(
        self,
        steer: float,
        hands_visible: int,
        *,
        active: bool,
        left_palm: str = "—",
        right_palm: str = "—",
        left_openness: float = 0.0,
        right_openness: float = 0.0,
    ) -> CarControlState:
        return CarControlState(
            steer=steer,
            throttle=self._smooth_throttle,
            brake=self._smooth_brake,
            mode=self.mode,
            hands_visible=hands_visible,
            active=active,
            steer_label=self._steer_text(steer),
            pedal_label=self._pedal_text(self._smooth_throttle, self._smooth_brake),
            left_palm=left_palm,
            right_palm=right_palm,
            left_openness=left_openness,
            right_openness=right_openness,
        )

    def _wheel_controls(self, hands: list[HandResult], dt: float) -> dict:
        left, right = self._pick_left_right(hands)
        left_read = self.palms.analyze(left, dt)
        right_read = self.palms.analyze(right, dt)

        lp = self._palm_center(left.landmarks)
        rp = self._palm_center(right.landmarks)
        angle = math.atan2(rp[1] - lp[1], rp[0] - lp[0])
        steer = self.steer.from_wheel_angle(angle, self.WHEEL_MAX_ANGLE, dt)

        throttle, brake = self._pedals_from_palms(left_read.state, right_read.state, dt)

        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "left_palm": left_read.label,
            "right_palm": right_read.label,
            "left_openness": left_read.openness,
            "right_openness": right_read.openness,
        }

    def _pointer_controls(self, hand: HandResult, dt: float) -> dict:
        reading = self.palms.analyze(hand, dt)
        palm_x = float(np.mean([hand.landmarks[i][0] for i in PALM_LANDMARKS]))
        steer = self.steer.from_pointer_x(palm_x, self.STEER_GAIN, dt)

        throttle, brake = self._pedals_from_palms(reading.state, reading.state, dt)

        label = reading.label
        side = hand.handedness
        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "left_palm": label if side == "Left" else "—",
            "right_palm": label if side == "Right" else "—",
            "left_openness": reading.openness if side == "Left" else 0.0,
            "right_openness": reading.openness if side == "Right" else 0.0,
        }

    def _pedals_from_palms(
        self, left: PalmState, right: PalmState, dt: float
    ) -> tuple[float, float]:
        if left == PalmState.CLOSED and right == PalmState.CLOSED:
            self._gas_time += dt
            self._brake_time = 0.0
        elif left == PalmState.OPEN and right == PalmState.OPEN:
            self._brake_time += dt
            self._gas_time = 0.0
        else:
            self._gas_time = max(0.0, self._gas_time - dt)
            self._brake_time = max(0.0, self._brake_time - dt)

        throttle = 1.0 if self._gas_time >= self.PEDAL_CONFIRM_SECONDS else 0.0
        brake = 1.0 if self._brake_time >= self.PEDAL_CONFIRM_SECONDS else 0.0
        if brake > 0.0:
            throttle = 0.0
        return throttle, brake

    @staticmethod
    def _palm_center(lms: np.ndarray) -> np.ndarray:
        return np.mean([lms[i][:2] for i in PALM_LANDMARKS], axis=0)

    @staticmethod
    def _pick_left_right(hands: list[HandResult]) -> tuple[HandResult, HandResult]:
        left = next((h for h in hands if h.handedness == "Left"), None)
        right = next((h for h in hands if h.handedness == "Right"), None)
        if left and right:
            return left, right
        ordered = sorted(hands, key=lambda h: h.landmarks[WRIST][0])
        return ordered[0], ordered[-1]

    def _decay_pedals(self, dt: float) -> None:
        self._smooth_throttle = lerp(self._smooth_throttle, 0.0, 0.5, dt)
        self._smooth_brake = lerp(self._smooth_brake, 0.0, 0.5, dt)
        self._gas_time = 0.0
        self._brake_time = 0.0

    @staticmethod
    def _steer_text(steer: float) -> str:
        if steer < -0.28:
            return "LEFT"
        if steer > 0.28:
            return "RIGHT"
        return "CENTER"

    @staticmethod
    def _pedal_text(throttle: float, brake: float) -> str:
        if brake > 0.45:
            return "BRAKE"
        if throttle > 0.45:
            return "FULL SPEED"
        return "COAST"
