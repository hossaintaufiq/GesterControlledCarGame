"""Map hand poses to car control signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

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


class GestureDriver:
    STEER_GAIN = 1.85
    WHEEL_MAX_ANGLE = 0.48
    MIN_HAND_SCORE = 0.62
    PEDAL_CONFIRM = 3

    def __init__(self) -> None:
        self.mode = ControlMode.WHEEL
        self.palms = PalmAnalyzer(min_hand_score=self.MIN_HAND_SCORE)
        self.steer = SteerSmoother()
        self._smooth_throttle = 0.0
        self._smooth_brake = 0.0
        self._gas_streak = 0
        self._brake_streak = 0

    def set_mode(self, mode: ControlMode) -> None:
        self.mode = mode
        self.palms.reset()
        self.steer.reset()
        self._gas_streak = 0
        self._brake_streak = 0

    def update(self, hands: list[HandResult], *, enabled: bool) -> CarControlState:
        reliable = [h for h in hands if h.score >= self.MIN_HAND_SCORE]

        if not enabled or not reliable:
            smooth_steer = self.steer.decay()
            self._decay_pedals()
            return self._state(
                smooth_steer,
                len(hands),
                active=False,
            )

        if self.mode == ControlMode.WHEEL and len(reliable) >= 2:
            raw = self._wheel_controls(reliable)
        else:
            raw = self._pointer_controls(reliable[0])

        self._smooth_throttle = self._lerp(self._smooth_throttle, raw["throttle"], 0.28)
        self._smooth_brake = self._lerp(self._smooth_brake, raw["brake"], 0.32)

        return self._state(
            raw["steer"],
            len(hands),
            active=True,
            left_palm=raw.get("left_palm", "—"),
            right_palm=raw.get("right_palm", "—"),
        )

    def _state(
        self,
        steer: float,
        hands_visible: int,
        *,
        active: bool,
        left_palm: str = "—",
        right_palm: str = "—",
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
        )

    def _wheel_controls(self, hands: list[HandResult]) -> dict:
        left, right = self._pick_left_right(hands)
        left_read = self.palms.analyze(left)
        right_read = self.palms.analyze(right)

        lp = self._palm_center(left.landmarks)
        rp = self._palm_center(right.landmarks)
        angle = math.atan2(rp[1] - lp[1], rp[0] - lp[0])
        steer = self.steer.from_wheel_angle(angle, self.WHEEL_MAX_ANGLE)

        throttle, brake = self._pedals_from_palms(left_read.state, right_read.state)

        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "left_palm": left_read.label,
            "right_palm": right_read.label,
        }

    def _pointer_controls(self, hand: HandResult) -> dict:
        reading = self.palms.analyze(hand)
        palm_x = float(np.mean([hand.landmarks[i][0] for i in PALM_LANDMARKS]))
        steer = self.steer.from_pointer_x(palm_x, self.STEER_GAIN)

        throttle, brake = 0.0, 0.0
        if reading.state == PalmState.CLOSED:
            self._gas_streak += 1
            self._brake_streak = 0
            if self._gas_streak >= self.PEDAL_CONFIRM:
                throttle = 1.0
        elif reading.state == PalmState.OPEN:
            self._brake_streak += 1
            self._gas_streak = 0
            if self._brake_streak >= self.PEDAL_CONFIRM:
                brake = 1.0
        else:
            self._gas_streak = max(0, self._gas_streak - 1)
            self._brake_streak = max(0, self._brake_streak - 1)

        label = reading.label
        side = hand.handedness
        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "left_palm": label if side == "Left" else "—",
            "right_palm": label if side == "Right" else "—",
        }

    def _pedals_from_palms(self, left: PalmState, right: PalmState) -> tuple[float, float]:
        both_closed = left == PalmState.CLOSED and right == PalmState.CLOSED
        both_open = left == PalmState.OPEN and right == PalmState.OPEN

        if both_closed:
            self._gas_streak += 1
            self._brake_streak = 0
        elif both_open:
            self._brake_streak += 1
            self._gas_streak = 0
        else:
            self._gas_streak = max(0, self._gas_streak - 1)
            self._brake_streak = max(0, self._brake_streak - 1)

        throttle = 1.0 if self._gas_streak >= self.PEDAL_CONFIRM else 0.0
        brake = 1.0 if self._brake_streak >= self.PEDAL_CONFIRM else 0.0
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

    def _decay_pedals(self) -> None:
        self._smooth_throttle = self._lerp(self._smooth_throttle, 0.0, 0.5)
        self._smooth_brake = self._lerp(self._smooth_brake, 0.0, 0.5)
        self._gas_streak = 0
        self._brake_streak = 0

    @staticmethod
    def _lerp(current: float, target: float, alpha: float) -> float:
        return current * (1.0 - alpha) + target * alpha

    @staticmethod
    def _steer_text(steer: float) -> str:
        if steer < -0.15:
            return "LEFT"
        if steer > 0.15:
            return "RIGHT"
        return "CENTER"

    @staticmethod
    def _pedal_text(throttle: float, brake: float) -> str:
        if brake > 0.45:
            return "BRAKE"
        if throttle > 0.45:
            return "FULL SPEED"
        return "COAST"
