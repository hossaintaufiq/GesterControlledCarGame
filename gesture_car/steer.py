"""Smooth steering filter — reduces jitter before keys are sent."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SteerSmoother:
    """Two-stage steer smoothing with circular angle handling."""

    output_alpha: float = 0.16
    angle_alpha: float = 0.18
    pointer_alpha: float = 0.20
    deadzone: float = 0.09
    max_rate: float = 0.12  # max change per frame

    _output: float = 0.0
    _sin_v: float = 0.0
    _cos_v: float = 1.0
    _pointer_x: float = 0.5
    _has_pointer: bool = False

    def from_wheel_angle(self, angle: float, max_angle: float) -> float:
        target = max(-1.0, min(1.0, angle / max_angle))
        self._sin_v = self._lerp(self._sin_v, math.sin(angle), self.angle_alpha)
        self._cos_v = self._lerp(self._cos_v, math.cos(angle), self.angle_alpha)
        filtered = math.atan2(self._sin_v, self._cos_v) / max_angle
        filtered = max(-1.0, min(1.0, filtered))
        return self._push_output(filtered)

    def from_pointer_x(self, palm_x: float, gain: float) -> float:
        if not self._has_pointer:
            self._pointer_x = palm_x
            self._has_pointer = True
        self._pointer_x = self._lerp(self._pointer_x, palm_x, self.pointer_alpha)
        target = max(-1.0, min(1.0, (self._pointer_x - 0.5) * gain))
        return self._push_output(target)

    def read(self) -> float:
        return self._output

    def decay(self) -> float:
        return self._push_output(0.0)

    def reset(self) -> None:
        self._output = 0.0
        self._sin_v = 0.0
        self._cos_v = 1.0
        self._pointer_x = 0.5
        self._has_pointer = False

    def _push_output(self, target: float) -> float:
        target = self._apply_deadzone(target)
        delta = target - self._output
        if abs(delta) > self.max_rate:
            target = self._output + math.copysign(self.max_rate, delta)
        self._output = self._lerp(self._output, target, self.output_alpha)
        return self._output

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return sign * max(0.0, min(1.0, scaled))

    @staticmethod
    def _lerp(current: float, target: float, alpha: float) -> float:
        return current * (1.0 - alpha) + target * alpha
