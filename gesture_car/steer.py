"""Smooth steering filter — reduces jitter before keys are sent."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SteerSmoother:
    """Two-stage steer smoothing with circular angle handling."""

    sensitivity: float = 1.0
    output_alpha: float = 0.11
    fast_alpha: float = 0.26
    angle_alpha: float = 0.20
    pointer_alpha: float = 0.22
    deadzone: float = 0.16
    max_rate: float = 0.07

    _output: float = 0.0
    _sin_v: float = 0.0
    _cos_v: float = 1.0
    _pointer_x: float = 0.5
    _has_pointer: bool = False

    def from_wheel_angle(self, angle: float, max_angle: float) -> float:
        self._sin_v = self._lerp(self._sin_v, math.sin(angle), self.angle_alpha)
        self._cos_v = self._lerp(self._cos_v, math.cos(angle), self.angle_alpha)
        filtered = math.atan2(self._sin_v, self._cos_v) / max_angle
        return self._push_output(filtered * self.sensitivity)

    def from_pointer_x(self, palm_x: float, gain: float) -> float:
        if not self._has_pointer:
            self._pointer_x = palm_x
            self._has_pointer = True
        self._pointer_x = self._lerp(self._pointer_x, palm_x, self.pointer_alpha)
        target = (self._pointer_x - 0.5) * gain * self.sensitivity
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
        target = self._apply_deadzone(max(-1.0, min(1.0, target)))
        delta = target - self._output
        # Suppress small landmark jitter while following deliberate turns quickly.
        movement = min(1.0, abs(delta) / 0.45)
        alpha = self.output_alpha + (self.fast_alpha - self.output_alpha) * movement
        next_output = self._lerp(self._output, target, alpha)
        step = next_output - self._output
        if abs(step) > self.max_rate:
            next_output = self._output + math.copysign(self.max_rate, step)
        self._output = next_output
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
