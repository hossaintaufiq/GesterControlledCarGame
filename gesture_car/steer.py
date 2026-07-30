"""Smooth steering filter — reduces jitter before keys are sent."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gesture_car.filters import REF_DT, clamp_dt, ema_alpha


@dataclass
class SteerSmoother:
    """Two-stage steer smoothing with circular angle handling."""

    sensitivity: float = 1.0
    output_alpha: float = 0.11
    fast_alpha: float = 0.26
    angle_alpha: float = 0.20
    pointer_alpha: float = 0.22
    deadzone: float = 0.16
    # Steering units per second (0.07 per frame at the 30 FPS reference).
    max_rate: float = 2.1

    _output: float = 0.0
    _sin_v: float = 0.0
    _cos_v: float = 1.0
    _pointer_x: float = 0.5
    _has_pointer: bool = False

    def from_wheel_angle(self, angle: float, max_angle: float, dt: float = REF_DT) -> float:
        alpha = ema_alpha(self.angle_alpha, dt)
        self._sin_v = self._blend(self._sin_v, math.sin(angle), alpha)
        self._cos_v = self._blend(self._cos_v, math.cos(angle), alpha)
        filtered = math.atan2(self._sin_v, self._cos_v) / max_angle
        return self._push_output(filtered * self.sensitivity, dt)

    def from_pointer_x(self, palm_x: float, gain: float, dt: float = REF_DT) -> float:
        if not self._has_pointer:
            self._pointer_x = palm_x
            self._has_pointer = True
        self._pointer_x = self._blend(
            self._pointer_x, palm_x, ema_alpha(self.pointer_alpha, dt)
        )
        target = (self._pointer_x - 0.5) * gain * self.sensitivity
        return self._push_output(target, dt)

    def read(self) -> float:
        return self._output

    def decay(self, dt: float = REF_DT) -> float:
        return self._push_output(0.0, dt)

    def reset(self) -> None:
        self._output = 0.0
        self._sin_v = 0.0
        self._cos_v = 1.0
        self._pointer_x = 0.5
        self._has_pointer = False

    def _push_output(self, target: float, dt: float) -> float:
        target = self._apply_deadzone(max(-1.0, min(1.0, target)))
        delta = target - self._output
        # Suppress small landmark jitter while following deliberate turns quickly.
        movement = min(1.0, abs(delta) / 0.45)
        reference = self.output_alpha + (self.fast_alpha - self.output_alpha) * movement
        next_output = self._blend(self._output, target, ema_alpha(reference, dt))

        step = next_output - self._output
        limit = self.max_rate * clamp_dt(dt)
        if abs(step) > limit:
            next_output = self._output + math.copysign(limit, step)
        self._output = next_output
        return self._output

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return sign * max(0.0, min(1.0, scaled))

    @staticmethod
    def _blend(current: float, target: float, alpha: float) -> float:
        return current * (1.0 - alpha) + target * alpha
