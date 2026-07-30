"""Send arrow / WASD keys to the focused window (browser games)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from pynput.keyboard import Controller, Key

if TYPE_CHECKING:
    from gesture_car.driver import CarControlState


class KeyScheme(Enum):
    ARROWS = auto()
    WASD = auto()


@dataclass
class KeyboardDriver:
    scheme: KeyScheme = KeyScheme.ARROWS
    steer_enter: float = 0.30
    steer_exit: float = 0.16
    pedal_threshold: float = 0.35
    # Partial steering is pulsed instead of held, so light turns stay gentle.
    pulse_period: float = 0.16
    min_duty: float = 0.30
    full_hold: float = 0.80
    _keyboard: Controller = field(default_factory=Controller)
    _pressed: set[str] = field(default_factory=set)
    _steer_side: str | None = None

    def _bindings(self) -> dict[str, str | Key]:
        if self.scheme == KeyScheme.WASD:
            return {"left": "a", "right": "d", "up": "w", "down": "s"}
        return {
            "left": Key.left,
            "right": Key.right,
            "up": Key.up,
            "down": Key.down,
        }

    def apply(self, state: "CarControlState", *, enabled: bool) -> set[str]:
        desired: set[str] = set()
        if enabled and state.active:
            desired |= self._steer_keys(state.steer)
            if state.throttle > self.pedal_threshold:
                desired.add("up")
            if state.brake > self.pedal_threshold:
                desired.add("down")
        else:
            self._steer_side = None

        bindings = self._bindings()
        for name in list(self._pressed):
            if name not in desired:
                self._release(bindings[name])
                self._pressed.discard(name)

        for name in desired:
            if name not in self._pressed:
                self._press(bindings[name])
                self._pressed.add(name)

        return desired.copy()

    def _steer_keys(self, steer: float) -> set[str]:
        """Hysteresis picks the direction; duty cycling sets the turn strength."""
        if self._steer_side == "left" and steer > -self.steer_exit:
            self._steer_side = None
        elif self._steer_side == "right" and steer < self.steer_exit:
            self._steer_side = None

        if self._steer_side is None:
            if steer <= -self.steer_enter:
                self._steer_side = "left"
            elif steer >= self.steer_enter:
                self._steer_side = "right"

        if self._steer_side is None:
            return set()
        return {self._steer_side} if self._pulse_on(abs(steer)) else set()

    def _pulse_on(self, magnitude: float) -> bool:
        if magnitude >= self.full_hold:
            return True
        duty = max(self.min_duty, magnitude / self.full_hold)
        phase = (time.perf_counter() % self.pulse_period) / self.pulse_period
        return phase < duty

    def release_all(self) -> None:
        bindings = self._bindings()
        for name in list(self._pressed):
            self._release(bindings[name])
        self._pressed.clear()
        self._steer_side = None

    def _press(self, key) -> None:
        try:
            self._keyboard.press(key)
        except Exception:
            pass

    def _release(self, key) -> None:
        try:
            self._keyboard.release(key)
        except Exception:
            pass
