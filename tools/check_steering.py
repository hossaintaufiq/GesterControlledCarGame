"""Report the steering response curve and pulsed key duty per hand tilt."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_car.driver import CarControlState, GestureDriver  # noqa: E402
from gesture_car.keyboard_out import KeyboardDriver  # noqa: E402
from gesture_car.steer import SteerSmoother  # noqa: E402


def settled_steer(tilt_deg: float, frames: int = 240) -> float:
    smoother = SteerSmoother()
    angle = math.radians(tilt_deg)
    value = 0.0
    for _ in range(frames):
        value = smoother.from_wheel_angle(angle, GestureDriver.WHEEL_MAX_ANGLE)
    return value


def measured_duty(steer: float, samples: int = 400) -> float:
    keyboard = KeyboardDriver()
    keyboard._press = lambda key: None  # avoid emitting real keystrokes
    keyboard._release = lambda key: None
    state = CarControlState(steer=steer, active=True)
    on = 0
    for _ in range(samples):
        if keyboard.apply(state, enabled=True):
            on += 1
        time.sleep(0.001)
    return on / samples


def main() -> int:
    print(f"wheel full-lock tilt: {math.degrees(GestureDriver.WHEEL_MAX_ANGLE):.0f} deg\n")
    print(f"{'tilt':>6} {'steer':>7} {'key':>6} {'duty':>6}")
    for tilt in (5, 10, 15, 20, 25, 30, 40, 50):
        steer = settled_steer(tilt)
        duty = measured_duty(steer)
        key = "right" if duty > 0 else "-"
        print(f"{tilt:>5}d {steer:>7.2f} {key:>6} {duty:>6.0%}")

    smoother = SteerSmoother()
    for _ in range(240):
        smoother.from_wheel_angle(math.radians(40), GestureDriver.WHEEL_MAX_ANGLE)
    for _ in range(240):
        smoother.decay()
    print(f"\nrecenter after release: {smoother.read():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
