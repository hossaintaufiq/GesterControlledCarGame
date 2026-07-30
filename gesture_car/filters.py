"""Frame-rate independent smoothing helpers.

All smoothing constants in this project are tuned against a 30 FPS reference.
Converting them with the real frame delta keeps the control feel identical
whether the app runs at 15 or 60 FPS.
"""

from __future__ import annotations

REF_DT = 1.0 / 30.0
MIN_DT = 1.0 / 240.0
MAX_DT = 1.0 / 5.0


def clamp_dt(dt: float) -> float:
    return min(MAX_DT, max(MIN_DT, dt))


def ema_alpha(reference_alpha: float, dt: float) -> float:
    """Rescale a per-frame EMA alpha tuned at 30 FPS to the current frame time."""
    alpha = min(max(reference_alpha, 1e-6), 0.999999)
    return 1.0 - (1.0 - alpha) ** (clamp_dt(dt) / REF_DT)


def lerp(current: float, target: float, reference_alpha: float, dt: float) -> float:
    alpha = ema_alpha(reference_alpha, dt)
    return current * (1.0 - alpha) + target * alpha
