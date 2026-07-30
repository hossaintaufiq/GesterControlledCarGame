"""Sanity check for the palm openness metric using synthetic hand poses."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_car.gestures import PalmAnalyzer, PalmState  # noqa: E402
from gesture_car.tracker import HandResult  # noqa: E402

# Finger chains as (mcp, pip, dip, tip) landmark indices.
CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
MCP = ((-0.22, -0.12), (-0.12, -0.30), (0.0, -0.34), (0.11, -0.31), (0.21, -0.26))
SEGMENTS = (
    (0.10, 0.08, 0.07),
    (0.13, 0.10, 0.07),
    (0.14, 0.11, 0.07),
    (0.12, 0.10, 0.07),
    (0.10, 0.08, 0.06),
)


def build_hand(curl_deg: float, *, rotation_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """curl_deg 0 = straight fingers, 90 = tightly curled fist."""
    lms = np.zeros((21, 3), dtype=np.float32)
    for (mcp, pip, dip, tip), base_xy, seg in zip(CHAINS, MCP, SEGMENTS):
        point = np.array(base_xy, dtype=np.float32)
        lms[mcp, :2] = point
        heading = -math.pi / 2  # fingers point up (negative y)
        for idx, length in zip((pip, dip, tip), seg):
            heading += math.radians(curl_deg)
            point = point + np.array(
                [math.cos(heading) * length, math.sin(heading) * length],
                dtype=np.float32,
            )
            lms[idx, :2] = point

    rot = math.radians(rotation_deg)
    matrix = np.array(
        [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]],
        dtype=np.float32,
    )
    lms[:, :2] = lms[:, :2] @ matrix.T * scale + np.array([0.5, 0.5], dtype=np.float32)
    return lms


def settle(
    analyzer: PalmAnalyzer,
    lms: np.ndarray,
    frames: int = 15,
    dt: float = 1 / 30,
) -> tuple[PalmState, float]:
    reading = None
    for _ in range(frames):
        reading = analyzer.analyze(HandResult(lms, "Right", 0.9), dt)
    assert reading is not None
    return reading.state, reading.openness


def main() -> int:
    cases = [
        ("open palm", 0.0, 0.0, 1.0, PalmState.OPEN),
        ("open palm rotated 35deg", 0.0, 35.0, 1.0, PalmState.OPEN),
        ("open palm far/small", 0.0, 0.0, 0.55, PalmState.OPEN),
        ("open palm close/large", 0.0, -20.0, 1.5, PalmState.OPEN),
        ("relaxed hand", 30.0, 0.0, 1.0, PalmState.NEUTRAL),
        ("fist", 85.0, 0.0, 1.0, PalmState.CLOSED),
        ("fist rotated 40deg", 85.0, 40.0, 1.0, PalmState.CLOSED),
        ("fist far/small", 85.0, 0.0, 0.55, PalmState.CLOSED),
    ]

    failures = 0
    for name, curl, rotation, scale, expected in cases:
        analyzer = PalmAnalyzer()
        state, openness = settle(analyzer, build_hand(curl, rotation_deg=rotation, scale=scale))
        ok = expected is None or state == expected
        failures += 0 if ok else 1
        flag = "ok " if ok else "FAIL"
        want = "-" if expected is None else expected.name
        print(f"{flag} {name:26s} openness={openness:.2f} state={state.name:7s} expected={want}")

    print("\nFrame-rate independence (same pose, different FPS)")
    for fps in (15, 30, 60):
        analyzer = PalmAnalyzer()
        state, openness = settle(
            analyzer, build_hand(0.0), frames=int(fps * 0.5), dt=1.0 / fps
        )
        print(f"  {fps:>3} fps -> openness={openness:.2f} state={state.name}")

    print("\nfailures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
