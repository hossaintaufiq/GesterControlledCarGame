"""Accurate palm open/closed detection with hysteresis and debouncing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from gesture_car.tracker import HandResult

WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP = 16
RING_PIP = 14
RING_MCP = 13
PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17

FINGERS = (
    (THUMB_TIP, THUMB_IP, THUMB_MCP),
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
)


class PalmState(Enum):
    NEUTRAL = auto()
    CLOSED = auto()
    OPEN = auto()


@dataclass
class PalmReading:
    state: PalmState
    openness: float
    confidence: float
    label: str


@dataclass
class _HandMemory:
    state: PalmState = PalmState.NEUTRAL
    closed_streak: int = 0
    open_streak: int = 0
    openness_ema: float = 0.5


@dataclass
class PalmAnalyzer:
    """Scale-invariant palm classifier with temporal stability."""

    min_hand_score: float = 0.62
    closed_enter: float = 0.28
    closed_exit: float = 0.40
    open_enter: float = 0.72
    open_exit: float = 0.60
    confirm_frames: int = 4
    openness_smooth: float = 0.35
    _memory: dict[str, _HandMemory] = field(default_factory=dict)

    def analyze(self, hand: HandResult) -> PalmReading:
        key = hand.handedness
        mem = self._memory.setdefault(key, _HandMemory())

        if hand.score < self.min_hand_score:
            return PalmReading(
                state=mem.state,
                openness=mem.openness_ema,
                confidence=hand.score,
                label=self._label(mem.state),
            )

        raw_open = self._measure_openness(hand.landmarks)
        mem.openness_ema = self._lerp(mem.openness_ema, raw_open, self.openness_smooth)
        openness = mem.openness_ema

        if mem.state == PalmState.CLOSED:
            if openness >= self.closed_exit:
                mem.closed_streak = 0
                if openness >= self.open_enter:
                    mem.open_streak += 1
                    if mem.open_streak >= self.confirm_frames:
                        mem.state = PalmState.OPEN
                        mem.open_streak = 0
                else:
                    mem.open_streak = 0
                    if openness <= self.closed_enter:
                        mem.state = PalmState.NEUTRAL
            else:
                mem.closed_streak += 1
                mem.open_streak = 0

        elif mem.state == PalmState.OPEN:
            if openness <= self.open_exit:
                mem.open_streak = 0
                if openness <= self.closed_enter:
                    mem.closed_streak += 1
                    if mem.closed_streak >= self.confirm_frames:
                        mem.state = PalmState.CLOSED
                        mem.closed_streak = 0
                else:
                    mem.closed_streak = 0
                    if openness >= self.open_enter:
                        mem.state = PalmState.NEUTRAL
            else:
                mem.open_streak += 1
                mem.closed_streak = 0

        else:
            if openness <= self.closed_enter:
                mem.closed_streak += 1
                mem.open_streak = 0
                if mem.closed_streak >= self.confirm_frames:
                    mem.state = PalmState.CLOSED
                    mem.closed_streak = 0
            elif openness >= self.open_enter:
                mem.open_streak += 1
                mem.closed_streak = 0
                if mem.open_streak >= self.confirm_frames:
                    mem.state = PalmState.OPEN
                    mem.open_streak = 0
            else:
                mem.closed_streak = max(0, mem.closed_streak - 1)
                mem.open_streak = max(0, mem.open_streak - 1)

        return PalmReading(
            state=mem.state,
            openness=openness,
            confidence=hand.score,
            label=self._label(mem.state),
        )

    def reset(self) -> None:
        self._memory.clear()

    @staticmethod
    def _measure_openness(lms: np.ndarray) -> float:
        """0 = fully closed fist, 1 = fully open spread palm."""
        palm_size = float(np.linalg.norm(lms[MIDDLE_MCP][:2] - lms[WRIST][:2]))
        palm_size = max(palm_size, 0.04)

        extensions: list[float] = []
        for tip, pip, mcp in FINGERS:
            tip_dist = float(np.linalg.norm(lms[tip][:2] - lms[WRIST][:2]))
            mcp_dist = float(np.linalg.norm(lms[mcp][:2] - lms[WRIST][:2]))
            pip_dist = float(np.linalg.norm(lms[pip][:2] - lms[WRIST][:2]))

            ratio = tip_dist / max(mcp_dist, 0.02)
            fold = pip_dist / max(tip_dist, 0.02)
            ext = float(np.clip((ratio - 0.95) / 0.55, 0.0, 1.0))
            if fold > 0.92:
                ext *= 0.35
            extensions.append(ext)

        spread = float(np.linalg.norm(lms[INDEX_TIP][:2] - lms[PINKY_TIP][:2]))
        spread_norm = float(np.clip((spread / palm_size - 0.55) / 0.85, 0.0, 1.0))

        finger_open = float(np.mean(extensions))
        return float(np.clip(finger_open * 0.78 + spread_norm * 0.22, 0.0, 1.0))

    @staticmethod
    def _label(state: PalmState) -> str:
        if state == PalmState.CLOSED:
            return "CLOSED"
        if state == PalmState.OPEN:
            return "OPEN"
        return "NEUTRAL"

    @staticmethod
    def _lerp(current: float, target: float, alpha: float) -> float:
        return current * (1.0 - alpha) + target * alpha
