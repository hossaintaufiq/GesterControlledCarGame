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
    pending: PalmState = PalmState.NEUTRAL
    pending_frames: int = 0
    openness_ema: float = 0.5
    samples: list[float] = field(default_factory=list)


@dataclass
class PalmAnalyzer:
    """Scale-invariant palm classifier with temporal stability."""

    min_hand_score: float = 0.62
    closed_enter: float = 0.28
    closed_exit: float = 0.40
    open_enter: float = 0.72
    open_exit: float = 0.60
    confirm_frames: int = 3
    openness_smooth: float = 0.42
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
        mem.samples.append(raw_open)
        if len(mem.samples) > 5:
            mem.samples.pop(0)
        median_open = float(np.median(mem.samples))
        mem.openness_ema = self._lerp(
            mem.openness_ema, median_open, self.openness_smooth,
        )
        openness = mem.openness_ema

        if mem.state == PalmState.CLOSED and openness < self.closed_exit:
            candidate = PalmState.CLOSED
        elif mem.state == PalmState.OPEN and openness > self.open_exit:
            candidate = PalmState.OPEN
        elif openness <= self.closed_enter:
            candidate = PalmState.CLOSED
        elif openness >= self.open_enter:
            candidate = PalmState.OPEN
        else:
            candidate = PalmState.NEUTRAL

        if candidate == mem.state:
            mem.pending = candidate
            mem.pending_frames = 0
        elif candidate == mem.pending:
            mem.pending_frames += 1
            if mem.pending_frames >= self.confirm_frames:
                mem.state = candidate
                mem.pending_frames = 0
        else:
            mem.pending = candidate
            mem.pending_frames = 1

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
        """Orientation- and scale-resistant palm openness score."""
        palm_size = float(np.linalg.norm(lms[MIDDLE_MCP][:2] - lms[WRIST][:2]))
        palm_size = max(palm_size, 0.04)
        palm_center = np.mean(
            lms[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2],
            axis=0,
        )

        extensions: list[float] = []
        for tip, pip, mcp in FINGERS[1:]:
            lower = lms[pip][:2] - lms[mcp][:2]
            upper = lms[tip][:2] - lms[pip][:2]
            denom = max(float(np.linalg.norm(lower) * np.linalg.norm(upper)), 1e-6)
            straightness = float(np.dot(lower, upper) / denom)
            straight_score = float(np.clip((straightness + 0.15) / 1.05, 0.0, 1.0))

            reach = float(np.linalg.norm(lms[tip][:2] - palm_center) / palm_size)
            reach_score = float(np.clip((reach - 0.72) / 1.05, 0.0, 1.0))
            ext = straight_score * 0.58 + reach_score * 0.42
            extensions.append(ext)

        thumb_reach = float(
            np.linalg.norm(lms[THUMB_TIP][:2] - palm_center) / palm_size
        )
        thumb_score = float(np.clip((thumb_reach - 0.55) / 0.9, 0.0, 1.0))
        spread = float(np.linalg.norm(lms[INDEX_TIP][:2] - lms[PINKY_TIP][:2]))
        spread_norm = float(np.clip((spread / palm_size - 0.55) / 0.85, 0.0, 1.0))

        finger_open = float(np.mean(extensions))
        return float(np.clip(
            finger_open * 0.72 + spread_norm * 0.18 + thumb_score * 0.10,
            0.0,
            1.0,
        ))

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
