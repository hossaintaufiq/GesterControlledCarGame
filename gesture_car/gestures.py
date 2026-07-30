"""Palm open/closed detection with hysteresis and debouncing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from gesture_car.tracker import HandResult

WRIST = 0
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

# (tip, pip) pairs for the four fingers that reliably distinguish fist vs palm.
CURL_PAIRS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

# Distance ratios (tip-to-wrist / pip-to-wrist) measured for curled vs straight.
CURLED_RATIO = 1.05
STRAIGHT_RATIO = 1.38


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
    """Rotation- and scale-invariant palm classifier with temporal stability."""

    # Handedness score only tells us left-vs-right certainty, so it must stay
    # low here or angled hands stop updating their gesture entirely.
    min_hand_score: float = 0.30
    closed_enter: float = 0.25
    closed_exit: float = 0.42
    open_enter: float = 0.72
    open_exit: float = 0.55
    confirm_frames: int = 2
    median_window: int = 5
    openness_smooth: float = 0.45
    _memory: dict[str, _HandMemory] = field(default_factory=dict)

    def analyze(self, hand: HandResult) -> PalmReading:
        mem = self._memory.setdefault(hand.handedness, _HandMemory())

        if hand.score < self.min_hand_score:
            return PalmReading(
                state=mem.state,
                openness=mem.openness_ema,
                confidence=hand.score,
                label=self._label(mem.state),
            )

        mem.samples.append(self._measure_openness(hand.landmarks))
        if len(mem.samples) > self.median_window:
            mem.samples.pop(0)
        median_open = float(np.median(mem.samples))
        mem.openness_ema = self._lerp(mem.openness_ema, median_open, self.openness_smooth)
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
        """0 = fist, 1 = open palm, independent of hand size and rotation."""
        wrist = lms[WRIST][:2]

        span = STRAIGHT_RATIO - CURLED_RATIO
        scores: list[float] = []
        for tip, pip in CURL_PAIRS:
            tip_dist = float(np.linalg.norm(lms[tip][:2] - wrist))
            pip_dist = float(np.linalg.norm(lms[pip][:2] - wrist))
            ratio = tip_dist / max(pip_dist, 1e-6)
            scores.append(float(np.clip((ratio - CURLED_RATIO) / span, 0.0, 1.0)))

        return float(np.clip(float(np.mean(scores)), 0.0, 1.0))

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
