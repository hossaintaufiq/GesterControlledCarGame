"""MediaPipe hand tracking with temporal smoothing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_car.filters import REF_DT, ema_alpha


@dataclass
class HandResult:
    landmarks: np.ndarray  # (21, 3) normalized x, y, z
    handedness: str
    score: float
    track_id: int = 0


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.32) -> None:
        self.alpha = alpha
        self._state: dict[str, np.ndarray] = {}

    def apply(self, key: str, landmarks: np.ndarray, dt: float = REF_DT) -> np.ndarray:
        prev = self._state.get(key)
        if prev is None:
            smooth = landmarks.copy()
        else:
            delta = float(np.mean(np.abs(landmarks[:, :2] - prev[:, :2])))
            blend = ema_alpha(min(0.82, self.alpha + delta * 2.0), dt)
            smooth = prev * (1.0 - blend) + landmarks * blend
        self._state[key] = smooth
        return smooth

    def prune(self, active: set[str]) -> None:
        for key in list(self._state.keys()):
            if key not in active:
                del self._state[key]


class HandTracker:
    def __init__(self, model_path: Path, max_hands: int = 2) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing model: {model_path}\nRun: python download_model.py"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.68,
            min_hand_presence_confidence=0.62,
            min_tracking_confidence=0.62,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._smoother = LandmarkSmoother()
        self._start_ms = int(time.perf_counter() * 1000)
        self._last_ts = -1

    def process(
        self,
        frame_bgr: np.ndarray,
        *,
        mirrored: bool = True,
        infer_max_width: int = 640,
        dt: float = REF_DT,
    ) -> list[HandResult]:
        import cv2
        from mediapipe import Image as MpImage, ImageFormat

        h, w = frame_bgr.shape[:2]
        if w > infer_max_width:
            scale = infer_max_width / float(w)
            small = cv2.resize(
                frame_bgr,
                (infer_max_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

        ts = int(time.perf_counter() * 1000) - self._start_ms
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            self._smoother.prune(set())
            return []

        hands: list[HandResult] = []
        seen: set[str] = set()
        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            handedness = "Right"
            score = 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = float(cat.score)

            if mirrored:
                handedness = "Left" if handedness == "Right" else "Right"

            key = handedness if handedness not in seen else f"{handedness}_{i}"
            seen.add(key)
            pts = self._smoother.apply(key, pts, dt)
            hands.append(
                HandResult(
                    landmarks=pts,
                    handedness=handedness,
                    score=score,
                    track_id=i,
                )
            )

        self._smoother.prune(seen)
        hands.sort(key=lambda hand: hand.score, reverse=True)
        return hands

    def close(self) -> None:
        self._landmarker.close()
