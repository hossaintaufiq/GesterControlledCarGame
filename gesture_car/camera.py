"""Threaded webcam capture that always serves the newest frame.

Capture and inference each cost ~34 ms, so running them sequentially halves
the frame rate. Grabbing in a background thread overlaps the two and drops
stale frames, which keeps input latency low.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        cap = self._open()
        if cap is None:
            return False
        self._cap = cap

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self._cap = None
            return False
        self._frame = frame
        self._seq = 1

        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True

    def _open(self) -> Optional[cv2.VideoCapture]:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def _pump(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq: int = -1) -> tuple[Optional[np.ndarray], int]:
        """Return the newest frame, or (None, seq) when nothing new arrived."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None, self._seq
            return self._frame, self._seq

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
