"""UI colors and drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np

BG = (18, 20, 24)
TEXT = (236, 238, 240)
MUTED = (150, 156, 164)
ACCENT = (90, 200, 255)
SUCCESS = (100, 220, 140)
WARN = (80, 180, 255)
DANGER = (80, 80, 240)


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color=TEXT,
    weight: int = 1,
) -> None:
    cv2.putText(
        img, text, (org[0] + 1, org[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA,
    )


def panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.65) -> None:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = BG
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, dst=roi)
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (60, 66, 72), 1, cv2.LINE_AA)
