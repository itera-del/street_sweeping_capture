from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class VideoPreview:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.duration_sec: float = 0.0
        self.effective_fps: float = 25.0

    def open(self, path: str, *, duration_sec: float, fps: float) -> bool:
        self.close()
        self.duration_sec = float(duration_sec)
        self.effective_fps = float(fps) if fps > 0 else 25.0
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False
        cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if cap_fps > 0:
            self.effective_fps = cap_fps
        return True

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def seek_seconds(self, t: float) -> None:
        if self.cap is None:
            return
        t = max(0.0, min(t, self.duration_sec))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)

    def read_frame_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def step_frames(self, delta_frames: int) -> None:
        if self.cap is None or delta_frames == 0:
            return
        cur_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        dt = delta_frames / self.effective_fps
        self.seek_seconds(cur_ms / 1000.0 + dt)
