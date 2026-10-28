"""Sign start / end detection in a continuous landmark stream (Sentence mode).

States: IDLE (hands down) -> SIGN (the higher wrist above SEG_WRIST_UP for SEG_START_S)
-> PAUSE (wrist down, waiting SEG_PAUSE_S to be sure) -> IDLE, and the segment is emitted.
A short drop of the wrist inside a sign (shorter than SEG_PAUSE_S) does not split it.

The segmenter is a pure function of the stream (frames + timestamps), so it is tested offline on
the recorded sessions; the app feeds it frame by frame. Wrist height comes from the pose model,
which estimates the wrists even when the hand itself is not found; hand presence is checked later
(classify() needs MIN_HAND_S of visible hand), and still phases inside a sign are cut by trim_idle.
"""

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from asl import config
from asl.landmarks_io import POSE

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


class State(enum.Enum):
    IDLE = "idle"
    SIGN = "sign"
    PAUSE = "pause"


@dataclass
class Params:
    wrist_up: float = config.SEG_WRIST_UP
    start_s: float = config.SEG_START_S
    pause_s: float = config.SEG_PAUSE_S
    min_sign_s: float = config.SEG_MIN_SIGN_S
    max_sign_s: float = config.SEG_MAX_SIGN_S
    preroll_s: float = config.SEG_PREROLL_S


@dataclass
class Segment:
    landmarks: np.ndarray  # (T, 543, 3)
    timestamps: np.ndarray  # (T,)
    start: int = -1  # frame indices in the stream (offline use)
    end: int = -1  # exclusive

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0


def wrist_height(frame: np.ndarray, aspect: float, shoulders=None) -> Tuple[float, Optional[tuple]]:
    """Height of the higher wrist above the shoulder line in shoulder widths (+ = above) and the
    shoulders used (mid y, width). Missing shoulders -> the last known ones (`shoulders`)."""
    pose = frame[POSE, :2].astype(np.float64)
    pose[:, 1] /= aspect
    sh = pose[[L_SHOULDER, R_SHOULDER]]
    if not np.isnan(sh).any():
        width = float(np.linalg.norm(sh[0] - sh[1]))
        if width > config.FEATURES_MIN_SCALE:
            shoulders = (float(sh[:, 1].mean()), width)
    wrist_y = np.nanmin(pose[[L_WRIST, R_WRIST], 1]) if not np.isnan(pose[[L_WRIST, R_WRIST], 1]).all() else np.nan
    if shoulders is None or np.isnan(wrist_y):
        return float("nan"), shoulders
    return (shoulders[0] - wrist_y) / shoulders[1], shoulders


class Segmenter:
    """Streaming: update(frame, t) -> a finished Segment or None; `state` for the overlay."""

    def __init__(self, aspect: float = config.CAMERA_ASPECT, params: Params = None):
        self.aspect, self.p = aspect, params or Params()
        self.state = State.IDLE
        self.shoulders = None
        self.frames, self.stamps, self.indices = [], [], []
        self.pre = deque()  # (frame, t, index) before the start, kept for preroll_s
        self.t_up = self.t_down = None  # since when the wrist is up (IDLE) / down (SIGN, PAUSE)
        self.n = 0  # frames seen

    def reset(self) -> None:
        self.__init__(self.aspect, self.p)

    def update(self, frame: np.ndarray, t: float) -> Optional[Segment]:
        height, self.shoulders = wrist_height(frame, self.aspect, self.shoulders)
        up = bool(height >= self.p.wrist_up)  # NaN -> False
        i, self.n = self.n, self.n + 1
        out = None
        if self.state is State.IDLE:
            self.pre.append((frame, t, i))
            if up:
                self.t_up = t if self.t_up is None else self.t_up
                if t - self.t_up >= self.p.start_s:
                    self.state = State.SIGN
                    keep = [x for x in self.pre if x[1] >= self.t_up - self.p.preroll_s]
                    self.frames, self.stamps, self.indices = [list(z) for z in zip(*keep)]
                    self.pre.clear()
            else:
                self.t_up = None
            while self.pre and self.pre[0][1] < t - self.p.preroll_s - self.p.start_s - 1.0:
                self.pre.popleft()
            return None
        # SIGN or PAUSE: frames are collected while up; the down tail is not part of the sign
        if up:
            self.state, self.t_down = State.SIGN, None
            self.frames.append(frame)
            self.stamps.append(t)
            self.indices.append(i)
            if t - self.stamps[0] >= self.p.max_sign_s:
                out = self._close()
        else:
            self.t_down = t if self.t_down is None else self.t_down
            self.state = State.PAUSE
            if t - self.t_down >= self.p.pause_s:
                out = self._close()
        return out

    def _close(self) -> Optional[Segment]:
        seg = Segment(np.stack(self.frames).astype(np.float32), np.asarray(self.stamps, np.float64),
                      self.indices[0], self.indices[-1] + 1)
        self.state, self.t_up, self.t_down = State.IDLE, None, None
        self.frames, self.stamps, self.indices = [], [], []
        return seg if seg.duration >= self.p.min_sign_s else None

    def flush(self) -> Optional[Segment]:
        """End of the stream: close a sign that is still open."""
        return self._close() if self.state is not State.IDLE and self.frames else None


def segment_clip(clip: np.ndarray, timestamps, aspect: float, params: Params = None) -> List[Segment]:
    """Offline: all segments of a recorded stream."""
    seg = Segmenter(aspect, params)
    out = [s for s in (seg.update(f, float(t)) for f, t in zip(clip, timestamps)) if s is not None]
    last = seg.flush()
    return out + ([last] if last is not None else [])
