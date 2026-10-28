"""Single word mode without Qt: frame checks, countdown + recording, classification, prediction log.

The Qt worker (asl.app.worker) drives these objects frame by frame; tests drive them with
RecordedSource, so the whole path is checked without a camera or a window.
"""

import csv
import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from asl import config, features
from asl.capture import save_recording
from asl.landmarks_io import POSE


@dataclass
class FrameStatus:
    hands: bool  # at least one hand found
    shoulders: bool  # both shoulders found and inside the image


def frame_status(landmarks: np.ndarray) -> FrameStatus:
    left, right = features.hand_presence(landmarks[None])
    sh = landmarks[POSE][[11, 12], :2]
    shoulders = not np.isnan(sh).any() and bool(((sh >= 0) & (sh <= 1)).all())
    return FrameStatus(hands=bool(left[0] or right[0]), shoulders=shoulders)


class HintTracker:
    """Hint keys (asl.app.strings) to show over the video: a part missing for longer than `delay`."""

    def __init__(self, delay: float = config.HINT_MISSING_S):
        self.delay = delay
        self.last_hands = self.last_shoulders = None

    def update(self, status: FrameStatus, now: float) -> List[str]:
        if self.last_hands is None:  # first frame: nothing has been missing for long yet
            self.last_hands = self.last_shoulders = now
        if status.hands:
            self.last_hands = now
        if status.shoulders:
            self.last_shoulders = now
        hints = []
        if now - self.last_shoulders > self.delay:
            hints.append("hint_no_shoulders")
        if now - self.last_hands > self.delay:
            hints.append("hint_no_hands")
        return hints


class Phase(enum.Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"


@dataclass
class Clip:
    landmarks: np.ndarray  # (T, 543, 3)
    timestamps: np.ndarray  # (T,) seconds

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0

    @property
    def fps(self) -> float:
        """Same definition as capture.save_recording(): (frames - 1) / duration."""
        return (len(self.timestamps) - 1) / self.duration if self.duration > 0 else config.DEFAULT_REPLAY_FPS


class SingleWordRecorder:
    """IDLE -> start() -> COUNTDOWN (countdown_s) -> RECORDING (record_s) -> Clip -> IDLE.
    `now` is any monotonic clock in seconds (frame timestamps in tests, perf_counter live)."""

    def __init__(self, countdown_s: float = config.SINGLE_COUNTDOWN_S):
        self.countdown_s = countdown_s
        self.phase = Phase.IDLE
        self.record_s = config.SINGLE_RECORD_SECONDS
        self.t_record = 0.0  # when the recording starts
        self.frames, self.stamps = [], []

    def start(self, now: float, record_s: float) -> bool:
        if self.phase is not Phase.IDLE:
            return False
        self.phase, self.record_s = Phase.COUNTDOWN, record_s
        self.t_record = now + self.countdown_s
        return True

    def cancel(self) -> None:
        self.phase, self.frames, self.stamps = Phase.IDLE, [], []

    def feed(self, landmarks: np.ndarray, timestamp: float, now: float) -> Optional[Clip]:
        if self.phase is Phase.COUNTDOWN and now >= self.t_record:
            self.phase, self.frames, self.stamps = Phase.RECORDING, [], []
        if self.phase is not Phase.RECORDING:
            return None
        self.frames.append(landmarks)
        self.stamps.append(timestamp)
        if now - self.t_record < self.record_s:
            return None
        clip = Clip(np.stack(self.frames).astype(np.float32), np.asarray(self.stamps, np.float64))
        self.cancel()
        return clip

    def overlay(self, now: float) -> dict:
        """What to draw over the video: countdown number or seconds of recording left."""
        if self.phase is Phase.COUNTDOWN:
            return {"phase": "countdown", "count": max(1, int(np.ceil(self.t_record - now)))}
        if self.phase is Phase.RECORDING:
            return {"phase": "recording", "left": max(0.0, self.record_s - (now - self.t_record))}
        return {"phase": "idle"}


@dataclass
class Prediction:
    top: List[Tuple[str, float]]  # best first; empty if the clip had no usable frames
    accepted: bool  # top-1 probability >= threshold
    threshold: float
    n_frames: int
    fps: float
    duration_s: float
    kept_s: float  # length after trim_idle
    hand_share: float  # share of recorded frames with a hand
    time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    attempt: Optional[str] = None  # file name of the saved attempt in logs/attempts/

    @property
    def sign(self) -> Optional[str]:
        return self.top[0][0] if self.accepted else None


def classify(model, clip: Clip, aspect: float, threshold: float, top_k: int = config.TOP_K) -> Prediction:
    """Live path of SignModel.predict_clip (trim_idle -> keep_moving_hand -> extract -> model)."""
    lm = clip.landmarks
    left, right = features.hand_presence(lm) if len(lm) else (np.zeros(0, bool), np.zeros(0, bool))
    hand_frames = int((left | right).sum())
    hand_share = hand_frames / len(lm) if len(lm) else 0.0
    info = dict(threshold=threshold, n_frames=len(lm), fps=round(clip.fps, 2), duration_s=round(clip.duration, 3),
                hand_share=round(hand_share, 3))
    if len(lm) < config.MIN_CLIP_FRAMES or hand_frames < max(1, np.ceil(config.MIN_HAND_S * clip.fps)):
        return Prediction(top=[], accepted=False, kept_s=0.0, **info)
    start, stop = features.trim_idle(lm, clip.fps, aspect)
    probs = model.predict_clip(lm, clip.fps, aspect)
    top = model.top_k(probs, top_k)
    return Prediction(top=top, accepted=top[0][1] >= threshold, kept_s=round((stop - start) / clip.fps, 2), **info)


class PredictionLog:
    """Every prediction -> one row of logs/predictions.csv; the recorded landmarks are kept in
    logs/attempts/ (same format as recordings/) so attempts can be scored again with later models."""

    FIELDS = ["time", "mode", "top1", "p1", "top2", "p2", "top3", "p3", "accepted", "threshold", "frames", "fps",
              "duration_s", "kept_s", "hand_share", "model", "features_version", "attempt"]

    RECORDING_KEYS = ("frames", "fps", "duration_s", "timestamps")  # written by save_recording itself

    def __init__(self, path=config.PREDICTIONS_LOG, attempts_dir=config.ATTEMPTS_DIR):
        self.path, self.attempts_dir = path, attempts_dir

    def append(self, pred: Prediction, clip: Clip, mode: str, model_name: str, capture: dict) -> None:
        capture = {k: v for k, v in capture.items() if k not in self.RECORDING_KEYS}
        if len(clip.landmarks):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            npy = self.attempts_dir / f"{mode}_{stamp}.npy"
            save_recording(npy, clip.landmarks, clip.timestamps, mode=mode, predicted=pred.top[0][0] if pred.top else None,
                           accepted=pred.accepted, model=model_name, **capture)
            pred.attempt = npy.name
        row = {"time": pred.time, "mode": mode, "accepted": pred.accepted, "threshold": pred.threshold,
               "frames": pred.n_frames, "fps": pred.fps, "duration_s": pred.duration_s, "kept_s": pred.kept_s,
               "hand_share": pred.hand_share, "model": model_name,
               "features_version": features.FEATURES_VERSION, "attempt": pred.attempt}
        for i in range(config.TOP_K):
            sign, p = pred.top[i] if i < len(pred.top) else ("", "")
            row[f"top{i + 1}"], row[f"p{i + 1}"] = sign, round(p, 4) if p != "" else ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
