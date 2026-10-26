"""Landmark sources: everything after landmarks works the same with a live camera or a recording.

A source yields LandmarkFrame objects whose `landmarks` is a (543, 3) float32 array in dataset order.
Recordings are `<name>.npy` (T, 543, 3) plus a `<name>.meta.json` sidecar with timestamps and settings.
"""

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from asl import config
from asl.landmarks_io import N_LANDMARKS


class CameraError(RuntimeError):
    """The camera cannot be opened or stopped delivering frames."""


@dataclass
class LandmarkFrame:
    landmarks: np.ndarray  # (543, 3) float32, NaN = missing
    timestamp: float  # seconds since the source started
    image: Optional[np.ndarray] = None  # BGR frame as fed to Holistic (live sources only)


class LandmarkSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[LandmarkFrame]:
        ...

    def close(self) -> None:
        """Ask a running iteration to stop after the current frame."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- Live camera ---------------------------------------------------------------

def open_camera(index: int, width: int, height: int, backends=config.CAMERA_BACKENDS):
    """First backend that opens the camera and delivers a frame (see cap.getBackendName())."""
    import cv2

    ids = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
    if sys.platform != "win32":
        backends = ("any",)
    for name in backends:
        cap = cv2.VideoCapture(index, ids[name])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise CameraError(f"Camera {index} not found or busy")


class _LatestFrameReader:
    """Reads the camera in a thread and keeps only the newest frame, so Holistic never waits
    for the camera and never works on a stale queued frame."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.frame_id, self.error = None, 0, None
        self.running = True
        self.thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok or frame is None:
                    self.error = "stopped delivering frames"
                    self.running = False
                else:
                    self.frame, self.frame_id = frame, self.frame_id + 1
                self.cond.notify_all()

    def next_after(self, last_id: int, min_step: int = 1):
        """Block until a frame with id >= last_id + min_step exists; return (frame, id) or (None, id)."""
        with self.cond:
            while self.frame_id < last_id + min_step and self.error is None:
                self.cond.wait(timeout=1.0)
            if self.frame_id < last_id + min_step:
                return None, self.frame_id
            return self.frame, self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


class LiveSource(LandmarkSource):
    """Camera + Holistic. `capture_factory(index, width, height)` can be replaced in tests.

    threaded=True reads the camera in a background thread and processes the newest frame
    (frames are dropped when Holistic is slower than the camera); threaded=False processes
    every camera frame in order (deterministic, used in tests).
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX, width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT, mirror_input: bool = config.MIRROR_INPUT,
                 process_every_n: int = config.PROCESS_EVERY_N_FRAMES, holistic_kwargs=None,
                 capture_factory=open_camera, threaded: bool = True):
        self.camera_index = camera_index
        self.width, self.height = width, height
        self.mirror_input = mirror_input
        self.process_every_n = max(1, process_every_n)
        self.holistic_kwargs = holistic_kwargs or {}
        self._capture_factory = capture_factory
        self.threaded = threaded
        self.backend_name = None  # filled in when the camera is opened
        self.frame_size = None  # (width, height) actually delivered; may differ from the request
        self._stop = False

    def settings(self) -> dict:
        """Capture settings, saved into recording sidecars and the FPS log."""
        return {
            "camera_index": self.camera_index, "backend": self.backend_name,
            "width": self.width, "height": self.height,  # requested
            "frame_width": self.frame_size[0] if self.frame_size else None,  # delivered
            "frame_height": self.frame_size[1] if self.frame_size else None,
            "mirror_input": self.mirror_input, "process_every_n": self.process_every_n,
            "model_complexity": self.holistic_kwargs.get("model_complexity",
                                                         config.HOLISTIC_MODEL_COMPLEXITY),
        }

    def __iter__(self) -> Iterator[LandmarkFrame]:
        import cv2

        from asl.holistic import create_holistic, process_bgr

        self._stop = False
        cap = self._capture_factory(self.camera_index, self.width, self.height)
        self.backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else None
        holistic = create_holistic(**self.holistic_kwargs)
        reader = _LatestFrameReader(cap) if self.threaded else None
        t0 = time.perf_counter()
        n = 0
        try:
            while not self._stop:
                if reader:
                    frame, n = reader.next_after(n, self.process_every_n)
                    if frame is None:
                        raise CameraError(f"Camera {self.camera_index} {reader.error}")
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise CameraError(f"Camera {self.camera_index} stopped delivering frames")
                    n += 1
                    if (n - 1) % self.process_every_n:
                        continue
                self.frame_size = (frame.shape[1], frame.shape[0])
                if self.mirror_input:
                    frame = cv2.flip(frame, 1)
                landmarks = process_bgr(holistic, frame)
                yield LandmarkFrame(landmarks, time.perf_counter() - t0, frame)
        finally:
            if reader:
                reader.stop()
            cap.release()
            holistic.close()

    def close(self) -> None:
        self._stop = True


# --- Recordings ----------------------------------------------------------------

def meta_path(npy_path) -> Path:
    p = Path(npy_path)
    return p.with_name(p.stem + config.RECORDING_META_SUFFIX)


def save_recording(npy_path, landmarks, timestamps, **meta) -> Path:
    """Save (T, 543, 3) landmarks and a sidecar with timestamps relative to the first frame."""
    npy_path = Path(npy_path)
    clip = np.asarray(landmarks, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"expected (T, {N_LANDMARKS}, 3), got {clip.shape}")
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) != len(clip):
        raise ValueError(f"{len(ts)} timestamps for {len(clip)} frames")
    ts = ts - ts[0] if len(ts) else ts
    duration = float(ts[-1]) if len(ts) > 1 else 0.0
    fps = (len(ts) - 1) / duration if duration > 0 else config.DEFAULT_REPLAY_FPS
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, clip)
    info = {"frames": len(clip), "fps": round(fps, 3), "duration_s": round(duration, 3),
            "timestamps": [round(float(t), 4) for t in ts], **meta}
    meta_path(npy_path).write_text(json.dumps(info, indent=1), encoding="utf-8")
    return npy_path


def load_recording(npy_path):
    """(clip (T, 543, 3) float32, meta dict). Without a sidecar, timestamps assume DEFAULT_REPLAY_FPS."""
    clip = np.load(npy_path).astype(np.float32, copy=False)
    mp = meta_path(npy_path)
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if len(meta.get("timestamps", [])) != len(clip):
        meta["timestamps"] = [i / config.DEFAULT_REPLAY_FPS for i in range(len(clip))]
        meta.setdefault("fps", config.DEFAULT_REPLAY_FPS)
    return clip, meta


class RecordedSource(LandmarkSource):
    """Replays a recording; with realtime=True frames come at their original pace."""

    def __init__(self, npy_path, realtime: bool = True, speed: float = 1.0, loop: bool = False):
        self.path = Path(npy_path)
        self.clip, self.meta = load_recording(self.path)
        self.timestamps = np.asarray(self.meta["timestamps"], dtype=np.float64)
        self.realtime, self.speed, self.loop = realtime, speed, loop
        self._stop = False

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    def __iter__(self) -> Iterator[LandmarkFrame]:
        self._stop = False
        offset = 0.0
        while True:
            start = time.perf_counter()
            for frame, ts in zip(self.clip, self.timestamps):
                if self._stop:
                    return
                if self.realtime:
                    delay = start + ts / self.speed - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                yield LandmarkFrame(frame, offset + float(ts))
            if not self.loop or not len(self.clip):
                return
            frame_step = 1.0 / self.fps if self.fps else 0.0
            offset += float(self.timestamps[-1]) + frame_step

    def close(self) -> None:
        self._stop = True
