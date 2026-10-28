"""Background worker (runs in a QThread): landmark source -> preview image + overlay state ->
clip -> classification -> prediction log. The UI thread only draws what the signals carry; the
model runs here, once per clip.

Modes (set_mode from the UI):
  single, sentence_manual  the button / Space starts countdown + fixed-length recording
                           (SingleWordRecorder); the segmenter is not used at all
  sentence_auto            every frame goes to the segmenter; each finished segment is classified
"""

import logging
import time

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from asl import config, skeleton
from asl.app.pipeline import Clip, HintTracker, PredictionLog, SingleWordRecorder, classify, frame_status
from asl.capture import CameraError, recording_aspect
from asl.fpslog import FpsLog
from asl.segmenter import Segmenter

log = logging.getLogger(__name__)

BLANK_RGB = (32, 34, 38)  # background when the source has no camera image (recordings)
MODES = ("single", "sentence_manual", "sentence_auto")


def to_qimage(bgr: np.ndarray) -> QImage:
    """BGR uint8 (H, W, 3) -> QImage that owns its pixels (safe to hand to the UI thread)."""
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


class PipelineWorker(QObject):
    frame_ready = Signal(QImage, dict)  # preview with skeleton; overlay: phase, count/left, hints, fps
    prediction_ready = Signal(object)  # pipeline.Prediction
    camera_failed = Signal(str)  # details
    failed = Signal(str)  # unexpected error, details
    finished = Signal()

    def __init__(self, source_factory, model, settings_fn, prediction_log: PredictionLog = None,
                 fps_tag: str = "app", fps_log_path=config.FPS_LOG, clock=time.perf_counter):
        """source_factory() -> LandmarkSource (opened inside the worker thread);
        settings_fn() -> current AppSettings (read on every use, so changes apply at once)."""
        super().__init__()
        self.source_factory, self.model, self.settings_fn = source_factory, model, settings_fn
        self.prediction_log = prediction_log or PredictionLog()
        self.fps_tag, self.fps_log_path, self.clock = fps_tag, fps_log_path, clock
        self.source = None
        self.recorder = SingleWordRecorder()
        self.segmenter = None  # created with the first frame's aspect ratio
        self.hints = HintTracker()
        self.mode = "single"
        self._mode_requested = None
        self._record_requested = False
        self._stop = False
        self.fps = 0.0

    def request_record(self) -> None:
        """Called from the UI thread; picked up with the next frame (ignored in sentence_auto)."""
        self._record_requested = True

    def set_mode(self, mode: str) -> None:
        """Called from the UI thread; applied with the next frame (a running recording or an open
        segment of the old mode is dropped)."""
        if mode not in MODES:
            raise ValueError(mode)
        self._mode_requested = mode

    def stop(self) -> None:
        self._stop = True
        if self.source is not None:
            self.source.close()

    def _preview(self, frame, mirrored: bool) -> np.ndarray:
        import cv2

        if frame.image is not None:
            img = skeleton.draw_cv2(frame.image.copy(), frame.landmarks)
            mirror_input = getattr(self.source, "mirror_input", config.MIRROR_INPUT)
            return cv2.flip(img, 1) if mirrored != mirror_input else img
        w = int(self._capture_info().get("frame_width") or config.CAMERA_WIDTH)
        h = int(self._capture_info().get("frame_height") or config.CAMERA_HEIGHT)
        img = skeleton.draw_cv2(np.full((h, w, 3), BLANK_RGB[::-1], np.uint8), frame.landmarks)
        return cv2.flip(img, 1) if mirrored else img

    def _capture_info(self) -> dict:
        """Capture settings saved with every attempt (camera settings or the replayed recording)."""
        if hasattr(self.source, "settings"):
            return self.source.settings()
        meta = getattr(self.source, "meta", {})
        keys = ("frame_width", "frame_height", "width", "height", "camera_index", "backend", "model_complexity")
        return {"replayed": str(getattr(self.source, "path", "")), **{k: meta[k] for k in keys if k in meta}}

    def _aspect(self, frame) -> float:
        if frame.image is not None:
            return frame.image.shape[1] / frame.image.shape[0]
        return recording_aspect(getattr(self.source, "meta", {}))

    @Slot()
    def run(self) -> None:
        fps_log = None
        try:
            self.source = self.source_factory()
            if self._stop:
                return
            fps_log = FpsLog(self._capture_info, self.fps_tag, self.fps_log_path)
            last = None
            for frame in self.source:
                if self._stop:
                    break
                now = self.clock()
                if last is not None and now > last:
                    inst = 1.0 / (now - last)
                    self.fps = inst if not self.fps else (1 - config.FPS_SMOOTHING) * self.fps + config.FPS_SMOOTHING * inst
                last = now
                status = frame_status(frame.landmarks)
                hints = self.hints.update(status, now)
                if frame.image is not None:
                    fps_log.tick(status.hands, frame.image.shape)
                settings = self.settings_fn()
                if self._mode_requested is not None:
                    self.mode, self._mode_requested = self._mode_requested, None
                    self.recorder.cancel()
                    if self.segmenter is not None:
                        self.segmenter.reset()
                if self.mode == "sentence_auto":
                    self._record_requested = False
                    clip, overlay = self._auto_step(frame)
                else:
                    if self._record_requested:
                        self._record_requested = False
                        if self.model is not None:
                            self.recorder.start(now, settings.record_seconds)
                    clip = self.recorder.feed(frame.landmarks, frame.timestamp, now)
                    overlay = self.recorder.overlay(now)
                overlay = {**overlay, "mode": self.mode, "hints": hints, "fps": self.fps}
                self.frame_ready.emit(to_qimage(self._preview(frame, settings.mirrored_preview)), overlay)
                if clip is not None:
                    self._classify(clip, self._aspect(frame), settings)
            else:  # the source ended (a replayed recording): a sign still open is classified too
                if self.mode == "sentence_auto" and self.segmenter is not None and not self._stop:
                    seg = self.segmenter.flush()
                    if seg is not None:
                        self._classify(Clip(seg.landmarks, seg.timestamps), self.segmenter.aspect, self.settings_fn())
        except CameraError as e:
            log.exception("camera failed")
            self.camera_failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - reported to the user, details in logs/app.log
            log.exception("worker failed")
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            if fps_log is not None:
                fps_log.close()
            if self.source is not None:
                self.source.close()
            self.finished.emit()

    def _auto_step(self, frame):
        """sentence_auto: feed the segmenter; a finished segment becomes a Clip."""
        if self.model is None:
            return None, {"phase": "seg_idle"}
        if self.segmenter is None:
            self.segmenter = Segmenter(self._aspect(frame))
        seg = self.segmenter.update(frame.landmarks, frame.timestamp)
        clip = Clip(seg.landmarks, seg.timestamps) if seg is not None else None
        return clip, {"phase": f"seg_{self.segmenter.state.value}"}

    def _classify(self, clip: Clip, aspect: float, settings) -> None:
        pred = classify(self.model, clip, aspect, settings.threshold)
        pred.mode = self.mode
        pred.stream_t = float(clip.timestamps[-1]) if len(clip.timestamps) else 0.0
        try:
            self.prediction_log.append(pred, clip, self.mode, self.model.metrics.get("preset", "?"), self._capture_info())
        except OSError:
            log.exception("cannot write the prediction log")
        log.info("prediction %s accepted=%s frames=%d fps=%.1f", pred.top, pred.accepted, pred.n_frames, pred.fps)
        self.prediction_ready.emit(pred)
