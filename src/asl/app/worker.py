"""Background worker (runs in a QThread): landmark source -> preview image + overlay state ->
countdown / recording -> classification -> prediction log. The UI thread only draws what the
signals carry; the model runs here, once per recorded clip."""

import logging
import time

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from asl import config, skeleton
from asl.app.pipeline import Clip, HintTracker, PredictionLog, SingleWordRecorder, classify, frame_status
from asl.capture import CameraError, recording_aspect
from asl.fpslog import FpsLog

log = logging.getLogger(__name__)

BLANK_RGB = (32, 34, 38)  # background when the source has no camera image (recordings)


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
        self.hints = HintTracker()
        self._record_requested = False
        self._stop = False
        self.fps = 0.0

    def request_record(self) -> None:
        """Called from the UI thread; picked up with the next frame."""
        self._record_requested = True

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
                if self._record_requested:
                    self._record_requested = False
                    if self.model is not None:
                        self.recorder.start(now, settings.record_seconds)
                clip = self.recorder.feed(frame.landmarks, frame.timestamp, now)
                overlay = {**self.recorder.overlay(now), "hints": hints, "fps": self.fps}
                self.frame_ready.emit(to_qimage(self._preview(frame, settings.mirrored_preview)), overlay)
                if clip is not None:
                    self._classify(clip, self._aspect(frame), settings)
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

    def _classify(self, clip: Clip, aspect: float, settings) -> None:
        pred = classify(self.model, clip, aspect, settings.threshold)
        try:
            self.prediction_log.append(pred, clip, "single", self.model.metrics.get("preset", "?"), self._capture_info())
        except OSError:
            log.exception("cannot write the prediction log")
        log.info("prediction %s accepted=%s frames=%d fps=%.1f", pred.top, pred.accepted, pred.n_frames, pred.fps)
        self.prediction_ready.emit(pred)
