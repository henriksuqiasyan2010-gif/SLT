"""Main window: tabs (Single word, Sentence, Learn), menu, the worker thread and its signals."""

import logging

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QMessageBox, QTabWidget, QWidget

from asl import config
from asl.app import settings as settings_io
from asl.app import strings
from asl.app.settings_dialog import SettingsDialog
from asl.app.speech import Speaker
from asl.app.strings import tr
from asl.app.widgets import ResultPanel, VideoView
from asl.app.worker import PipelineWorker
from asl.capture import LiveSource

log = logging.getLogger(__name__)

VIDEO_STRETCH, PANEL_STRETCH = 3, 2
WORKER_STOP_TIMEOUT_MS = 5000


def show_error(parent, text: str) -> None:
    """Every error dialog of the app goes through here (tests replace QMessageBox.critical)."""
    QMessageBox.critical(parent, tr("error_title"), text)


class MainWindow(QMainWindow):
    speech_failed = Signal(str)  # emitted from the speech thread, shown in the UI thread

    def __init__(self, settings, model, vocab: dict, source_factory=None, settings_path=config.SETTINGS_JSON,
                 prediction_log=None, fps_tag: str = "app", fps_log_path=config.FPS_LOG, start: bool = True):
        """source_factory() -> LandmarkSource; default: the camera from the settings (LiveSource)."""
        super().__init__()
        self.settings, self.model, self.vocab = settings, model, vocab
        self.settings_path = settings_path
        self.source_factory = source_factory
        self.prediction_log, self.fps_tag, self.fps_log_path = prediction_log, fps_tag, fps_log_path
        self.thread = self.worker = None
        self.last_prediction = None
        self.speaker = Speaker(on_error=self.speech_failed.emit)
        self.speech_failed.connect(lambda d: self.statusBar().showMessage(tr("error_speech", details=d)))

        self.video = VideoView()
        self.panel = ResultPanel(vocab)
        single = QWidget()
        lay = QHBoxLayout(single)
        lay.addWidget(self.video, VIDEO_STRETCH)
        lay.addWidget(self.panel, PANEL_STRETCH)
        self.placeholders = [QLabel(), QLabel()]
        for label in self.placeholders:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs = QTabWidget()
        self.tabs.addTab(single, "")
        for label in self.placeholders:
            self.tabs.addTab(label, "")
        self.setCentralWidget(self.tabs)

        self.menu_file = self.menuBar().addMenu("")
        self.act_exit = QAction(self)
        self.act_exit.triggered.connect(self.close)
        self.menu_file.addAction(self.act_exit)
        self.menu_settings = self.menuBar().addMenu("")
        self.act_settings = QAction(self)
        self.act_settings.triggered.connect(self.open_settings)
        self.menu_settings.addAction(self.act_settings)
        self.menu_help = self.menuBar().addMenu("")
        self.act_about = QAction(self)
        self.act_about.triggered.connect(self.show_about)
        self.menu_help.addAction(self.act_about)

        self.panel.btn_record.clicked.connect(self.record)
        self.panel.btn_speak.clicked.connect(self.speak)
        self.shortcut_record = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_record.activated.connect(self.record)
        self.panel.btn_record.setEnabled(model is not None)
        self.retranslate()
        if start:
            self.start_worker()

    # --- worker ---------------------------------------------------------------------------
    def _make_source(self):
        if self.source_factory is not None:
            return self.source_factory()
        return LiveSource(camera_index=self.settings.camera_index)

    def start_worker(self) -> None:
        self.video.set_message("hint_camera_starting")
        self.thread = QThread(self)
        self.worker = PipelineWorker(self._make_source, self.model, lambda: self.settings, self.prediction_log,
                                     self.fps_tag, self.fps_log_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.prediction_ready.connect(self.on_prediction)
        self.worker.camera_failed.connect(self.on_camera_failed)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.thread.quit()
        if not self.thread.wait(WORKER_STOP_TIMEOUT_MS):
            log.warning("worker thread did not stop within %d ms", WORKER_STOP_TIMEOUT_MS)
        self.worker = self.thread = None

    # --- slots ----------------------------------------------------------------------------
    def record(self) -> None:
        if self.model is not None and self.worker is not None:
            self.worker.request_record()

    def on_frame(self, image, overlay: dict) -> None:
        if self.sender() is not self.worker:  # a frame from a worker that is being replaced
            return
        self.video.set_frame(image, overlay)
        self.panel.btn_record.setEnabled(self.model is not None and overlay.get("phase") == "idle")

    def on_prediction(self, prediction) -> None:
        self.last_prediction = prediction
        self.panel.show_prediction(prediction)
        if prediction.accepted and self.settings.sound:
            self.speak()

    def speak(self) -> None:
        pred = self.last_prediction
        if pred is not None and pred.accepted:
            self.speaker.speak(self.panel.words(pred.top[0][0])["en"])

    def on_camera_failed(self, details: str) -> None:
        self.video.set_message("hint_camera_missing")
        self.panel.btn_record.setEnabled(False)
        show_error(self, tr("error_camera", index=self.settings.camera_index, details=details))

    def on_failed(self, details: str) -> None:
        show_error(self, tr("error_internal", details=details))

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != SettingsDialog.DialogCode.Accepted:
            return
        self.apply_settings(dlg.settings())

    def apply_settings(self, new) -> None:
        old, self.settings = self.settings, new
        try:
            settings_io.save(new, self.settings_path)
        except OSError as e:
            log.exception("cannot save settings")
            show_error(self, tr("error_internal", details=e))
        if new.language != old.language:
            strings.set_language(new.language)
            self.retranslate()
        if new.camera_index != old.camera_index:
            self.stop_worker()
            self.start_worker()

    def show_about(self) -> None:
        m = self.model.metrics if self.model is not None else {}
        orig = m.get("orig", {})
        pct = (lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—")
        QMessageBox.about(self, tr("about_title"), tr("about_text", n_signs=config.NUM_CLASSES,
                                                     model=m.get("preset", "—"), top1=pct(orig.get("top1")),
                                                     top3=pct(orig.get("top3"))))

    def retranslate(self) -> None:
        self.setWindowTitle(tr("app_title"))
        for i, key in enumerate(("tab_single", "tab_sentence", "tab_learn")):
            self.tabs.setTabText(i, tr(key))
        for label in self.placeholders:
            label.setText(tr("tab_coming_soon"))
        self.menu_file.setTitle(tr("menu_file"))
        self.act_exit.setText(tr("menu_exit"))
        self.menu_settings.setTitle(tr("menu_settings"))
        self.act_settings.setText(tr("menu_settings_open"))
        self.menu_help.setTitle(tr("menu_help"))
        self.act_about.setText(tr("menu_about"))
        self.panel.retranslate()
        self.video.update()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self.stop_worker()
        super().closeEvent(event)
