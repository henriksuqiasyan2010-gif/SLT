"""Main window ("Фиолетовый вечер"): the header (logo, tabs, HY / RU / EN, settings, help), the live
video on the left (Single word, Sentence) and the page of the current tab; the Learn page takes the
whole width. The current tab (and the Sentence "manual" toggle) sets the worker mode: single,
sentence_manual (button, no segmenter), sentence_auto (segmenter) or learn (button; the attempt is
scored against the selected sign and counted in progress.json)."""

import csv
import logging
from dataclasses import replace

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from asl import config
from asl.app import settings as settings_io
from asl.app import strings, theme
from asl.app.header import Header, TabSwitch
from asl.app.learn import LearnPanel, attempt_display_clip
from asl.app.pipeline import target_score
from asl.app.progress import Progress, defense_set, export_defense_set
from asl.app.sentence import Sentence, log_correction
from asl.app.settings_dialog import SettingsDialog
from asl.app.strings import tr
from asl.app.widgets import PANEL_WIDTH, ResultPanel, SentencePanel, VideoView
from asl.app.worker import PipelineWorker
from asl.audio import AudioPlayer
from asl.capture import LiveSource, recording_aspect
from asl.reference import ReferenceLibrary

log = logging.getLogger(__name__)

WORKER_STOP_TIMEOUT_MS = 5000
TAB_SINGLE, TAB_SENTENCE, TAB_LEARN = 0, 1, 2
STATUS_MS = 3000  # status bar messages
PAGE_MARGINS = (26, 22, 26, 22)
GAP = 22
QWIDGETSIZE_MAX = 16777215
PANEL_MAX_WIDTH = int(PANEL_WIDTH * 1.25)  # a long translation may widen the right column this much


def show_error(parent, text: str) -> None:
    """Every error dialog of the app goes through here (tests replace QMessageBox.critical)."""
    QMessageBox.critical(parent, tr("error_title"), text)


def about_box(window) -> QMessageBox:
    """"About SLT": the logo, what the app does, the model's accuracy, the data credit."""
    m = window.model.metrics if window.model is not None else {}
    orig = m.get("orig", {})
    pct = (lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—")  # noqa: E731
    box = QMessageBox(window)
    box.setWindowTitle(tr("about_title"))
    box.setIconPixmap(theme.logo_pixmap())
    box.setText(tr("about_text", n_signs=config.NUM_CLASSES, model=m.get("preset", "—"),
                   top1=pct(orig.get("top1")), top3=pct(orig.get("top3"))))
    return box


class MainWindow(QMainWindow):
    speech_failed = Signal(str)  # emitted from the audio thread, shown in the UI thread

    def __init__(self, settings, model, vocab: dict, source_factory=None, settings_path=config.SETTINGS_JSON,
                 prediction_log=None, fps_tag: str = "app", fps_log_path=config.FPS_LOG, start: bool = True,
                 corrections_log=config.CORRECTIONS_LOG, progress=None, references=None,
                 defense_set_path=config.DEFENSE_SET_CSV, audio=None):
        """source_factory() -> LandmarkSource; default: the camera from the settings (LiveSource)."""
        super().__init__()
        self.settings, self.model, self.vocab = settings, model, vocab
        self.settings_path = settings_path
        self.source_factory = source_factory
        self.prediction_log, self.fps_tag, self.fps_log_path = prediction_log, fps_tag, fps_log_path
        self.corrections_log = corrections_log
        self.progress = progress if progress is not None else Progress.load()
        self.references = references if references is not None else ReferenceLibrary()
        self.defense_set_path = defense_set_path
        self.thread = self.worker = None
        self.last_prediction = None
        self._idle = None  # last record-button state: changed only when it changes (the buttons have shadows)
        self.audio = audio if audio is not None else AudioPlayer(vocab, on_error=self.speech_failed.emit)
        self.speech_failed.connect(lambda d: self.statusBar().showMessage(tr("error_speech", details=d)))

        self.header = Header()
        self.video = VideoView()
        self.panel = ResultPanel(vocab)
        self.sentence = Sentence()
        self.sentence_panel = SentencePanel(vocab, self.sentence)
        self.learn_panel = LearnPanel(vocab, self.progress, self.references)
        self.stack = QStackedWidget()
        for page in (self.panel, self.sentence_panel, self.learn_panel):
            self.stack.addWidget(page)
        self.tabs = TabSwitch(self.header.tabs, self.stack, self)
        self.header.tabs.set_current("tab_single")
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(*PAGE_MARGINS)
        outer.setSpacing(GAP)
        outer.addWidget(self.header)
        content = QHBoxLayout()
        content.setSpacing(GAP)
        content.addWidget(self.video, 1)
        content.addWidget(self.stack)
        outer.addLayout(content, 1)
        self.setCentralWidget(root)
        self.setWindowIcon(theme.app_icon())

        self.header.lang.set_current(settings.language)
        self.header.lang.changed.connect(lambda lang: self.apply_settings(replace(self.settings, language=lang)))
        self.header.btn_settings.clicked.connect(self.open_settings)
        self.header.btn_help.clicked.connect(self.show_about)
        self.panel.btn_record.clicked.connect(self.record)
        self.panel.btn_speak.clicked.connect(self.speak)
        sp = self.sentence_panel
        sp.btn_record.clicked.connect(self.record)
        sp.btn_delete.clicked.connect(self.sentence_delete_last)
        sp.btn_clear.clicked.connect(self.sentence_clear)
        sp.btn_speak.clicked.connect(self.sentence_speak)
        sp.btn_copy.clicked.connect(self.sentence_copy)
        for b in sp.alt_buttons:
            b.clicked.connect(lambda _=False, b=b: self.sentence_replace(b.property("sign")))
        sp.manual.toggled.connect(lambda _: self.update_mode())
        self.learn_panel.btn_try.clicked.connect(self.record)
        self.learn_panel.btn_export.clicked.connect(self.export_defense_set)
        self.tabs.currentChanged.connect(self.on_tab)
        self.shortcut_record = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_record.activated.connect(self.record)
        self.set_record_enabled(model is not None)
        self.on_tab(TAB_SINGLE)
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
        self.worker.set_mode(self.mode())
        self.thread.start()

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.thread.quit()
        if not self.thread.wait(WORKER_STOP_TIMEOUT_MS):
            log.warning("worker thread did not stop within %d ms", WORKER_STOP_TIMEOUT_MS)
        self.worker = self.thread = None

    # --- modes ----------------------------------------------------------------------------
    def mode(self) -> str:
        if self.tabs.currentIndex() == TAB_SENTENCE:
            return "sentence_manual" if self.sentence_panel.manual.isChecked() else "sentence_auto"
        if self.tabs.currentIndex() == TAB_LEARN:
            return "learn"
        return "single"

    def update_mode(self) -> None:
        if self.worker is not None:
            self.worker.set_mode(self.mode())

    def on_tab(self, index: int) -> None:
        learn = index == TAB_LEARN
        self.video.setVisible(not learn)  # Learn has its own players
        self.stack.setMinimumWidth(0 if learn else PANEL_WIDTH)
        self.stack.setMaximumWidth(QWIDGETSIZE_MAX if learn else PANEL_MAX_WIDTH)
        self.update_mode()

    # --- slots ----------------------------------------------------------------------------
    def record(self) -> None:
        """Button or Space: a fixed-length recording in Single word, manual Sentence and Learn."""
        if self.model is None or self.worker is None:
            return
        if self.mode() == "learn":
            if self.learn_panel.sign is None:
                return
            self.worker.target = self.learn_panel.sign
        if self.mode() in ("single", "sentence_manual", "learn"):
            self.worker.request_record()

    def set_record_enabled(self, on: bool) -> None:
        self._idle = on
        self.panel.btn_record.setEnabled(on)
        self.sentence_panel.btn_record.setEnabled(on)
        self.learn_panel.btn_try.setEnabled(on and self.learn_panel.sign is not None)

    def on_frame(self, payload, overlay: dict) -> None:
        if self.sender() is not self.worker:  # a frame from a worker that is being replaced
            return
        if self.tabs.currentIndex() == TAB_LEARN:
            self.learn_panel.show_live(payload, overlay)
        else:
            self.video.set_frame(payload, overlay, self.settings.mirrored_preview)
        idle = self.model is not None and overlay.get("phase") == "idle"
        if idle != self._idle:
            self.set_record_enabled(idle)

    def on_prediction(self, prediction) -> None:
        if prediction.mode == "learn":
            self.on_learn_attempt(prediction)
            return
        if prediction.mode == "single":
            self.last_prediction = prediction
            self.panel.show_prediction(prediction)
            if prediction.accepted and self.settings.sound:
                self.speak()
            return
        self.sentence.add(prediction, prediction.stream_t)
        self.sentence_panel.status_key = None if prediction.accepted else "sent_not_recognized"
        self.sentence_panel.refresh()

    def on_learn_attempt(self, prediction) -> None:
        sign = prediction.target
        if sign is None:
            return
        words = self.panel.words
        if not prediction.top:  # no hand in the recording: not counted as an attempt
            self.learn_panel.show_attempt(tr("learn_result_none"), "error")
            return
        p, rank = target_score(prediction, self.model.labels, sign)
        success = self.progress.record(sign, p, rank, self.settings.threshold)
        word, pct = words(sign)["en"], f"{p:.0%}"
        if success:
            text, tone = "✓ " + tr("learn_result_ok", word=word, p=pct), "success"
        elif rank == 1:
            text, tone = tr("learn_result_low", word=word, p=pct, threshold=f"{self.settings.threshold:.0%}"), "error"
        else:
            other, p_other = prediction.top[0]
            text = tr("learn_result_other", other=words(other)["en"], p_other=f"{p_other:.0%}", word=word, p=pct,
                      rank=rank)
            tone = "error"
        self.learn_panel.show_attempt(text, tone, attempt_display_clip(prediction), self._attempt_aspect())

    def _attempt_aspect(self) -> float:
        """Frame width / height of the live source (camera or replayed recording)."""
        src = self.worker.source if self.worker is not None else None
        if src is not None and getattr(src, "frame_size", None):
            return src.frame_size[0] / src.frame_size[1]
        return recording_aspect(getattr(src, "meta", {})) if src is not None else config.CAMERA_ASPECT

    def export_defense_set(self) -> None:
        preset = self.model.metrics.get("preset", "") if self.model is not None else ""
        per_sign = config.REPORTS_DIR / preset / f"{preset}_T32" / "per_sign_accuracy.csv"
        val = {}
        if preset and per_sign.exists():
            with open(per_sign, encoding="utf-8") as f:
                val = {r["sign"]: round(float(r["accuracy"]), 3) for r in csv.DictReader(f)}
        try:
            n = export_defense_set(defense_set(self.progress, self.vocab, val), self.defense_set_path)
        except OSError as e:
            log.exception("defense set export")
            show_error(self, tr("error_internal", details=e))
            return
        self.statusBar().showMessage(tr("learn_exported", n=n, path=self.defense_set_path), STATUS_MS * 2)

    def sentence_delete_last(self) -> None:
        self.sentence.delete_last()
        self.sentence_panel.status_key = None
        self.sentence_panel.refresh()

    def sentence_clear(self) -> None:
        self.sentence.clear()
        self.sentence_panel.status_key = None
        self.sentence_panel.refresh()

    def sentence_replace(self, sign: str) -> None:
        word = self.sentence.replace_last(sign)
        if word is None:
            return
        try:
            log_correction(word, self.mode(), self.corrections_log)
        except OSError:
            log.exception("cannot write the corrections log")
        self.sentence_panel.refresh()

    def sentence_speak(self) -> None:
        self.audio.play_sentence(self.sentence.signs())

    def sentence_copy(self) -> None:
        QGuiApplication.clipboard().setText(self.sentence.text(self.vocab, self.sentence_panel.word_lang))
        self.statusBar().showMessage(tr("sent_copied"), STATUS_MS)

    def speak(self) -> None:
        pred = self.last_prediction
        if pred is not None and pred.accepted:
            self.audio.play_word(pred.top[0][0])

    def on_camera_failed(self, details: str) -> None:
        self.video.set_message("hint_camera_missing")
        self.set_record_enabled(False)
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
            self.header.lang.set_current(new.language)
            self.retranslate()
        if new.camera_index != old.camera_index:
            self.stop_worker()
            self.start_worker()

    def show_about(self) -> None:
        about_box(self).exec()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("app_title"))
        self.header.retranslate()
        self.panel.retranslate()
        self.sentence_panel.retranslate()
        self.learn_panel.retranslate()
        self.video.update()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self.stop_worker()
        super().closeEvent(event)
