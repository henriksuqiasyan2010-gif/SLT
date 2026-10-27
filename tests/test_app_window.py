"""pytest-qt: the window opens, records from a replayed recording, survives a missing camera,
switches the interface language, and the UI font can draw every Armenian letter we show."""

import csv
import glob
import sys
import threading
import types

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asl import config
from asl.app import strings
from asl.app.fonts import armenian_font_family, missing_glyphs
from asl.app.main_window import MainWindow
from asl.app.pipeline import PredictionLog
from asl.app.settings import AppSettings
from asl.app.settings_dialog import SettingsDialog
from asl.app.speech import Speaker
from asl.capture import CameraError, LiveSource, RecordedSource
from asl.vocab import load_vocab

WAIT_MS = 15000


class FakeModel:
    metrics = {"preset": "fake", "orig": {"top1": 0.627, "top3": 0.769}}

    def predict_clip(self, clip, fps, aspect):
        return np.array([0.9, 0.06, 0.04])

    def top_k(self, probs, k):
        return list(zip(["hello", "hat", "cowboy"], probs.tolist()))[:k]


@pytest.fixture(autouse=True)
def armenian_interface():
    strings.set_language("hy")
    yield
    strings.set_language(config.UI_LANGUAGES[0])


@pytest.fixture
def recording():
    files = sorted(glob.glob(str(config.RECORDINGS_DIR / "hello_*.npy")))
    if not files:
        pytest.skip("no recordings/hello_*.npy")
    return files[0]


@pytest.fixture
def make_window(qtbot, tmp_path):
    windows = []

    def make(source_factory, model=FakeModel(), settings=None, start=True):
        w = MainWindow(settings or AppSettings(sound=False), model, load_vocab(), source_factory=source_factory,
                       settings_path=tmp_path / "settings.json",
                       prediction_log=PredictionLog(tmp_path / "predictions.csv", tmp_path / "attempts"),
                       fps_log_path=tmp_path / "fps.csv", start=start)
        qtbot.addWidget(w)
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.close()


def test_window_records_a_sign_from_a_recording(qtbot, make_window, recording, tmp_path):
    w = make_window(lambda: RecordedSource(recording, loop=True))
    assert w.windowTitle() == strings.STRINGS["app_title"]["hy"]
    qtbot.waitUntil(lambda: w.video.image is not None, timeout=WAIT_MS)
    w.worker.recorder.countdown_s = 0.2  # keep the test short
    w.shortcut_record.activated.emit()  # the Space key
    qtbot.waitUntil(lambda: w.last_prediction is not None, timeout=WAIT_MS)
    pred = w.last_prediction
    assert pred.accepted and pred.sign == "hello"
    assert w.panel.word_en.text() == "hello" and w.panel.word_hy.text() == load_vocab()["hello"]["hy"]
    assert w.panel.btn_speak.isEnabled()
    rows = list(csv.DictReader(open(tmp_path / "predictions.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["top1"] == "hello" and (tmp_path / "attempts" / rows[0]["attempt"]).exists()


def test_missing_camera_shows_an_error_instead_of_crashing(qtbot, make_window, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: shown.append(text))

    def no_camera(index, width, height):
        raise CameraError(f"Camera {index} cannot be opened")

    w = make_window(lambda: LiveSource(camera_index=3, capture_factory=no_camera))
    qtbot.waitUntil(lambda: bool(shown), timeout=WAIT_MS)
    assert "3" in shown[0] and "Camera 3 cannot be opened" in shown[0]
    assert w.video.message_key == "hint_camera_missing"
    assert not w.panel.btn_record.isEnabled()


def test_without_model_recording_is_disabled(make_window):
    w = make_window(None, model=None, start=False)
    assert not w.panel.btn_record.isEnabled()
    w.record()  # no worker, no model: nothing happens


def test_language_switch_is_saved_and_applied(make_window, tmp_path):
    w = make_window(None, start=False)
    w.apply_settings(AppSettings(language="ru", sound=False))
    assert strings.language() == "ru"
    assert w.windowTitle() == "Переводчик ASL" and w.tabs.tabText(0) == "Одно слово"
    assert w.panel.btn_record.text() == strings.STRINGS["btn_record"]["ru"]
    assert '"language": "ru"' in (tmp_path / "settings.json").read_text(encoding="utf-8")
    w.apply_settings(AppSettings(language="en", sound=False))
    assert w.menu_settings.title() == "&Settings"


def test_settings_dialog_values(qtbot):
    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    assert dlg.language.currentData() == "hy"
    dlg.camera.setValue(2)
    dlg.record.setValue(3.5)
    dlg.threshold.setValue(55)
    dlg.sound.setChecked(False)
    dlg.language.setCurrentIndex(config.UI_LANGUAGES.index("en"))
    s = dlg.settings()
    assert (s.camera_index, s.record_seconds, s.threshold, s.sound, s.language) == (2, 3.5, 0.55, False, "en")


def test_ui_font_draws_all_armenian_text(qapp):
    family = armenian_font_family()
    assert family is not None
    text = "".join(t["hy"] for t in strings.STRINGS.values()) + "".join(v["hy"] for v in load_vocab().values())
    assert missing_glyphs(family, text) == ""


def test_speaker_does_not_block_and_skips_while_busy(monkeypatch):
    spoken, release = [], threading.Event()

    class Engine:
        def say(self, text):
            spoken.append(text)

        def runAndWait(self):  # noqa: N802 - pyttsx3 API
            release.wait(5)

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=lambda: Engine()))
    sp = Speaker()
    assert sp.speak("hello")
    assert not sp.speak("again")  # still speaking
    release.set()  # the first word ends; the next call is accepted once its thread is done
    for _ in range(100):
        if sp.speak("again"):
            break
        threading.Event().wait(0.02)
    for _ in range(100):
        if len(spoken) == 2:
            break
        threading.Event().wait(0.02)
    assert spoken == ["hello", "again"]


def test_speaker_reports_errors(monkeypatch):
    errors, done = [], threading.Event()

    def boom():
        raise RuntimeError("no voice")

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=boom))
    sp = Speaker(on_error=lambda d: (errors.append(d), done.set()))
    assert sp.speak("hello")
    done.wait(5)
    assert errors and "no voice" in errors[0]
