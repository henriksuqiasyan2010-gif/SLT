"""pytest-qt: the window opens, records from a replayed recording, survives a missing camera,
switches the interface language, the UI font can draw every Armenian letter we show, and the
Sentence tab works in both modes (manual = button, automatic = segmenter) independently."""

import csv
import glob

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asl import config
from asl.app import strings
from asl.app.fonts import armenian_font_family, missing_glyphs
from asl.app.main_window import MainWindow
from asl.app.pipeline import PredictionLog
from asl.app.progress import Progress
from asl.app.settings import AppSettings
from asl.app.settings_dialog import SettingsDialog
from asl.capture import CameraError, LiveSource, RecordedSource
from asl.vocab import load_vocab

WAIT_MS = 15000


class FakeModel:
    metrics = {"preset": "fake", "orig": {"top1": 0.627, "top3": 0.769}}
    labels = {0: "hello", 1: "hat", 2: "cowboy", 3: "dad"}

    def predict_clip(self, clip, fps, aspect):
        return np.array([0.9, 0.06, 0.04, 0.0])

    def top_k(self, probs, k):
        return list(zip(["hello", "hat", "cowboy"], probs.tolist()))[:k]


@pytest.fixture(autouse=True, scope="module")
def design(qapp):
    """The app's fonts and stylesheet, as asl.app.__main__ sets them."""
    from asl.app import theme
    theme.apply(qapp)


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
                       fps_log_path=tmp_path / "fps.csv", start=start,
                       corrections_log=tmp_path / "corrections.csv",
                       progress=Progress.load(tmp_path / "progress.json"),
                       defense_set_path=tmp_path / "defense_set.csv")
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
    assert w.windowTitle() == "SLT" and w.tabs.tabText(0) == "Одно слово"
    assert w.panel.btn_record.text() == strings.STRINGS["btn_record"]["ru"]
    assert '"language": "ru"' in (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert w.header.lang.current() == "ru"
    w.header.lang.buttons["en"].click()  # the HY / RU / EN switch in the header
    assert strings.language() == "en" and w.header.btn_settings.toolTip() == "Settings"
    assert '"language": "en"' in (tmp_path / "settings.json").read_text(encoding="utf-8")


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


@pytest.fixture
def session():
    files = sorted(glob.glob(str(config.SESSIONS_DIR / "session1.npy")))
    if not files:
        pytest.skip("no recordings/sessions/session1.npy")
    return files[0]


def test_sentence_manual_mode_uses_the_button_and_never_the_segmenter(qtbot, make_window, recording, tmp_path):
    w = make_window(lambda: RecordedSource(recording, loop=True))
    w.tabs.setCurrentIndex(1)
    w.sentence_panel.manual.setChecked(True)
    assert w.mode() == "sentence_manual" and w.sentence_panel.btn_record.isVisible() is not None
    qtbot.waitUntil(lambda: w.video.image is not None and w.worker.mode == "sentence_manual", timeout=WAIT_MS)
    w.worker.recorder.countdown_s = 0.2
    for n in (1, 2):
        w.shortcut_record.activated.emit()  # Space
        qtbot.waitUntil(lambda: len(w.sentence.words) == n, timeout=WAIT_MS)
        qtbot.waitUntil(lambda: w.sentence_panel.btn_record.isEnabled(), timeout=WAIT_MS)
    assert w.sentence.signs() == ["hello", "hello"]  # two button presses = two words, seconds apart
    assert w.worker.segmenter is None  # the automatic path was never touched
    assert w.sentence_panel.text() == "hello hello"
    rows = list(csv.DictReader(open(tmp_path / "predictions.csv", encoding="utf-8")))
    assert [r["mode"] for r in rows] == ["sentence_manual"] * 2
    assert w.panel.prediction is None  # Single word panel untouched


def test_sentence_auto_mode_segments_a_session(qtbot, make_window, session, tmp_path):
    w = make_window(lambda: RecordedSource(session, speed=4.0))
    w.tabs.setCurrentIndex(1)
    assert w.mode() == "sentence_auto"
    w.record()  # the button does nothing in automatic mode
    qtbot.waitUntil(lambda: w.worker is None or w.worker.source is None or len(w.sentence.words) >= 5,
                    timeout=4 * WAIT_MS)
    qtbot.waitUntil(lambda: len(w.sentence.words) == 5, timeout=WAIT_MS)  # the last one comes from flush()
    assert w.worker.recorder.phase.value == "idle"
    rows = list(csv.DictReader(open(tmp_path / "predictions.csv", encoding="utf-8")))
    assert [r["mode"] for r in rows] == ["sentence_auto"] * 5


def test_replace_last_word_from_its_alternatives(qtbot, make_window, recording, tmp_path):
    w = make_window(lambda: RecordedSource(recording, loop=True))
    w.tabs.setCurrentIndex(1)
    w.sentence_panel.manual.setChecked(True)
    qtbot.waitUntil(lambda: w.video.image is not None and w.worker.mode == "sentence_manual", timeout=WAIT_MS)
    w.worker.recorder.countdown_s = 0.2
    w.record()
    qtbot.waitUntil(lambda: len(w.sentence.words) == 1, timeout=WAIT_MS)
    alts = w.sentence_panel.alt_buttons
    assert [b.property("sign") for b in alts] == ["hello", "hat", "cowboy"] and not alts[0].isEnabled()
    alts[1].click()
    assert w.sentence.signs() == ["hat"]
    rows = list(csv.DictReader(open(tmp_path / "corrections.csv", encoding="utf-8")))
    assert rows[0]["predicted"] == "hello" and rows[0]["corrected_to"] == "hat"
    w.sentence_panel.btn_delete.click()
    assert w.sentence.signs() == [] and not w.sentence_panel.btn_clear.isEnabled()


def select_learn(qtbot, w, sign):
    w.tabs.setCurrentIndex(2)
    w.learn_panel.search.setText(sign)
    items = [w.learn_panel.words.item(i) for i in range(w.learn_panel.words.count())]
    item = next(i for i in items if i.data(256) == sign)  # Qt.ItemDataRole.UserRole
    w.learn_panel.words.setCurrentItem(item)
    qtbot.waitUntil(lambda: w.worker.mode == "learn" and w.learn_panel.you_view.live is not None, timeout=WAIT_MS)
    w.worker.recorder.countdown_s = 0.2


def test_learn_attempt_is_scored_shown_side_by_side_and_counted(qtbot, make_window, recording, tmp_path):
    w = make_window(lambda: RecordedSource(recording, loop=True))
    if "dad" not in w.references:
        pytest.skip("no assets/reference")
    select_learn(qtbot, w, "dad")
    lp = w.learn_panel
    assert lp.sign == "dad" and lp.ref_view.clip is not None and lp.playing  # the reference plays
    assert lp.you_view.live is not None and lp.you_view.clip is None  # your attempt player shows you live
    w.shortcut_record.activated.emit()  # Space = Try
    qtbot.waitUntil(lambda: w.progress.get("dad").attempts == 1, timeout=WAIT_MS)
    assert "hello" in lp.result.text() and lp.result.property("chip") == "error"  # "Looks like hello"
    assert lp.showing_attempt and lp.you_view.clip is not None  # the attempt replays next to the reference
    assert (lp.ref_view.name, lp.you_view.name) == (strings.tr("learn_reference"), strings.tr("learn_you"))
    assert w.progress.get("dad").successes == 0 and not w.progress.is_learned("dad")
    rows = list(csv.DictReader(open(tmp_path / "predictions.csv", encoding="utf-8")))
    assert rows[-1]["mode"] == "learn" and rows[-1]["target"] == "dad"
    assert Progress.load(tmp_path / "progress.json").get("dad").attempts == 1  # saved at once


def test_three_successes_mark_a_sign_learned_and_export(qtbot, make_window, recording, tmp_path):
    w = make_window(lambda: RecordedSource(recording, loop=True))
    if "hello" not in w.references:
        pytest.skip("no assets/reference")
    select_learn(qtbot, w, "hello")
    for n in (1, 2, 3):
        qtbot.waitUntil(lambda: w.learn_panel.btn_try.isEnabled(), timeout=WAIT_MS)
        w.record()
        qtbot.waitUntil(lambda: w.progress.get("hello").attempts == n, timeout=WAIT_MS)
    lp = w.learn_panel
    assert w.progress.is_learned("hello")
    assert lp.progress_label.text() == strings.tr("learn_progress_done", n=3)
    assert lp.result.property("chip") == "success"
    lp.search.setText("")
    assert lp.count.text() == strings.tr("learn_count", n=1, total=len(load_vocab()))
    lp.filter.set_current("learned")
    assert [w.learn_panel.words.item(i).data(256) for i in range(w.learn_panel.words.count())] == ["hello"]
    w.learn_panel.btn_export.click()
    rows = list(csv.DictReader(open(tmp_path / "defense_set.csv", encoding="utf-8-sig")))
    assert rows[0]["sign"] == "hello" and rows[0]["learned"] == "True"



def test_armenian_text_is_drawn(qapp):
    """PHASES.md Phase 7: an Armenian string renders (real pixels, not empty boxes)."""
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
    family = armenian_font_family()
    font = QFont(family, 28)
    text = "բարև ձեզ"
    img = QImage(400, 80, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    p.setFont(font)
    p.setPen(QColor("black"))
    p.drawText(10, 55, text)
    p.end()
    dark = sum(1 for x in range(0, 400, 2) for y in range(0, 80, 2) if img.pixelColor(x, y).lightness() < 128)
    assert dark > 50 and QFontMetrics(font).horizontalAdvance(text) > 0
    assert missing_glyphs(family, text) == ""


def test_design_fonts_logo_and_shadow_flag(qapp, monkeypatch):
    """Phase 7.5: bundled Rubik + Noto Sans Armenian, Armenian drawn with Noto, logo / icon files, the
    shadow flag in config, and the record button grows with a long label instead of cutting it."""
    from PySide6.QtGui import QFontInfo
    from asl.app import theme, ui_kit
    families = theme.load_fonts()
    assert {"Rubik", "Noto Sans Armenian"} <= set(families)
    assert armenian_font_family() == "Noto Sans Armenian"
    assert QFontInfo(theme.font(16, 800)).family() == "Rubik"
    assert missing_glyphs("Noto Sans Armenian", "".join(v["hy"] for v in load_vocab().values())) == ""
    for name in ("slt_logo.png", "slt_icon.png", "slt.ico"):
        assert (config.BRANDING_DIR / name).exists(), name
    card = ui_kit.Card()
    assert card.graphicsEffect() is not None
    monkeypatch.setattr(config, "UI_SHADOWS", False)
    assert ui_kit.Card().graphicsEffect() is None
    short, long = ui_kit.button("OK", "primary"), ui_kit.button("Տեսագրել ժեստը, խնդրում եմ, հիմա", "primary")
    assert long.minimumSizeHint().width() > short.minimumSizeHint().width()



def test_long_armenian_labels_are_not_cut(qtbot, make_window):
    """Phase 7.5: buttons widen with longer Armenian texts instead of cutting them."""
    from PySide6.QtWidgets import QPushButton
    strings.set_language("hy")
    w = make_window(None, start=False)
    w.resize(1280, 800)
    w.show()
    for tab in range(3):
        w.tabs.setCurrentIndex(tab)
        qtbot.wait(100)  # let the layout settle
        for b in w.findChildren(QPushButton):
            if b.isVisible() and b.text():
                assert b.width() >= b.fontMetrics().horizontalAdvance(b.text()), (tab, b.text())
