"""Demo mode (Phase 8): the bundled sessions, a replay through the whole pipeline without any camera,
demo runs logged apart from the user's attempts and never counted in progress.json, the help menu
and the About text (version, author)."""

import json
import shutil

import numpy as np
import pytest

from asl import config
from asl.app import strings
from asl.app.demo import demo_sessions
from asl.app.main_window import TAB_LEARN, TAB_SENTENCE, MainWindow, about_box
from asl.app.pipeline import PredictionLog
from asl.app.progress import Progress
from asl.app.settings import AppSettings
from asl.capture import CameraError, LiveSource, load_recording, save_recording
from asl.vocab import load_vocab

WAIT_MS = 20000
FIRST_SIGN_END_S = 4.5  # session 1: "hello" is segmented at 0.64-3.05 s, the hands are down by 4.5 s
RECORDING_KEYS = ("frames", "fps", "duration_s", "timestamps")


class FakeModel:
    metrics = {"preset": "fake", "orig": {"top1": 0.627, "top3": 0.769}}
    labels = {0: "hello", 1: "hat", 2: "cowboy"}

    def predict_clip(self, clip, fps, aspect):
        return np.array([0.9, 0.06, 0.04])

    def top_k(self, probs, k):
        return list(zip(["hello", "hat", "cowboy"], probs.tolist()))[:k]


def no_camera(index, width, height):
    raise CameraError(f"Camera {index} cannot be opened")


@pytest.fixture(autouse=True, scope="module")
def design(qapp):
    from asl.app import theme
    theme.apply(qapp)


@pytest.fixture(autouse=True)
def russian_interface():
    strings.set_language("ru")
    yield
    strings.set_language(config.UI_LANGUAGES[0])


@pytest.fixture
def short_demo(tmp_path):
    """assets/demo/session1 cut after its first sign, so a replay takes ~4.5 s."""
    full = demo_sessions()[0]
    clip, meta = load_recording(full.path)
    ts = np.asarray(meta["timestamps"])
    keep = ts <= FIRST_SIGN_END_S
    root = tmp_path / "demo"
    save_recording(root / "session1.npy", clip[keep], ts[keep],
                   **{k: v for k, v in meta.items() if k not in RECORDING_KEYS})
    (root / "session1.json").write_text(json.dumps({"signs": ["hello"]}), encoding="utf-8")
    return root


@pytest.fixture
def window(qtbot, tmp_path, short_demo, monkeypatch):
    monkeypatch.setattr(config, "DEMO_RESTART_S", 0.2)
    w = MainWindow(AppSettings(sound=False, language="ru"), FakeModel(), load_vocab(),
                   source_factory=lambda: LiveSource(camera_index=7, capture_factory=no_camera),
                   settings_path=tmp_path / "settings.json",
                   prediction_log=PredictionLog(tmp_path / "predictions.csv", tmp_path / "attempts"),
                   fps_log_path=tmp_path / "fps.csv", start=False, corrections_log=tmp_path / "corrections.csv",
                   progress=Progress.load(tmp_path / "progress.json"), defense_set_path=tmp_path / "d.csv",
                   demo_dir=short_demo,
                   demo_log=PredictionLog(tmp_path / "predictions_demo.csv", tmp_path / "attempts_demo"),
                   demo_corrections_log=tmp_path / "corrections_demo.csv")
    qtbot.addWidget(w)
    yield w
    w.close()


def test_the_three_sessions_are_bundled_with_their_signs():
    sessions = demo_sessions()
    assert [s.name for s in sessions] == list(config.DEMO_SESSIONS)
    assert sessions[0].signs == ("hello", "please", "happy", "hat", "shower")
    for s in sessions:
        clip, meta = load_recording(s.path)
        assert clip.shape[1:] == (543, 3) and len(meta["timestamps"]) == len(clip) and len(s.signs) == 5


def test_a_missing_session_is_skipped(tmp_path):
    for suffix in (".npy", ".meta.json", ".json"):
        shutil.copy(config.DEMO_DIR / f"session2{suffix}", tmp_path / f"session2{suffix}")
    assert [s.name for s in demo_sessions(tmp_path)] == ["session2"]


def test_demo_replays_a_session_through_the_pipeline_without_a_camera(qtbot, window, tmp_path):
    window.start_demo(window.demo_sessions[0])
    assert window.tabs.currentIndex() == TAB_SENTENCE and window.mode() == "sentence_auto"
    assert "hello" in window.windowTitle() and "hello" in window.video.demo_text
    qtbot.waitUntil(lambda: window.sentence.signs() == ["hello"], timeout=WAIT_MS)
    # logged apart from the user's own attempts
    assert (tmp_path / "predictions_demo.csv").exists() and not (tmp_path / "predictions.csv").exists()
    # after the end the session is replayed: the sentence starts again from the first word
    first_worker = window.worker
    qtbot.waitUntil(lambda: window.worker is not first_worker, timeout=WAIT_MS)
    assert window.sentence.signs() in ([], ["hello"])
    window.stop_demo()
    assert window.demo is None and window.video.demo_text == "" and "hello" not in window.windowTitle()


def test_a_learn_attempt_in_demo_is_scored_but_not_counted(qtbot, window, tmp_path):
    window.start_demo(window.demo_sessions[0])
    window.tabs.setCurrentIndex(TAB_LEARN)
    window.learn_panel.select("hello")
    qtbot.waitUntil(lambda: window.worker is not None and window.worker.mode == "learn", timeout=WAIT_MS)
    window.worker.recorder.countdown_s = 0.2
    window.record()
    qtbot.waitUntil(lambda: window.learn_panel.showing_attempt, timeout=WAIT_MS)
    assert "hello" not in window.progress.signs or window.progress.signs["hello"].attempts == 0
    assert not (tmp_path / "progress.json").exists()


def test_help_menu_offers_the_demos_the_camera_and_about(window):
    labels = [a.text() for a in window.help_menu().actions() if not a.isSeparator()]
    assert labels[0] == strings.tr("menu_demo", n=1, signs="hello")
    assert labels[-2:] == [strings.tr("menu_camera"), strings.tr("about_title")]
    camera = window.help_menu().actions()[1]
    assert not camera.isEnabled()  # already on the camera


def test_about_shows_version_and_author_only(window):
    text = about_box(window).text()
    assert config.APP_VERSION in text and config.APP_AUTHOR in text and "62.7%" in text
    assert "Isolated Sign Language Recognition" in text
