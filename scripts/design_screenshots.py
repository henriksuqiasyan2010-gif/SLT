"""Screenshots of every mode, taken with QWidget.grab(): the design review (Phase 7.5) and the team kit
(Phase 8).

    python scripts/design_screenshots.py [--langs hy ru] [--out reports/design]
    python scripts/design_screenshots.py --langs en hy --out reports/team_kit/screenshots

No camera: Single word and Learn replay the user's recordings, Sentence replays session 1, the demo
mode session 2. The real progress.json is copied and logs go to a temporary folder, never to logs/.
Files: <out>/<n>_<mode>_<lang>.png at 1280x800 (the mockups' size).
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtWidgets import QApplication  # noqa: E402

from asl import config  # noqa: E402
from asl.app import settings as S  # noqa: E402
from asl.app import strings, theme  # noqa: E402
from asl.app.main_window import MainWindow, about_box  # noqa: E402
from asl.app.pipeline import PredictionLog  # noqa: E402
from asl.app.progress import Progress  # noqa: E402
from asl.app.settings_dialog import SettingsDialog  # noqa: E402
from asl.capture import RecordedSource  # noqa: E402
from asl.inference import SignModel  # noqa: E402
from asl.vocab import load_vocab  # noqa: E402

SIZE = (1280, 800)
SINGLE = config.RECORDINGS_DIR / "hello_20261026_192257.npy"
SESSION = config.SESSIONS_DIR / "session1.npy"
LEARN = config.RECORDINGS_DIR / "dad_20261028_141638.npy"
LEARN_SIGN = "dad"
DEMO_SESSION = 1  # index in assets/demo: session 2 (airplane high find sick mom)
TIMEOUT_S = 30
COUNTDOWN_S = 0.3  # shorter than in the app, so the screenshots do not wait 3 s


def wait(app, condition, timeout=TIMEOUT_S) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def pause(app, seconds: float) -> None:
    wait(app, lambda: False, seconds)


def shoot(app, out: Path, lang: str, model, vocab) -> list:
    strings.set_language(lang)
    tmp = Path(tempfile.mkdtemp())
    if config.PROGRESS_JSON.exists():
        shutil.copy(config.PROGRESS_JSON, tmp / "progress.json")
    source = {"path": SINGLE, "loop": True, "speed": 1.0}
    w = MainWindow(S.AppSettings(language=lang, sound=False), model, vocab,
                   source_factory=lambda: RecordedSource(source["path"], loop=source["loop"], speed=source["speed"]),
                   settings_path=tmp / "settings.json", prediction_log=PredictionLog(tmp / "p.csv", tmp / "att"),
                   fps_log_path=tmp / "fps.csv", corrections_log=tmp / "c.csv",
                   progress=Progress.load(tmp / "progress.json"), defense_set_path=tmp / "d.csv",
                   demo_log=PredictionLog(tmp / "pd.csv", tmp / "att_demo"), demo_corrections_log=tmp / "cd.csv")
    w.resize(*SIZE)
    w.show()
    saved = []

    def grab(name):
        path = out / f"{name}_{lang}.png"
        w.grab().save(str(path))
        saved.append(path)

    def restart(path, loop=True, speed=1.0, tab=None):
        w.stop_worker()
        source.update(path=path, loop=loop, speed=speed)
        if tab is not None:
            w.tabs.setCurrentIndex(tab)
        w.start_worker()
        wait(app, lambda: w.worker is not None and w.worker.mode == w.mode() and w.worker.source is not None)

    # 1. Single word: record "hello"
    wait(app, lambda: w.video.payload is not None)
    w.worker.recorder.countdown_s = COUNTDOWN_S
    w.record()
    wait(app, lambda: w.last_prediction is not None)
    pause(app, 0.5)
    grab("1_single")
    # 2. Sentence (automatic): session 1 at 2x until all five words are in
    restart(SESSION, loop=False, speed=2.0, tab=1)
    wait(app, lambda: len(w.sentence.words) >= 3)
    grab("2_sentence_signing")
    wait(app, lambda: len(w.sentence.words) >= 5)
    pause(app, 0.3)
    grab("2_sentence")
    # 3. Learn: dad, then one attempt (the user's dad recording)
    restart(LEARN, tab=2)
    lp = w.learn_panel
    lp.search.setText(LEARN_SIGN)
    lp.words.setCurrentRow(0)
    lp.search.setText("")
    pause(app, 0.8)
    grab("3_learn")
    w.worker.recorder.countdown_s = COUNTDOWN_S
    w.record()
    wait(app, lambda: lp.showing_attempt)
    pause(app, 0.6)
    grab("3_learn_attempt")
    # 4. dialogs: Settings and About, in the same style
    dlg = SettingsDialog(w.settings, w)
    dlg.show()
    pause(app, 0.3)
    path = out / f"4_settings_{lang}.png"
    dlg.grab().save(str(path))
    saved.append(path)
    dlg.close()
    box = about_box(w)
    box.show()
    pause(app, 0.3)
    path = out / f"5_about_{lang}.png"
    box.grab().save(str(path))
    saved.append(path)
    box.close()
    # 6. demo mode (no camera): session 2 until its five words are in
    w.start_demo(w.demo_sessions[DEMO_SESSION])
    wait(app, lambda: len(w.sentence.words) >= 5, timeout=TIMEOUT_S * 2)
    pause(app, 0.3)
    grab("6_demo")
    w.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hy", "ru"])
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "design")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    theme.apply(app)
    model, vocab = SignModel.load(), load_vocab()
    for lang in args.langs:
        for path in shoot(app, args.out, lang, model, vocab):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
