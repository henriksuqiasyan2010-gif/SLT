"""Start the app:  python -m asl.app

    python -m asl.app --recording recordings/hello_x.npy     replay a recording in a loop instead of the camera
    python -m asl.app --quit-after 330 --auto-record 20      FPS measurement: press Record every 20 s, quit after 330 s
"""

import argparse
import logging
import sys
import traceback
from dataclasses import replace

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from asl import config
from asl.app import settings as settings_io
from asl.app import strings
from asl.app.fonts import apply_ui_font
from asl.app.main_window import MainWindow, show_error
from asl.app.pipeline import PredictionLog
from asl.app.strings import tr
from asl.capture import RecordedSource
from asl.inference import SignModel
from asl.vocab import VocabError, load_vocab, validate_vocab

log = logging.getLogger("asl.app")
MS = 1000


def setup_logging() -> None:
    config.LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(filename=config.APP_LOG, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("comtypes").setLevel(logging.WARNING)  # pyttsx3 / SAPI5 logs every COM cache lookup


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m asl.app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", help="replay this .npy recording in a loop instead of the camera")
    ap.add_argument("--quit-after", type=float, default=0, help="close the app after N seconds (measurements)")
    ap.add_argument("--auto-record", type=float, default=0,
                    help="press Record every N seconds (measurements; logged to logs/predictions_auto.csv)")
    args = ap.parse_args(argv)

    setup_logging()
    log.info("start %s", " ".join(sys.argv))
    app = QApplication(sys.argv[:1])
    apply_ui_font(app)
    settings = settings_io.load()
    if args.auto_record:  # measurement runs never speak (the setting itself is not changed)
        settings = replace(settings, sound=False)
    strings.set_language(settings.language)

    errors = []
    try:
        vocab = load_vocab()
    except VocabError as e:
        log.exception("vocab")
        vocab, errors = {}, errors + [tr("error_vocab", details=e)]
    try:
        model = SignModel.load()
    except Exception as e:  # noqa: BLE001 - shown to the user, the window still opens
        log.exception("model")
        model, errors = None, errors + [tr("error_model", details=e)]
    if vocab and model is not None:  # every class of the model needs its three translations
        problems = validate_vocab(vocab, model.labels)
        if problems:
            log.error("vocab: %s", "; ".join(problems))
            errors.append(tr("error_vocab", details="; ".join(problems)))

    source_factory = (lambda: RecordedSource(args.recording, loop=True)) if args.recording else None
    measuring = bool(args.auto_record)
    window = MainWindow(
        settings, model, vocab, source_factory=source_factory,
        prediction_log=PredictionLog(config.LOGS_DIR / "predictions_auto.csv", config.LOGS_DIR / "attempts_auto")
        if measuring else None,
        fps_tag="app-auto-record" if measuring else "app")

    def excepthook(kind, value, tb):  # exceptions in Qt slots: log + message instead of a silent stderr line
        log.error("uncaught exception:\n%s", "".join(traceback.format_exception(kind, value, tb)))
        show_error(window, tr("error_internal", details=f"{kind.__name__}: {value}"))

    sys.excepthook = excepthook
    window.resize(config.CAMERA_WIDTH * 2, int(config.CAMERA_HEIGHT * 1.4))
    window.show()
    for text in errors:
        show_error(window, text)
    if args.auto_record:
        timer = QTimer(window)
        timer.timeout.connect(window.record)
        timer.start(int(args.auto_record * MS))
    if args.quit_after:
        QTimer.singleShot(int(args.quit_after * MS), window.close)
    code = app.exec()
    log.info("exit %s", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
