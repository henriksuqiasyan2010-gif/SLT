"""Record the Armenian word for every sign (Phase 7, option a: recorded voice). Not used yet:
the team decides later whether Armenian audio is needed.

    python scripts/record_audio.py                 # the signs without a file, alphabetical
    python scripts/record_audio.py --all           # every sign (re-record)
    python scripts/record_audio.py --start hello   # begin at a sign

A window shows the Armenian word (with en and ru). Keys:
    Space   start / stop recording (the default microphone, 44.1 kHz mono 16-bit)
    P       play the saved file          Enter / Right   next word
    Left    previous word                Esc             quit
Each take is saved as assets/audio/hy/<sign>.wav and post-processed at once (silence trimmed,
loudness normalised); recording again overwrites it. Also writes reports/audio_script.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

import numpy as np  # noqa: E402

from asl import audio, config  # noqa: E402
from asl.app.fonts import apply_ui_font  # noqa: E402
from asl.vocab import load_vocab  # noqa: E402

WORD_POINT_SIZE = 48
HELP = "Space: record / stop    P: play    Enter / →: next    ←: previous    Esc: quit"


class Recorder(QWidget):
    def __init__(self, vocab: dict, signs: list):
        super().__init__()
        self.vocab, self.signs, self.i = vocab, signs, 0
        self.player = audio.AudioPlayer(vocab)
        self.source = self.device = None
        self.buffer = bytearray()
        self.word, self.info, self.status = QLabel(), QLabel(), QLabel()
        f = QFont(self.font())
        f.setPointSize(WORD_POINT_SIZE)
        self.word.setFont(f)
        for label in (self.word, self.info, self.status):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.word, 1)
        lay.addWidget(self.status)
        lay.addWidget(QLabel(HELP))
        self.setWindowTitle("Armenian audio recording")
        self.resize(900, 420)
        self.show_word()

    @property
    def sign(self) -> str:
        return self.signs[self.i]

    def show_word(self, note: str = "") -> None:
        v = self.vocab[self.sign]
        path = audio.audio_path(self.sign)
        self.word.setText(v["hy"])
        self.info.setText(f"{self.i + 1} / {len(self.signs)}    {self.sign}    en: {v['en']}    ru: {v['ru']}")
        state = "● RECORDING" if self.source else ("saved ✓" if path.exists() else "not recorded")
        self.status.setText(f"{state}    {note}")

    def keyPressEvent(self, e):  # noqa: N802 - Qt API
        key = e.key()
        if key == Qt.Key.Key_Space:
            self.stop() if self.source else self.start()
        elif key == Qt.Key.Key_P and not self.source:
            self.player.play_word(self.sign)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Right) and not self.source:
            self.i = min(self.i + 1, len(self.signs) - 1)
            self.show_word()
        elif key == Qt.Key.Key_Left and not self.source:
            self.i = max(self.i - 1, 0)
            self.show_word()
        elif key == Qt.Key.Key_Escape:
            self.close()

    def start(self) -> None:
        fmt = QAudioFormat()
        fmt.setSampleRate(config.AUDIO_SAMPLE_RATE)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = QMediaDevices.defaultAudioInput()
        if dev.isNull() or not dev.isFormatSupported(fmt):
            self.show_word("no microphone, or it does not support 44.1 kHz mono 16-bit")
            return
        self.buffer = bytearray()
        self.source = QAudioSource(dev, fmt, self)
        self.device = self.source.start()
        self.device.readyRead.connect(lambda: self.buffer.extend(bytes(self.device.readAll().data())))
        self.show_word()

    def stop(self) -> None:
        self.buffer.extend(bytes(self.device.readAll().data()))
        self.source.stop()
        self.source = self.device = None
        samples = np.frombuffer(bytes(self.buffer), dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) == 0:
            self.show_word("nothing was recorded")
            return
        path = audio.audio_path(self.sign)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio.write_wav(path, samples, config.AUDIO_SAMPLE_RATE)
        info = audio.postprocess_file(path)
        self.show_word(f"{info['before_s']} s recorded -> {info['after_s']} s after trimming, "
                       f"level {info['rms_db']} dB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every sign, also those already recorded")
    ap.add_argument("--start", help="begin at this sign")
    args = ap.parse_args()
    vocab = load_vocab()
    n = audio.write_audio_script(vocab)
    print(f"{n} words -> {config.AUDIO_SCRIPT_CSV}")
    signs = sorted(vocab) if args.all else audio.missing_signs(vocab)
    if args.start:
        signs = [s for s in signs if s >= args.start]
    if not signs:
        print("every sign already has a file (use --all to record again)")
        return 0
    app = QApplication(sys.argv[:1])
    apply_ui_font(app)
    w = Recorder(vocab, signs)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
