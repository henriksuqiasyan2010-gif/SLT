"""Speaking the recognized words, without blocking the interface.

    player = AudioPlayer(vocab)
    player.play_word("hello")                  # Armenian file if it exists, else English via pyttsx3
    player.play_sentence(["hello", "please"])  # word by word, AUDIO_SENTENCE_GAP_S apart

Everything runs in one background thread fed by a queue; a new request replaces whatever was still
waiting and stops the file being played. Armenian files: AUDIO_DIR/<sign>.wav (16-bit WAV, recorded
with scripts/record_audio.py and cleaned by postprocess()). Playback and speech are injectable, so
tests need no sound card.

postprocess(): trim leading/trailing silence, normalise loudness (RMS target, peak limit), 16-bit WAV.
"""

import csv
import logging
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from asl import config

log = logging.getLogger(__name__)


def audio_path(sign: str, root=config.AUDIO_DIR) -> Path:
    return Path(root) / f"{sign}.wav"


def play_wav_blocking(path) -> None:
    """Windows' own player (no extra dependency); returns when the file has been played."""
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)


def stop_wav() -> None:
    import winsound

    winsound.PlaySound(None, 0)


def say_blocking(text: str) -> None:
    """English text with pyttsx3 (SAPI5 needs COM initialised in this thread)."""
    try:
        import comtypes
        comtypes.CoInitialize()
    except (ImportError, OSError):
        pass
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


class AudioPlayer:
    def __init__(self, vocab: dict, root=config.AUDIO_DIR, play_wav=play_wav_blocking, say=say_blocking,
                 stop=stop_wav, on_error=None, gap_s: float = config.AUDIO_SENTENCE_GAP_S):
        """on_error(details) is called from the audio thread when a word cannot be played."""
        self.vocab, self.root, self.gap_s = vocab, Path(root), gap_s
        self._play_wav, self._say, self._stop, self.on_error = play_wav, say, stop, on_error
        self._queue = queue.Queue()
        self._generation = 0  # bumped by every request; older items are skipped
        self._lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        threading.Thread(target=self._run, daemon=True, name="audio").start()

    def source(self, sign: str) -> tuple:
        """("wav", path) if the Armenian file exists, else ("tts", English word)."""
        path = audio_path(sign, self.root)
        if path.exists():
            return "wav", path
        return "tts", self.vocab.get(sign, {}).get("en", sign)

    def play_word(self, sign: str) -> None:
        self.play_sentence([sign])

    def play_sentence(self, signs) -> None:
        """Returns at once; replaces anything still being said."""
        with self._lock:
            self._generation += 1
            gen = self._generation
        try:
            self._stop()
        except Exception:  # noqa: BLE001 - nothing playing / no sound device
            pass
        self.idle.clear()
        self._queue.put((gen, [self.source(s) for s in signs]))

    def _run(self) -> None:
        while True:
            gen, items = self._queue.get()
            for i, (kind, what) in enumerate(items):
                if gen != self._generation:
                    break
                if i:
                    time.sleep(self.gap_s)
                try:
                    self._play_wav(what) if kind == "wav" else self._say(what)
                except Exception as e:  # noqa: BLE001 - no sound must never break the app
                    log.exception("audio: cannot play %s %s", kind, what)
                    if self.on_error:
                        self.on_error(f"{type(e).__name__}: {e}")
            if self._queue.empty():
                self.idle.set()


# --- Post-processing of recorded words ---------------------------------------------------

def read_wav(path):
    """(samples float32 in [-1, 1], mono; sample rate)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit audio, expected 16-bit")
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, channels).mean(axis=1), rate


def write_wav(path, samples, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def postprocess(samples, rate: int):
    """Trim silence (frames quieter than peak + AUDIO_TRIM_DB, AUDIO_PAD_S kept), then scale to the
    RMS target without letting the peak exceed AUDIO_PEAK_DB. Returns the new samples."""
    x = np.asarray(samples, dtype=np.float32)
    if len(x) == 0 or np.abs(x).max() == 0:
        return x
    n = max(1, int(rate * config.AUDIO_FRAME_S))
    frames = np.abs(x[: len(x) // n * n].reshape(-1, n)).max(axis=1)
    loud = np.flatnonzero(frames >= np.abs(x).max() * 10 ** (config.AUDIO_TRIM_DB / 20))
    pad = int(rate * config.AUDIO_PAD_S)
    x = x[max(0, loud[0] * n - pad): min(len(x), (loud[-1] + 1) * n + pad)]
    rms = float(np.sqrt(np.mean(x ** 2)))
    gain = min(10 ** (config.AUDIO_TARGET_RMS_DB / 20) / max(rms, 1e-12),
               10 ** (config.AUDIO_PEAK_DB / 20) / float(np.abs(x).max()))
    return (x * gain).astype(np.float32)


def postprocess_file(path) -> dict:
    """Post-process a WAV in place; returns before/after duration and levels (for the report)."""
    x, rate = read_wav(path)
    y = postprocess(x, rate)
    write_wav(path, y, rate)
    return {"before_s": round(len(x) / rate, 2), "after_s": round(len(y) / rate, 2),
            "rms_db": round(db(np.sqrt(np.mean(y ** 2))) if len(y) else -120.0, 1),
            "peak_db": round(db(np.abs(y).max()) if len(y) else -120.0, 1)}


# --- Recording plan -----------------------------------------------------------------------

def missing_signs(vocab: dict, root=config.AUDIO_DIR) -> list:
    """Signs without an Armenian file, in alphabetical order."""
    return [s for s in sorted(vocab) if not audio_path(s, root).exists()]


def check_file(path) -> str:
    """'' if the file is a playable 16-bit mono WAV with sound in it, else what is wrong."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return f"{8 * w.getsampwidth()}-bit, {w.getnchannels()} channels (expected 16-bit mono)"
            if w.getnframes() == 0:
                return "empty"
    except (OSError, wave.Error, EOFError) as e:
        return f"unreadable: {e}"
    return ""


def write_audio_script(vocab: dict, path=config.AUDIO_SCRIPT_CSV) -> int:
    """The recording script: one row per sign (file name, Armenian word to say, en and ru for context)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "hy", "en", "ru"])
        for sign in sorted(vocab):
            w.writerow([audio_path(sign, Path(".")).name, vocab[sign]["hy"], vocab[sign]["en"], vocab[sign]["ru"]])
    return len(vocab)

