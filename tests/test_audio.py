import threading
import time

import numpy as np
import pytest

from asl import audio, config

VOCAB = {"hello": {"en": "hello", "hy": "բարև", "ru": "привет"}, "please": {"en": "please", "hy": "x", "ru": "y"}}
WAIT_S = 5


class Fake:
    """Records what would be played; each item takes `delay` seconds."""

    def __init__(self, delay=0.0):
        self.calls, self.delay, self.stops = [], delay, 0
        self.lock = threading.Lock()

    def play_wav(self, path):
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(("wav", path.name, time.perf_counter()))

    def say(self, text):
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(("tts", text, time.perf_counter()))

    def stop(self):
        self.stops += 1


def player(tmp_path, fake, **kw):
    return audio.AudioPlayer(VOCAB, root=tmp_path, play_wav=fake.play_wav, say=fake.say, stop=fake.stop, **kw)


def test_armenian_file_if_present_else_english(tmp_path):
    audio.write_wav(tmp_path / "hello.wav", np.zeros(100), 16000)
    fake = Fake()
    p = player(tmp_path, fake)
    assert p.source("hello") == ("wav", tmp_path / "hello.wav")
    assert p.source("please") == ("tts", "please")
    assert p.source("unknown") == ("tts", "unknown")
    p.play_word("hello")
    assert p.idle.wait(WAIT_S)
    p.play_word("please")
    assert p.idle.wait(WAIT_S)
    assert [(k, w) for k, w, _ in fake.calls] == [("wav", "hello.wav"), ("tts", "please")]


def test_does_not_block_and_sentence_has_gaps(tmp_path):
    fake = Fake(delay=0.2)
    p = player(tmp_path, fake, gap_s=0.3)
    t0 = time.perf_counter()
    p.play_sentence(["hello", "please"])
    assert time.perf_counter() - t0 < 0.05  # returns at once, the UI thread is free
    assert p.idle.wait(WAIT_S)
    (_, w1, t1), (_, w2, t2) = fake.calls
    assert (w1, w2) == ("hello", "please") and t2 - t1 >= 0.3 + 0.2 - 0.02  # gap + the second word


def test_new_request_replaces_the_old_one(tmp_path):
    fake = Fake(delay=0.3)
    p = player(tmp_path, fake, gap_s=0.0)
    p.play_sentence(["hello", "please", "hello", "please"])
    time.sleep(0.1)
    p.play_word("please")
    assert p.idle.wait(WAIT_S)
    words = [w for _, w, _ in fake.calls]
    assert words[-1] == "please" and len(words) <= 3  # the rest of the old sentence was dropped
    assert fake.stops >= 2  # a playing file is stopped by every new request


def test_errors_are_reported_not_raised(tmp_path):
    errors = []

    def broken(text):
        raise RuntimeError("no voice")

    p = audio.AudioPlayer(VOCAB, root=tmp_path, say=broken, stop=lambda: None, on_error=errors.append)
    p.play_word("hello")
    assert p.idle.wait(WAIT_S) and "no voice" in errors[0]


def test_postprocess_trims_silence_and_normalises(tmp_path):
    rate = 16000
    t = np.arange(int(0.4 * rate)) / rate
    word = 0.05 * np.sin(2 * np.pi * 220 * t)  # a quiet 0.4 s "word"
    x = np.concatenate([np.zeros(rate), word, np.zeros(rate)]) + 1e-5 * np.random.default_rng(0).standard_normal(
        2 * rate + len(word))
    audio.write_wav(tmp_path / "w.wav", x, rate)
    info = audio.postprocess_file(tmp_path / "w.wav")
    y, r = audio.read_wav(tmp_path / "w.wav")
    assert r == rate and info["before_s"] == pytest.approx(2.4, abs=0.01)
    assert 0.4 <= info["after_s"] <= 0.4 + 2 * config.AUDIO_PAD_S + 0.03
    assert info["peak_db"] <= config.AUDIO_PEAK_DB + 0.1
    assert info["rms_db"] == pytest.approx(config.AUDIO_TARGET_RMS_DB, abs=0.5)  # quiet word: the RMS target wins
    assert audio.postprocess(np.zeros(10), rate).tolist() == [0.0] * 10  # silence stays silence
    spiky = 0.01 * np.sin(2 * np.pi * 220 * t)
    spiky[len(t) // 2] = 0.5  # a click: reaching the RMS target would clip, so the peak limit wins
    out = audio.postprocess(spiky, rate)
    assert audio.db(np.abs(out).max()) == pytest.approx(config.AUDIO_PEAK_DB, abs=0.05)
    assert audio.db(np.sqrt(np.mean(out ** 2))) < config.AUDIO_TARGET_RMS_DB


def test_recording_plan(tmp_path):
    vocab = {"b": {"en": "b", "hy": "բ", "ru": "б"}, "a": {"en": "a", "hy": "ա", "ru": "а"}}
    audio.write_wav(tmp_path / "a.wav", 0.1 * np.ones(800), 16000)
    assert audio.missing_signs(vocab, tmp_path) == ["b"]
    assert audio.check_file(tmp_path / "a.wav") == ""
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    assert audio.check_file(tmp_path / "bad.wav").startswith("unreadable")
    n = audio.write_audio_script(vocab, tmp_path / "script.csv")
    lines = (tmp_path / "script.csv").read_text(encoding="utf-8-sig").splitlines()
    assert n == 2 and lines == ["file,hy,en,ru", "a.wav,ա,a,а", "b.wav,բ,b,б"]
