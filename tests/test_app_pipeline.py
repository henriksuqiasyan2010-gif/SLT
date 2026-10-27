import csv
import glob

import numpy as np
import pytest

from asl import config
from asl.app.pipeline import (Clip, FrameStatus, HintTracker, Phase, PredictionLog, SingleWordRecorder, classify,
                              frame_status)
from asl.capture import RecordedSource, load_recording, recording_aspect
from asl.landmarks_io import POSE, RIGHT_HAND


def empty_frame():
    return np.full((543, 3), np.nan, np.float32)


def test_frame_status():
    f = empty_frame()
    assert frame_status(f) == FrameStatus(hands=False, shoulders=False)
    f[POSE.start + 11, :2] = (0.6, 0.7)
    f[POSE.start + 12, :2] = (0.4, 0.7)
    f[RIGHT_HAND.start, :2] = (0.5, 0.5)
    assert frame_status(f) == FrameStatus(hands=True, shoulders=True)
    f[POSE.start + 12, 1] = 1.2  # shoulder below the image
    assert not frame_status(f).shoulders


def test_hint_tracker_waits_before_complaining():
    t = HintTracker(delay=1.0)
    nothing = FrameStatus(hands=False, shoulders=False)
    assert t.update(nothing, 10.0) == []
    assert t.update(nothing, 10.9) == []
    assert t.update(nothing, 11.1) == ["hint_no_shoulders", "hint_no_hands"]
    assert t.update(FrameStatus(hands=True, shoulders=False), 11.2) == ["hint_no_shoulders"]


def test_recorder_countdown_then_fixed_length():
    rec = SingleWordRecorder(countdown_s=3)
    assert rec.start(0.0, record_s=2.0)
    assert not rec.start(0.5, record_s=2.0)  # already running
    assert rec.overlay(0.1) == {"phase": "countdown", "count": 3}
    assert rec.overlay(2.5)["count"] == 1
    clip = None
    times = np.round(np.arange(0.0, 6.0, 0.1), 3)
    for t in times:
        clip = rec.feed(empty_frame(), t, t) or clip
    assert rec.phase is Phase.IDLE and clip is not None
    assert clip.timestamps[0] == pytest.approx(3.0) and clip.timestamps[-1] == pytest.approx(5.0)
    assert clip.fps == pytest.approx(10.0)


class FakeModel:
    metrics = {"preset": "fake"}

    def __init__(self, p_top=0.9):
        self.p_top = p_top
        self.calls = 0

    def predict_clip(self, clip, fps, aspect):
        self.calls += 1
        return np.array([self.p_top, 0.06, 0.04])

    def top_k(self, probs, k):
        return list(zip(["hello", "hat", "cowboy"], probs.tolist()))[:k]


def hand_clip(n=20, fps=15.0):
    lm = np.stack([empty_frame() for _ in range(n)])
    lm[:, RIGHT_HAND, :2] = 0.5 + 0.01 * np.arange(n)[:, None, None]
    return Clip(lm, np.arange(n) / fps)


def test_classify_threshold():
    pred = classify(FakeModel(0.9), hand_clip(), aspect=4 / 3, threshold=0.4)
    assert pred.accepted and pred.sign == "hello" and len(pred.top) == 3
    assert pred.fps == pytest.approx(15.0) and pred.hand_share == 1.0
    low = classify(FakeModel(0.3), hand_clip(), aspect=4 / 3, threshold=0.4)
    assert not low.accepted and low.sign is None and low.top[0] == ("hello", 0.3)


def test_classify_without_hands_or_frames_skips_the_model():
    model = FakeModel()
    no_hands = Clip(np.stack([empty_frame()] * 10), np.arange(10) / 15.0)
    one_hand_frame = hand_clip(44, fps=21.0)  # live: a hand seen in 1 of 44 frames gave "noisy 60%"
    one_hand_frame.landmarks[1:, RIGHT_HAND] = np.nan
    for clip in (no_hands, one_hand_frame, Clip(np.zeros((0, 543, 3), np.float32), np.zeros(0))):
        pred = classify(model, clip, aspect=4 / 3, threshold=0.4)
        assert pred.top == [] and not pred.accepted
    assert model.calls == 0
    enough = hand_clip(44, fps=15.0)
    enough.landmarks[3:, RIGHT_HAND] = np.nan  # 3 frames = 0.2 s at 15 fps
    assert classify(model, enough, aspect=4 / 3, threshold=0.4).top and model.calls == 1


def test_prediction_log_row_and_attempt(tmp_path):
    clip = hand_clip()
    pred = classify(FakeModel(), clip, 4 / 3, 0.4)
    log = PredictionLog(tmp_path / "p.csv", tmp_path / "att")
    log.append(pred, clip, "single", "fake", {"frame_width": 640, "frame_height": 480})
    log.append(pred, clip, "single", "fake", {"frame_width": 640, "frame_height": 480})
    rows = list(csv.DictReader(open(tmp_path / "p.csv", encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["top1"] == "hello" and rows[0]["p1"] == "0.9" and rows[0]["accepted"] == "True"
    saved, meta = load_recording(tmp_path / "att" / rows[0]["attempt"])
    np.testing.assert_array_equal(saved, clip.landmarks)
    assert meta["predicted"] == "hello" and recording_aspect(meta) == pytest.approx(4 / 3)


def recordings():
    files = sorted(glob.glob(str(config.RECORDINGS_DIR / "*.npy")))
    if not files:
        pytest.skip("no recordings/*.npy")
    return files


def test_full_pipeline_on_recordings_without_camera(tmp_path):
    """RecordedSource -> recorder -> classify -> log, as the worker does it, on every recording."""
    from asl.inference import ModelError, SignModel
    try:
        model = SignModel.load()
    except ModelError:
        pytest.skip("no model in models/")
    log = PredictionLog(tmp_path / "p.csv", tmp_path / "att")
    for path in recordings():
        src = RecordedSource(path, realtime=False)
        rec = SingleWordRecorder(countdown_s=0.0)
        rec.start(0.0, record_s=src.timestamps[-1])  # record the whole file
        clip = None
        for frame in src:
            clip = rec.feed(frame.landmarks, frame.timestamp, frame.timestamp) or clip
        assert clip is not None and len(clip.landmarks) == len(src.clip)
        aspect = recording_aspect(src.meta)
        pred = classify(model, clip, aspect, config.CONFIDENCE_THRESHOLD)
        expected = model.top_k(model.predict_clip(src.clip, src.fps, aspect), config.TOP_K)
        assert [s for s, _ in pred.top] == [s for s, _ in expected]
        log.append(pred, clip, "single", model.metrics["preset"], dict(src.meta))
    rows = list(csv.DictReader(open(tmp_path / "p.csv", encoding="utf-8")))
    assert len(rows) == len(recordings())
