import glob
import json

import numpy as np
import pytest

from asl import config
from asl.capture import load_recording, recording_aspect
from asl.landmarks_io import POSE
from asl.segmenter import Params, Segmenter, State, segment_clip, wrist_height

FPS = 15.0
REST, UP = -1.2, 0.3  # wrist height in shoulder widths: hands down / signing


def frame(height):
    """Synthetic frame: shoulders at y = 0.5 (width 0.3 in x), right wrist `height` widths above them."""
    f = np.full((543, 3), np.nan, np.float32)
    f[POSE.start + 11, :2] = (0.65, 0.5)
    f[POSE.start + 12, :2] = (0.35, 0.5)
    f[POSE.start + 16, :2] = (0.4, 0.5 - height * 0.3)  # aspect 1
    return f


def stream(plan):
    """plan: [(height, seconds), ...] -> clip, timestamps at FPS."""
    heights = np.concatenate([np.full(int(round(s * FPS)), h) for h, s in plan])
    return np.stack([frame(h) for h in heights]), np.arange(len(heights)) / FPS


def test_wrist_height():
    h, sh = wrist_height(frame(0.5), aspect=1.0)
    assert h == pytest.approx(0.5) and sh == (pytest.approx(0.5), pytest.approx(0.3))
    no_shoulders = frame(0.5)
    no_shoulders[POSE.start + 11] = np.nan
    assert wrist_height(no_shoulders, 1.0, sh)[0] == pytest.approx(0.5)  # last known shoulders
    assert np.isnan(wrist_height(no_shoulders, 1.0)[0])


def test_three_signs_with_pauses():
    clip, ts = stream([(REST, 1), (UP, 1.5), (REST, 1), (UP, 2), (REST, 0.6), (UP, 1), (REST, 1)])
    segs = segment_clip(clip, ts, aspect=1.0)
    assert len(segs) == 3
    assert [round(s.duration, 1) for s in segs] == [1.4, 1.9, 0.9]  # the down tail is not included


def test_short_drop_does_not_split_and_twitch_is_dropped():
    p = Params()
    clip, ts = stream([(REST, 1), (UP, 1), (REST, p.pause_s / 2), (UP, 1), (REST, 1),
                       (UP, p.min_sign_s / 2), (REST, 1)])
    segs = segment_clip(clip, ts, aspect=1.0)
    assert len(segs) == 1 and segs[0].duration > 2


def test_long_sign_is_closed_at_max_length_and_open_sign_is_flushed():
    p = Params(max_sign_s=2.0)
    clip, ts = stream([(REST, 0.5), (UP, 3.0)])
    segs = segment_clip(clip, ts, aspect=1.0, params=p)
    assert len(segs) == 2 and segs[0].duration == pytest.approx(2.0, abs=0.1)  # closed, then flushed


def test_states_for_the_overlay():
    seg = Segmenter(aspect=1.0)
    states = []
    clip, ts = stream([(REST, 0.5), (UP, 1.0), (REST, 1.0)])
    for f, t in zip(clip, ts):
        seg.update(f, t)
        states.append(seg.state)
    assert states[0] is State.IDLE and State.SIGN in states and State.PAUSE in states and states[-1] is State.IDLE


def sessions():
    files = sorted(glob.glob(str(config.SESSIONS_DIR / "session*.npy")))
    if not files:
        pytest.skip("no recordings/sessions/session*.npy")
    return files


def test_segment_count_on_every_session():
    """PHASES.md Phase 5: the number of detected segments equals the true count +- 1."""
    for path in sessions():
        clip, meta = load_recording(path)
        truth = json.loads(open(path.replace(".npy", ".json"), encoding="utf-8").read())["signs"]
        segs = segment_clip(clip, meta["timestamps"], recording_aspect(meta))
        assert abs(len(segs) - len(truth)) <= 1, (path, len(segs), len(truth))
