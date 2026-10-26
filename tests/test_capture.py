import time

import numpy as np
import pytest

from asl import capture
from asl.landmarks_io import LEFT_HAND, RIGHT_HAND


def make_clip(n=12, seed=0):
    return np.random.default_rng(seed).random((n, 543, 3)).astype(np.float32)


def test_save_and_load_roundtrip(tmp_path):
    clip = make_clip()
    ts = 100.0 + np.arange(len(clip)) / 20.0
    path = capture.save_recording(tmp_path / "x.npy", clip, ts, sign="hello")
    loaded, meta = capture.load_recording(path)
    np.testing.assert_array_equal(loaded, clip)
    assert meta["sign"] == "hello" and meta["frames"] == len(clip)
    assert meta["fps"] == pytest.approx(20.0, rel=1e-3)
    assert meta["timestamps"][0] == 0.0
    assert capture.meta_path(path).name == "x.meta.json"


def test_save_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        capture.save_recording(tmp_path / "x.npy", np.zeros((3, 540, 3)), [0, 1, 2])


def test_recorded_source_replays_frames(tmp_path):
    clip = make_clip()
    path = capture.save_recording(tmp_path / "x.npy", clip, np.arange(len(clip)) / 20.0)
    frames = list(capture.RecordedSource(path, realtime=False))
    assert len(frames) == len(clip)
    np.testing.assert_array_equal(np.stack([f.landmarks for f in frames]), clip)
    assert all(f.image is None for f in frames)


def test_recorded_source_keeps_original_pace(tmp_path):
    clip = make_clip(n=11)
    path = capture.save_recording(tmp_path / "x.npy", clip, np.arange(11) / 25.0)  # 0.4 s
    t0 = time.perf_counter()
    n = sum(1 for _ in capture.RecordedSource(path, realtime=True))
    elapsed = time.perf_counter() - t0
    assert n == 11
    assert 0.38 <= elapsed <= 0.6


def test_recording_without_sidecar_uses_default_fps(tmp_path):
    clip = make_clip(n=4)
    np.save(tmp_path / "bare.npy", clip)
    src = capture.RecordedSource(tmp_path / "bare.npy", realtime=False)
    assert src.fps == capture.config.DEFAULT_REPLAY_FPS
    assert [f.timestamp for f in src][1] == pytest.approx(1 / capture.config.DEFAULT_REPLAY_FPS)


def test_recorded_source_close_stops(tmp_path):
    path = capture.save_recording(tmp_path / "x.npy", make_clip(), np.arange(12) / 20.0)
    src = capture.RecordedSource(path, realtime=False, loop=True)
    got = []
    for f in src:
        got.append(f)
        if len(got) == 30:
            src.close()
    assert len(got) == 30


class FakeCapture:
    def __init__(self, n_frames):
        self.left = n_frames
        self.released = False

    def read(self):
        if self.left == 0:
            return False, None
        self.left -= 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True

    def getBackendName(self):
        return "FAKE"


def test_live_source_threaded_with_fake_camera():
    cam = FakeCapture(5)
    src = capture.LiveSource(capture_factory=lambda *a: cam, threaded=True)
    frames = []
    with pytest.raises(capture.CameraError, match="stopped"):
        for f in src:
            frames.append(f)
    assert 1 <= len(frames) <= 5  # the reader may drop frames while Holistic works
    assert cam.released
    assert all(f.landmarks.shape == (543, 3) for f in frames)


def test_settings_report_backend_and_frame_size_once_running():
    src = capture.LiveSource(capture_factory=lambda *a: FakeCapture(2), threaded=False)
    before = src.settings()
    assert before["backend"] is None and before["frame_width"] is None
    next(iter(src))
    after = src.settings()
    assert after["backend"] == "FAKE"
    assert (after["frame_width"], after["frame_height"]) == (640, 480)
    src.close()


def test_live_source_with_fake_camera():
    cam = FakeCapture(6)
    src = capture.LiveSource(capture_factory=lambda *a: cam, process_every_n=2, threaded=False)
    frames = []
    with pytest.raises(capture.CameraError):
        for f in src:
            frames.append(f)
    assert len(frames) == 3  # every second of 6 frames
    assert cam.released
    for f in frames:
        assert f.landmarks.shape == (543, 3) and f.image.shape == (480, 640, 3)
        assert np.isnan(f.landmarks[LEFT_HAND]).all() and np.isnan(f.landmarks[RIGHT_HAND]).all()


def test_missing_camera_raises_camera_error():
    def no_camera(*a):
        raise capture.CameraError("Camera 7 not found or busy")

    with pytest.raises(capture.CameraError, match="not found"):
        next(iter(capture.LiveSource(camera_index=7, capture_factory=no_camera)))
