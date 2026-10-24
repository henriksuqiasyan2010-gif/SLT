import numpy as np

from asl import skeleton
from asl.landmarks_io import load_clip


def test_segments_skip_missing_points():
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    assert list(skeleton.segments(frame)) == []
    frame[522:543] = np.random.default_rng(0).random((21, 3))  # right hand only
    segs = list(skeleton.segments(frame))
    assert len(segs) == len(skeleton.HAND_EDGES)
    assert {c for c, _, _ in segs} == {skeleton.COLORS["right_hand"]}


def test_save_gif(tmp_path, sample_parquets):
    clip = load_clip(sample_parquets[0])[:5]
    out = tmp_path / "clip.gif"
    skeleton.save_gif(clip, out, title="test")
    assert out.read_bytes()[:6] in (b"GIF89a", b"GIF87a")


def test_face_is_drawn_with_lips(monkeypatch):
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[:468] = np.random.default_rng(0).random((468, 3))
    colors = {c for c, _, _ in skeleton.segments(frame)}
    assert {skeleton.COLORS["face"], skeleton.COLORS["lips"]} <= colors
    monkeypatch.setattr(skeleton, "DRAW_FACE", False)
    assert list(skeleton.segments(frame)) == []
