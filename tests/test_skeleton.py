import numpy as np

from asl import skeleton
from asl.landmarks_io import FACE, POSE, load_clip


def test_segments_skip_missing_points():
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    assert list(skeleton.segments(frame)) == []
    frame[522:543] = np.random.default_rng(0).random((21, 3))  # right hand only
    segs = list(skeleton.segments(frame))
    assert len(segs) == len(skeleton.HAND_EDGES)
    assert {part for part, _, _ in segs} == {"right_hand"}


def test_every_part_has_a_style():
    frame = np.random.default_rng(0).random((543, 3)).astype(np.float32)
    frame[468:489] = np.nan  # left hand missing -> stub
    parts = {part for part, _, _ in skeleton.segments(frame)}
    assert parts <= set(skeleton.STYLES)
    assert {"body", "head", "face", "lips", "right_hand", "left_hand_stub"} <= parts


def test_face_uses_body_ink():
    assert skeleton.STYLES["face"]["color"] == skeleton.STYLES["body"]["color"]
    assert skeleton.STYLES["head"]["color"] == skeleton.STYLES["body"]["color"]


def test_face_is_drawn_with_lips(monkeypatch):
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[FACE] = np.random.default_rng(0).random((468, 3))
    parts = {part for part, _, _ in skeleton.segments(frame)}
    assert {"head", "face", "lips"} <= parts
    monkeypatch.setattr(skeleton, "DRAW_FACE", False)
    assert list(skeleton.segments(frame)) == []


def test_neck_needs_shoulders_and_chin():
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[POSE.start + 11] = (0.4, 0.6, 0)
    frame[POSE.start + 12] = (0.6, 0.6, 0)
    frame[FACE.start + skeleton.CHIN] = (0.5, 0.4, 0)
    segs = list(skeleton.segments(frame))
    neck = [s for s in segs if s[0] == "body" and s[1][1] == 0.5]
    assert len(neck) == 1


def test_save_gif(tmp_path, sample_parquets):
    clip = load_clip(sample_parquets[0])[:5]
    out = tmp_path / "clip.gif"
    skeleton.save_gif(clip, out, title="test")
    assert out.read_bytes()[:6] in (b"GIF89a", b"GIF87a")
