import numpy as np
import pytest

from asl import skeleton
from asl.landmarks_io import FACE, POSE, load_clip


def test_segments_skip_missing_points():
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    assert list(skeleton.segments(frame)) == []
    frame[522:543] = np.random.default_rng(0).random((21, 3))  # right hand only
    segs = list(skeleton.segments(frame))
    assert len(segs) == len(skeleton.HAND_EDGES)
    assert {part for part, _, _ in segs} == {"right_hand"}


def blank_frame():
    return np.full((543, 3), np.nan, dtype=np.float32)


def body_segments(frame):
    return [(pytest.approx(xs, abs=1e-6), pytest.approx(ys, abs=1e-6))
            for part, xs, ys in skeleton.segments(frame) if part == "body"]


def test_every_part_has_a_style():
    frame = np.random.default_rng(0).random((543, 3)).astype(np.float32)
    frame[468:489] = np.nan  # left hand lost
    parts = {part for part, _, _ in skeleton.segments(frame)}
    assert parts <= set(skeleton.STYLES)
    assert {"body", "head", "face", "lips", "right_hand"} <= parts


def test_lost_hand_is_not_drawn():
    frame = np.random.default_rng(0).random((543, 3)).astype(np.float32)
    frame[468:489] = np.nan
    frame[522:543] = np.nan
    parts = {part for part, _, _ in skeleton.segments(frame)}
    assert not parts & {"left_hand", "right_hand"}
    assert not any("stub" in part for part in parts)


def test_pose_points_outside_the_frame_are_not_drawn():
    frame = blank_frame()
    frame[POSE.start + 11] = (0.6, 0.4, 0)   # person's left shoulder, inside
    frame[POSE.start + 13] = (0.7, 0.7, 0)   # left elbow, inside
    frame[POSE.start + 15] = (0.7, 1.4, 0)   # left wrist, below the frame
    frame[POSE.start + 23] = (0.6, 1.6, 0)   # left hip, below the frame
    segs = body_segments(frame)
    assert segs == [([0.6, 0.7], [0.4, 0.7])]  # only shoulder-elbow


def test_forearm_stays_while_the_hand_is_lost():
    frame = blank_frame()
    frame[POSE.start + 14] = (0.3, 0.7, 0)   # right elbow
    frame[POSE.start + 16] = (0.35, 0.3, 0)  # right wrist, inside
    assert body_segments(frame) == [([0.3, 0.35], [0.7, 0.3])]
    frame[POSE.start + 16] = (0.35, 1.2, 0)  # wrist leaves the frame -> forearm disappears
    assert body_segments(frame) == []


def test_forearm_ends_at_the_detected_hand_wrist():
    frame = blank_frame()
    frame[POSE.start + 14] = (0.3, 0.7, 0)
    frame[POSE.start + 16] = (0.35, 0.3, 0)
    frame[522:543] = 0.5
    frame[522] = (0.36, 0.31, 0)  # hand wrist
    assert body_segments(frame) == [([0.3, 0.36], [0.7, 0.31])]


def test_display_bounds_ignore_pose_points_outside_the_frame():
    clip = np.full((2, 543, 3), np.nan, dtype=np.float32)
    clip[:, POSE.start + 11] = (0.4, 0.4, 0)
    clip[:, POSE.start + 12] = (0.6, 0.4, 0)
    clip[:, POSE.start + 23] = (0.4, 1.8, 0)  # hip far below the frame
    x0, x1, y0, y1 = skeleton.display_bounds(clip, margin=0)
    assert (x0, x1, y0, y1) == pytest.approx((0.4, 0.6, 0.4, 0.4))


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


def test_draw_cv2_draws_in_pixels(sample_parquets):
    clip = load_clip(sample_parquets[0])
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = skeleton.draw_cv2(img, clip[len(clip) // 2])
    assert out is img and img.any()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    skeleton.draw_cv2(blank, np.full((543, 3), np.nan, dtype=np.float32))
    assert not blank.any()
