import glob
import time

import numpy as np
import pandas as pd
import pytest

from asl import config
from asl import features as F
from asl.capture import load_recording
from asl.landmarks_io import FACE, LEFT_HAND, POSE, RIGHT_HAND, load_clip


@pytest.fixture(scope="module")
def sample_clips(sample_parquets):
    return [load_clip(p) for p in sample_parquets]


@pytest.fixture(scope="module")
def recordings():
    files = sorted(glob.glob(str(config.RECORDINGS_DIR / "*.npy")))
    if not files:
        pytest.skip("no recordings/*.npy")
    return [load_recording(f)[0] for f in files]


def right_dominant(clips, min_frames=0):
    return [c for c in clips if not F.dominant_is_left(c) and len(c) >= min_frames
            and F.hand_presence(c)[1].sum() > 0]


def mirror_clip(clip):
    """What the camera would see if the person signed with the other hand: x -> 1-x and every
    left/right landmark pair swapped (hands, pose, face), as Holistic labels sides by the body."""
    out = clip.copy()
    out[..., 0] = 1.0 - out[..., 0]
    swap = [(LEFT_HAND.start + k, RIGHT_HAND.start + k) for k in range(21)]
    swap += [(POSE.start + a, POSE.start + b) for a, b in F.POSE_PAIRS + [(1, 4), (2, 5), (3, 6), (7, 8),
                                                                         (9, 10), (23, 24), (25, 26), (27, 28),
                                                                         (29, 30), (31, 32)]]
    swap += [(FACE.start + a, FACE.start + b) for a, b in F.FACE_PAIRS]
    for a, b in swap:
        out[:, [a, b]] = out[:, [b, a]]
    return out


# --- Basic contract ------------------------------------------------------------------------------

def test_subset_size_and_layout():
    assert F.N_POINTS == 130 and len(set(F.SUBSET.tolist())) == 130
    assert len(F.FACE_POINTS) == 76 and len(F.POSE_POINTS) == 12
    assert sorted(F.MIRROR_PERM.tolist()) == list(range(130))
    np.testing.assert_array_equal(F.MIRROR_PERM[F.MIRROR_PERM], np.arange(130))  # an involution
    assert F.feature_dim() == 2 * 130 * 2 + 2 + 20


def test_shape_dtype_no_nan(sample_clips):
    for clip in sample_clips:
        feats, mask = F.extract(clip)
        assert feats.shape == (config.T_FIX, F.feature_dim()) and feats.dtype == np.float32
        assert mask.shape == (config.T_FIX,) and mask.dtype == bool and mask.all()
        assert np.isfinite(feats).all()


@pytest.mark.parametrize("t_fix", [32, 64])
def test_t_fix_is_a_parameter(sample_clips, t_fix):
    feats, mask = F.extract(sample_clips[0], t_fix=t_fix)
    assert feats.shape[0] == t_fix and mask.shape == (t_fix,)


def test_deterministic(sample_clips):
    a, _ = F.extract(sample_clips[3])
    b, _ = F.extract(sample_clips[3].copy())
    np.testing.assert_array_equal(a, b)


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError):
        F.extract(np.zeros((5, 540, 3), np.float32))


def test_empty_clip():
    feats, mask = F.extract(np.zeros((0, 543, 3), np.float32))
    assert not mask.any() and not feats.any()


def test_clip_without_hands():
    clip = load_clip(sorted(config.SAMPLE_DIR.glob("train_landmark_files/*/*.parquet"))[0]).copy()
    clip[:, LEFT_HAND] = np.nan
    clip[:, RIGHT_HAND] = np.nan
    feats, mask = F.extract(clip)
    assert mask.all() and np.isfinite(feats).all()
    n = F.N_POINTS * 2
    presence = feats[:, 2 * n:2 * n + 2]
    assert not presence.any()
    hands = feats[:, :n].reshape(config.T_FIX, F.N_POINTS, 2)[:, :42]
    assert not hands.any()


# --- Invariances -------------------------------------------------------------------------------

def test_translation_and_scale_invariance(sample_clips):
    clip = sample_clips[5]
    base, _ = F.extract(clip)
    moved = clip.copy()
    moved[..., :2] = moved[..., :2] * 0.6 + np.array([0.15, -0.1], np.float32)  # same factor on x and y
    feats, _ = F.extract(moved)
    np.testing.assert_allclose(feats, base, atol=1e-4)


def test_aspect_ratio_is_removed(sample_clips):
    """The same scene filmed with another frame aspect gives the same features when aspect is passed."""
    clip = sample_clips[2]
    base, _ = F.extract(clip, aspect=config.DATASET_ASPECT)
    # Pixels are fixed; a 4:3 frame of the same width shows y squeezed by the ratio of the aspects.
    other = clip.copy()
    other[..., 1] *= config.CAMERA_ASPECT / config.DATASET_ASPECT
    feats, _ = F.extract(other, aspect=config.CAMERA_ASPECT)
    np.testing.assert_allclose(feats, base, atol=1e-4)


def test_left_dominant_clip_matches_right_dominant(sample_clips):
    clip = right_dominant(sample_clips, min_frames=10)[0]
    assert not F.dominant_is_left(clip)
    mirrored = mirror_clip(clip)
    assert F.dominant_is_left(mirrored)
    a, _ = F.extract(clip)
    b, _ = F.extract(mirrored)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_tie_keeps_right_and_does_not_mirror():
    clip = np.full((4, 543, 3), np.nan, np.float32)
    clip[:2, LEFT_HAND] = 0.3
    clip[2:, RIGHT_HAND] = 0.7
    assert not F.dominant_is_left(clip)
    clip[2, RIGHT_HAND] = np.nan  # now left 2 > right 1
    assert F.dominant_is_left(clip)


def test_dominant_hand_lands_in_the_right_slot(sample_clips):
    lefties = [c for c in sample_clips if F.dominant_is_left(c)]
    assert lefties, "sample has left-dominant clips"
    n = F.N_POINTS * 2
    for c in lefties:
        feats, _ = F.extract(c)
        dom, other = feats[:, 2 * n], feats[:, 2 * n + 1]
        assert dom.sum() >= other.sum()


MAX_HEAD_ROLL = 0.1  # eye-line slope (y difference / interocular distance)


def test_face_mirror_pairs_match_the_data(sample_clips, recordings):
    """Each FACE_PAIRS pair lies symmetric about the face midline in real frames."""
    pts = np.concatenate([c[:, FACE, :2] for c in sample_clips + recordings])
    pts = pts[~np.isnan(pts).any(axis=(1, 2))]
    roll = (pts[:, 33, 1] - pts[:, 263, 1]) / np.abs(pts[:, 33, 0] - pts[:, 263, 0])
    pts = pts[np.abs(roll) < MAX_HEAD_ROLL]  # a tilted head (dad: hand at the forehead) is not asymmetry
    mid = pts[:, [168, 6, 1, 152], 0].mean(axis=1, keepdims=True)
    io = np.abs(pts[:, 33, 0] - pts[:, 263, 0])[:, None]
    x = np.median((pts[..., 0] - mid) / io, axis=0)
    y = np.median((pts[..., 1] - pts[:, [1], 1]) / io, axis=0)
    for a, b in F.FACE_PAIRS:
        assert abs(x[a] + x[b]) < 0.06 and abs(y[a] - y[b]) < 0.06, (a, b)
    for m in (0, 13, 14, 17, 1, 2):
        assert abs(x[m]) < 0.06, m


# --- Trimming, FPS independence ------------------------------------------------------------------

def test_trim_to_hands():
    clip = np.full((10, 543, 3), np.nan, np.float32)
    clip[3, RIGHT_HAND] = 0.5
    clip[6, LEFT_HAND] = 0.5
    assert F.trim_to_hands(clip) == (3, 7)
    assert F.trim_to_hands(np.full((5, 543, 3), np.nan, np.float32)) == (0, 5)


def test_independent_of_fps(sample_clips):
    """A clip and the same clip with every second frame dropped (30 -> 15 fps) give close features:
    the distance between them is small compared with the distance between different clips.
    Positions must hold for every clip; velocity, which is small and partly Holistic jitter, on average."""
    clips = right_dominant(sample_clips, min_frames=30)
    assert len(clips) >= 3
    n = F.N_POINTS * 2
    full = [F.extract(c)[0] for c in clips]
    half = [F.extract(c[::2])[0] for c in clips]
    vel_ratios = []
    for i in range(len(clips)):
        between = [np.abs(full[i] - full[j]) for j in range(len(clips)) if j != i]
        pos_between = np.median([b[:, :n].mean() for b in between])
        vel_between = np.median([b[:, n:2 * n].mean() for b in between])
        pos_within = np.abs(full[i][:, :n] - half[i][:, :n]).mean()
        vel_within = np.abs(full[i][:, n:2 * n] - half[i][:, n:2 * n]).mean()
        assert pos_within < 0.5 * pos_between, (i, pos_within, pos_between)
        vel_ratios.append(vel_within / vel_between)
        # dropping frames removes some jitter, so |velocity| may shrink, but not collapse
        ratio = np.abs(half[i][:, n:2 * n]).mean() / np.abs(full[i][:, n:2 * n]).mean()
        assert 0.5 < ratio < 1.2, (i, ratio)
    assert np.mean(vel_ratios) < 0.5, vel_ratios


def test_velocity_smoothing_keeps_nan_and_constant():
    x = np.ones((10, 3, 2))
    x[4, 1] = np.nan
    y = F._smooth_time(x, 2.0)
    assert np.isnan(y[4, 1]).all() and np.allclose(np.nan_to_num(y, nan=1.0), 1.0)


# --- Real data and speed -------------------------------------------------------------------------

def test_user_recordings(recordings):
    for clip in recordings:
        feats, mask = F.extract(clip, aspect=config.CAMERA_ASPECT)
        assert feats.shape == (config.T_FIX, F.feature_dim()) and np.isfinite(feats).all() and mask.all()
        n = F.N_POINTS * 2
        assert feats[:, 2 * n].any()  # the dominant hand is there


def test_shoulders_are_unit_width(sample_clips):
    for clip in sample_clips:
        feats, _ = F.extract(clip)
        pos = feats[:, :F.N_POINTS * 2].reshape(-1, F.N_POINTS, 2)
        width = np.linalg.norm(pos[:, F.L_SHOULDER] - pos[:, F.R_SHOULDER], axis=1)
        assert np.median(width) == pytest.approx(1.0, abs=0.15)


def test_fallback_without_shoulders(sample_clips):
    clip = sample_clips[0].copy()
    clip[:, POSE] = np.nan
    feats, _ = F.extract(clip)
    assert np.isfinite(feats).all()
    assert F.extract_info(clip)["shoulders_used"] is False


def test_speed(sample_clips):
    clip = max(sample_clips, key=len)
    F.extract(clip)
    batches = []
    for _ in range(5):  # best of 5 batches: the laptop throttles and other tests load the CPU
        t0 = time.perf_counter()
        for _ in range(20):
            F.extract(clip)
        batches.append((time.perf_counter() - t0) / 20)
    per_clip = min(batches)
    assert per_clip < 0.005, f"{per_clip * 1000:.2f} ms for {len(clip)} frames"


# --- trim_idle ------------------------------------------------------------------------------------

def moving_hand_clip(n_still_before, n_moving, n_still_after, fps=30.0):
    """Shoulders 0.2 apart; the right hand still, then moving 1 shoulder width/s, then still."""
    t = n_still_before + n_moving + n_still_after
    clip = np.full((t, 543, 3), np.nan, np.float32)
    clip[:, POSE.start + 11] = (0.6, 0.5, 0)
    clip[:, POSE.start + 12] = (0.4, 0.5, 0)
    x = np.concatenate([np.zeros(n_still_before), np.arange(1, n_moving + 1) * 0.2 / fps,
                        np.full(n_still_after, n_moving * 0.2 / fps)])
    clip[:, RIGHT_HAND, 0] = 0.3 + x[:, None]
    clip[:, RIGHT_HAND, 1] = 0.4
    clip[:, RIGHT_HAND, 2] = 0.0
    return clip


def test_trim_idle_cuts_still_start_and_end():
    clip = moving_hand_clip(20, 30, 25)
    start, stop = F.trim_idle(clip, fps=30.0, aspect=1.0)
    margin = round(config.TRIM_IDLE_MARGIN_S * 30)
    assert abs(start - (20 - margin)) <= 3 and abs(stop - (50 + margin)) <= 3, (start, stop)


def test_trim_idle_speed_is_per_second():
    """The same motion recorded at 15 fps is cut at the same times."""
    clip = moving_hand_clip(20, 30, 24)
    s30 = F.trim_idle(clip, fps=30.0, aspect=1.0)
    s15 = F.trim_idle(clip[::2], fps=15.0, aspect=1.0)
    assert abs(s15[0] * 2 - s30[0]) <= 4 and abs(s15[1] * 2 - s30[1]) <= 4, (s30, s15)


def test_trim_idle_without_motion_falls_back_to_hands():
    clip = moving_hand_clip(10, 0, 10)
    clip[:3, RIGHT_HAND] = np.nan
    assert F.trim_idle(clip, fps=30.0, aspect=1.0) == F.trim_to_hands(clip) == (3, 20)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6, 8])
def test_trim_idle_short_clips(sample_clips, n):
    clip = sample_clips[0][:n] if len(sample_clips[0]) >= n else sample_clips[0]
    start, stop = F.trim_idle(clip, fps=30.0)
    assert 0 <= start < stop <= len(clip) or len(clip) == 0


def test_trim_idle_runs_on_every_sample_clip(sample_clips):
    for clip in sample_clips:
        start, stop = F.trim_idle(clip, fps=config.DATASET_FPS_ASSUMED)
        h0, h1 = F.trim_to_hands(clip)
        assert h0 <= start < stop <= h1


def from_subset(xy):
    """(T, 130, 2) subset x, y -> (T, 543, 3) clip with everything else NaN (what the train kernel does)."""
    clip = np.full((len(xy), 543, 3), np.nan, np.float32)
    clip[:, F.SUBSET, :2] = xy
    return clip


def test_extract_needs_only_the_subset(sample_clips):
    """Training rebuilds clips from the stored subset; the features must not change."""
    for clip in sample_clips:
        full, _ = F.extract(clip)
        sub, _ = F.extract(from_subset(clip[:, F.SUBSET, :2]))
        np.testing.assert_array_equal(full, sub)
        assert F.trim_idle(clip, 30.0) == F.trim_idle(from_subset(clip[:, F.SUBSET, :2]), 30.0)
        half, _ = F.extract(from_subset(clip[:, F.SUBSET, :2].astype(np.float16)))
        assert np.abs(half - full).max() < 0.05  # float16 storage of raw coordinates


def test_dataset_aspect_per_participant():
    assert F.dataset_aspect(30680) == config.DATASET_ASPECT_BY_PARTICIPANT[30680]
    assert F.dataset_aspect("62590") == config.DATASET_ASPECT_BY_PARTICIPANT[62590]
    assert F.dataset_aspect(16069) == config.DATASET_ASPECT


# --- keep_moving_hand (live path only) ------------------------------------------------------------

def with_still_other_hand(clip):
    """Right-dominant clip + a still left hand in every frame (a copy of the first right-hand pose)."""
    out = clip.copy()
    first = clip[F.hand_presence(clip)[1]][0, RIGHT_HAND]
    out[:, LEFT_HAND] = first + np.array([0.2, 0.0, 0.0], np.float32)
    return out


def test_keep_moving_hand_one_hand_is_untouched(sample_clips):
    for clip in sample_clips:  # the sample never shows both hands in one frame
        assert F.keep_moving_hand(clip) is clip


def test_keep_moving_hand_drops_the_still_hand(sample_clips):
    for clip in right_dominant(sample_clips, min_frames=10):
        both = with_still_other_hand(clip)
        assert (F.hand_presence(both)[0] & F.hand_presence(both)[1]).any()
        kept = F.keep_moving_hand(both)
        assert np.isnan(kept[:, LEFT_HAND]).all()
        np.testing.assert_array_equal(kept[:, RIGHT_HAND], clip[:, RIGHT_HAND])
        np.testing.assert_array_equal(F.extract(kept)[0], F.extract(clip)[0])
        assert not np.array_equal(F.extract(both)[0], F.extract(clip)[0])  # extract() itself keeps both hands
        # the same with the sides swapped: the still hand is the right one
        swapped = both.copy()
        swapped[:, LEFT_HAND], swapped[:, RIGHT_HAND] = both[:, RIGHT_HAND], both[:, LEFT_HAND]
        assert np.isnan(F.keep_moving_hand(swapped)[:, RIGHT_HAND]).all()
        assert not np.isnan(both[:, LEFT_HAND]).all()  # input not modified


def test_hand_paths_are_aspect_corrected():
    clip = np.full((3, 543, 3), np.nan, np.float32)
    clip[:, RIGHT_HAND, :2] = 0.5
    clip[:, RIGHT_HAND, 1] = [[0.0], [0.1], [0.2]]
    left, right = F.hand_paths(clip, aspect=0.5)
    assert left == 0.0 and right == pytest.approx(0.4)


def test_keep_moving_hand_on_two_handed_sick():
    files = sorted(glob.glob(str(config.RECORDINGS_DIR / "sick-2h_*.npy")))
    if not files:
        pytest.skip("no recordings/sick-2h_*.npy")
    for f in files:  # right hand at the forehead moves, the left one waits at the stomach
        clip, _ = load_recording(f)
        kept = F.keep_moving_hand(clip, config.CAMERA_ASPECT)
        assert np.isnan(kept[:, LEFT_HAND]).all() and not np.isnan(kept[:, RIGHT_HAND]).all()
