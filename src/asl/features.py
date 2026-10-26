"""THE feature function: the only code that turns landmarks into model input.

    features, mask = extract(clip)          # clip: (T, 543, 3) float32, NaN = missing

Kaggle kernels and the app import this module; never re-implement any of it elsewhere.
NumPy only, deterministic. Steps, in order:

1. y is converted to the unit of x (fraction of the image width) with the frame aspect ratio,
   because the dataset videos are portrait 3:4 and our camera is landscape 4:3.
2. Leading and trailing frames without any hand are trimmed (none are trimmed if no frame has a hand).
3. Centre = median mid-point of the shoulders, scale = median shoulder width over the clip.
   Without shoulders: mean of the available points and their robust spread.
4. Dominant hand = hand seen in more frames (tie -> right). If it is the left hand, x is mirrored
   and every left/right pair (hands, pose arms, eyes, lips) is swapped, so the dominant hand is
   always in the right-hand slot. Both hands stay in the features.
5. The 130-point subset is resampled to t_fix frames by linear interpolation in time - always,
   also for short clips, so a clip recorded at 15 fps and the same clip at 30 fps give the same
   features.
6. Velocity = difference of neighbouring frames, computed after step 5 (FPS independent) on
   positions lightly smoothed in time (FEATURES_VELOCITY_SMOOTH_SIGMA), because Holistic jitter
   differs between 30 fps and 15 fps; the position features are not smoothed.
7. NaN -> 0; per-frame hand-presence flags tell the model where 0 means "missing".

Per frame (D = 2, or 3 with FEATURES_USE_Z): positions 130*D | velocity 130*D |
hand present (dominant, other) 2 | fingertip distances 10 per hand (FEATURES_FINGERTIP_DISTANCES).
The mask is True for every frame of a non-empty clip (all frames are real after resampling) and
all False for a clip without frames.
"""

import numpy as np

from asl import config
from asl.landmarks_io import FACE, LEFT_HAND, N_LANDMARKS, POSE, RIGHT_HAND

# Bump on ANY change of the output for the same input. Saved in model metadata (Phase 3);
# the app refuses a model trained with another version.
FEATURES_VERSION = 1

# --- Landmark subset (130 points), indices into the 543-point frame ---------------------------
# Face part follows the 1st-place solution of the Kaggle competition (lips, nose, eyes); pose part
# adds both arms (11-22: shoulders, elbows, wrists, pinky/index/thumb), which that solution lacked.
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # outer
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # inner
NOSE = [1, 2, 98, 327]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]  # person's right
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
FACE_POINTS = LIPS + NOSE + RIGHT_EYE + LEFT_EYE  # 76
POSE_POINTS = list(range(11, 23))  # 12
HAND_POINTS = 21

SUBSET = np.array(
    [LEFT_HAND.start + i for i in range(HAND_POINTS)]
    + [RIGHT_HAND.start + i for i in range(HAND_POINTS)]
    + [POSE.start + i for i in POSE_POINTS]
    + [FACE.start + i for i in FACE_POINTS]
)
N_POINTS = len(SUBSET)  # 130
SUB_LEFT_HAND = slice(0, HAND_POINTS)
SUB_RIGHT_HAND = slice(HAND_POINTS, 2 * HAND_POINTS)
FINGERTIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(i, j) for i in range(len(FINGERTIPS)) for j in range(i + 1, len(FINGERTIPS))]  # 10, into FINGERTIPS

# --- Left/right pairs, swapped together with the x-mirror ------------------------------------
POSE_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
# Face-mesh mirror pairs taken from the order of the MediaPipe eye and lip contours
# (checked against data in tests/test_features.py). Midline points 0, 13, 14, 17, 1, 2 map to themselves.
FACE_PAIRS = [
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),  # outer lip, lower
    (185, 409), (40, 270), (39, 269), (37, 267),  # outer lip, upper
    (78, 308), (95, 324), (88, 318), (178, 402), (87, 317),  # inner lip, lower
    (191, 415), (80, 310), (81, 311), (82, 312),  # inner lip, upper
    (98, 327),  # nostrils
    (33, 263), (7, 249), (163, 390), (144, 373), (145, 374), (153, 380), (154, 381), (155, 382),
    (133, 362), (246, 466), (161, 388), (160, 387), (159, 386), (158, 385), (157, 384), (173, 398),
]


def _mirror_permutation() -> np.ndarray:
    """perm[i] = subset position whose data goes to position i after a left/right swap."""
    pos = {int(p): i for i, p in enumerate(SUBSET)}
    perm = np.arange(N_POINTS)
    swaps = [(LEFT_HAND.start + k, RIGHT_HAND.start + k) for k in range(HAND_POINTS)]
    swaps += [(POSE.start + a, POSE.start + b) for a, b in POSE_PAIRS]
    swaps += [(FACE.start + a, FACE.start + b) for a, b in FACE_PAIRS]
    for a, b in swaps:
        perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    return perm


MIRROR_PERM = _mirror_permutation()
_SUB_POSE = {p: N_POINTS - len(FACE_POINTS) - len(POSE_POINTS) + i for i, p in enumerate(POSE_POINTS)}
L_SHOULDER, R_SHOULDER = _SUB_POSE[11], _SUB_POSE[12]


def feature_dim(use_z: bool = config.FEATURES_USE_Z,
                fingertips: bool = config.FEATURES_FINGERTIP_DISTANCES) -> int:
    d = 3 if use_z else 2
    return 2 * N_POINTS * d + 2 + (2 * len(TIP_PAIRS) if fingertips else 0)


# --- Steps -------------------------------------------------------------------------------------

def hand_presence(clip: np.ndarray):
    """(left, right) boolean arrays (T,): the hand block has at least one non-NaN x."""
    return (~np.isnan(clip[:, LEFT_HAND, 0]).all(axis=1),
            ~np.isnan(clip[:, RIGHT_HAND, 0]).all(axis=1))


def trim_to_hands(clip: np.ndarray):
    """(start, stop) of the frames from the first to the last frame with any hand; whole clip if none."""
    left, right = hand_presence(clip)
    idx = np.flatnonzero(left | right)
    if len(idx) == 0:
        return 0, len(clip)
    return int(idx[0]), int(idx[-1]) + 1


def dominant_is_left(clip: np.ndarray) -> bool:
    """Left hand seen in strictly more frames than the right one (tie -> right, no mirroring)."""
    left, right = hand_presence(clip)
    return int(left.sum()) > int(right.sum())


def _center_scale(pts: np.ndarray):
    """pts: (T, N_POINTS, 2) x,y of the subset. Centre (2,) and scale (float) of the clip."""
    sh = pts[:, [L_SHOULDER, R_SHOULDER]]
    ok = ~np.isnan(sh).any(axis=(1, 2))
    if ok.any():
        mid = sh[ok].mean(axis=1)
        width = np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1)
        center, scale = np.median(mid, axis=0), float(np.median(width))
        if scale > config.FEATURES_MIN_SCALE:
            return center, scale
    flat = pts.reshape(-1, 2)
    flat = flat[~np.isnan(flat).any(axis=1)]
    if len(flat) == 0:
        return np.zeros(2, dtype=np.float32), 1.0
    center = flat.mean(axis=0)
    spread = float(np.median(np.linalg.norm(flat - center, axis=1)))
    return center, max(spread * config.FEATURES_FALLBACK_SCALE_K, config.FEATURES_MIN_SCALE)


def _resample(x: np.ndarray, t_out: int) -> np.ndarray:
    """Linear interpolation along axis 0 to t_out frames. Where one neighbour is NaN the other is
    used; NaN only where both are NaN."""
    t_in = len(x)
    if t_in == 1:
        return np.repeat(x, t_out, axis=0)
    pos = np.linspace(0.0, t_in - 1, t_out)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t_in - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    a, b = x[i0], x[i1]
    out = (1 - w) * a + w * b
    out = np.where(np.isnan(a), b, out)
    out = np.where(np.isnan(b), a, out)
    return out


def _smooth_time(x: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along axis 0 (normalized convolution); NaN stays NaN."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    valid = ~np.isnan(x)
    pad = [(r, r)] + [(0, 0)] * (x.ndim - 1)
    xv = np.pad(np.where(valid, x, 0.0), pad)
    vv = np.pad(valid.astype(np.float64), pad)
    t = len(x)
    num = sum(w[k] * xv[k:k + t] for k in range(2 * r + 1))
    den = sum(w[k] * vv[k:k + t] for k in range(2 * r + 1))
    out = num / np.maximum(den, 1e-12)
    out[~valid] = np.nan
    return out


def extract(clip: np.ndarray, t_fix: int = config.T_FIX, aspect: float = config.DATASET_ASPECT,
            use_z: bool = config.FEATURES_USE_Z, fingertips: bool = config.FEATURES_FINGERTIP_DISTANCES):
    """(T, 543, 3) landmarks -> (features (t_fix, F) float32, mask (t_fix,) bool).

    aspect: frame width / height of the source (config.DATASET_ASPECT for the dataset,
    config.CAMERA_ASPECT for the live camera and our recordings).
    """
    clip = np.asarray(clip, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"expected (T, {N_LANDMARKS}, 3), got {clip.shape}")
    dim = feature_dim(use_z, fingertips)
    if len(clip) == 0:
        return np.zeros((t_fix, dim), np.float32), np.zeros(t_fix, bool)

    start, stop = trim_to_hands(clip)
    clip = clip[start:stop]
    mirror = dominant_is_left(clip)

    d = 3 if use_z else 2
    pts = clip[:, SUBSET, :d].astype(np.float64)
    pts[..., 1] /= aspect  # y in units of the image width
    center, scale = _center_scale(pts[..., :2])
    pts[..., :2] = (pts[..., :2] - center) / scale
    if use_z:
        pts[..., 2] /= scale  # Holistic z is in roughly the same unit as x
    if mirror:
        pts = pts[:, MIRROR_PERM]
        pts[..., 0] = -pts[..., 0]

    pts = _resample(pts, t_fix)  # (t_fix, N_POINTS, d)
    smooth = _smooth_time(pts, config.FEATURES_VELOCITY_SMOOTH_SIGMA)
    vel = np.diff(smooth, axis=0, prepend=smooth[:1])
    other = ~np.isnan(pts[:, SUB_LEFT_HAND, 0]).all(axis=1)  # after mirroring: left slot = other hand
    dom = ~np.isnan(pts[:, SUB_RIGHT_HAND, 0]).all(axis=1)

    parts = [np.nan_to_num(pts).reshape(t_fix, -1), np.nan_to_num(vel).reshape(t_fix, -1),
             np.stack([dom, other], axis=1).astype(np.float64)]
    if fingertips:
        for hand in (SUB_RIGHT_HAND, SUB_LEFT_HAND):  # dominant first
            tips = pts[:, hand][:, FINGERTIPS, :2]
            dist = np.stack([np.linalg.norm(tips[:, i] - tips[:, j], axis=-1) for i, j in TIP_PAIRS], axis=1)
            parts.append(np.nan_to_num(dist))
    features = np.concatenate(parts, axis=1).astype(np.float32)
    assert features.shape == (t_fix, dim)
    return features, np.ones(t_fix, dtype=bool)


def extract_info(clip: np.ndarray, aspect: float = config.DATASET_ASPECT) -> dict:
    """What extract() decides about a clip, for reports and tests (not model input)."""
    clip = np.asarray(clip, dtype=np.float32)
    start, stop = trim_to_hands(clip)
    trimmed = clip[start:stop]
    left, right = hand_presence(trimmed)
    pts = trimmed[:, SUBSET, :2].astype(np.float64)
    pts[..., 1] /= aspect
    sh = pts[:, [L_SHOULDER, R_SHOULDER]]
    return {
        "frames": len(clip), "trim_start": start, "trim_stop": stop, "trimmed_frames": stop - start,
        "left_frames": int(left.sum()), "right_frames": int(right.sum()),
        "mirrored": dominant_is_left(trimmed),
        "shoulders_used": bool((~np.isnan(sh).any(axis=(1, 2))).any()),
    }
