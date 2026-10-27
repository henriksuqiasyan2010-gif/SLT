"""Training augmentation of raw landmarks (before features.extract()), shared by the Kaggle kernels.

Works on the stored subset: xy (T, 130, 2) = x, y of features.SUBSET in normalized image
coordinates, aspect not applied. Time changes first (camera fps 12-15, uneven speed, dropped
frames, hands lost for a while), then geometry in isotropic units (rotation, separate x/y stretch
for the residual aspect error). extract() itself removes translation, scale, handedness and the
overall duration, so those are not augmented.
"""

import numpy as np

from asl import config, features


def thin_index(n: int, fps_dst: float, fps_src: float = config.DATASET_FPS_ASSUMED) -> np.ndarray:
    """Frame indices kept when an fps_src clip of n frames is recorded at fps_dst instead."""
    idx = np.unique(np.round(np.arange(0.0, n, fps_src / fps_dst)).astype(int))
    return idx[idx < n]


def subset_to_clip(xy: np.ndarray) -> np.ndarray:
    """(T, 130, 2) subset x, y -> (T, 543, 3) clip for extract(); other points NaN."""
    clip = np.full((len(xy), 543, 3), np.nan, np.float32)
    clip[:, features.SUBSET, :2] = xy
    return clip


def augment(xy: np.ndarray, aspect: float, rng: np.random.Generator, p: dict = None) -> np.ndarray:
    """Random time and geometry changes of (T, 130, 2) raw landmarks; returns (T', 130, 2)."""
    p = p or config.AUGMENT
    xy = np.array(xy, dtype=np.float32)
    if rng.random() < p["p_fps"] and len(xy) > 2:
        xy = xy[thin_index(len(xy), rng.uniform(*p["fps_range"]))]
    t = len(xy)
    if rng.random() < p["p_warp"] and t >= 4:
        steps = rng.uniform(*p["warp_speed"], size=t)
        pos = np.concatenate([[0.0], np.cumsum(steps[:-1])])
        pos = pos / pos[-1] * (t - 1)
        xy = xy[np.round(pos).astype(int)]
    if rng.random() < p["p_drop"] and t > 4:
        keep = rng.random(t) >= rng.uniform(0, p["drop_max"])
        keep[[0, -1]] = True
        xy = xy[keep]
    t = len(xy)
    if rng.random() < p["p_hand_gap"] and t >= 4:
        n = max(1, int(t * rng.uniform(*p["hand_gap"])))
        s = rng.integers(0, t - n + 1)
        xy[s:s + n, :2 * features.HAND_POINTS] = np.nan  # both hand blocks: Holistic lost the hands

    # geometry in isotropic units (y / aspect) around the shoulder centre
    xy[..., 1] /= aspect
    sh = xy[:, [features.L_SHOULDER, features.R_SHOULDER]].reshape(-1, 2)
    sh = sh[~np.isnan(sh).any(axis=1)]
    if len(sh):
        center = sh.mean(axis=0)
    else:
        flat = xy.reshape(-1, 2)
        flat = flat[~np.isnan(flat).any(axis=1)]
        center = flat.mean(axis=0) if len(flat) else np.zeros(2, np.float32)
    pts = xy - center
    if rng.random() < p["p_rotate"]:
        ang = np.deg2rad(rng.uniform(-p["rotate_deg"], p["rotate_deg"]))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        pts = pts @ rot.T
    if rng.random() < p["p_stretch"]:
        pts = pts * rng.uniform(1 - p["stretch"], 1 + p["stretch"], size=2).astype(np.float32)
    xy = pts + center
    xy[..., 1] *= aspect
    return xy.astype(np.float32)
