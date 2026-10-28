"""Choice of a reference (example) clip for a sign: what the user is shown to learn the sign.

Rule (user's decision, 2026-10-26; also for the Learn-mode references in Phase 6):
1. the hand must be found in most frames: dominant-hand frames / clip frames > REFERENCE_MIN_HAND_SHARE;
2. among those, the clip whose length is closest to the median length of that sign in the dataset;
3. right-dominant clips are preferred because the user signs with the right hand
   (left-dominant ones are used only if no right-dominant clip passes rule 1);
4. ties: higher hand share, then path (deterministic).
"""

import json

import numpy as np
import pandas as pd

from asl import config
from asl.landmarks_io import LEFT_HAND, N_LANDMARKS, RIGHT_HAND
from asl.skeleton import DISPLAY_INDICES


def load_stats(path=config.DATASET_STATS_CSV) -> pd.DataFrame:
    """Per-clip statistics from the asl-stats kernel (reports/stats/stats.csv)."""
    return pd.read_csv(path)


def pick_reference(stats: pd.DataFrame, sign: str, prefer_right: bool = True) -> pd.Series:
    """Row of `stats` (path, participant_id, n_frames, left_frames, right_frames, ...) chosen by the rule."""
    clips = stats[stats["sign"] == sign].copy()
    if clips.empty:
        raise KeyError(f"no clips for sign '{sign}'")
    median = clips["n_frames"].median()
    dom = clips[["left_frames", "right_frames"]].max(axis=1)
    clips["hand_share"] = dom / clips["n_frames"]
    clips["right_dominant"] = clips["right_frames"] >= clips["left_frames"]
    clips["length_gap"] = (clips["n_frames"] - median).abs()

    good = clips[clips["hand_share"] > config.REFERENCE_MIN_HAND_SHARE]
    if good.empty:
        good = clips
    if prefer_right and good["right_dominant"].any():
        good = good[good["right_dominant"]]
    order = good.sort_values(["length_gap", "hand_share", "path"], ascending=[True, False, True])
    row = order.iloc[0].copy()
    row["sign_median_frames"] = median
    return row


# --- Learn-mode references (kernel asl-references, Phase 6) ----------------------------------

def reference_candidates(stats: pd.DataFrame, sign: str,
                         participants=config.VAL_PARTICIPANTS) -> pd.DataFrame:
    """Held-out clips of `sign` that may become its Learn-mode reference (the kernel then keeps the
    one the model is most confident about): rule 1 (hand share), rule 3 (right-dominant, if any)
    and a length within REFERENCE_LENGTH_TOL of the sign's median over the whole dataset; at least
    the REFERENCE_MIN_CANDIDATES clips closest to the median."""
    all_clips = stats[stats["sign"] == sign]
    if all_clips.empty:
        raise KeyError(f"no clips for sign '{sign}'")
    median = all_clips["n_frames"].median()
    clips = all_clips[all_clips["participant_id"].isin(participants)].copy()
    clips["hand_share"] = clips[["left_frames", "right_frames"]].max(axis=1) / clips["n_frames"]
    clips["right_dominant"] = clips["right_frames"] >= clips["left_frames"]
    clips["length_gap"] = (clips["n_frames"] - median).abs()
    clips["sign_median_frames"] = median
    good = clips[clips["hand_share"] > config.REFERENCE_MIN_HAND_SHARE]
    if good.empty:
        good = clips
    if good["right_dominant"].any():
        good = good[good["right_dominant"]]
    good = good.sort_values(["length_gap", "hand_share", "path"], ascending=[True, False, True])
    tol = max(config.REFERENCE_LENGTH_TOL_MIN, config.REFERENCE_LENGTH_TOL * median)
    near = good[good["length_gap"] <= tol]
    return near if len(near) >= config.REFERENCE_MIN_CANDIDATES else good.head(config.REFERENCE_MIN_CANDIDATES)


def mirror_for_display(clip):
    """A left-handed clip shown as a right-handed one: x -> 1 - x and the two hand blocks swapped
    (a geometric mirror; only for drawing, never for the model)."""
    out = np.array(clip, dtype=np.float32, copy=True)
    out[..., 0] = 1.0 - out[..., 0]
    out[:, LEFT_HAND], out[:, RIGHT_HAND] = out[:, RIGHT_HAND].copy(), out[:, LEFT_HAND].copy()
    return out


def to_display(clip):
    """(T, 543, 3) -> (T, K, 2) float16 of skeleton.DISPLAY_INDICES (the stored reference)."""
    return np.asarray(clip, dtype=np.float32)[:, DISPLAY_INDICES, :2].astype(np.float16)


def from_display(points):
    """(T, K, 2) stored points -> (T, 543, 3) float32 clip, NaN elsewhere (z is NaN)."""
    points = np.asarray(points, dtype=np.float32)
    clip = np.full((len(points), N_LANDMARKS, 3), np.nan, np.float32)
    clip[:, DISPLAY_INDICES, :2] = points
    return clip


def load_reference_index(root=config.REFERENCE_DIR) -> dict:
    """{sign: metadata} written by the asl-references kernel (references.json)."""
    return json.loads((root / "references.json").read_text(encoding="utf-8"))


def load_reference(sign: str, root=config.REFERENCE_DIR, index: dict = None):
    """(clip (T, 543, 3), meta) of a sign's reference; meta has fps, aspect, source, model prob."""
    index = index if index is not None else load_reference_index(root)
    meta = index[sign]
    return from_display(np.load(root / f"{sign}.npy")), meta


class ReferenceLibrary:
    """sign -> (clip, meta), loaded from assets/reference/ on first use. An empty library (no
    references.json) answers `sign in lib` with False, so Learn mode can say "no reference"."""

    def __init__(self, root=config.REFERENCE_DIR):
        self.root = root
        self.index = load_reference_index(root) if (root / "references.json").exists() else {}
        self._cache = {}

    def __contains__(self, sign) -> bool:
        return sign in self.index and (self.root / f"{sign}.npy").exists()

    def __getitem__(self, sign):
        if sign not in self._cache:
            self._cache[sign] = load_reference(sign, self.root, self.index)
        return self._cache[sign]

    def __len__(self) -> int:
        return len(self.index)
