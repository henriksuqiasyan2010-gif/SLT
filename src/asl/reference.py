"""Choice of a reference (example) clip for a sign: what the user is shown to learn the sign.

Rule (user's decision, 2026-10-26; also for the Learn-mode references in Phase 6):
1. the hand must be found in most frames: dominant-hand frames / clip frames > REFERENCE_MIN_HAND_SHARE;
2. among those, the clip whose length is closest to the median length of that sign in the dataset;
3. right-dominant clips are preferred because the user signs with the right hand
   (left-dominant ones are used only if no right-dominant clip passes rule 1);
4. ties: higher hand share, then path (deterministic).
"""

import pandas as pd

from asl import config


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
