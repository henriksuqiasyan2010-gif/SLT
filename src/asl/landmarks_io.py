"""Load dataset parquet clips into the canonical landmark array (T, 543, 3) float32.

Row order inside every frame of the dataset is constant:
face (468) -> left_hand (21) -> pose (33) -> right_hand (21).
This is verified by tests/test_landmarks_io.py against the `type` and
`landmark_index` columns, which is what makes the fast reshape path valid.
"""

from pathlib import Path

import numpy as np

from asl.config import LANDMARK_COLUMNS

# (type name in the parquet, number of points), in dataset order.
BLOCK_SPEC = [("face", 468), ("left_hand", 21), ("pose", 33), ("right_hand", 21)]


def _block_slices():
    slices, start = {}, 0
    for name, size in BLOCK_SPEC:
        slices[name] = slice(start, start + size)
        start += size
    return slices, start


_SLICES, N_LANDMARKS = _block_slices()
FACE = _SLICES["face"]
LEFT_HAND = _SLICES["left_hand"]
POSE = _SLICES["pose"]
RIGHT_HAND = _SLICES["right_hand"]


def load_clip(path) -> np.ndarray:
    """Fast path: read x, y, z and reshape. Returns (T, 543, 3) float32, NaN for missing points."""
    import pandas as pd  # here, not at the top: the app never reads parquet and is built without pandas
    data = pd.read_parquet(Path(path), columns=LANDMARK_COLUMNS).to_numpy(dtype=np.float32)
    if data.shape[0] % N_LANDMARKS:
        raise ValueError(f"{path}: {data.shape[0]} rows is not a multiple of {N_LANDMARKS}")
    return data.reshape(-1, N_LANDMARKS, 3)


def load_clip_checked(path) -> np.ndarray:
    """Slow path: place every row by (frame, type, landmark_index). Used to validate load_clip."""
    import pandas as pd
    df = pd.read_parquet(Path(path))
    frames = np.sort(df["frame"].unique())
    frame_pos = {f: i for i, f in enumerate(frames)}
    offset = df["type"].map({name: s.start for name, s in _SLICES.items()})
    if offset.isna().any():
        raise ValueError(f"{path}: unknown landmark type {set(df['type'][offset.isna()])}")
    point = (offset + df["landmark_index"]).to_numpy(dtype=np.int64)
    t = df["frame"].map(frame_pos).to_numpy(dtype=np.int64)
    out = np.full((len(frames), N_LANDMARKS, 3), np.nan, dtype=np.float32)
    out[t, point] = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float32)
    return out
