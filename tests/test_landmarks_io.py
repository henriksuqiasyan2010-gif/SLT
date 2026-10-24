import numpy as np
import pandas as pd

from asl import landmarks_io as lio


def test_block_constants():
    assert lio.N_LANDMARKS == 543
    assert (lio.FACE, lio.LEFT_HAND, lio.POSE, lio.RIGHT_HAND) == (
        slice(0, 468), slice(468, 489), slice(489, 522), slice(522, 543))


def test_row_order_matches_block_constants(sample_parquets):
    expected_type = np.empty(543, dtype=object)
    expected_index = np.empty(543, dtype=np.int64)
    for name, size in lio.BLOCK_SPEC:
        s = getattr(lio, name.upper())
        expected_type[s] = name
        expected_index[s] = np.arange(size)
    for path in sample_parquets:
        df = pd.read_parquet(path, columns=["frame", "type", "landmark_index"])
        n_frames = df["frame"].nunique()
        assert len(df) == n_frames * 543, path
        types = df["type"].to_numpy().reshape(n_frames, 543)
        idx = df["landmark_index"].to_numpy().reshape(n_frames, 543)
        frames = df["frame"].to_numpy().reshape(n_frames, 543)
        assert (types == expected_type).all(), path
        assert (idx == expected_index).all(), path
        assert (frames == frames[:, :1]).all(), path


def test_fast_load_equals_checked_load(sample_parquets):
    for path in sample_parquets:
        fast = lio.load_clip(path)
        slow = lio.load_clip_checked(path)
        assert fast.dtype == np.float32
        assert fast.ndim == 3 and fast.shape[1:] == (543, 3)
        np.testing.assert_array_equal(fast, slow)


def test_missing_points_are_nan(sample_parquets):
    clips = [lio.load_clip(p) for p in sample_parquets]
    # In every clip at least one hand is missing in some frame; face is mostly present.
    assert any(np.isnan(c[:, lio.LEFT_HAND]).all(axis=(1, 2)).any() or
               np.isnan(c[:, lio.RIGHT_HAND]).all(axis=(1, 2)).any() for c in clips)
    for c in clips:
        hand = c[:, lio.RIGHT_HAND]
        # A hand is either entirely present or entirely missing in a frame.
        per_frame = np.isnan(hand).all(axis=(1, 2)) | ~np.isnan(hand).any(axis=(1, 2))
        assert per_frame.all()
