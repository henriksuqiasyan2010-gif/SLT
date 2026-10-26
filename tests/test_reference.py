import pandas as pd
import pytest

from asl import reference


def make_stats(rows):
    return pd.DataFrame(rows, columns=["path", "participant_id", "sign", "n_frames", "left_frames", "right_frames"])


def test_picks_clip_near_median_with_hand_in_most_frames():
    stats = make_stats([
        ("a", 1, "hi", 20, 0, 18),   # median length, hand 90%
        ("b", 2, "hi", 21, 0, 5),    # near median but hand only 24%
        ("c", 3, "hi", 60, 0, 60),   # long
        ("d", 4, "hi", 10, 0, 10),
        ("e", 5, "hi", 22, 0, 12),   # same distance to the median as "a", lower hand share
        ("x", 6, "other", 20, 0, 20),
    ])
    row = reference.pick_reference(stats, "hi")
    assert row["path"] == "a" and row["sign_median_frames"] == 21


def test_prefers_right_dominant():
    stats = make_stats([
        ("left_exact", 1, "hi", 20, 20, 0),
        ("right_close", 2, "hi", 23, 0, 23),
        ("right_far", 3, "hi", 17, 0, 15),
    ])
    assert reference.pick_reference(stats, "hi")["path"] == "right_close"
    assert reference.pick_reference(stats, "hi", prefer_right=False)["path"] == "left_exact"


def test_falls_back_when_no_clip_has_hand_in_most_frames():
    stats = make_stats([("a", 1, "hi", 20, 0, 5), ("b", 2, "hi", 30, 0, 6)])
    assert reference.pick_reference(stats, "hi")["path"] in ("a", "b")


def test_unknown_sign():
    with pytest.raises(KeyError):
        reference.pick_reference(make_stats([("a", 1, "hi", 20, 0, 20)]), "nope")


def test_real_stats_if_present():
    from asl import config
    if not config.DATASET_STATS_CSV.exists():
        pytest.skip("reports/stats/stats.csv not downloaded")
    stats = reference.load_stats()
    for sign in ("hello", "please", "mom"):
        row = reference.pick_reference(stats, sign)
        assert row["hand_share"] > 0.5
        assert abs(row["n_frames"] - row["sign_median_frames"]) <= 3
