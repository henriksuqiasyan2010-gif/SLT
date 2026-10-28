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


# --- Learn-mode references (Phase 6) ------------------------------------------------------------

def test_candidates_are_held_out_right_dominant_and_near_the_median():
    val = 27610  # one of config.VAL_PARTICIPANTS
    stats = make_stats([
        ("a", val, "hi", 20, 0, 18),   # median length, right
        ("b", val, "hi", 22, 0, 20),   # within 20% of the median
        ("c", val, "hi", 21, 19, 0),   # left-dominant: right ones exist, so dropped
        ("d", val, "hi", 40, 0, 40),   # too long
        ("e", 1, "hi", 20, 0, 20),     # a training participant: never a candidate
        ("f", val, "hi", 19, 0, 5),    # hand in too few frames
        ("g", 2, "hi", 20, 0, 20),
    ])
    c = reference.reference_candidates(stats, "hi")
    # a and b are within the tolerance; fewer than REFERENCE_MIN_CANDIDATES, so the next closest (d) joins
    assert list(c["path"]) == ["a", "b", "d"]
    assert c["sign_median_frames"].iloc[0] == 20


def test_display_roundtrip_and_mirror():
    import numpy as np
    from asl.landmarks_io import LEFT_HAND, RIGHT_HAND
    from asl.skeleton import DISPLAY_INDICES
    clip = np.random.default_rng(0).random((5, 543, 3)).astype(np.float32)
    back = reference.from_display(reference.to_display(clip))
    np.testing.assert_allclose(back[:, DISPLAY_INDICES, :2], clip[:, DISPLAY_INDICES, :2], atol=1e-3)
    assert np.isnan(back[:, 0, 0]).all() or 0 in DISPLAY_INDICES
    m = reference.mirror_for_display(clip)
    np.testing.assert_allclose(m[:, RIGHT_HAND, 0], 1 - clip[:, LEFT_HAND, 0])
    np.testing.assert_allclose(m[:, LEFT_HAND, 1], clip[:, RIGHT_HAND, 1])


def test_reference_files_exist_for_all_250_signs():
    """PHASES.md Phase 6 Done when: 250 reference files."""
    from asl import config
    from asl.vocab import load_vocab
    lib = reference.ReferenceLibrary()
    if not len(lib):
        pytest.skip("no assets/reference/references.json")
    signs = set(load_vocab())
    assert len(signs) == config.NUM_CLASSES and set(lib.index) == signs
    assert all(s in lib for s in signs)
    clip, meta = lib["dad"]
    assert clip.shape[1:] == (543, 3) and len(clip) == meta["frames"] and meta["fps"] > 0
    assert meta["participant_id"] in config.VAL_PARTICIPANTS and meta["p"] > 0
