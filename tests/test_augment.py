import numpy as np
import pytest

from asl import augment as A
from asl import config, features as F
from asl.landmarks_io import load_clip

ALWAYS = {**config.AUGMENT, **{k: 1.0 for k in config.AUGMENT if k.startswith("p_")}}
NEVER = {**config.AUGMENT, **{k: 0.0 for k in config.AUGMENT if k.startswith("p_")}}


@pytest.fixture(scope="module")
def xys(sample_parquets):
    return [load_clip(p)[:, F.SUBSET, :2] for p in sample_parquets]


def test_thin_index():
    idx = A.thin_index(60, 15.0)
    assert idx[0] == 0 and np.all(np.diff(idx) == 2) and len(idx) == 30
    assert len(A.thin_index(60, 13.0)) == 26
    assert A.thin_index(1, 13.0).tolist() == [0]


def test_subset_to_clip_roundtrip(xys):
    clip = A.subset_to_clip(xys[0])
    assert clip.shape == (len(xys[0]), 543, 3)
    np.testing.assert_array_equal(clip[:, F.SUBSET, :2], xys[0])


def test_no_augmentation_is_identity(xys):
    rng = np.random.default_rng(0)
    for xy in xys:
        np.testing.assert_allclose(A.augment(xy, 0.75, rng, NEVER), xy, atol=1e-6)


def test_augmented_clips_go_through_extract(xys):
    rng = np.random.default_rng(1)
    for xy in xys:
        for _ in range(5):
            out = A.augment(xy, 0.75, rng, ALWAYS)
            assert out.ndim == 3 and out.shape[1:] == (130, 2) and 1 <= len(out) <= len(xy)
            feats, mask = F.extract(A.subset_to_clip(out), t_fix=32, aspect=0.75)
            assert np.isfinite(feats).all() and mask.all()


def test_deterministic_for_a_seed(xys):
    a = A.augment(xys[5], 0.75, np.random.default_rng(7), ALWAYS)
    b = A.augment(xys[5], 0.75, np.random.default_rng(7), ALWAYS)
    np.testing.assert_array_equal(a, b)


def test_rotation_keeps_isotropic_shoulder_width(xys):
    p = {**NEVER, "p_rotate": 1.0}
    xy = next(x for x in xys if len(x) > 10)
    out = A.augment(xy, 0.75, np.random.default_rng(3), p)

    def width(v):
        v = v.astype(np.float64).copy()
        v[..., 1] /= 0.75
        return np.nanmedian(np.linalg.norm(v[:, F.L_SHOULDER] - v[:, F.R_SHOULDER], axis=1))

    assert width(out) == pytest.approx(width(xy), rel=1e-4)


def test_stretch_stays_within_ten_percent(xys):
    p = {**NEVER, "p_stretch": 1.0}
    xy = xys[2]
    for seed in range(20):
        out = A.augment(xy, 0.75, np.random.default_rng(seed), p)
        ratio = np.nanstd(out[..., 0]) / np.nanstd(xy[..., 0])
        assert 0.9 - 1e-6 <= ratio <= 1.1 + 1e-6


def test_hand_gap_removes_both_hands_for_a_while(xys):
    p = {**NEVER, "p_hand_gap": 1.0}
    xy = next(x for x in xys if len(x) >= 20)
    out = A.augment(xy, 0.75, np.random.default_rng(0), p)
    lost = np.isnan(out[:, :42, 0]).all(axis=1) & ~np.isnan(xy[:, :42, 0]).all(axis=1)
    assert lost.sum() >= int(len(xy) * 0.1) - 1
    assert not np.isnan(out[:, 42:, 0]).all()  # pose and face untouched
