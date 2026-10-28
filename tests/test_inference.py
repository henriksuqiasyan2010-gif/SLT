import json

import numpy as np
import pytest

from asl import config, features
from asl.inference import ModelError, SignModel


class FakeSession:
    def __init__(self, n_classes=250):
        self.n = n_classes
        self.last = None

    def run(self, names, inputs):
        self.last = inputs
        b = inputs["features"].shape[0]
        p = np.zeros((b, self.n), np.float32)
        p[:, 7] = 0.7
        p[:, 3] = 0.2
        p[:, 9] = 0.1
        return [p]


def metrics(**kw):
    m = {"FEATURES_VERSION": features.FEATURES_VERSION, "T_FIX": 64, "F": features.feature_dim(), "preset": "test"}
    m.update(kw)
    return m


LABELS = {str(i): f"sign{i}" for i in range(250)}


def test_top_k_and_shapes():
    model = SignModel(FakeSession(), metrics(), LABELS)
    probs = model.predict(np.zeros((64, features.feature_dim()), np.float32), np.ones(64, bool))
    assert probs.shape == (250,)
    assert model.top_k(probs, 3) == [("sign7", pytest.approx(0.7)), ("sign3", pytest.approx(0.2)),
                                     ("sign9", pytest.approx(0.1))]
    assert model.session.last["mask"].dtype == bool


def test_refuses_other_features_version():
    with pytest.raises(ModelError, match="FEATURES_VERSION"):
        SignModel(FakeSession(), metrics(FEATURES_VERSION=features.FEATURES_VERSION + 1), LABELS)


def test_refuses_other_feature_dim():
    with pytest.raises(ModelError, match="F="):
        SignModel(FakeSession(), metrics(F=10), LABELS)


def test_predict_clip_uses_the_model_t_fix(sample_parquets):
    from asl.landmarks_io import load_clip
    model = SignModel(FakeSession(), metrics(T_FIX=32), LABELS)
    model.predict_clip(load_clip(sample_parquets[3]), fps=30.0, aspect=config.DATASET_ASPECT)
    assert model.session.last["features"].shape == (1, 32, features.feature_dim())


def test_load_reports_missing_files(tmp_path):
    with pytest.raises(ModelError, match="missing"):
        SignModel.load(tmp_path)


def test_real_model_if_present():
    if not (config.MODELS_DIR / "model.onnx").exists():
        pytest.skip("no models/model.onnx yet")
    meta = json.loads((config.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    if meta["FEATURES_VERSION"] != features.FEATURES_VERSION:  # an older model must be refused, not used
        with pytest.raises(ModelError, match="FEATURES_VERSION"):
            SignModel.load()
        return
    model = SignModel.load()
    probs = model.predict(np.zeros((model.t_fix, model.n_features), np.float32), np.ones(model.t_fix, bool))
    assert probs.shape == (len(model.labels),) and abs(float(probs.sum()) - 1) < 1e-3


def test_predict_clip_drops_a_still_second_hand(sample_parquets):
    """keep_moving_hand is part of the live path: a still extra hand does not reach the model."""
    from asl.landmarks_io import LEFT_HAND, RIGHT_HAND, load_clip
    model = SignModel(FakeSession(), metrics(T_FIX=32), LABELS)
    clip = next(c for c in map(load_clip, sample_parquets)
                if features.hand_presence(c)[1].any() and not features.hand_presence(c)[0].any())
    right = features.hand_presence(clip)[1]
    both = clip.copy()
    both[:, LEFT_HAND] = clip[right][0, RIGHT_HAND] + np.array([0.2, 0.0, 0.0], np.float32)
    seen = []
    for c, keep in ((clip, True), (both, True), (both, False)):
        model.predict_clip(c, 30.0, config.DATASET_ASPECT, keep_moving_hand=keep)
        seen.append(model.session.last["features"].copy())
    np.testing.assert_array_equal(seen[1], seen[0])
    assert not np.array_equal(seen[2], seen[0])
