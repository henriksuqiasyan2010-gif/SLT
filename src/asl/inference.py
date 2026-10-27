"""onnxruntime wrapper for the trained sign classifier.

    model = SignModel.load()                      # models/model.onnx + metrics.json + labels.json
    probs = model.predict(features, mask)         # (250,) for one clip from features.extract()
    model.top_k(probs, 3)                         # [(sign, prob), ...]
    model.predict_clip(clip, fps, aspect)         # trim_idle -> extract -> probs

The model is refused if it was trained with another FEATURES_VERSION or another T_FIX / F.
"""

import json
from pathlib import Path

import numpy as np

from asl import config, features


class ModelError(RuntimeError):
    """The model files are missing or do not match the feature code."""


class SignModel:
    def __init__(self, session, metrics: dict, labels: dict):
        self.session = session
        self.metrics = metrics
        self.labels = {int(k): v for k, v in labels.items()}
        self.t_fix = int(metrics["T_FIX"])
        self.n_features = int(metrics["F"])
        if int(metrics["FEATURES_VERSION"]) != features.FEATURES_VERSION:
            raise ModelError(f"model was trained with FEATURES_VERSION {metrics['FEATURES_VERSION']}, "
                             f"the app computes version {features.FEATURES_VERSION}")
        if self.n_features != features.feature_dim():
            raise ModelError(f"model expects F={self.n_features}, extract() gives {features.feature_dim()}")

    @classmethod
    def load(cls, model_dir=config.MODELS_DIR):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        files = [model_dir / n for n in ("model.onnx", "metrics.json", "labels.json")]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise ModelError(f"missing in {model_dir}: {', '.join(missing)}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.INFERENCE_THREADS
        session = ort.InferenceSession(str(files[0]), sess_options=opts, providers=["CPUExecutionProvider"])
        return cls(session, json.loads(files[1].read_text(encoding="utf-8")),
                   json.loads(files[2].read_text(encoding="utf-8")))

    def predict_batch(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """(B, T, F), (B, T) -> (B, n_classes) probabilities."""
        return self.session.run(["probs"], {"features": np.asarray(feats, np.float32),
                                            "mask": np.asarray(mask, bool)})[0]

    def predict(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.predict_batch(feats[None], mask[None])[0]

    def predict_clip(self, clip: np.ndarray, fps: float, aspect: float = config.CAMERA_ASPECT) -> np.ndarray:
        """Live path: cut the idle start/end (trim_idle), extract features, classify."""
        start, stop = features.trim_idle(clip, fps, aspect)
        feats, mask = features.extract(clip[start:stop], t_fix=self.t_fix, aspect=aspect)
        return self.predict(feats, mask)

    def top_k(self, probs: np.ndarray, k: int = 3):
        order = np.argsort(-probs)[:k]
        return [(self.labels[int(i)], float(probs[i])) for i in order]
