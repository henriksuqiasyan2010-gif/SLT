"""onnxruntime CPU latency of models/model.onnx on this laptop: median of 100 runs -> logs/bench.txt.

Also times the whole live path for one clip (trim_idle + extract + inference) on a recording.
"""

import glob
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402

RUNS = 100


def timed(fn, runs=RUNS):
    fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    model = SignModel.load()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((model.t_fix, model.n_features)).astype(np.float32)
    mask = np.ones(model.t_fix, bool)
    med, p95 = timed(lambda: model.predict(feats, mask))
    lines = [f"{datetime.now().isoformat(timespec='seconds')} {platform.processor()}",
             f"model: {model.metrics.get('preset')} T={model.t_fix} F={model.n_features} "
             f"threads={config.INFERENCE_THREADS}",
             f"inference: median {med:.2f} ms, p95 {p95:.2f} ms over {RUNS} runs "
             f"({'OK' if med < config.MAX_INFERENCE_MS else 'TOO SLOW'}, limit {config.MAX_INFERENCE_MS} ms)"]
    recs = sorted(glob.glob(str(config.RECORDINGS_DIR / "*.npy")))
    if recs:
        clip, meta = load_recording(recs[0])
        aspect = recording_aspect(meta)
        med2, p952 = timed(lambda: model.predict_clip(clip, meta["fps"], aspect))
        lines.append(f"full live path (trim_idle + extract + inference) on {Path(recs[0]).name}: "
                     f"median {med2:.2f} ms, p95 {p952:.2f} ms")
    text = "\n".join(lines)
    print(text)
    config.LOGS_DIR.mkdir(exist_ok=True)
    with open(config.LOGS_DIR / "bench.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()
