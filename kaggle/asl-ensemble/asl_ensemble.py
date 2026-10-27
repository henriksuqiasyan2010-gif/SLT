"""Kaggle kernel asl-ensemble (CPU): would averaging trained models help? Measured, not guessed.

Loads the ONNX models of finished asl-train kernels (MODELS, mounted as kernel sources), runs them
with onnxruntime on the same held-out participants as training (GroupKFold(5), fold 0) on the
original and on the 13 fps clips, and reports top-1 / top-3 for every model alone and for the
average of the probabilities of each combination of models. No GPU: one pass over 18,528 clips.

The models share SEED 42 (same initialisation, same batch order), so the gain measured here is a
lower bound for an ensemble trained with different seeds.

Outputs in /kaggle/working: ensemble.json, per_participant.csv (top-1 per val participant and model).
"""

import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"main": "main_T32", "main-reg": "main-reg_T32", "main-reg30": "main-reg30_T32"}
N_FOLDS = 5
VAL_FOLD = 0
EVAL_FPS = 13.0
BATCH = 512


def find_one(pattern):
    hits = sorted(Path("/kaggle/input").glob(pattern))
    if not hits:
        raise FileNotFoundError(f"/kaggle/input/{pattern}")
    return hits[0]


# Kaggle unpacks the asl-src zip without its top folder, so rebuild the `asl` package under /tmp.
SRC_DIR = Path("/tmp/asl_src")
shutil.copytree(find_one("**/landmarks_io.py").parent, SRC_DIR / "asl", dirs_exist_ok=True)
sys.path.insert(0, str(SRC_DIR))
from asl import augment as A, features as F  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
    import onnxruntime as ort

PRE_DIR = find_one("**/X32.npy").parent
OUT_DIR = Path("/kaggle/working")
_RAW = None


def _thinned_one(args):
    """Same as val_thinned_features() in asl_train.py: raw subset -> 13 fps -> extract()."""
    global _RAW
    if _RAW is None:
        _RAW = (np.load(PRE_DIR / "raw_xy.npy", mmap_mode="r"), np.load(PRE_DIR / "raw_offsets.npy"))
    i, group, t = args
    xy, off = _RAW
    xy = np.asarray(xy[off[i]:off[i + 1]], dtype=np.float32)
    xy = xy[A.thin_index(len(xy), EVAL_FPS)]
    return F.extract(A.subset_to_clip(xy), t_fix=t, aspect=F.dataset_aspect(group))


def predict(sess, X, M):
    out = []
    for s in range(0, len(X), BATCH):
        out.append(sess.run(["probs"], {"features": np.asarray(X[s:s + BATCH], np.float32),
                                        "mask": np.asarray(M[s:s + BATCH], bool)})[0])
    return np.concatenate(out)


def topk(p, y, k):
    return float((np.argsort(-p, axis=1)[:, :k] == y[:, None]).any(1).mean())


def main():
    t0 = time.time()
    from sklearn.model_selection import GroupKFold

    y = np.load(PRE_DIR / "y.npy").astype(np.int64)
    groups = np.load(PRE_DIR / "groups.npy")
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups))
    val = np.sort(folds[VAL_FOLD][1])
    yv, gv = y[val], groups[val]
    X = np.array(np.load(PRE_DIR / "X32.npy", mmap_mode="r")[val])
    M = np.load(PRE_DIR / "mask32.npy")[val]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(_thinned_one, [(int(i), int(groups[i]), 32) for i in val], chunksize=64)
    X13, M13 = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
    print(f"val clips={len(val)} participants={sorted(int(g) for g in np.unique(gv))} "
          f"features ready in {time.time() - t0:.0f}s, cpus={os.cpu_count()}", flush=True)

    probs = {}
    for name, folder in MODELS.items():
        path = find_one(f"**/{folder}/model.onnx")
        meta = json.loads((path.parent / "metrics.json").read_text())
        if meta["FEATURES_VERSION"] != F.FEATURES_VERSION or meta["T_FIX"] != 32:
            raise RuntimeError(f"{name}: FEATURES_VERSION {meta['FEATURES_VERSION']}, T {meta['T_FIX']}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count()
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        probs[name] = (predict(sess, X, M), predict(sess, X13, M13))
        # must reproduce the number the training kernel reported for this model
        print(f"{name}: top-1 {topk(probs[name][0], yv, 1):.4f} (training kernel: {meta['orig']['top1']:.4f})",
              flush=True)

    rows, per_part = [], []
    for r in range(1, len(MODELS) + 1):
        for combo in itertools.combinations(MODELS, r):
            p = np.mean([probs[n][0] for n in combo], axis=0)
            p13 = np.mean([probs[n][1] for n in combo], axis=0)
            rows.append({"models": "+".join(combo), "top1": topk(p, yv, 1), "top3": topk(p, yv, 3),
                         "top1_13fps": topk(p13, yv, 1), "top3_13fps": topk(p13, yv, 3)})
            hit = p.argmax(1) == yv
            for g in np.unique(gv):
                per_part.append({"models": "+".join(combo), "participant": int(g),
                                 "n": int((gv == g).sum()), "top1": float(hit[gv == g].mean())})
            print(json.dumps(rows[-1]), flush=True)
    (OUT_DIR / "ensemble.json").write_text(json.dumps({"val_participants": sorted(int(g) for g in np.unique(gv)),
                                                      "n_val": int(len(val)), "results": rows,
                                                      "runtime_s": round(time.time() - t0, 1)}, indent=1))
    pd.DataFrame(per_part).to_csv(OUT_DIR / "per_participant.csv", index=False)


if __name__ == "__main__":
    main()
