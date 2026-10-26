"""Kaggle kernel asl-preprocess (CPU): asl.features.extract() on every clip of the dataset.

Outputs in /kaggle/working (float16 features, written through memmaps):
  X64.npy (N, 64, F), X32.npy (N, 32, F)  - features for T_FIX = 64 and 32 (Phase 3 compares them)
  mask64.npy, mask32.npy                  - (N, T) bool
  y.npy (N,) int16 class id, groups.npy (N,) int32 participant_id, paths.csv
  raw_xy.npy (sum of clip lengths, 130, 2) float16 (raw normalized x, y; aspect not applied) + raw_offsets.npy (N+1,) int64 - x, y of the
      features.SUBSET points of every frame, untrimmed; the train kernel augments these and calls
      extract() on the fly, so augmented features come from the same function
  meta.json                               - FEATURES_VERSION, F, T_FIX values, aspect/fps used, counts
"""

import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


def find_one(pattern):
    hits = sorted(Path("/kaggle/input").glob(pattern))
    if not hits:
        raise FileNotFoundError(f"/kaggle/input/{pattern}")
    return hits[0]


# Kaggle unpacks the asl-src zip without its top folder, so rebuild the `asl` package under /tmp.
SRC_DIR = Path("/tmp/asl_src")
shutil.copytree(find_one("**/landmarks_io.py").parent, SRC_DIR / "asl", dirs_exist_ok=True)
sys.path.insert(0, str(SRC_DIR))
from asl import config, features as F  # noqa: E402
from asl.landmarks_io import load_clip  # noqa: E402

COMP_DIR = find_one("**/sign_to_prediction_index_map.json").parent
OUT_DIR = Path("/kaggle/working")
T_VALUES = (64, 32)


def one(item):
    rel_path, participant_id = item
    clip = load_clip(COMP_DIR / rel_path)
    aspect = F.dataset_aspect(participant_id)
    feats = [F.extract(clip, t_fix=t, aspect=aspect) for t in T_VALUES]
    return feats, clip[:, F.SUBSET, :2].astype(np.float16)


def main():
    t0 = time.time()
    train = pd.read_csv(COMP_DIR / "train.csv")
    labels = json.loads((COMP_DIR / "sign_to_prediction_index_map.json").read_text())
    n, dim = len(train), F.feature_dim()
    print(f"clips={n} F={dim} FEATURES_VERSION={F.FEATURES_VERSION} cpus={os.cpu_count()}", flush=True)

    X = {t: np.lib.format.open_memmap(OUT_DIR / f"X{t}.npy", mode="w+", dtype=np.float16, shape=(n, t, dim))
         for t in T_VALUES}
    M = {t: np.zeros((n, t), dtype=bool) for t in T_VALUES}
    worst = 0.0
    raw_parts, offsets = [], [0]
    with Pool(os.cpu_count()) as pool:
        for i, (res, raw) in enumerate(pool.imap(one, zip(train["path"], train["participant_id"]), chunksize=32)):
            raw_parts.append(raw)
            offsets.append(offsets[-1] + len(raw))
            for t, (feats, mask) in zip(T_VALUES, res):
                worst = max(worst, float(np.abs(feats).max()))
                X[t][i] = feats.astype(np.float16)
                M[t][i] = mask
            if i % 10000 == 0:
                print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    for t in T_VALUES:
        X[t].flush()
        np.save(OUT_DIR / f"mask{t}.npy", M[t])
    np.save(OUT_DIR / "raw_xy.npy", np.concatenate(raw_parts))
    np.save(OUT_DIR / "raw_offsets.npy", np.asarray(offsets, dtype=np.int64))
    del raw_parts
    np.save(OUT_DIR / "y.npy", train["sign"].map(labels).to_numpy(np.int16))
    np.save(OUT_DIR / "groups.npy", train["participant_id"].to_numpy(np.int32))
    train[["path", "participant_id", "sequence_id", "sign"]].to_csv(OUT_DIR / "paths.csv", index=False)

    # float16 overflows above 65504; features are in shoulder widths, so this is a sanity check
    finite = all(np.isfinite(np.asarray(X[t][::97])).all() for t in T_VALUES)
    meta = {
        "FEATURES_VERSION": F.FEATURES_VERSION, "F": dim, "T_FIX": list(T_VALUES), "N": n,
        "num_classes": len(labels), "aspect": config.DATASET_ASPECT, "aspect_by_participant": config.DATASET_ASPECT_BY_PARTICIPANT,
        "max_abs_feature": worst, "float16_finite_sampled": bool(finite),
        "sizes_gb": {name: round(os.path.getsize(OUT_DIR / f"{name}.npy") / 1e9, 2)
                     for name in [f"X{t}" for t in T_VALUES] + ["raw_xy"]},
        "subset": F.SUBSET.tolist(),
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "subset"}, indent=2))


if __name__ == "__main__":
    main()
