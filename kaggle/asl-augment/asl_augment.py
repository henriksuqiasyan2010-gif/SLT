"""Kaggle kernel asl-augment-<shard> (CPU): precomputed augmented features for training.

Each shard writes VERSIONS_PER_SHARD versions; version k = augment() with seed (SEED, k, clip) on
the raw landmarks of every clip, then features.extract(t_fix=T). The GPU kernel takes one version
per epoch, so the GPU does not wait for augmentation + extract() on its 4 CPUs (measured in
asl-train-main-trial: 66 s per epoch, GPU busy 15%).

Outputs in /kaggle/working: Xaug<T>_v<k>.npy (N, T, F) float16 for k in this shard, aug_meta_<shard>.json.
"""

SHARD = 0  # replaced by scripts/push_kernel.py
VERSIONS_PER_SHARD = 5
T = 32
SEED = 42

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def find_one(pattern):
    hits = sorted(Path("/kaggle/input").glob(pattern))
    if not hits:
        raise FileNotFoundError(f"/kaggle/input/{pattern}")
    return hits[0]


SRC_DIR = Path("/tmp/asl_src")
shutil.copytree(find_one("**/landmarks_io.py").parent, SRC_DIR / "asl", dirs_exist_ok=True)
sys.path.insert(0, str(SRC_DIR))
from asl import augment as A, config, features as F  # noqa: E402

PRE_DIR = find_one("**/raw_xy.npy").parent
OUT_DIR = Path("/kaggle/working")
_RAW = None


def one(args):
    global _RAW
    if _RAW is None:
        _RAW = (np.load(PRE_DIR / "raw_xy.npy", mmap_mode="r"), np.load(PRE_DIR / "raw_offsets.npy"),
                np.load(PRE_DIR / "groups.npy"))
    xy_all, off, groups = _RAW
    version, i = args
    aspect = F.dataset_aspect(groups[i])
    xy = A.augment(np.asarray(xy_all[off[i]:off[i + 1]], np.float32), aspect,
                   np.random.default_rng([SEED, version, i]))
    feats, _ = F.extract(A.subset_to_clip(xy), t_fix=T, aspect=aspect)
    return feats.astype(np.float16)


def main():
    t0 = time.time()
    meta = json.loads((PRE_DIR / "meta.json").read_text())
    if meta["FEATURES_VERSION"] != F.FEATURES_VERSION:
        raise RuntimeError(f"preprocess FEATURES_VERSION {meta['FEATURES_VERSION']} != code {F.FEATURES_VERSION}")
    n, dim = meta["N"], F.feature_dim()
    versions = [SHARD * VERSIONS_PER_SHARD + k for k in range(VERSIONS_PER_SHARD)]
    print(f"shard {SHARD}: versions {versions}, N={n}, T={T}, F={dim}, cpus={os.cpu_count()}", flush=True)
    timings = {}
    with Pool(os.cpu_count()) as pool:
        for v in versions:
            tv = time.time()
            X = np.lib.format.open_memmap(OUT_DIR / f"Xaug{T}_v{v}.npy", mode="w+", dtype=np.float16, shape=(n, T, dim))
            for i, feats in enumerate(pool.imap(one, ((v, i) for i in range(n)), chunksize=64)):
                X[i] = feats
            X.flush()
            del X
            timings[v] = round(time.time() - tv, 1)
            print(f"version {v}: {timings[v]}s", flush=True)
    info = {"shard": SHARD, "versions": versions, "T": T, "N": n, "F": dim, "seed": SEED,
            "FEATURES_VERSION": F.FEATURES_VERSION, "augment": config.AUGMENT,
            "seconds_per_version": timings, "runtime_s": round(time.time() - t0, 1)}
    (OUT_DIR / f"aug_meta_{SHARD}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
