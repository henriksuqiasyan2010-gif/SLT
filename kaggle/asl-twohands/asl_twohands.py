"""Kaggle kernel asl-twohands (CPU): does features.keep_moving_hand() hurt the model on real data?

keep_moving_hand() is meant for the live path only: when both hands are visible together, the hand
that travels less becomes NaN before extract(). Before it is switched on in the app, this kernel
measures main-reg30 on the held-out participants (GroupKFold(5), fold 0) on the clips where it
changes anything (both hands visible in at least one frame), with and without it, on the original
clips and at 13 fps. Clips where it changes nothing give the same prediction by construction.

Outputs in /kaggle/working: twohands.json (summary), twohands_clips.csv (one row per changed clip).
"""

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

MODEL = "main-reg30_T32"
N_FOLDS = 5
VAL_FOLD = 0
EVAL_FPS = 13.0
T = 32


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

PRE_DIR = find_one("**/raw_xy.npy").parent
COMP_DIR = find_one("**/sign_to_prediction_index_map.json").parent
OUT_DIR = Path("/kaggle/working")
_RAW = None


def raw_clip(i):
    global _RAW
    if _RAW is None:
        _RAW = (np.load(PRE_DIR / "raw_xy.npy", mmap_mode="r"), np.load(PRE_DIR / "raw_offsets.npy"))
    xy, off = _RAW
    return A.subset_to_clip(np.asarray(xy[off[i]:off[i + 1]], dtype=np.float32))


def one(args):
    """Features with and without keep_moving_hand(), at the original rate and at 13 fps."""
    i, group = args
    aspect = F.dataset_aspect(group)
    clip = raw_clip(i)
    thin = clip[A.thin_index(len(clip), EVAL_FPS)]
    out = {"i": i}
    for name, c in (("orig", clip), ("13fps", thin)):
        kept = F.keep_moving_hand(c, aspect)
        out[name] = (F.extract(c, t_fix=T, aspect=aspect), F.extract(kept, t_fix=T, aspect=aspect))
        if name == "orig":
            left, right = F.hand_presence(c)
            pl, pr = F.hand_paths(c, aspect)
            out.update(both_frames=int((left & right).sum()), left_frames=int(left.sum()),
                       right_frames=int(right.sum()), path_left=pl, path_right=pr,
                       dropped="left" if np.isnan(kept[:, F.LEFT_HAND, 0]).all() else "right",
                       extract_dominant="left" if F.dominant_is_left(c) else "right")
    return out


def main():
    t0 = time.time()
    from sklearn.model_selection import GroupKFold

    y = np.load(PRE_DIR / "y.npy").astype(np.int64)
    groups = np.load(PRE_DIR / "groups.npy")
    signs = {v: k for k, v in json.loads((COMP_DIR / "sign_to_prediction_index_map.json").read_text()).items()}
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups))
    val = np.sort(folds[VAL_FOLD][1])

    # select the val clips with both hands in one frame from the raw subset (hand blocks 0-41)
    xy = np.load(PRE_DIR / "raw_xy.npy", mmap_mode="r")
    off = np.load(PRE_DIR / "raw_offsets.npy")
    both_any, ever_both = [], 0
    for i in val:
        h = ~np.isnan(xy[off[i]:off[i + 1], :2 * F.HAND_POINTS, 0])
        left, right = h[:, :F.HAND_POINTS].any(1), h[:, F.HAND_POINTS:].any(1)
        ever_both += bool(left.any() and right.any())
        if (left & right).any():
            both_any.append(int(i))
    print(f"val clips={len(val)}; both hands in one frame: {len(both_any)}; both hands at some time: {ever_both}",
          flush=True)

    with Pool(os.cpu_count()) as pool:
        res = pool.map(one, [(i, int(groups[i])) for i in both_any], chunksize=8)
    path = find_one(f"**/{MODEL}/model.onnx")
    meta = json.loads((path.parent / "metrics.json").read_text())
    if meta["FEATURES_VERSION"] != F.FEATURES_VERSION or meta["T_FIX"] != T:
        raise RuntimeError(f"model FEATURES_VERSION {meta['FEATURES_VERSION']}, T {meta['T_FIX']}")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def probs(key, k):
        X = np.stack([r[key][k][0] for r in res]).astype(np.float32)
        M = np.stack([r[key][k][1] for r in res])
        return sess.run(["probs"], {"features": X, "mask": M})[0]

    yb = y[both_any]
    summary = {"model": MODEL, "val_participants": sorted(int(g) for g in np.unique(groups[val])),
               "n_val": int(len(val)), "n_both_in_one_frame": len(both_any), "n_both_at_some_time": ever_both}
    rows = pd.DataFrame([{k: v for k, v in r.items() if k not in ("orig", "13fps")} for r in res])
    rows["sign"] = [signs[int(c)] for c in yb]
    for key in ("orig", "13fps"):
        p_without, p_with = probs(key, 0), probs(key, 1)
        top3_wo, top3_w = np.argsort(-p_without, 1)[:, :3], np.argsort(-p_with, 1)[:, :3]
        summary[key] = {
            "top1_without": float((top3_wo[:, 0] == yb).mean()), "top1_with": float((top3_w[:, 0] == yb).mean()),
            "top3_without": float((top3_wo == yb[:, None]).any(1).mean()),
            "top3_with": float((top3_w == yb[:, None]).any(1).mean()),
        }
        rows[f"correct_without_{key}"] = top3_wo[:, 0] == yb
        rows[f"correct_with_{key}"] = top3_w[:, 0] == yb
    summary["dropped_hand_was_extract_dominant"] = int((rows.dropped == rows.extract_dominant).sum())
    per_sign = rows.groupby("sign").agg(n=("sign", "size"), without=("correct_without_orig", "mean"),
                                        with_=("correct_with_orig", "mean"))
    per_sign["delta"] = per_sign.with_ - per_sign.without
    ranked = per_sign[per_sign.n >= 3].sort_values("delta").round(3).reset_index()
    summary["signs_most_changed"] = {"worse": ranked.head(8).to_dict("records"),
                                     "better": ranked.tail(8).iloc[::-1].to_dict("records")}
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "twohands.json").write_text(json.dumps(summary, indent=1))
    rows.to_csv(OUT_DIR / "twohands_clips.csv", index=False)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
