"""Kaggle kernel asl-check (CPU): two checks on every clip of the dataset.

1. Frame aspect ratio per participant: face proportions (eye-mouth height / interocular distance,
   and / shoulder width) in normalized coordinates depend on the frame width/height.
2. How much asl.features.trim_idle() cuts compared with trim_to_hands().

Outputs in /kaggle/working: check.csv (one row per clip) and summary.json.
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
from asl.landmarks_io import FACE, POSE, load_clip  # noqa: E402

COMP_DIR = find_one("**/sign_to_prediction_index_map.json").parent
OUT_DIR = Path("/kaggle/working")


def one(rel_path):
    c = load_clip(COMP_DIR / rel_path)
    h0, h1 = F.trim_to_hands(c)
    i0, i1 = F.trim_idle(c, config.DATASET_FPS_ASSUMED)
    f = c[:, FACE, :2]
    eye_y = (f[:, 33, 1] + f[:, 263, 1]) / 2
    mouth_y = (f[:, 13, 1] + f[:, 14, 1]) / 2
    io = np.abs(f[:, 33, 0] - f[:, 263, 0])
    sw = np.abs(c[:, POSE.start + 11, 0] - c[:, POSE.start + 12, 0])
    with np.errstate(all="ignore"):
        r_face = np.nanmedian((mouth_y - eye_y) / io)
        r_shoulder = np.nanmedian((mouth_y - eye_y) / sw)
    return {"path": rel_path, "n_frames": len(c), "hands_start": h0, "hands_stop": h1,
            "idle_start": i0, "idle_stop": i1, "r_face": float(r_face), "r_shoulder": float(r_shoulder)}


def main():
    t0 = time.time()
    train = pd.read_csv(COMP_DIR / "train.csv")
    print(f"clips={len(train)} cpus={os.cpu_count()} FEATURES_VERSION={F.FEATURES_VERSION}", flush=True)
    rows = []
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap(one, train["path"], chunksize=64)):
            rows.append(r)
            if i % 10000 == 0:
                print(f"{i}/{len(train)} {time.time() - t0:.0f}s", flush=True)
    df = train.merge(pd.DataFrame(rows), on="path")
    df.to_csv(OUT_DIR / "check.csv", index=False)

    df["hands_len"] = df.hands_stop - df.hands_start
    df["idle_len"] = df.idle_stop - df.idle_start
    extra = 1 - df.idle_len / df.hands_len
    per_part = df.groupby("participant_id").agg(r_face=("r_face", "median"), r_shoulder=("r_shoulder", "median"),
                                                r_face_iqr=("r_face", lambda x: float(x.quantile(.75) - x.quantile(.25))),
                                                clips=("path", "size"))
    summary = {
        "clips": int(len(df)),
        "trim": {
            "frames_total": int(df.n_frames.sum()),
            "kept_by_trim_to_hands": float(df.hands_len.sum() / df.n_frames.sum()),
            "kept_by_trim_idle": float(df.idle_len.sum() / df.n_frames.sum()),
            "extra_cut_by_idle_share_of_hand_frames": float(1 - df.idle_len.sum() / df.hands_len.sum()),
            "clips_with_extra_cut": {f">{int(t * 100)}%": float((extra > t).mean()) for t in (0.0, 0.1, 0.25, 0.5)},
            "extra_cut_percentiles": {f"p{p}": float(np.percentile(extra, p)) for p in (50, 90, 95, 99)},
            "idle_len_percentiles": {f"p{p}": float(np.percentile(df.idle_len, p)) for p in (5, 25, 50, 75, 95)},
            "worst_signs_by_mean_extra_cut": extra.groupby(df.sign).mean().sort_values(ascending=False).head(10).round(3).to_dict(),
        },
        "per_participant": {str(k): {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in per_part.iterrows()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
