"""Kaggle kernel asl-stats (CPU): per-clip frame counts and hand presence for the whole dataset.

Outputs in /kaggle/working: stats.csv (one row per clip) and summary.json.
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
from asl.landmarks_io import LEFT_HAND, N_LANDMARKS, RIGHT_HAND  # noqa: E402

COMP_DIR = find_one("**/sign_to_prediction_index_map.json").parent
OUT_DIR = Path("/kaggle/working")
PERCENTILES = [5, 10, 25, 50, 75, 90, 95, 99]


def clip_stats(rel_path):
    df = pd.read_parquet(COMP_DIR / rel_path, columns=["frame", "x"])
    n_rows = len(df)
    frames = df["frame"].to_numpy()
    x = df["x"].to_numpy(dtype=np.float32).reshape(-1, N_LANDMARKS)
    left = ~np.isnan(x[:, LEFT_HAND]).all(axis=1)
    right = ~np.isnan(x[:, RIGHT_HAND]).all(axis=1)
    return {
        "path": rel_path,
        "n_frames": n_rows // N_LANDMARKS,
        "rows_ok": n_rows % N_LANDMARKS == 0,
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "left_frames": int(left.sum()),
        "right_frames": int(right.sum()),
        "hand_frames": int((left | right).sum()),
    }


def main():
    t0 = time.time()
    train = pd.read_csv(COMP_DIR / "train.csv")
    print(f"src={SRC_DIR} comp={COMP_DIR} clips={len(train)} cpus={os.cpu_count()}", flush=True)
    rows = []
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap(clip_stats, train["path"], chunksize=64)):
            rows.append(r)
            if i % 5000 == 0:
                print(f"{i}/{len(train)} {time.time() - t0:.0f}s", flush=True)
    stats = train.merge(pd.DataFrame(rows), on="path")
    stats.to_csv(OUT_DIR / "stats.csv", index=False)

    n = stats["n_frames"]
    per_sign = stats["sign"].value_counts()
    per_part = stats["participant_id"].value_counts()
    summary = {
        "n_clips": int(len(stats)),
        "n_signs": int(stats["sign"].nunique()),
        "n_participants": int(stats["participant_id"].nunique()),
        "all_rows_multiple_of_543": bool(stats["rows_ok"].all()),
        "frames_contiguous_share": float((stats["frame_max"] - stats["frame_min"] + 1 == n).mean()),
        "length": {"mean": float(n.mean()), "min": int(n.min()), "max": int(n.max()),
                   **{f"p{p}": float(np.percentile(n, p)) for p in PERCENTILES}},
        "hand_frames_length": {f"p{p}": float(np.percentile(stats["hand_frames"], p)) for p in PERCENTILES},
        "clips_without_hands": int((stats["hand_frames"] == 0).sum()),
        "dominant_hand_share": {
            "right": float((stats["right_frames"] > stats["left_frames"]).mean()),
            "left": float((stats["left_frames"] > stats["right_frames"]).mean()),
        },
        "clips_per_sign": {"min": int(per_sign.min()), "median": float(per_sign.median()),
                           "max": int(per_sign.max()), "counts": per_sign.to_dict()},
        "clips_per_participant": {str(k): int(v) for k, v in per_part.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("clips_per_sign", "clips_per_participant")},
                     indent=2))


if __name__ == "__main__":
    main()
