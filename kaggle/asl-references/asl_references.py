"""Kaggle kernel asl-references (CPU): one reference clip per sign for Learn mode.

Candidates per sign = held-out clips (the validation participants of the trained model) that pass
the reference rule (asl.reference.reference_candidates: hand in > 50% of the frames, right-dominant
if possible, length close to the sign's median). The model (main-reg30, live path without trim_idle:
keep_moving_hand -> extract) scores every candidate, and the one with the highest probability of
its own sign is kept (user's spec, Phase 6). A left-dominant clip (only if a sign has no
right-dominant candidate) is mirrored for display.

Outputs in /kaggle/working/references/: <sign>.npy (T, K, 2) float16 x, y of skeleton.DISPLAY_INDICES,
references.json (per sign: source, fps, aspect, model probability and rank, ...), summary.json.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "main-reg30_T32"
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
from asl import config, features as F  # noqa: E402
from asl.landmarks_io import load_clip  # noqa: E402
from asl.reference import mirror_for_display, reference_candidates, to_display  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
    import onnxruntime as ort

COMP_DIR = find_one("**/sign_to_prediction_index_map.json").parent
OUT_DIR = Path("/kaggle/working/references")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(find_one("**/stats.csv"))
    model_path = find_one(f"**/{MODEL}/model.onnx")
    meta = json.loads((model_path.parent / "metrics.json").read_text())
    if meta["FEATURES_VERSION"] != F.FEATURES_VERSION or meta["T_FIX"] != T:
        raise RuntimeError(f"model FEATURES_VERSION {meta['FEATURES_VERSION']} / T {meta['T_FIX']}")
    labels = {int(k): v for k, v in json.loads((model_path.parent / "labels.json").read_text()).items()}
    index_of = {v: k for k, v in labels.items()}
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    print(f"clips={len(stats)} signs={stats.sign.nunique()} model={MODEL} FEATURES_VERSION={F.FEATURES_VERSION} "
          f"val participants={config.VAL_PARTICIPANTS}", flush=True)

    index, n_cand = {}, 0
    for sign in sorted(stats["sign"].unique()):
        cands = reference_candidates(stats, sign)
        clips, feats, masks = [], [], []
        for row in cands.itertuples():
            clip = load_clip(COMP_DIR / row.path)
            aspect = F.dataset_aspect(row.participant_id)
            f, m = F.extract(F.keep_moving_hand(clip, aspect), t_fix=T, aspect=aspect)
            clips.append(clip)
            feats.append(f)
            masks.append(m)
        probs = sess.run(["probs"], {"features": np.stack(feats).astype(np.float32), "mask": np.stack(masks)})[0]
        k = index_of[sign]
        p_true = probs[:, k]
        best = int(np.argmax(p_true))
        row = cands.iloc[best]
        clip = clips[best] if row.right_dominant else mirror_for_display(clips[best])
        np.save(OUT_DIR / f"{sign}.npy", to_display(clip))
        order = np.argsort(-probs[best])
        index[sign] = {
            "source": row.path, "participant_id": int(row.participant_id), "frames": int(row.n_frames),
            "fps": config.REFERENCE_FPS, "aspect": F.dataset_aspect(row.participant_id),
            "right_dominant": bool(row.right_dominant), "mirrored": not bool(row.right_dominant),
            "hand_share": round(float(row.hand_share), 3), "sign_median_frames": float(row.sign_median_frames),
            "length_gap": float(row.length_gap), "candidates": int(len(cands)),
            "model": MODEL, "p": round(float(p_true[best]), 4), "rank": int(np.where(order == k)[0][0]) + 1,
            "top3": [[labels[int(i)], round(float(probs[best, i]), 4)] for i in order[:3]],
            "candidate_p": [round(float(x), 4) for x in p_true],
        }
        n_cand += len(cands)
    (OUT_DIR / "references.json").write_text(json.dumps(index, indent=1))
    ps = np.array([v["p"] for v in index.values()])
    summary = {"signs": len(index), "candidates": n_cand, "rank1": int(sum(v["rank"] == 1 for v in index.values())),
               "p_median": round(float(np.median(ps)), 3), "p_below_0.4": int((ps < 0.4).sum()),
               "mirrored": int(sum(v["mirrored"] for v in index.values())),
               "frames_median": float(np.median([v["frames"] for v in index.values()])),
               "runtime_s": round(time.time() - t0, 1)}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
