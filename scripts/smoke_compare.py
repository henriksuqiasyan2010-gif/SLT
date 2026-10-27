"""Compare one model with an ensemble (averaged probabilities) on the user's recordings.

    python scripts/smoke_compare.py --model models --ensemble DIR DIR ... [--per-sign CSV]

For every recording in recordings/ (not recordings/old): top-3 of both, the rank of the true sign,
and the model's validation accuracy for that sign on the dataset (per_sign_accuracy.csv).
Same live path as smoke_test.py: trim_idle -> extract -> model. -> stdout and logs/smoke_compare.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, features  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402


def rank_of(top, sign):
    return next((i + 1 for i, (s, _) in enumerate(top) if s == sign), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--ensemble", type=Path, nargs="+", required=True)
    ap.add_argument("--per-sign", type=Path, required=True, help="per_sign_accuracy.csv of --model")
    ap.add_argument("--recordings", type=Path, default=config.RECORDINGS_DIR)
    args = ap.parse_args()

    single = SignModel.load(args.model)
    members = [SignModel.load(d) for d in args.ensemble]
    if any(m.labels != single.labels for m in members):
        raise SystemExit("models have different label maps")
    with open(args.per_sign, encoding="utf-8") as f:
        val_acc = {r["sign"]: float(r["accuracy"]) for r in csv.DictReader(f)}

    rows = []
    for path in sorted(args.recordings.glob("*.npy")):
        clip, meta = load_recording(path)
        sign = meta.get("sign") or path.stem.split("_")[0]
        aspect = recording_aspect(meta)
        start, stop = features.trim_idle(clip, meta["fps"], aspect)
        top_s = single.top_k(single.predict_clip(clip, meta["fps"], aspect), 3)
        p_ens = np.mean([m.predict_clip(clip, meta["fps"], aspect) for m in members], axis=0)
        top_e = single.top_k(p_ens, 3)
        rows.append({"recording": path.stem, "sign": sign, "kept_s": round((stop - start) / meta["fps"], 2),
                     "single": ", ".join(f"{s} {p:.0%}" for s, p in top_s), "single_rank": rank_of(top_s, sign),
                     "ensemble": ", ".join(f"{s} {p:.0%}" for s, p in top_e), "ensemble_rank": rank_of(top_e, sign),
                     "dataset_val_acc": val_acc.get(sign)})
        r = rows[-1]
        print(f"{r['recording']:26s} {sign:8s} val {r['dataset_val_acc']:.0%} | {r['single']:38s} "
              f"[{r['single_rank'] or '-'}] | {r['ensemble']:38s} [{r['ensemble_rank'] or '-'}]")
    n = len(rows)
    for key in ("single", "ensemble"):
        print(f"{key}: top-1 {sum(r[f'{key}_rank'] == 1 for r in rows)}/{n}, "
              f"top-3 {sum(r[f'{key}_rank'] is not None for r in rows)}/{n}")
    config.LOGS_DIR.mkdir(exist_ok=True)
    with open(config.LOGS_DIR / "smoke_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
