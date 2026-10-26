"""Run the model on the user's recordings: top-3 per recording -> stdout and logs/smoke_test.csv.

    python scripts/smoke_test.py [--model-dir models] [--recordings recordings]

Live path: trim_idle(clip, fps, aspect) -> extract(aspect) -> model. The sign is read from the
recording's sidecar (or its file name).
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, features  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--recordings", type=Path, default=config.RECORDINGS_DIR)
    args = ap.parse_args()

    model = SignModel.load(args.model_dir)
    rows, hits1, hits3 = [], 0, 0
    files = sorted(args.recordings.glob("*.npy"))
    for path in files:
        clip, meta = load_recording(path)
        sign = meta.get("sign") or path.stem.split("_")[0]
        aspect = recording_aspect(meta)
        start, stop = features.trim_idle(clip, meta["fps"], aspect)
        top = model.top_k(model.predict_clip(clip, meta["fps"], aspect), 3)
        rank = next((i + 1 for i, (s, _) in enumerate(top) if s == sign), None)
        hits1 += rank == 1
        hits3 += rank is not None
        rows.append({"time": datetime.now().isoformat(timespec="seconds"), "model": model.metrics.get("preset"),
                     "recording": path.name, "sign": sign, "fps": meta["fps"],
                     "kept_s": round((stop - start) / meta["fps"], 2),
                     **{f"top{i + 1}": f"{s} {p:.2f}" for i, (s, p) in enumerate(top)}, "rank": rank or "-"})
        print(f"{path.name:32s} {sign:8s} kept {rows[-1]['kept_s']:.2f} s  ->  "
              + ", ".join(f"{s} {p:.0%}" for s, p in top) + f"   {'OK' if rank == 1 else f'rank {rank}' if rank else 'MISS'}")
    if files:
        print(f"top-1 {hits1}/{len(files)}, top-3 {hits3}/{len(files)}")
        config.LOGS_DIR.mkdir(exist_ok=True)
        out = config.LOGS_DIR / "smoke_test.csv"
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, rows[0].keys())
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
