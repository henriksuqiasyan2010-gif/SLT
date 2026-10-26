"""Skeleton animation of a sign at its real speed.

    python scripts/view_sample.py --sign <word> [--save out.gif]
    python scripts/view_sample.py --path <file.parquet|file.npy> [--save out.gif]

--sign picks the reference clip by the rule in asl/reference.py (hand in most frames, length near
the median of that sign, right hand preferred) from reports/stats/stats.csv and downloads it from
Kaggle if it is not in data/sample. Dataset clips play at config.DATASET_FPS_ASSUMED, recordings at
their own frame rate. Without --save the GIF goes to reports/gifs/<name>.gif.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, reference, skeleton  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.landmarks_io import load_clip  # noqa: E402


def reference_clip_path(sign: str) -> Path:
    from _kaggle import download_competition_file

    row = reference.pick_reference(reference.load_stats(), sign)
    print(f"reference for '{sign}': {row['path']} (participant {row['participant_id']}), "
          f"{row['n_frames']} frames (median {row['sign_median_frames']:.0f}), hand in {row['hand_share']:.0%} of frames")
    return download_competition_file(row["path"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sign")
    src.add_argument("--path", type=Path)
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()

    path = args.path or reference_clip_path(args.sign)
    if path.suffix == ".npy":
        clip, meta = load_recording(path)
        fps, aspect = float(meta["fps"]), recording_aspect(meta)
    else:
        clip, fps, aspect = load_clip(path), config.DATASET_FPS_ASSUMED, config.DATASET_ASPECT
    name = args.sign or path.stem
    out = args.save or config.REPORTS_DIR / "gifs" / f"{name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    skeleton.save_gif(clip, out, title=name, fps=fps, aspect=aspect)
    print(f"{path} -> {out} ({len(clip)} frames at {fps:.1f} fps = {len(clip) / fps:.2f} s, "
          f"{out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
