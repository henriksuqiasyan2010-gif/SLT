"""Skeleton animation of a dataset clip.

    python scripts/view_sample.py --sign <word> [--save out.gif]
    python scripts/view_sample.py --path <file.parquet|file.npy> [--save out.gif]

Without --save the GIF goes to reports/gifs/<name>.gif.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, skeleton  # noqa: E402
from asl.landmarks_io import load_clip  # noqa: E402


def find_sample_for_sign(sign: str) -> Path:
    train = pd.read_csv(config.SAMPLE_DIR / "train.csv")
    for rel in train.loc[train["sign"] == sign, "path"]:
        p = config.SAMPLE_DIR / rel
        if p.exists():
            return p
    local = sorted(set(train.loc[train["path"].map(lambda r: (config.SAMPLE_DIR / r).exists()), "sign"]))
    raise SystemExit(f"No local clip for sign '{sign}'. Signs in data/sample: {', '.join(local)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sign")
    src.add_argument("--path", type=Path)
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()

    path = args.path or find_sample_for_sign(args.sign)
    clip = np.load(path) if path.suffix == ".npy" else load_clip(path)
    name = args.sign or path.stem
    out = args.save or config.REPORTS_DIR / "gifs" / f"{name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    skeleton.save_gif(clip, out, title=name)
    print(f"{path} -> {out} ({len(clip)} frames, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
