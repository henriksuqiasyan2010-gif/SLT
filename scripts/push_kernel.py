"""Push the training kernel with a preset as its own Kaggle kernel and log it in STATUS.md.

    python scripts/push_kernel.py --preset baseline        # -> <KAGGLE_USER>/asl-train-baseline

The kernel folder is generated in build/kernels/asl-train-<preset>/ from kaggle/asl-train/asl_train.py
(only the PRESET line differs), so the outputs of different presets never overwrite each other.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from _kaggle import run_kaggle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402

SOURCE = config.ROOT / "kaggle" / "asl-train" / "asl_train.py"
STATUS = config.ROOT / "STATUS.md"
KERNELS_HEADER = "## Kaggle kernels"


def log_kernel(slug: str, purpose: str, result: str) -> None:
    """Append a row to the Kaggle kernels table in STATUS.md."""
    text = STATUS.read_text(encoding="utf-8")
    if KERNELS_HEADER not in text:
        text = text.rstrip("\n") + f"\n\n{KERNELS_HEADER}\n\n| Date | Kernel | Purpose | Result |\n|---|---|---|---|\n"
    text = text.rstrip("\n") + f"\n| {date.today().isoformat()} | `{slug}` | {purpose} | {result} |\n"
    STATUS.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--purpose", default="")
    args = ap.parse_args()

    src = SOURCE.read_text(encoding="utf-8")
    presets = re.findall(r'^    "([\w-]+)": dict\(', src, flags=re.M)
    if args.preset not in presets:
        raise SystemExit(f"unknown preset {args.preset}; known: {presets}")
    src, n = re.subn(r'^PRESET = "[\w-]+"', f'PRESET = "{args.preset}"', src, count=1, flags=re.M)
    assert n == 1

    name = f"asl-train-{args.preset}"
    slug = f"{config.KAGGLE_USER}/{name}"
    folder = config.ROOT / "build" / "kernels" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "asl_train.py").write_text(src, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": slug, "title": name, "code_file": "asl_train.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,  # internet: pip onnx/onnxruntime if missing
        "dataset_sources": [f"{config.KAGGLE_USER}/{config.KAGGLE_SRC_DATASET}"],
        "competition_sources": [config.KAGGLE_COMPETITION],
        "kernel_sources": [f"{config.KAGGLE_USER}/asl-preprocess"],
    }, indent=2), encoding="utf-8")

    res = run_kaggle("kernels", "push", "-p", str(folder))
    out = (res.stdout + res.stderr).strip()
    print(out[-300:])
    if res.returncode != 0 or "successfully pushed" not in out:
        return 1
    version = re.search(r"Kernel version (\d+)", out)
    log_kernel(f"{slug} v{version.group(1) if version else '?'}", args.purpose or f"train, preset {args.preset}",
               "running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
