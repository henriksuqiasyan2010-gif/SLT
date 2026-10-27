"""Push a parameterised Kaggle kernel and log it in STATUS.md.

    python scripts/push_kernel.py --kernel asl-train --preset baseline    # -> <user>/asl-train-baseline (GPU)
    python scripts/push_kernel.py --kernel asl-augment --preset 0         # -> <user>/asl-augment-0 (CPU)

The kernel folder is generated in build/kernels/<name>/ from kaggle/<kernel>/<script>.py; only the
parameter line (PRESET = "..." or SHARD = N) differs, so the outputs of different presets never
overwrite each other. asl-train presets with augment="precomputed" also mount the outputs of
aug_versions / 5 asl-augment-<shard> kernels (main, main-reg: 0..2; main-reg30: 0..5).
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

STATUS = config.ROOT / "STATUS.md"
KERNELS_HEADER = "## Kaggle kernels"
VERSIONS_PER_SHARD = 5  # augmented versions written by one asl-augment-<shard> kernel

KERNELS = {
    "asl-train": {"script": "asl_train.py", "param": r'^PRESET = "[\w-]+"', "fmt": 'PRESET = "{}"', "gpu": True},
    "asl-augment": {"script": "asl_augment.py", "param": r"^SHARD = \d+", "fmt": "SHARD = {}", "gpu": False},
}


def log_kernel(slug: str, purpose: str, result: str) -> None:
    """Append a row to the Kaggle kernels table in STATUS.md."""
    text = STATUS.read_text(encoding="utf-8")
    if KERNELS_HEADER not in text:
        text = text.rstrip("\n") + f"\n\n{KERNELS_HEADER}\n\n| Date | Kernel | Purpose | Result |\n|---|---|---|---|\n"
    text = text.rstrip("\n") + f"\n| {date.today().isoformat()} | `{slug}` | {purpose} | {result} |\n"
    STATUS.write_text(text, encoding="utf-8")


def train_presets(src: str) -> dict:
    """PRESETS of asl_train.py: the plain-literal head of the script (up to SEED) is executed alone,
    without the imports and the Kaggle-only code below it."""
    ns = {}
    exec(src[:src.index("\nSEED = ")], ns)
    return ns["PRESETS"]


def augment_shards(cfg: dict) -> int:
    """Number of asl-augment-<shard> kernels a preset trains on (0 = none)."""
    if cfg.get("augment") != "precomputed":
        return 0
    assert cfg["aug_versions"] % VERSIONS_PER_SHARD == 0, cfg["aug_versions"]
    return cfg["aug_versions"] // VERSIONS_PER_SHARD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="asl-train", choices=sorted(KERNELS))
    ap.add_argument("--preset", required=True)
    ap.add_argument("--purpose", default="")
    args = ap.parse_args()

    spec = KERNELS[args.kernel]
    src = (config.ROOT / "kaggle" / args.kernel / spec["script"]).read_text(encoding="utf-8")
    shards = 0
    if args.kernel == "asl-train":
        presets = train_presets(src)
        if args.preset not in presets:
            raise SystemExit(f"unknown preset {args.preset}; known: {list(presets)}")
        shards = augment_shards(presets[args.preset])
    src, n = re.subn(spec["param"], spec["fmt"].format(args.preset), src, count=1, flags=re.M)
    assert n == 1, "parameter line not found"

    name = f"{args.kernel}-{args.preset}"
    slug = f"{config.KAGGLE_USER}/{name}"
    sources = [f"{config.KAGGLE_USER}/asl-preprocess"]
    sources += [f"{config.KAGGLE_USER}/asl-augment-{k}" for k in range(shards)]
    folder = config.ROOT / "build" / "kernels" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / spec["script"]).write_text(src, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": slug, "title": name, "code_file": spec["script"], "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": spec["gpu"],
        "enable_internet": spec["gpu"],  # train kernel: pip onnx/onnxruntime if missing
        "dataset_sources": [f"{config.KAGGLE_USER}/{config.KAGGLE_SRC_DATASET}"],
        "competition_sources": [config.KAGGLE_COMPETITION],
        "kernel_sources": sources,
    }, indent=2), encoding="utf-8")

    res = run_kaggle("kernels", "push", "-p", str(folder))
    out = (res.stdout + res.stderr).strip()
    print(out[-300:])
    if res.returncode != 0 or "successfully pushed" not in out:
        return 1
    version = re.search(r"Kernel version (\d+)", out)
    log_kernel(f"{slug} v{version.group(1) if version else '?'}", args.purpose or f"{args.kernel}, preset {args.preset}",
               "running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
