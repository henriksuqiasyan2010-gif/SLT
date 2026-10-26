"""Publish src/asl as the private Kaggle dataset <KAGGLE_USER>/asl-src and wait until it is ready.

    python scripts/publish_src.py -m "message"

First run creates the dataset, later runs add a version. Kernels rebuild the `asl` package from it.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from _kaggle import run_kaggle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402

BUILD = config.ROOT / "build" / "kaggle_src"
SLUG = f"{config.KAGGLE_USER}/{config.KAGGLE_SRC_DATASET}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait for 'ready'")
    args = ap.parse_args()

    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.copytree(config.ROOT / "src" / "asl", BUILD / "asl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "app"))
    (BUILD / "dataset-metadata.json").write_text(json.dumps(
        {"title": config.KAGGLE_SRC_DATASET, "id": SLUG, "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}))
    files = sorted(p.name for p in (BUILD / "asl").glob("*.py"))
    print(f"publishing {SLUG}: {', '.join(files)}")

    exists = run_kaggle("datasets", "status", SLUG).returncode == 0
    cmd = (["datasets", "version", "-p", str(BUILD), "-m", args.message] if exists
           else ["datasets", "create", "-p", str(BUILD)])
    res = run_kaggle(*cmd, "--dir-mode", "zip")
    print((res.stdout + res.stderr).strip().splitlines()[-1] if (res.stdout + res.stderr).strip() else "")

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        status = run_kaggle("datasets", "status", SLUG).stdout.strip()
        if status == "ready":
            print(f"ready after {time.time() - t0:.0f} s")
            return 0
        time.sleep(10)
    print(f"not ready after {args.timeout} s (last status: {status})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
