"""Count the user's earlier tests in progress.json, so Learn mode does not start from zero (user, Phase 6).

    python scripts/seed_progress.py [--signs hello please ...] [--dry-run]

Evidence with a known sign, scored with the current model (live path, as in the app):
  * recordings/*.npy (not recordings/old) - the sign is in the sidecar;
  * the labelled app attempts (reports/user_tests/phase*_*.csv);
  * the Phase 5 sessions: segmenter -> words, matched in order to the true list when the counts agree.
A success = the sign is the model's top-1 and reaches config.CONFIDENCE_THRESHOLD. The counts go into
the "seeded" fields of each sign (Learn-mode attempts are kept apart); 3 successes = learned.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.app.progress import Progress  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402
from asl.segmenter import segment_clip  # noqa: E402

TESTED = ["hello", "please", "airplane", "high", "find", "sick", "happy", "hat", "shower", "mom"]


def evidence():
    """(source, sign, clip, fps, aspect) for every labelled clip of the user."""
    for path in sorted(config.RECORDINGS_DIR.glob("*.npy")):
        clip, meta = load_recording(path)
        yield "recordings", meta.get("sign") or path.stem.split("_")[0], clip, meta["fps"], recording_aspect(meta)
    for table in sorted((config.REPORTS_DIR / "user_tests").glob("phase*_*.csv")):
        with open(table, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clip, meta = load_recording(config.ATTEMPTS_DIR / row["attempt"])
                yield "attempts", row["sign"], clip, meta["fps"], recording_aspect(meta)
    for path in sorted(config.SESSIONS_DIR.glob("session?.npy")):
        clip, meta = load_recording(path)
        truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["signs"]
        aspect = recording_aspect(meta)
        segs = segment_clip(clip, meta["timestamps"], aspect)
        if len(segs) != len(truth):
            print(f"{path.name}: {len(segs)} segments for {len(truth)} signs, skipped")
            continue
        for seg, sign in zip(segs, truth):
            yield "sessions", sign, seg.landmarks, (len(seg.timestamps) - 1) / seg.duration, aspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", nargs="+", default=TESTED)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write progress.json")
    args = ap.parse_args()
    model = SignModel.load()
    stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "best_p": 0.0, "sources": defaultdict(int)})
    for source, sign, clip, fps, aspect in evidence():
        if sign not in args.signs:
            continue
        probs = model.predict_clip(clip, fps, aspect)
        top_sign, top_p = model.top_k(probs, 1)[0]
        p = float(probs[[k for k, v in model.labels.items() if v == sign][0]])
        st = stats[sign]
        st["attempts"] += 1
        st["successes"] += top_sign == sign and top_p >= config.CONFIDENCE_THRESHOLD
        st["best_p"] = max(st["best_p"], p)
        st["sources"][source] += 1
    progress = Progress.load()
    print(f"{'sign':10s} attempts successes best_p  learned  sources")
    for sign in args.signs:
        st = stats[sign]
        source = "tests: " + ", ".join(f"{k} {v}" for k, v in sorted(st["sources"].items()))
        learned = st["successes"] >= config.LEARN_SUCCESSES
        print(f"{sign:10s} {st['attempts']:8d} {st['successes']:9d} {st['best_p']:6.2f}  {str(learned):7s}  {source}")
        if not args.dry_run and st["attempts"]:
            progress.seed(sign, st["attempts"], st["successes"], st["best_p"], f"{source} ({model.metrics['preset']})")
    if not args.dry_run:
        print(f"-> {progress.path}")


if __name__ == "__main__":
    main()
