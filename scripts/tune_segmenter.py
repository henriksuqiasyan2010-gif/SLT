"""Grid search of the segmenter thresholds on the recorded sessions.

    python scripts/tune_segmenter.py [--model-dir models]

For each parameter set and session (recordings/sessions/<name>.npy + <name>.json with the true
signs): number of segments, words appended (live path: classify -> Sentence.add with the
confidence threshold and the repeat rule) and correct words (order kept). -> reports/segmenter/
tuning.csv and the best sets on stdout. The config defaults are marked.
"""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.app.pipeline import Clip, classify  # noqa: E402
from asl.app.sentence import Sentence, word_matches  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402
from asl.segmenter import Params, segment_clip  # noqa: E402

GRID = {
    "wrist_up": [-1.0, -0.9, -0.85, -0.75, -0.6, -0.4],
    "pause_s": [0.2, 0.35, 0.5, 0.8],
    "start_s": [0.0, 0.1, 0.2],
    "preroll_s": [0.0, 0.2],
}
OUT = config.REPORTS_DIR / "segmenter" / "tuning.csv"


def run(model, sessions, params):
    rows = []
    for name, clip, meta, truth in sessions:
        aspect = recording_aspect(meta)
        segs = segment_clip(clip, meta["timestamps"], aspect, params)
        sentence = Sentence()
        for s in segs:
            pred = classify(model, Clip(s.landmarks, s.timestamps), aspect, config.CONFIDENCE_THRESHOLD)
            sentence.add(pred, float(s.timestamps[-1]))
        rows.append({"session": name, "segments": len(segs), "true_words": len(truth),
                     "words": len(sentence.words), "correct": word_matches(sentence.signs(), truth),
                     "sentence": " ".join(sentence.signs())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    args = ap.parse_args()
    model = SignModel.load(args.model_dir)
    sessions = []
    for path in sorted(config.SESSIONS_DIR.glob("session*.npy")):
        clip, meta = load_recording(path)
        truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["signs"]
        sessions.append((path.stem, clip, meta, truth))
    default = asdict(Params())
    table = []
    for values in itertools.product(*GRID.values()):
        params = Params(**{**default, **dict(zip(GRID, values))})
        rows = run(model, sessions, params)
        table.append({**dict(zip(GRID, values)), "default": dict(zip(GRID, values)) == {k: default[k] for k in GRID},
                      "segments_off": sum(abs(r["segments"] - r["true_words"]) for r in rows),
                      "correct": sum(r["correct"] for r in rows), "true_words": sum(r["true_words"] for r in rows),
                      "sessions_ok": sum(r["correct"] >= 3 for r in rows),
                      **{f"{r['session']}": f"{r['segments']} seg, {r['correct']}/{r['true_words']}: {r['sentence']}"
                         for r in rows}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, table[0].keys())
        w.writeheader()
        w.writerows(table)
    best = sorted(table, key=lambda r: (-r["correct"], r["segments_off"]))
    n_best = sum(r["correct"] == best[0]["correct"] and r["segments_off"] == best[0]["segments_off"] for r in table)
    print(f"{len(table)} parameter sets x {len(sessions)} sessions -> {OUT}")
    print(f"best: {best[0]['correct']}/{best[0]['true_words']} words, segment count off by {best[0]['segments_off']} "
          f"({n_best} sets reach it)")
    d = [r for r in table if r["default"]][0]
    print(f"config defaults {dict((k, default[k]) for k in GRID)}: {d['correct']}/{d['true_words']} words, "
          f"off by {d['segments_off']}")
    for k in GRID:  # robustness: worst result over the other parameters for each value of k
        print(f"  {k}: " + ", ".join(f"{v}: {min(r['correct'] for r in table if r[k] == v)}-"
                                      f"{max(r['correct'] for r in table if r[k] == v)}" for v in GRID[k]))


if __name__ == "__main__":
    main()
