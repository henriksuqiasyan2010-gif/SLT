"""Score a model on everything the user has recorded with a known sign:

  * recordings/*.npy (not recordings/old): the sign comes from the sidecar
  * the app attempts listed in reports/user_tests/*.csv (attempt file in logs/attempts/, sign)

    python scripts/score_user.py [--model-dir models] [--out reports/user_tests/score_<preset>.csv]

Live path as in the app (SignModel.predict_clip). Prints top-1 / top-3 / accepted per group.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402

USER_TESTS = config.REPORTS_DIR / "user_tests"


def labelled_clips():
    for path in sorted(config.RECORDINGS_DIR.glob("*.npy")):
        clip, meta = load_recording(path)
        yield "recording", path.stem, meta.get("sign") or path.stem.split("_")[0], clip, meta
    for table in sorted(USER_TESTS.glob("phase*_*.csv")):
        with open(table, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clip, meta = load_recording(config.ATTEMPTS_DIR / row["attempt"])
                yield f"attempt:{table.stem}", Path(row["attempt"]).stem, row["sign"], clip, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    model = SignModel.load(args.model_dir)
    preset = model.metrics.get("preset", "model")
    rows = []
    for group, name, sign, clip, meta in labelled_clips():
        probs = model.predict_clip(clip, meta["fps"], recording_aspect(meta))
        top = model.top_k(probs, config.TOP_K)
        rank = next((i + 1 for i, (s, _) in enumerate(top) if s == sign), None)
        p_true = float(probs[[k for k, v in model.labels.items() if v == sign][0]])
        rows.append({"group": group, "name": name, "sign": sign, "rank": rank or "", "p_true": round(p_true, 4),
                     "accepted": rank == 1 and top[0][1] >= config.CONFIDENCE_THRESHOLD,
                     "top3": ", ".join(f"{s} {p:.2f}" for s, p in top)})
    out = args.out or USER_TESTS / f"score_{preset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for group in sorted({r["group"] for r in rows}):
        g = [r for r in rows if r["group"] == group]
        print(f"{preset} {group}: top-1 {sum(r['rank'] == 1 for r in g)}/{len(g)}, "
              f"top-3 {sum(r['rank'] != '' for r in g)}/{len(g)}, accepted {sum(r['accepted'] for r in g)}/{len(g)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
