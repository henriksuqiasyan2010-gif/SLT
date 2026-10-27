"""All 250 signs the model knows, with translations and the model's validation accuracy per sign.

    python scripts/export_signs.py [--per-sign reports/<run>/<run>_T32/per_sign_accuracy.csv]

-> reports/signs_250.csv (UTF-8 with BOM, opens in Excel). Default accuracy source: the run of the
model in models/ (metrics.json "preset").
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.vocab import load_vocab  # noqa: E402

OUT = config.REPORTS_DIR / "signs_250.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", type=Path, help="per_sign_accuracy.csv of a training run")
    args = ap.parse_args()
    preset = json.loads((config.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))["preset"]
    per_sign = args.per_sign or config.REPORTS_DIR / preset / f"{preset}_T32" / "per_sign_accuracy.csv"
    with open(per_sign, encoding="utf-8") as f:
        acc = {r["sign"]: r for r in csv.DictReader(f)}
    vocab = load_vocab()
    rows = []
    for i, sign in enumerate(sorted(vocab), 1):
        a = acc.get(sign, {})
        rows.append({"#": i, "sign": sign, "en": vocab[sign]["en"], "hy": vocab[sign]["hy"], "ru": vocab[sign]["ru"],
                     f"accuracy % ({preset})": round(100 * float(a["accuracy"])) if a else "",
                     "accuracy % at 13 fps": round(100 * float(a["accuracy_13fps"])) if a else "",
                     "val clips": a.get("n_val", "")})
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} signs -> {OUT} (accuracy from {per_sign})")


if __name__ == "__main__":
    main()
