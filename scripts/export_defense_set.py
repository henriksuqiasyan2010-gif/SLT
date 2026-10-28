"""Export reports/defense_set.csv: the signs the user performs most reliably (from progress.json).

    python scripts/export_defense_set.py

Same as the "Export the defense set" button in Learn mode. The model's validation accuracy per sign
comes from reports/<preset>/<preset>_T32/per_sign_accuracy.csv of the model in models/.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.app.progress import Progress, defense_set, export_defense_set  # noqa: E402
from asl.vocab import load_vocab  # noqa: E402

if __name__ == "__main__":
    preset = json.loads((config.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))["preset"]
    with open(config.REPORTS_DIR / preset / f"{preset}_T32" / "per_sign_accuracy.csv", encoding="utf-8") as f:
        val = {r["sign"]: round(float(r["accuracy"]), 3) for r in csv.DictReader(f)}
    rows = defense_set(Progress.load(), load_vocab(), val)
    n = export_defense_set(rows)
    print(f"{n} signs -> {config.DEFENSE_SET_CSV}")
    for r in rows:
        print(f"  {r['sign']:10s} learned={r['learned']!s:5s} {r['successes']}/{r['attempts']}  val {r['val_accuracy']}")
