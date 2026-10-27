"""Export every interface text to reports/ui_strings.csv (key, en, hy, ru) for the user's review.

    python scripts/export_ui_strings.py

Edit the texts in src/asl/app/strings.py, not in the CSV (the CSV is regenerated).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402
from asl.app import strings  # noqa: E402

if __name__ == "__main__":
    n = strings.export_csv()
    print(f"{n} strings -> {config.UI_STRINGS_CSV}")
