"""Learn-mode progress in progress.json (git-ignored), and the defense set export.

A sign counts as learned after LEARN_SUCCESSES successful attempts: the sign is the model's top-1
and its probability reaches the confidence threshold. Learned stays learned. Signs the user had
already shown reliably in earlier tests can be seeded from that evidence (scripts/seed_progress.py),
marked as such. The file is written atomically, and a broken file is kept aside and never stops the app.
"""

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict

from asl import config

log = logging.getLogger(__name__)
FORMAT_VERSION = 1


@dataclass
class SignProgress:
    attempts: int = 0  # Learn-mode attempts
    successes: int = 0
    seeded_attempts: int = 0  # earlier test evidence counted by seed_progress.py (kept apart)
    seeded_successes: int = 0
    seeded_from: str = ""
    best_p: float = 0.0
    learned: bool = False
    last: str = ""  # ISO time of the last Learn-mode attempt

    @property
    def total_attempts(self) -> int:
        return self.attempts + self.seeded_attempts

    @property
    def total_successes(self) -> int:
        return self.successes + self.seeded_successes

    @property
    def success_rate(self) -> float:
        return self.total_successes / self.total_attempts if self.total_attempts else 0.0


@dataclass
class Progress:
    path: object = config.PROGRESS_JSON
    signs: Dict[str, SignProgress] = field(default_factory=dict)

    @classmethod
    def load(cls, path=config.PROGRESS_JSON) -> "Progress":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            signs = {s: SignProgress(**v) for s, v in data["signs"].items()}
        except FileNotFoundError:
            return cls(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            broken = path.with_name(f"{path.stem}.broken-{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
            log.warning("progress: cannot read %s (%s); kept as %s, starting empty", path, e, broken.name)
            try:
                path.replace(broken)
            except OSError:
                pass
            return cls(path)
        return cls(path, signs)

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"version": FORMAT_VERSION, "signs": {s: asdict(v) for s, v in self.signs.items()}},
                                  indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: an interrupted save never leaves half a file

    def get(self, sign: str) -> SignProgress:
        return self.signs.get(sign, SignProgress())

    def is_learned(self, sign: str) -> bool:
        return self.get(sign).learned

    def record(self, sign: str, p: float, rank: int, threshold: float) -> bool:
        """One Learn-mode attempt at `sign`; returns True if it was a success. Saved at once."""
        sp = self.signs.setdefault(sign, SignProgress())
        success = rank == 1 and p >= threshold
        sp.attempts += 1
        sp.successes += success
        sp.best_p = max(sp.best_p, round(float(p), 4))
        sp.learned = sp.learned or sp.total_successes >= config.LEARN_SUCCESSES
        sp.last = datetime.now().isoformat(timespec="seconds")
        self.save()
        return success

    def seed(self, sign: str, attempts: int, successes: int, best_p: float, source: str) -> SignProgress:
        """Count earlier test evidence for a sign (replaces an earlier seed; Learn attempts are kept)."""
        sp = self.signs.setdefault(sign, SignProgress())
        sp.seeded_attempts, sp.seeded_successes, sp.seeded_from = attempts, successes, source
        sp.best_p = max(sp.best_p, round(float(best_p), 4))
        sp.learned = sp.learned or sp.total_successes >= config.LEARN_SUCCESSES
        self.save()
        return sp


def defense_set(progress: Progress, vocab: dict, val_accuracy: dict, limit: int = config.DEFENSE_SET_MAX) -> list:
    """The signs the user performs most reliably: at least one success, learned first, then success
    rate, successes and best probability; the model's validation accuracy is shown alongside."""
    rows = []
    for sign, sp in progress.signs.items():
        if sp.total_successes == 0:
            continue
        v = vocab.get(sign, {})
        rows.append({"sign": sign, "en": v.get("en", sign), "hy": v.get("hy", ""), "ru": v.get("ru", ""),
                     "learned": sp.learned, "attempts": sp.total_attempts, "successes": sp.total_successes,
                     "success_rate": round(sp.success_rate, 2), "best_p": sp.best_p,
                     "val_accuracy": val_accuracy.get(sign, ""), "seeded_from": sp.seeded_from})
    rows.sort(key=lambda r: (not r["learned"], -r["success_rate"], -r["successes"], -r["best_p"], r["sign"]))
    return rows[:limit]


def export_defense_set(rows: list, path=config.DEFENSE_SET_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sign", "en", "hy", "ru", "learned", "attempts", "successes", "success_rate", "best_p", "val_accuracy",
              "seeded_from"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
