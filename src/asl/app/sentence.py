"""The sentence line of Sentence mode, without Qt.

Words are appended from predictions (automatic segments or button recordings alike): only if the
top-1 probability reaches the threshold, and not the same word again within SENTENCE_REPEAT_S.
The user can replace the last word by one of its top-3 alternatives; replacements are logged
(logs/corrections.csv) because they say what the user actually signed.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from asl import config


@dataclass
class Word:
    sign: str
    prob: float
    top: List[Tuple[str, float]]  # the model's top-3 for this word, for the replacement buttons
    time: float  # stream time in seconds (for the repeat rule)
    attempt: Optional[str] = None  # saved attempt file (logs/attempts/)
    replaced_from: Optional[str] = None


@dataclass
class Sentence:
    repeat_s: float = config.SENTENCE_REPEAT_S
    words: List[Word] = field(default_factory=list)

    def add(self, prediction, t: float) -> bool:
        """Append the prediction's top-1 if accepted and not a repeat; True if appended."""
        if not prediction.accepted or not prediction.top:
            return False
        sign, prob = prediction.top[0]
        last = self.words[-1] if self.words else None
        if last is not None and last.sign == sign and t - last.time < self.repeat_s:
            return False
        self.words.append(Word(sign, prob, list(prediction.top), t, prediction.attempt))
        return True

    def delete_last(self) -> Optional[Word]:
        return self.words.pop() if self.words else None

    def clear(self) -> None:
        self.words.clear()

    def replace_last(self, sign: str) -> Optional[Word]:
        """Replace the last word by one of its alternatives; returns the changed word."""
        if not self.words:
            return None
        w = self.words[-1]
        alternatives = dict(w.top)
        if sign not in alternatives or sign == w.sign:
            return None
        w.replaced_from = w.replaced_from or w.sign
        w.sign, w.prob = sign, alternatives[sign]
        return w

    def signs(self) -> List[str]:
        return [w.sign for w in self.words]

    def text(self, vocab: dict, lang: str) -> str:
        return " ".join(vocab.get(s, {}).get(lang, s) for s in self.signs())


def log_correction(word: Word, mode: str, path=config.CORRECTIONS_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "mode", "predicted", "corrected_to", "attempt"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), mode, word.replaced_from, word.sign, word.attempt])


def word_matches(predicted: List[str], truth: List[str]) -> int:
    """Correct words of a sentence: length of the longest common subsequence (order kept, so an
    extra or a missing segment does not shift all later words)."""
    n, m = len(predicted), len(truth)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            best[i + 1][j + 1] = best[i][j] + 1 if predicted[i] == truth[j] else max(best[i][j + 1], best[i + 1][j])
    return best[n][m]
