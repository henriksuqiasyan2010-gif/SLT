"""Translations of the 250 signs from data/vocab.csv (sign, class_id, en, hy, ru, notes).

    vocab = load_vocab()
    vocab["hello"]["hy"]        # -> 'բարև'

validate_vocab() is the start-up check against the model's labels.json (asl.app.__main__).
"""

import csv

from asl import config

LANGUAGES = ("en", "hy", "ru")


class VocabError(RuntimeError):
    """vocab.csv is missing, malformed or incomplete."""


def load_vocab(path=config.VOCAB_CSV) -> dict:
    """{sign: {"en": ..., "hy": ..., "ru": ...}}; raises VocabError on missing columns or empty cells."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise VocabError(f"cannot read {path}: {e}") from e
    if not rows or not {"sign", *LANGUAGES} <= set(rows[0]):
        raise VocabError(f"{path}: expected columns sign, {', '.join(LANGUAGES)}")
    vocab = {}
    for r in rows:
        empty = [lang for lang in LANGUAGES if not r[lang].strip()]
        if empty:
            raise VocabError(f"{path}: sign {r['sign']!r} has no {', '.join(empty)} translation")
        vocab[r["sign"]] = {lang: r[lang].strip() for lang in LANGUAGES}
    return vocab


def validate_vocab(vocab: dict, labels: dict, path=config.VOCAB_CSV) -> list:
    """Start-up check against the model's labels.json ({class id: sign}): every class present with
    the same class_id, no extra signs, NUM_CLASSES rows. Returns a list of problems (empty = OK)."""
    problems = []
    if len(vocab) != config.NUM_CLASSES:
        problems.append(f"{len(vocab)} signs instead of {config.NUM_CLASSES}")
    missing = sorted(set(labels.values()) - set(vocab))
    if missing:
        problems.append(f"no row for: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    extra = sorted(set(vocab) - set(labels.values()))
    if extra:
        problems.append(f"not in the model: {', '.join(extra[:10])}{' ...' if len(extra) > 10 else ''}")
    with open(path, encoding="utf-8", newline="") as f:
        ids = {r["sign"]: r.get("class_id", "") for r in csv.DictReader(f)}
    wrong = [s for k, s in labels.items() if s in ids and str(ids[s]).strip() != str(k)]
    if wrong:
        problems.append(f"class_id differs from labels.json for: {', '.join(wrong[:10])}")
    return problems
