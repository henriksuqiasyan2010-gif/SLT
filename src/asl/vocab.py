"""Translations of the 250 signs from data/vocab.csv (sign, class_id, en, hy, ru, notes).

    vocab = load_vocab()
    vocab["hello"]["hy"]        # -> 'բարև'

Phase 7 adds the full start-up validation against labels.json.
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
