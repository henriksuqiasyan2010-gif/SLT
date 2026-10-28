import pytest

from asl import config
from asl.vocab import LANGUAGES, VocabError, load_vocab


def test_load_vocab():
    vocab = load_vocab()
    assert len(vocab) == config.NUM_CLASSES
    assert set(vocab["hello"]) == set(LANGUAGES)
    assert all(all(v[lang] for lang in LANGUAGES) for v in vocab.values())


def test_errors(tmp_path):
    with pytest.raises(VocabError, match="cannot read"):
        load_vocab(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("sign,en\nhello,hello\n", encoding="utf-8")
    with pytest.raises(VocabError, match="columns"):
        load_vocab(bad)
    empty = tmp_path / "empty.csv"
    empty.write_text("sign,class_id,en,hy,ru,notes\nhello,0,hello,,привет,\n", encoding="utf-8")
    with pytest.raises(VocabError, match="no hy"):
        load_vocab(empty)


def test_validate_against_the_model_labels():
    import csv
    import json
    from asl.vocab import validate_vocab
    path = config.MODELS_DIR / "labels.json"
    if path.exists():
        labels = {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    else:  # no model: the class ids of vocab.csv itself
        with open(config.VOCAB_CSV, encoding="utf-8") as f:
            labels = {int(r["class_id"]): r["sign"] for r in csv.DictReader(f)}
    vocab = load_vocab()
    assert validate_vocab(vocab, labels) == []
    broken = dict(vocab)
    broken.pop("hello")
    broken["selfie"] = {"en": "selfie", "hy": "x", "ru": "y"}
    problems = validate_vocab(broken, labels)
    assert any("no row for: hello" in p for p in problems) and any("selfie" in p for p in problems)
    swapped = dict(labels)
    swapped[0], swapped[1] = labels[1], labels[0]
    assert any("class_id differs" in p for p in validate_vocab(vocab, swapped))
