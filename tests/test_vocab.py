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
