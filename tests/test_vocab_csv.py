import json

import pandas as pd
import pytest

from asl import config


@pytest.fixture(scope="module")
def vocab():
    return pd.read_csv(config.VOCAB_CSV, keep_default_na=False)


def test_vocab_shape(vocab):
    assert list(vocab.columns) == ["sign", "class_id", "en", "hy", "ru", "notes"]
    assert len(vocab) == config.NUM_CLASSES
    assert vocab["sign"].is_unique
    assert sorted(vocab["class_id"]) == list(range(config.NUM_CLASSES))


def test_vocab_no_empty_translations(vocab):
    for col in ("en", "hy", "ru"):
        assert (vocab[col].str.strip() != "").all(), col


def test_hy_is_armenian_script(vocab):
    # Armenian block U+0530..U+058F; allow spaces and punctuation.
    for word in vocab["hy"]:
        letters = [c for c in word if c.isalpha()]
        assert letters and all("԰" <= c <= "֏" for c in letters), word


def test_vocab_matches_label_map(vocab):
    path = config.SAMPLE_DIR / "sign_to_prediction_index_map.json"
    if not path.exists():
        pytest.skip("label map not downloaded")
    label_map = json.loads(path.read_text())
    assert dict(zip(vocab["sign"], vocab["class_id"])) == label_map
