import csv
import re

import pytest

from asl import config
from asl.app import strings

ALLOWED_LATIN = {"ASL", "FPS", "OK", "Google", "Isolated", "Sign", "Language", "Recognition", "Kaggle", "top",
                 "logs", "app", "log", "data", "vocab", "csv", "English"}


@pytest.fixture(autouse=True)
def restore_language():
    yield
    strings.set_language(config.UI_LANGUAGES[0])


def test_every_key_has_three_languages_and_same_placeholders():
    for key, texts in strings.STRINGS.items():
        assert set(texts) == set(config.UI_LANGUAGES), key
        assert all(t.strip() for t in texts.values()), key
        assert len({frozenset(strings.placeholders(t)) for t in texts.values()}) == 1, key


def test_armenian_texts_use_armenian_letters():
    for key, texts in strings.STRINGS.items():
        if key.startswith("lang_name_") or texts["hy"] == texts["en"]:  # language names, "FPS {fps}"
            continue
        hy = re.sub(r"\{[^}]*\}", " ", texts["hy"])  # placeholders like {left:.1f}
        words = "".join(c if c.isalpha() else " " for c in hy).split()
        latin = [w for w in words if all(c.isascii() for c in w) and w not in ALLOWED_LATIN]
        assert not latin, (key, latin)
        assert any("Ա" <= c <= "֏" for c in hy), key


def test_default_language_is_armenian_and_switch():
    assert strings.language() == "hy" == config.UI_LANGUAGES[0]
    strings.set_language("ru")
    assert strings.tr("btn_speak") == "Произнести"
    assert strings.tr("overlay_recording", left=1.25) == "Запись: 1.2 с"
    with pytest.raises(ValueError):
        strings.set_language("de")


def test_export_csv(tmp_path):
    n = strings.export_csv(tmp_path / "ui.csv")
    with open(tmp_path / "ui.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert n == len(rows) == len(strings.STRINGS)
    assert list(rows[0]) == ["key", "en", "hy", "ru"]
    assert {r["key"]: r["hy"] for r in rows}["btn_speak"] == strings.STRINGS["btn_speak"]["hy"]
