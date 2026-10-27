import json

from asl import config
from asl.app import settings as S


def test_missing_file_gives_defaults(tmp_path):
    s = S.load(tmp_path / "nope.json")
    assert s == S.AppSettings()
    assert s.language == "hy" and s.record_seconds == config.SINGLE_RECORD_SECONDS
    assert s.threshold == config.CONFIDENCE_THRESHOLD and s.sound and s.mirrored_preview


def test_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    s = S.AppSettings(camera_index=1, record_seconds=3.0, threshold=0.55, sound=False, mirrored_preview=False,
                      language="ru")
    S.save(s, path)
    assert S.load(path) == s
    assert json.loads(path.read_text(encoding="utf-8"))["language"] == "ru"


def test_bad_values_fall_back_one_by_one(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"camera_index": -1, "record_seconds": 99, "threshold": "high", "sound": "yes",
                                "language": "de", "mirrored_preview": False, "extra": 1}), encoding="utf-8")
    s = S.load(path)
    d = S.AppSettings()
    assert (s.camera_index, s.record_seconds, s.threshold, s.sound, s.language) == \
        (d.camera_index, d.record_seconds, d.threshold, d.sound, d.language)
    assert s.mirrored_preview is False  # the valid value is kept


def test_broken_json_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert S.load(path) == S.AppSettings()
    path.write_text("[1, 2]", encoding="utf-8")
    assert S.load(path) == S.AppSettings()
