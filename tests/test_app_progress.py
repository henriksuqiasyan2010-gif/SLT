import csv
import json

from asl import config
from asl.app.progress import Progress, SignProgress, defense_set, export_defense_set


def test_learned_after_three_successes_and_persists(tmp_path):
    path = tmp_path / "progress.json"
    pr = Progress.load(path)
    assert pr.get("dad") == SignProgress() and not path.exists()
    assert not pr.record("dad", 0.30, 1, 0.4)  # top-1 but below the threshold
    assert not pr.record("dad", 0.80, 2, 0.4)  # another sign won
    for _ in range(config.LEARN_SUCCESSES):
        assert not pr.is_learned("dad")
        assert pr.record("dad", 0.75, 1, 0.4)
    assert pr.is_learned("dad")
    again = Progress.load(path)  # a new run of the app
    sp = again.get("dad")
    assert (sp.attempts, sp.successes, sp.best_p, sp.learned) == (5, 3, 0.8, True)
    assert again.record("dad", 0.1, 5, 0.4) is False and again.is_learned("dad")  # learned stays learned


def test_seed_is_kept_apart_from_learn_attempts(tmp_path):
    pr = Progress.load(tmp_path / "p.json")
    pr.record("mom", 0.9, 1, 0.4)
    pr.seed("mom", attempts=4, successes=2, best_p=0.87, source="tests")
    sp = pr.get("mom")
    assert (sp.total_attempts, sp.total_successes, sp.learned) == (5, 3, True)
    pr.seed("mom", attempts=4, successes=1, best_p=0.5, source="tests again")  # replaces the earlier seed
    sp = Progress.load(tmp_path / "p.json").get("mom")
    assert (sp.attempts, sp.successes, sp.seeded_attempts, sp.seeded_successes) == (1, 1, 4, 1)
    assert sp.learned  # learned once, stays learned


def test_broken_file_is_kept_aside(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{broken", encoding="utf-8")
    pr = Progress.load(path)
    assert pr.signs == {} and list(tmp_path.glob("progress.broken-*.json"))
    pr.record("hello", 0.9, 1, 0.4)
    assert json.loads(path.read_text(encoding="utf-8"))["signs"]["hello"]["successes"] == 1


def test_defense_set_order_and_export(tmp_path):
    pr = Progress.load(tmp_path / "p.json")
    pr.seed("hello", 5, 5, 0.87, "tests")
    pr.seed("mom", 4, 2, 0.87, "tests")
    pr.seed("dad", 2, 0, 0.1, "tests")
    pr.record("hat", 0.9, 1, 0.4)
    vocab = {"hello": {"en": "hello", "hy": "բարև", "ru": "привет"}}
    rows = defense_set(pr, vocab, {"hello": 0.70})
    assert [r["sign"] for r in rows] == ["hello", "hat", "mom"]  # learned first; dad never succeeded
    assert rows[0]["learned"] and rows[0]["val_accuracy"] == 0.70 and rows[1]["hy"] == ""
    n = export_defense_set(rows, tmp_path / "defense.csv")
    with open(tmp_path / "defense.csv", encoding="utf-8-sig") as f:
        out = list(csv.DictReader(f))
    assert n == 3 and out[0]["ru"] == "привет" and out[2]["successes"] == "2"
