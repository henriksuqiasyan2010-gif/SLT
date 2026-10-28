import csv

from asl.app.pipeline import Prediction
from asl.app.sentence import Sentence, log_correction, word_matches


def pred(sign, p=0.9, accepted=True, alts=(("hat", 0.05), ("cowboy", 0.03))):
    return Prediction(top=[(sign, p), *alts], accepted=accepted, threshold=0.4, n_frames=30, fps=15.0,
                      duration_s=2.0, kept_s=1.5, hand_share=0.8, attempt=f"{sign}.npy")


def test_add_threshold_and_repeat_rule():
    s = Sentence(repeat_s=1.0)
    assert s.add(pred("hello"), 0.0)
    assert not s.add(pred("hello"), 0.5)  # the same word again within 1 s
    assert s.add(pred("hello"), 1.6)  # later: a real repetition
    assert not s.add(pred("please", 0.3, accepted=False), 3.0)  # below the threshold
    assert s.add(pred("please"), 3.5)
    assert s.signs() == ["hello", "hello", "please"]


def test_delete_clear_and_text():
    s = Sentence()
    for i, w in enumerate(("hello", "please", "hat")):
        s.add(pred(w), 2.0 * i)
    vocab = {"hello": {"en": "hello", "hy": "բարև", "ru": "привет"}}
    assert s.text(vocab, "ru") == "привет please hat"  # missing words fall back to the sign name
    assert s.delete_last().sign == "hat" and s.signs() == ["hello", "please"]
    s.clear()
    assert s.signs() == [] and s.delete_last() is None


def test_replace_last_with_an_alternative_and_log(tmp_path):
    s = Sentence()
    s.add(pred("hello"), 0.0)
    assert s.replace_last("dad") is None  # not one of the top-3
    w = s.replace_last("hat")
    assert w.sign == "hat" and w.prob == 0.05 and w.replaced_from == "hello"
    assert s.replace_last("cowboy").replaced_from == "hello"  # the original prediction is kept
    log_correction(w, "sentence_auto", tmp_path / "c.csv")
    rows = list(csv.DictReader(open(tmp_path / "c.csv", encoding="utf-8")))
    assert rows[0]["predicted"] == "hello" and rows[0]["corrected_to"] == "cowboy"
    assert Sentence().replace_last("hat") is None


def test_word_matches_keeps_order():
    truth = ["hello", "please", "happy", "hat", "shower"]
    assert word_matches(truth, truth) == 5
    assert word_matches(["hello", "happy", "hat", "shower"], truth) == 4  # one missed segment
    assert word_matches(["hello", "x", "please", "happy", "hat", "shower"], truth) == 5  # one extra
    assert word_matches(["shower", "hat"], truth) == 1
