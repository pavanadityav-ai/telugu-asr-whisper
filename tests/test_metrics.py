"""Tests for the metric and text code. Run with:  python -m pytest -q"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import align, cer, char_confusions, error_counts, wer, word_hits  # noqa: E402
from src.text_utils import normalize_text, script_profile  # noqa: E402


# ---------- metrics ----------

def test_perfect_match_is_zero():
    assert wer(["నేను ఇంటికి వెళ్ళాను"], ["నేను ఇంటికి వెళ్ళాను"]) == 0.0
    assert cer(["నేను"], ["నేను"]) == 0.0


def test_one_wrong_word_out_of_four():
    ref = ["the cat sat down"]
    hyp = ["the bat sat down"]
    assert wer(ref, hyp) == 0.25


def test_counts_sub_del_ins():
    c = error_counts("a b c d".split(), "a x c".split())
    assert (c["S"], c["D"], c["I"]) == (1, 1, 0)
    c = error_counts("a b".split(), "a b c".split())
    assert (c["S"], c["D"], c["I"]) == (0, 0, 1)


def test_wer_can_go_above_one():
    # Many inserted words: more errors than reference words.
    assert wer(["hello"], ["hello there my friend"]) == 3.0


def test_cer_matches_hand_calculation():
    # kitten -> sitting needs 3 edits; "kitten" has 6 characters.
    assert abs(cer(["kitten"], ["sitting"]) - 3 / 6) < 1e-9


def test_corpus_rate_is_weighted_by_length():
    # 1 error in a 1-word sentence + 0 errors in a 3-word sentence = 1/4
    assert wer(["a", "b c d"], ["x", "b c d"]) == 0.25


def test_empty_reference_is_skipped():
    assert wer(["", "a b"], ["junk", "a b"]) == 0.0


def test_align_operations():
    ops = align(list("abc"), list("axc"))
    assert [op for op, _, _ in ops] == ["C", "S", "C"]


def test_telugu_char_confusion():
    # క (ka) heard as ఖ (kha): a classic aspiration error
    conf = char_confusions(["కథ"], ["ఖథ"])
    assert conf[("S", "క", "ఖ")] == 1


def test_word_hits():
    hits = word_hits(["a b c"], ["a x c"])
    assert hits == [("a", True), ("b", False), ("c", True)]


# ---------- text normalisation ----------

def test_keeps_telugu_vowel_signs():
    # Whisper's default normaliser would delete the vowel signs here.
    assert normalize_text("నాయకత్వంలో") == "నాయకత్వంలో"


def test_removes_punctuation_and_extra_spaces():
    assert normalize_text("  నేను,  వచ్చాను!  ") == "నేను వచ్చాను"
    assert normalize_text("ఇది ఒక వాక్యం।") == "ఇది ఒక వాక్యం"


def test_lowercases_english_and_drops_invisible_chars():
    assert normalize_text("Hello\u200cWorld") == "helloworld"


def test_script_profile():
    # "నా" is 2 code points (న + vowel sign ా), "ab" is 2 Latin letters.
    p = script_profile("నా ab")
    assert p["telugu"] == 0.5
    assert p["latin"] == 0.5
    assert p["other"] == 0.0
