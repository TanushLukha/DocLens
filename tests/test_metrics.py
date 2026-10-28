import random

import pytest

from doclens.metrics import Scores, align, levenshtein, score
from doclens.text import normalize, wrong_script


def naive(a, b):
    """The textbook recursion with memoisation: slow, obviously right."""
    from functools import lru_cache

    @lru_cache(None)
    def d(i, j):
        if i == 0 or j == 0:
            return i + j
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
    return d(len(a), len(b))


def test_textbook_cases():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_matches_naive_and_rapidfuzz_on_random_strings():
    rf = pytest.importorskip("rapidfuzz.distance").Levenshtein
    rng = random.Random(0)
    alphabet = "abक्षिा "
    for _ in range(300):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        assert levenshtein(a, b) == naive(a, b) == rf.distance(a, b)


def test_alignment_reproduces_both_strings_and_the_distance():
    rng = random.Random(1)
    for _ in range(200):
        a = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 10)))
        b = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 10)))
        ops = align(a, b)
        assert "".join(r for op, r, _ in ops if r is not None) == a
        assert "".join(h for op, _, h in ops if h is not None) == b
        assert sum(op != "ok" for op, _, _ in ops) == levenshtein(a, b)


def test_cer_counts_code_points_after_nfc():
    # the same visible letter as one precomposed code point or as base + nukta must score 0.
    # Nukta letters are composition exclusions: NFC writes both as base + nukta.
    precomposed, decomposed = chr(0x095B), chr(0x091C) + chr(0x093C)
    assert normalize(precomposed) == normalize(decomposed) == decomposed
    assert score([precomposed], [decomposed])["cer"] == 0.0


def test_corpus_rates_weigh_by_length_and_report_catastrophes():
    sc = Scores()
    sc.add("abcdefghij", "abcdefghij")   # 0 errors in 10
    sc.add("ab", "xyzw")                  # 4 errors in 2: a catastrophe
    s = sc.summary()
    assert s["cer"] == pytest.approx(4 / 12)
    assert s["catastrophic"] == 0.5 and s["exact"] == 0.5
    assert s["median_cer"] == pytest.approx(1.0)


def test_wrong_script_flags_only_letters_of_a_missing_script():
    assert wrong_script("कमरे", "FAR")                 # Hindi read as Latin letters
    assert wrong_script("TOTAL", "टोटल")               # English read as Devanagari
    assert not wrong_script("कमरे", "कमर")             # a misreading, same script
    assert not wrong_script("24,000", "24,O00")        # numbers have no letters to flip
    assert not wrong_script("IIT दिल्ली", "IIT दिल")   # both scripts in the truth
    sc = Scores()
    sc.add("कमरे", "FAR")
    sc.add("कमरे", "कमरे")
    assert sc.summary()["wrong_script"] == 0.5


def test_wer_counts_words():
    assert score(["the cat sat"], ["the bat sat"])["wer"] == pytest.approx(1 / 3)
