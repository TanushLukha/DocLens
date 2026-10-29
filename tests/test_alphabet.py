from collections import Counter

import pytest

from doclens.alphabet import BLANK, Alphabet, build


def test_round_trip_and_blank():
    abc = Alphabet(["क", "्", "ष", "a", " "])
    ids = abc.encode("क्ष a")
    assert BLANK not in ids and len(abc) == 6
    assert abc.decode(ids) == "क्ष a"
    assert abc.decode([BLANK] + ids + [BLANK]) == "क्ष a"


def test_missing_characters_are_reported_and_refused():
    abc = Alphabet(["a", "b"])
    assert not abc.covers("abc") and abc.missing("abc") == ["c"]
    with pytest.raises(KeyError):
        abc.encode("abc")


def test_build_keeps_real_characters_and_frequent_drawable_ones(tmp_path):
    synthetic = Counter({"क": 500, "a": 500, "中": 5, "\u0007": 900, "ঞ": 300})
    real = Counter({"।": 3, "॥": 1})
    abc = build(synthetic, real, drawable=lambda ch: ch != "ঞ", min_count=50)
    assert set(abc.chars) == {"क", "a", "।", "॥"}        # rare, control and undrawable characters are out
    abc.save(tmp_path / "a.json")
    assert Alphabet.load(tmp_path / "a.json").chars == abc.chars


def test_duplicates_and_multi_code_point_symbols_are_refused():
    with pytest.raises(ValueError):
        Alphabet(["a", "a"])
    with pytest.raises(ValueError):
        Alphabet(["क्ष"])
