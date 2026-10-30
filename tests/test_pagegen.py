import unicodedata

from doclens.pagegen import wrap


def test_wrap_fills_lines_by_width():
    assert wrap("aa bb cc dd".split(), 5, len) == ["aa bb", "cc dd"]
    assert wrap("aa bb cc".split(), 5, len, indent=2) == ["aa", "bb cc"]


def test_a_word_wider_than_a_line_is_broken_without_a_hyphen():
    lines = wrap(["ab", "abcdefghij", "cd"], 4, len)
    assert lines == ["ab", "abcd", "efgh", "ij", "cd"]
    assert all(len(t) <= 4 for t in lines)
    assert wrap(["abcdefghij"], 6, len, indent=2) == ["abcd", "efghij"]   # the first line is indented


def test_a_broken_hindi_word_keeps_its_vowel_signs():
    word = "विश्वविद्यालयों"
    lines = wrap([word], 3, len)
    assert "".join(lines) == word
    assert len(lines) > 1
    assert not any(unicodedata.category(t[0]).startswith("M") for t in lines)   # no piece starts with a sign
