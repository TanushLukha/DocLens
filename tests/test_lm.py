from pathlib import Path

import pytest

from doclens import REPO, lm


def test_pick_and_tune_prefer_the_language_model_only_when_it_helps():
    refs = ["राजधानी", "दिल्ली", "TOTAL 45,000"]
    nbest = [[("राजधाली", -1.0), ("राजधानी", -1.2)],     # the beam's best has a wrong consonant
             [("दिल्ली", -0.5), ("दिल्लि", -0.9)],
             [("TOTAL 45,000", -0.3), ("TOTAL 45.000", -0.4)]]
    lms = [[-30.0, -12.0], [-10.0, -25.0], [-40.0, -38.0]]
    w = lm.tune(refs, nbest, lms)
    assert w["deva"]["lam"] > 0 and w["deva"]["edits"] == 0 < w["deva"]["edits_plain_beam"]
    assert w["latn"]["lam"] == 0.0                     # rescoring would break the receipt line
    assert lm.apply(nbest, lms, w) == refs


def test_group_puts_mixed_lines_with_hindi():
    assert lm.group([("हाइड्रोजन (Hydrogen)", 0.0)]) == "deva"
    assert lm.group([("Hydrogen", 0.0)]) == "latn"


BUNDLE = REPO.parent / "TinyGPT" / "models" / "lm"


@pytest.mark.skipif(not (BUNDLE / "config.json").exists(), reason="TinyGPT's model bundle is not next to DocLens")
def test_tinygpt_prefers_real_hindi():
    pytest.importorskip("tinygpt")
    scorer = lm.TinyGPTScorer(Path(BUNDLE))
    texts = ["भारत की राजधानी नई दिल्ली है।", "भारत की राजधनी नई दिल्ली है।", "kal subah 7 baje ka alarm"]
    good, bad, hinglish = scorer.score(texts)
    assert good > bad
    one_by_one = scorer.score(texts, batch_size=1)          # padding and batch size change nothing
    assert all(abs(a - b) < 1e-3 for a, b in zip([good, bad, hinglish], one_by_one))
