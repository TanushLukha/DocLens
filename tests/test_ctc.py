import itertools
import math

import numpy as np

from doclens.ctc import beam_search, greedy, path_logprob


def logp_from(probs) -> np.ndarray:
    p = np.asarray(probs, np.float64)
    return np.log(p / p.sum(axis=1, keepdims=True))


def test_greedy_merges_repeats_and_drops_blanks():
    # best path: a a _ a b b, read as a a b
    best = [1, 1, 0, 1, 2, 2]
    logp = np.full((len(best), 3), -10.0)
    logp[np.arange(len(best)), best] = 0.0
    assert greedy(logp) == [1, 1, 2]


def all_labelings(classes: int, max_len: int):
    for n in range(max_len + 1):
        yield from itertools.product(range(1, classes), repeat=n)


def test_probabilities_of_all_texts_sum_to_one():
    rng = np.random.default_rng(0)
    logp = logp_from(rng.random((4, 3)) + 0.05)
    total = sum(math.exp(path_logprob(logp, list(l))) for l in all_labelings(3, 4))
    assert abs(total - 1.0) < 1e-9


def test_beam_search_finds_the_most_probable_text_where_greedy_does_not():
    # Greedy reads the blank in both columns; "a" spread over its alignments is more probable.
    logp = logp_from([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]])
    assert greedy(logp) == []
    best = max(all_labelings(3, 2), key=lambda l: path_logprob(logp, list(l)))
    (top, score), *_ = beam_search(logp, beam=10, prune=-50)
    assert list(top) == list(best) == [1]
    assert abs(score - path_logprob(logp, [1])) < 1e-9


def test_beam_scores_are_exact_when_the_beam_is_wide():
    rng = np.random.default_rng(1)
    logp = logp_from(rng.random((5, 4)) + 0.05)
    for labels, score in beam_search(logp, beam=200, prune=-50)[:10]:
        assert abs(score - path_logprob(logp, list(labels))) < 1e-9
