"""
Unit tests for the paired bootstrap (src/eval/stats.py).

These pin the properties a reported confidence interval has to have: it must be
reproducible, it must be centred on the observed effect, it must not claim
significance for an effect that is not there, and it must not silently compare
mismatched question sets.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.eval.stats import (
    compare_configs, correct_key, f1_key, hallucination_key, mcnemar_exact,
    paired_bootstrap, unpaired_bootstrap,
)


def _rec(f1=0.0, outcome="correct", score=None):
    return {"f1": f1, "score": f1 if score is None else score, "outcome": outcome,
            "abstained": outcome == "abstention", "em": 0.0}


class TestPairedBootstrap:
    def test_empty_input_is_not_significant(self):
        r = paired_bootstrap([])
        assert r["n"] == 0 and r["significant"] is False and r["p_value"] == 1.0

    def test_delta_is_the_observed_mean(self):
        r = paired_bootstrap([1.0, 2.0, 3.0], n_resamples=500)
        assert r["delta"] == pytest.approx(2.0)

    def test_reproducible_across_calls(self):
        # A CI that moves every run is not a number a reader can check.
        a = paired_bootstrap([0.4, -0.1, 0.9, 0.2] * 10, n_resamples=1000)
        b = paired_bootstrap([0.4, -0.1, 0.9, 0.2] * 10, n_resamples=1000)
        assert (a["ci_low"], a["ci_high"], a["p_value"]) == \
               (b["ci_low"], b["ci_high"], b["p_value"])

    def test_seed_changes_the_interval(self):
        a = paired_bootstrap([0.4, -0.3, 0.9, -0.2] * 5, n_resamples=1000, seed=0)
        b = paired_bootstrap([0.4, -0.3, 0.9, -0.2] * 5, n_resamples=1000, seed=99)
        assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])

    def test_ci_brackets_the_observed_delta(self):
        r = paired_bootstrap([0.3, 0.5, 0.4, 0.6, 0.5] * 8, n_resamples=2000)
        assert r["ci_low"] <= r["delta"] <= r["ci_high"]

    def test_large_consistent_effect_is_significant(self):
        r = paired_bootstrap([0.5] * 60, n_resamples=2000)
        assert r["significant"] is True
        assert r["p_value"] < 0.05
        assert r["ci_low"] > 0

    def test_no_effect_is_not_significant(self):
        # Symmetric around zero: the honest verdict is "no difference shown".
        r = paired_bootstrap([1.0, -1.0] * 40, n_resamples=2000)
        assert r["significant"] is False
        assert r["ci_low"] < 0 < r["ci_high"]

    def test_negative_effect_reported_as_negative(self):
        r = paired_bootstrap([-0.4] * 50, n_resamples=1000)
        assert r["delta"] < 0 and r["ci_high"] < 0 and r["significant"] is True

    def test_noisy_small_sample_is_not_significant(self):
        # Three questions cannot establish anything; the CI must say so rather
        # than ranking cells on n=3.
        r = paired_bootstrap([0.9, -0.8, 0.1], n_resamples=2000)
        assert r["significant"] is False


class TestCompareConfigs:
    def test_matches_on_question_id(self):
        a = {"q1": _rec(1.0), "q2": _rec(1.0)}
        b = {"q1": _rec(0.0), "q2": _rec(0.0)}
        r = compare_configs(a, b, f1_key, n_resamples=500)
        assert r["n"] == 2 and r["delta"] == pytest.approx(1.0)

    def test_non_overlapping_questions_are_reported_not_hidden(self):
        a = {"q1": _rec(1.0), "q2": _rec(1.0), "q3": _rec(1.0)}
        b = {"q1": _rec(0.0)}
        r = compare_configs(a, b, f1_key, n_resamples=200)
        assert r["n"] == 1
        assert r["n_dropped"] == 2, "a silently shrinking comparison is a trap"

    def test_result_does_not_depend_on_dict_insertion_order(self):
        pairs = [("q1", 0.2), ("q2", 0.9), ("q3", 0.5), ("q4", 0.1)]
        a1 = {q: _rec(v) for q, v in pairs}
        a2 = {q: _rec(v) for q, v in reversed(pairs)}
        b = {q: _rec(0.3) for q, _ in pairs}
        r1 = compare_configs(a1, b, f1_key, n_resamples=1000)
        r2 = compare_configs(a2, b, f1_key, n_resamples=1000)
        assert (r1["ci_low"], r1["ci_high"]) == (r2["ci_low"], r2["ci_high"])

    def test_hallucination_key_counts_only_wrong_answers(self):
        a = {"q1": _rec(outcome="hallucination"), "q2": _rec(outcome="abstention"),
             "q3": _rec(outcome="correct")}
        b = {q: _rec(outcome="correct") for q in ("q1", "q2", "q3")}
        r = compare_configs(a, b, hallucination_key, n_resamples=500)
        # One of three questions is a confident error on the a-side, none on b.
        assert r["delta"] == pytest.approx(1 / 3)

    def test_abstention_does_not_count_as_hallucination(self):
        # The whole point of the measure: declining is not the same as erring.
        a = {"q1": _rec(outcome="abstention"), "q2": _rec(outcome="abstention")}
        b = {"q1": _rec(outcome="correct"), "q2": _rec(outcome="correct")}
        r = compare_configs(a, b, hallucination_key, n_resamples=500)
        assert r["delta"] == 0.0

    def test_correct_key_tracks_the_correct_bucket(self):
        a = {"q1": _rec(outcome="correct"), "q2": _rec(outcome="hallucination")}
        b = {"q1": _rec(outcome="hallucination"), "q2": _rec(outcome="hallucination")}
        r = compare_configs(a, b, correct_key, n_resamples=500)
        assert r["delta"] == pytest.approx(0.5)


class TestMcnemarExact:
    """The exact binomial sensitivity check for binary paired outcomes."""

    def _maps(self, pairs):
        # pairs: list of (a_event, b_event) per question, as 0/1.
        a = {f"q{i}": _rec(outcome="hallucination" if x else "correct")
             for i, (x, _) in enumerate(pairs)}
        b = {f"q{i}": _rec(outcome="hallucination" if y else "correct")
             for i, (_, y) in enumerate(pairs)}
        return a, b

    def test_known_small_case(self):
        # n10=1, n01=5: p = 2 * (C(6,0)+C(6,1)) / 2^6 = 14/64 = 0.21875 exactly.
        pairs = [(1, 0)] + [(0, 1)] * 5 + [(0, 0)] * 4
        a, b = self._maps(pairs)
        r = mcnemar_exact(a, b, hallucination_key)
        assert (r["n10"], r["n01"]) == (1, 5)
        assert r["p_value"] == pytest.approx(0.21875)

    def test_balanced_discordance_clamps_to_one(self):
        a, b = self._maps([(1, 0)] * 3 + [(0, 1)] * 3)
        assert mcnemar_exact(a, b, hallucination_key)["p_value"] == 1.0

    def test_no_discordant_pairs_is_p_one(self):
        a, b = self._maps([(1, 1), (0, 0), (1, 1)])
        r = mcnemar_exact(a, b, hallucination_key)
        assert r["p_value"] == 1.0 and r["n01"] == r["n10"] == 0

    def test_orientation_and_delta_sign(self):
        # A has the event where B does not -> n10, and delta is positive
        # (A minus B), matching compare_configs on the same key.
        a, b = self._maps([(1, 0), (1, 0), (0, 0), (0, 0)])
        r = mcnemar_exact(a, b, hallucination_key)
        assert (r["n10"], r["n01"]) == (2, 0)
        assert r["delta"] == pytest.approx(0.5)

    def test_large_discordant_counts_do_not_overflow(self):
        # 1,500 discordant pairs: comb(1500, 750) has ~450 digits, so a float
        # `comb * 0.5**n` overflows to inf*0.0 = nan. The integer-sum path
        # must return a sane probability (normal approx puts it near .01).
        pairs = [(1, 0)] * 800 + [(0, 1)] * 700 + [(0, 0)] * 100
        a, b = self._maps(pairs)
        r = mcnemar_exact(a, b, hallucination_key)
        assert 0.005 <= r["p_value"] <= 0.02

    def test_non_binary_key_raises(self):
        a = {"q1": _rec(f1=0.5)}
        b = {"q1": _rec(f1=0.5)}
        with pytest.raises(ValueError):
            mcnemar_exact(a, b, f1_key)

    def test_bool_key_is_accepted(self):
        a = {"q1": _rec(outcome="hallucination")}
        b = {"q1": _rec(outcome="correct")}
        r = mcnemar_exact(a, b, lambda s: s["outcome"] == "hallucination")
        assert (r["n10"], r["n01"]) == (1, 0)


class TestUnpairedBootstrap:
    """Two-sample CI for disjoint groups (correct-vs-wrong faithfulness)."""

    def test_empty_group_is_null_result_not_crash(self):
        r = unpaired_bootstrap([], [0.5, 0.6])
        assert r["p_value"] == 1.0 and r["significant"] is False

    def test_delta_is_difference_of_group_means(self):
        r = unpaired_bootstrap([1.0, 0.0], [0.5, 0.5], n_resamples=200)
        assert r["delta"] == pytest.approx(0.0)

    def test_reproducible_across_calls(self):
        xs, ys = [0.9, 0.7, 0.8] * 10, [0.4, 0.5, 0.3] * 10
        a = unpaired_bootstrap(xs, ys, n_resamples=1000)
        b = unpaired_bootstrap(xs, ys, n_resamples=1000)
        assert (a["ci_low"], a["ci_high"], a["p_value"]) == \
               (b["ci_low"], b["ci_high"], b["p_value"])

    def test_clear_separation_is_significant(self):
        r = unpaired_bootstrap([0.9] * 30, [0.1] * 30, n_resamples=1000)
        assert r["significant"] is True and r["ci_low"] > 0

    def test_full_overlap_is_not_significant(self):
        r = unpaired_bootstrap([0.4, 0.6] * 20, [0.5, 0.5] * 20,
                               n_resamples=1000)
        assert r["significant"] is False
