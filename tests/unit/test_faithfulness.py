"""
Unit tests for src/eval/faithfulness.py — the groundedness measure.

All LLM calls are mocked; no API key or network needed.

What is pinned here is mostly about what the scorer must NOT do. It is an LLM
judging an LLM, so the failure modes are silent by nature: a parser that quietly
returns no claims scores nothing, a default that treats an unreadable verdict as
support inflates exactly the answers the judge found hardest, and a missing
context reported as 0.0 is indistinguishable from a fabrication. Each of those
would produce a plausible number rather than an error.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_faithfulness.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch

import pytest

from src.eval import faithfulness


def _replies(*texts):
    """Patch the module's single call seam to return these texts in order."""
    return patch.object(faithfulness, "_complete", side_effect=list(texts))


class TestDecompose:
    def test_parses_dash_bullets(self):
        with _replies("- Paris is the capital of France.\n- France is in Europe."):
            claims = faithfulness.decompose("Where?", "Paris, in Europe")
        assert claims == ["Paris is the capital of France.", "France is in Europe."]

    def test_accepts_numbered_and_star_bullets(self):
        """
        The prompt asks for "- ", and models drift to "1." or "*" anyway. A
        strict parser would return no claims and silently make the answer
        unscoreable rather than raising.
        """
        with _replies("1. Alpha is true.\n* Beta is true.\n2) Gamma is true."):
            claims = faithfulness.decompose("Q?", "A")
        assert claims == ["Alpha is true.", "Beta is true.", "Gamma is true."]

    def test_strips_markdown_emphasis(self):
        with _replies("- **There are 7 continents.**"):
            assert faithfulness.decompose("Q?", "7") == ["There are 7 continents."]

    def test_single_unbulleted_line_is_kept(self):
        """A model that ignored the format but gave one usable claim beats an
        empty list, which would drop the question from the denominator."""
        with _replies("Deenie is a Judy Blume book."):
            assert faithfulness.decompose("Q?", "Deenie") == [
                "Deenie is a Judy Blume book."]

    def test_empty_answer_makes_no_call(self):
        with patch.object(faithfulness, "_complete") as m:
            assert faithfulness.decompose("Q?", "   ") == []
        m.assert_not_called()


class TestVerify:
    """
    `verify` returns a RECORD, not a bare list, and a verdict may be None.

    It used to coerce an unreadable verdict to False and fold it into the score,
    so an answer the judge failed to rule on came back as a confident 0.0 with
    status "ok" -- "we could not measure this" reported as "the model made it
    up". That is the exact conflation the None-not-0.0 discipline in the rest of
    this module exists to end; it simply had not been applied here.
    """

    def test_reads_verdicts_in_order(self):
        with _replies("1. SUPPORTED\n2. UNSUPPORTED\n3. SUPPORTED"):
            v = faithfulness.verify(["a", "b", "c"], "ctx")
        assert v["verdicts"] == [True, False, True]
        assert v["n_unparsed"] == 0 and v["parse_coverage"] == 1.0
        assert v["repair_used"] is False

    def test_tolerates_commentary_around_the_verdict(self):
        with _replies("1. The context states this - SUPPORTED\n2. UNSUPPORTED"):
            assert faithfulness.verify(["a", "b"], "ctx")["verdicts"] == [True, False]

    def test_an_unreadable_verdict_is_none_after_a_failed_repair(self):
        """None, never False: absent evidence is not evidence of absence."""
        with _replies("1. SUPPORTED\nI could not assess the second claim.",
                      "1. SUPPORTED"):
            v = faithfulness.verify(["a", "b"], "ctx")
        assert v["verdicts"] == [True, None]
        assert v["n_unparsed"] == 1 and v["parse_coverage"] == 0.5
        assert v["repair_used"] is True

    def test_a_format_repair_recovers_a_verdict_that_was_present(self):
        with _replies("Claim one is SUPPORTED and claim two is UNSUPPORTED",
                      "1. SUPPORTED\n2. UNSUPPORTED"):
            v = faithfulness.verify(["a", "b"], "ctx")
        assert v["verdicts"] == [True, False]
        assert v["repair_used"] is True and v["n_unparsed"] == 0

    def test_a_repair_may_not_overturn_a_verdict_already_read(self):
        """Format-only. A second opinion is a new measurement, not a recovery."""
        with _replies("1. SUPPORTED\nnothing for two",
                      "1. UNSUPPORTED\n2. SUPPORTED"):
            v = faithfulness.verify(["a", "b"], "ctx")
        assert v["verdicts"] == [True, True]     # claim 1 keeps its ORIGINAL verdict

    def test_a_conflicting_duplicate_index_is_none_not_last_wins(self):
        with _replies("1. SUPPORTED\n1. UNSUPPORTED\n2. SUPPORTED", "2. SUPPORTED"):
            v = faithfulness.verify(["a", "b"], "ctx")
        assert v["verdicts"] == [None, True]

    def test_no_claims_makes_no_call(self):
        with patch.object(faithfulness, "_complete") as m:
            v = faithfulness.verify([], "ctx")
        assert v["verdicts"] == [] and v["parse_coverage"] == 1.0
        m.assert_not_called()


class TestScoreMissingness:
    """An unmeasurable answer must never enter the headline mean."""

    def test_unparsed_verdicts_make_the_score_none(self):
        with _replies("- claim one\n- claim two",
                      "1. SUPPORTED\nno verdict for two", "1. SUPPORTED"):
            out = faithfulness.score("q", "a", "ctx")
        assert out["score"] is None
        assert out["status"] == "unparsed_verdicts"
        assert out["n_unparsed"] == 1
        # Kept for diagnosis, never aggregated: scoring the parsed subset would
        # change the denominator per answer and bias towards legible replies.
        assert out["partial_score"] == 1.0

    def test_n_supported_counts_only_true(self):
        with _replies("- claim one\n- claim two",
                      "1. SUPPORTED\nnothing", "nothing either"):
            out = faithfulness.score("q", "a", "ctx")
        assert out["n_supported"] == 1 and out["verdicts"] == [True, None]

    def test_a_fully_parsed_answer_scores_normally(self):
        with _replies("- claim one\n- claim two", "1. SUPPORTED\n2. UNSUPPORTED"):
            out = faithfulness.score("q", "a", "ctx")
        assert out["status"] == "ok" and out["score"] == 0.5
        assert out["parse_coverage"] == 1.0

class TestScore:
    def test_supported_fraction(self):
        with _replies("- a\n- b\n- c\n- d", "1. SUPPORTED\n2. SUPPORTED\n"
                                            "3. UNSUPPORTED\n4. UNSUPPORTED"):
            out = faithfulness.score("Q?", "A", "context")
        assert out["score"] == 0.5
        assert (out["n_claims"], out["n_supported"]) == (4, 2)
        assert out["status"] == "ok"

    @pytest.mark.parametrize("answer,context,status", [
        ("an answer", "", "no_context"),
        ("", "a context", "no_answer"),
    ])
    def test_unmeasurable_cases_score_None_not_zero(self, answer, context, status):
        """
        None, never 0.0. "We could not measure this" and "the model made it up"
        are different findings, and averaging a missing context in as 0.0 would
        report a config as unfaithful for having retrieved nothing — which is a
        coverage failure, not a groundedness one.
        """
        out = faithfulness.score("Q?", answer, context)
        assert out["score"] is None
        assert out["status"] == status

    def test_no_claims_is_unscored_rather_than_zero(self):
        with _replies("(the answer states nothing checkable)\nsecond line"):
            out = faithfulness.score("Q?", "hmm", "context")
        assert out["score"] is None and out["status"] == "no_claims"


class TestScopeInvariants:
    def test_base_llm_is_excluded_by_construction(self):
        """
        Configuration 1 retrieves nothing, so faithfulness is undefined for it.
        It must never appear in the scored set: a 0.0 there would read as
        "unfaithful" rather than "not applicable", and the headline comparison
        against the baseline would look like a groundedness result when it
        cannot be one.
        """
        assert "base_llm_abstain" not in faithfulness.RETRIEVAL_CONFIGS
        assert set(faithfulness.RETRIEVAL_CONFIGS) == {"rag", "graph_rag", "rerank"}

    def test_scoring_budget_is_independent_of_the_experiment_cap(self):
        """
        llm_config.MAX_TOKENS is 128 because Mintaka ANSWERS are short, and it
        is held identical across configs so it cannot confound them. A scoring
        call must emit a claim list, and inheriting 128 would truncate the
        judgement — the same defect the condensing call shipped with, where a
        budget chosen for answers was silently reused for prose.
        """
        from src import llm_config
        assert faithfulness.DECOMPOSE_MAX_TOKENS > llm_config.MAX_TOKENS
        assert faithfulness.VERIFY_MAX_TOKENS > llm_config.MAX_TOKENS

    def test_judging_does_not_inherit_the_brevity_instruction(self):
        """
        The answer-role framing appends "Answer with the fact only - no
        explanation", which would sabotage a call whose whole output is a list.
        The condense role carries the reasoning-mode switch without it.
        """
        from src import llm_config
        answer_style = llm_config.prompt_style(
            "nvidia/llama-3.3-nemotron-super-49b-v1", role="answer")
        condense_style = llm_config.prompt_style(
            "nvidia/llama-3.3-nemotron-super-49b-v1", role="condense")
        assert "fact only" in answer_style["preamble"]
        assert "fact only" not in condense_style["preamble"]
        assert condense_style["system"] == "detailed thinking off"


class TestContractInstrumentation:
    """
    Out-of-range verdict indices: counted, stamped, and NEVER scored.

    `_parse_verdicts` silently ignores an index outside 1..n, so an
    out-of-range line marks a degenerate judge generation, not a wrong
    verdict. The instrumentation must count those (distinct indices, per
    stage) without changing a single verdict — the frozen stored files were
    scored by exactly the ignore-them behaviour.
    """

    def test_out_of_range_indices_are_distinct_and_sorted(self):
        raw = "77. SUPPORTED\n0. SUPPORTED\n77. UNSUPPORTED\n1. SUPPORTED"
        assert faithfulness.out_of_range_indices(raw, 1) == [0, 77]

    def test_in_range_indices_are_not_flagged(self):
        assert faithfulness.out_of_range_indices(
            "1. SUPPORTED\n2. UNSUPPORTED", 2) == []

    def test_none_or_empty_text_is_empty(self):
        assert faithfulness.out_of_range_indices(None, 3) == []
        assert faithfulness.out_of_range_indices("", 3) == []

    def test_verify_counts_but_does_not_score_out_of_range_lines(self):
        with _replies("1. SUPPORTED\n77. UNSUPPORTED"):
            v = faithfulness.verify(["a"], "ctx")
        assert v["verdicts"] == [True], "index 77 must never enter the verdicts"
        assert v["n_out_of_range_initial"] == 1
        assert v["n_out_of_range_repair"] is None, "no repair call was made"

    def test_verify_counts_repair_stage_separately(self):
        # Initial reply leaves claim 2 unparsed -> repair runs; the repair text
        # carries its own out-of-range line.
        with _replies("1. SUPPORTED\nno verdict for two",
                      "1. SUPPORTED\n2. UNSUPPORTED\n9. SUPPORTED"):
            v = faithfulness.verify(["a", "b"], "ctx")
        assert v["verdicts"] == [True, False]
        assert v["n_out_of_range_initial"] == 0
        assert v["n_out_of_range_repair"] == 1

    def test_score_stamps_the_ok_path(self):
        with _replies("- Alpha is true.", "1. SUPPORTED"):
            r = faithfulness.score("Q?", "A", "ctx")
        assert r["status"] == "ok"
        assert r["instrument_version"] == faithfulness.INSTRUMENT_VERSION
        assert r["n_out_of_range_initial"] == 0
        assert r["n_out_of_range_repair"] is None

    def test_score_stamps_every_blank_path(self):
        # no_context / no_answer / no_claims all skip verify(): the counters
        # are None ("not measured"), never 0, and the stamp is still present.
        cases = [
            ("Q?", "A", "   "),          # no_context
            ("Q?", "   ", "ctx"),        # no_answer
        ]
        for q, a, ctx in cases:
            r = faithfulness.score(q, a, ctx)
            assert r["score"] is None
            assert r["instrument_version"] == faithfulness.INSTRUMENT_VERSION
            assert r["n_out_of_range_initial"] is None
            assert r["n_out_of_range_repair"] is None
        with _replies("no bullets here\nat all\nreally"):
            r = faithfulness.score("Q?", "A", "ctx")
        assert r["status"] == "no_claims"
        assert r["instrument_version"] == faithfulness.INSTRUMENT_VERSION
