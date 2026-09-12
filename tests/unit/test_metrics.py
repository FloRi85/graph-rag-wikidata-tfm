"""Unit tests for the type-aware scorer (numeric + date fixes).

No network / API. Run:
    PYTHONPATH=. venv/Scripts/python -m pytest tests/unit/test_metrics.py -v
"""

import pytest

from src.eval import metrics
from src.eval.parse_answers import build_gold_answer
from src.prompts import ABSTAIN_SENTINEL

from src.eval.metrics import (
    numeric_match, date_match, score_answer, is_abstention, set_match,
    score_results, failed_configs, present_configs, context_words, CONFIGS,
    gold_forms, parsed_gold_forms, entity_match, rival_candidates,
    question_entity_labels,
    token_f1, per_question_scores, is_attempted, ATTEMPTED_OUTCOMES,
    CORRECT, HALLUCINATION, ABSTENTION, OTHER,
    _parse_numbers, _extract_numbers, _extract_years, _extract_full_dates,
    _token_span_index, _parse_boolean, boolean_match,
)


# --------------------------------------------------------------------------
# numeric_match — spelled-out numbers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold", [
    ("America has had one Civil War", "1"),          # word, no digit anywhere
    ("Three 6 Mafia has won one Academy Award", "1"),  # word answer, digit decoy in name
    ("Alabama has 7 US House seats.", "7"),           # plain digit
    ("Columbus set sail with three ships on August 3, 1492", "3"),  # word + decoy dates
    ("They released twenty-three albums", "23"),      # hyphenated tens+unit
    ("about one million copies", "1000000"),          # scale word
    ("won the fifth title", "5"),                     # ordinal
    ("1,000 people attended", "1000"),                # comma-grouped digits
])
def test_numeric_match_correct(pred, gold):
    assert numeric_match(pred, gold) == 1.0


@pytest.mark.parametrize("pred,gold", [
    ("Ivan Lendl did not win 6 Grand Slam finals.", "7"),  # genuinely wrong
    ("America has had two Civil Wars", "1"),               # wrong word
    ("in 2001 he was born", "1"),                          # year is not the answer
    ("I cannot find the number", "3"),                     # no number
])
def test_numeric_match_wrong(pred, gold):
    assert numeric_match(pred, gold) == 0.0


def test_numeric_match_empty_gold():
    assert numeric_match("three", "") == 0.0


# --------------------------------------------------------------------------
# _parse_boolean — module-level so tools share the scorer's yes/no vocabulary
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Yes, because the treaty was signed in 1848.", True),   # first-word wins
    ("No, he never played for them.", False),
    ("The claim is false.", False),                          # fallback scan
    ("That is correct.", True),
    ("Paris is the capital of France.", None),               # no boolean asserted
    ("", None),
])
def test_parse_boolean(text, expected):
    assert _parse_boolean(text) is expected


def test_boolean_match_uses_parse_boolean():
    """The hoist is behaviour-identical: boolean_match still reads the same forms."""
    assert boolean_match("Yes, absolutely", "yes") == 1.0
    assert boolean_match("No, it does not", "yes") == 0.0
    assert boolean_match("cannot say", "yes") == 0.0


def test_parse_numbers_words():
    assert _parse_numbers("one hundred and five") == [105.0]
    assert _parse_numbers("one hundred twenty three") == [123.0]
    assert _parse_numbers("twenty three") == [23.0]
    assert _parse_numbers("twenty-three") == [23.0]
    assert _parse_numbers("two hundred thousand") == [200000.0]
    assert _parse_numbers("won three and lost twenty two") == [3.0, 22.0]
    assert _parse_numbers("no numbers here") == []


def test_extract_numbers_mixed():
    assert _extract_numbers("3 ships and three more") == {3.0}
    assert _extract_numbers("won 5 in 2017") == {5.0, 2017.0}


# --------------------------------------------------------------------------
# Digit-plus-scale composition.
#
# Regression suite for the 2026-08-13 grammar rewrite. The old extractor
# unioned a digit regex with a word scanner, so a phrase emitted its COMPONENTS
# rather than its value: "$1 million" -> {1.0, 1000000.0} matched gold
# "$2 million" (they collide on the stray 10^6) while "2000000" did not.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # Composition: only the composed value, never the components.
    ("1.5 million",                       [1500000.0]),
    ("$1 million",                        [1000000.0]),
    ("3.967 million",                     [3967000.0]),
    ("8.8 million",                       [8800000.0]),
    ("2.873 million",                     [2873000.0]),
    ("325 million years",                 [325000000.0]),
    # Descending scales combine into one quantity.
    ("2 million 300 thousand",            [2300000.0]),
    ("2 million and 300 thousand",        [2300000.0]),
    # Equal or rising scales are two quantities, NOT one sum.
    ("2 million and 3 million",           [2000000.0, 3000000.0]),
    ("2 thousand and 3 million",          [2000.0, 3000000.0]),
    # A range names both endpoints.
    ("260 million to 325 million",        [260000000.0, 325000000.0]),
    # Two bare quantities never sum.
    ("Three 6 Mafia",                     [3.0, 6.0]),
    ("three six",                         [3.0, 6.0]),
    ("nineteen eighty four",              [19.0, 84.0]),
    ("20 3 times",                        [20.0, 3.0]),
    # Punctuation in the gap breaks the phrase; a hyphen does not.
    ("2 million, 3 thousand",             [2000000.0, 3000.0]),
    # ⚠️ LOAD-BEARING for the verbose configs: a SENTENCE must not compose.
    ("the population is 2 million and the area is 300 thousand",
                                          [2000000.0, 300000.0]),
    # Unit-conversion answers keep every figure separate.
    ("110 short tons (100 t; 98 long tons)", [110.0, 100.0, 98.0]),
    ("-5 degrees",                        [-5.0]),
    ("1,000",                             [1000.0]),
])
def test_parse_numbers_scale_phrases(text, expected):
    assert _parse_numbers(text) == expected


@pytest.mark.parametrize("pred,gold,expected", [
    ("$1 million",   "$2 million", 0.0),   # was 1.0 — collided on a stray 10^6
    ("2000000",      "$2 million", 1.0),   # was 0.0 — gold never composed
    ("$2 million",   "$2 million", 1.0),
    ("1.5 million",  "2 million",  0.0),
    ("3.967 million", "3967000",   1.0),   # both gold surface forms now agree
    ("8.8 million",  "8800000",    1.0),
])
def test_numeric_match_digit_plus_scale(pred, gold, expected):
    assert numeric_match(pred, gold) == expected


# --------------------------------------------------------------------------
# date_match
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold", [
    ("released in 2004", "2004"),                          # year-only gold, year in pred
    ("The College Dropout came out February 10, 2004", "2004"),  # prose date, year gold
    ("born on 1970-08-02", "1970-08-02"),                 # exact ISO
    ("born on August 2, 1970", "1970-08-02"),             # prose == full gold
    ("born 2 August 1970", "1970-08-02"),                 # D Month YYYY
    ("it was 1970", "1970-08-02"),                         # year-only pred, full gold -> lenient
])
def test_date_match_correct(pred, gold):
    assert date_match(pred, gold) == 1.0


@pytest.mark.parametrize("pred,gold", [
    ("released in 2003", "2004"),                          # wrong year
    ("born on August 5, 1970", "1970-08-02"),             # right year, wrong day committed
    ("no date given", "2004"),
])
def test_date_match_wrong(pred, gold):
    assert date_match(pred, gold) == 0.0


def test_extract_years_and_dates():
    assert _extract_years("in 1970 and 2004") == {1970, 2004}
    assert (1970, 8, 2) in _extract_full_dates("August 2, 1970")
    assert (1970, 8, 2) in _extract_full_dates("1970-08-02")


# --------------------------------------------------------------------------
# dispatch through score_answer
# --------------------------------------------------------------------------

def test_score_answer_numeric_dispatch():
    r = score_answer("America has had one Civil War", "1", "numerical")
    assert r["score"] == 1.0 and not r["abstained"]


def test_score_answer_date_dispatch():
    r = score_answer("released in 2004", "2004", "date")
    assert r["score"] == 1.0


def test_score_answer_multi_gold_numeric():
    # best over gold forms
    assert score_answer("three", ["3", "III"], "numerical")["score"] == 1.0


# --- date golds: Mintaka's mention is often an unparseable abbreviation -------
# Scoring dates against `expected` alone graded correct answers as
# hallucinations on ~22% (test) / ~29% (dev) of date questions, and the error
# fell entirely on the base LLM and C2 — i.e. it inflated the hallucination rate of
# exactly the configs the thesis argues against.

def test_date_mention_abbreviation_alone_fails():
    """The bug, pinned: the mention on its own cannot score a correct answer."""
    assert score_answer("Madden 18 was released on August 25, 2017.",
                        ["25-Aug-17"], "date")["score"] < 0.5


def test_gold_forms_adds_answer_value_for_dates():
    row = {"id": "q1", "expected": "25-Aug-17", "answer_type": "date",
           "answer_value": "2017-08-25"}
    forms, _ = gold_forms(row)
    assert "2017-08-25" in forms and "25-Aug-17" in forms
    assert score_answer("Madden 18 was released on August 25, 2017.",
                        forms, "date")["score"] == 1.0


def test_gold_forms_recovers_answer_value_from_questions_file():
    """Result files predating the field must re-score without a re-run."""
    row = {"id": "q1", "expected": "25-Aug-17", "answer_type": "date"}
    qbi = {"q1": {"answer_type": "date", "answer_value": "2017-08-25"}}
    forms, _ = gold_forms(row, qbi)
    assert "2017-08-25" in forms


def test_gold_forms_skips_answer_value_for_numerical():
    """Mention is the BETTER gold here: '2 million' must not have to match
    the raw literal 2000000."""
    row = {"id": "q1", "expected": "$2 million", "answer_type": "numerical",
           "answer_value": 2000000}
    forms, _ = gold_forms(row)
    assert "2000000" not in forms
    assert score_answer("about 2 million", forms, "numerical")["score"] == 1.0


class TestParsedGoldForms:
    """`parsed_gold_forms` — gold from a GoldAnswer instead of a result row.

    The historical migration comparison found no score movement on the stored
    runs. These tests pin the properties that argument
    rests on, so a future edit to either definition breaks a test rather than a
    published number.
    """

    def test_forms_and_members_come_from_the_gold_answer(self):
        g = build_gold_answer({
            "answerType": "entity", "mention": "The Rock",
            "answer": [{"name": "Q10738", "label": {"en": "Dwayne Johnson"}}],
        })
        forms, members = parsed_gold_forms(g)
        assert forms == ["Dwayne Johnson", "The Rock"]
        # One label is a surface variant, not a set: best-of-forms, not set F1.
        assert members == []

    def test_two_distinct_labels_are_set_members(self):
        g = build_gold_answer({
            "answerType": "entity", "mention": "Super Bowl XL, Super Bowl XLIX",
            "answer": [{"name": "Q1", "label": {"en": "Super Bowl XL"}},
                       {"name": "Q2", "label": {"en": "Super Bowl XLIX"}}],
        })
        forms, members = parsed_gold_forms(g)
        assert members == ["Super Bowl XL", "Super Bowl XLIX"]
        assert score_answer("Super Bowl XL", forms, "entity", members)["set_answer"]

    def test_empty_gold_never_yields_an_empty_form_list(self):
        """`score_answer` iterates the forms; [] would make max() raise."""
        g = build_gold_answer({"answerType": "string", "mention": "", "answer": []})
        forms, _ = parsed_gold_forms(g)
        assert forms == [""]

    def test_numerical_keeps_the_typed_payload_beside_the_mention(self):
        """The one substantive difference from the legacy definition.

        Legacy withheld the payload on numerical answers, so a prose mention was
        the only gold and a model answering with the bare number scored 0. 69 of
        4,000 TEST questions are shaped this way.
        """
        answer = {"answerType": "numerical", "mention": "166 pounds", "answer": [166]}
        forms, _ = parsed_gold_forms(build_gold_answer(answer))
        assert forms == ["166", "166 pounds"]

        legacy, _ = gold_forms({"id": "q", "expected": "166 pounds",
                                "answer_type": "numerical", "answer_value": 166})
        assert "166" not in legacy

    def test_order_cannot_change_a_score(self):
        """Canonical label leads here and the mention led before; every matcher
        takes the max over forms, so the reordering is inert by construction."""
        forms = ["Dwayne Johnson", "The Rock"]
        assert (score_answer("The Rock", forms, "entity")["score"]
                == score_answer("The Rock", forms[::-1], "entity")["score"])


class TestScoringWithSuppliedGolds:

    ROWS = [{
        "id": "q1", "expected": "wrong-gold-frozen-into-the-row",
        "answer_type": "entity",
        "answers": {"base_llm_abstain": "42"},
    }]

    def _gold(self):
        return {"q1": build_gold_answer(
            {"answerType": "numerical", "mention": "42", "answer": [42]})}

    def test_supplied_gold_overrides_the_row(self):
        scored = per_question_scores(self.ROWS, golds=self._gold())
        rec = scored["base_llm_abstain"]["q1"]
        assert rec["raw_score"] == 1.0
        # answer_type comes from the gold too, so the matcher and the strings it
        # dispatches on cannot come from different places.
        assert rec["answer_type"] == "numerical"

    def test_row_gold_is_used_when_no_golds_supplied(self):
        rec = per_question_scores(self.ROWS)["base_llm_abstain"]["q1"]
        assert rec["raw_score"] < 1.0

    def test_a_missing_id_raises_rather_than_falling_back(self):
        """Half a run under one gold definition and half under another is two
        experiments in one table, not a graceful degradation."""
        with pytest.raises(KeyError, match="no gold answer"):
            per_question_scores(self.ROWS, golds={})


def test_score_answer_abstention_unaffected():
    r = score_answer("I cannot answer from the context.", "3", "numerical")
    assert r["abstained"] is True and r["score"] == 0.0


# --------------------------------------------------------------------------
# is_abstention — negation with intervening words
#
# The adjacent-only patterns matched "not stated" but not "not *explicitly*
# stated", so real abstentions were scored as attempted answers. That inflated
# coverage and depressed F1@attempted, i.e. it corrupted the faithfulness
# columns specifically.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "The context does not explicitly state the answer.",
    "It is not possible to determine from the given context.",
    "There is no relevant information in the context.",
    "The provided facts do not allow me to answer.",
    "Unfortunately, the context lacks this information.",
    "I could not find the answer in the context.",
    "The answer is not in the context.",          # adjacent form still works
    "I cannot determine the answer.",
    "There is insufficient information provided.",
    "I am unable to confirm this from the facts.",
    # Contracted negations (2026-08-05). "cannot|can't|could not|couldn't"
    # always covered both forms; the do/is families did not, so "doesn't
    # provide" scored as an ATTEMPTED answer while "does not provide" scored as
    # an abstention. Conversational models write the contraction.
    "The context doesn't provide the answer.",
    "The facts don't mention her birth date.",
    "That isn't stated in the context.",
    "The birth date wasn't provided in the given facts.",
])
def test_is_abstention_detected(answer):
    assert is_abstention(answer) is True


@pytest.mark.parametrize("answer", [
    "Matt Damon",
    "Paris",
    "42",
    "Yes",
    "No",
    "The Matrix was released in 1999.",
    # Negations that are part of a real answer, not a refusal:
    "Barack Obama, who was not the first president, served two terms.",
    "She does not have a Nobel Prize but won an Oscar.",
    # Contracted form of the same real answer: "have" is deliberately absent
    # from the verb list, so adding contractions must not start catching these.
    "She doesn't have a Nobel Prize but won an Oscar.",
])
def test_is_abstention_no_false_positive(answer):
    assert is_abstention(answer) is False


# --------------------------------------------------------------------------
# set_match — multi-entity gold answers
#
# Previously every member was treated as an alternative COMPLETE gold answer
# and the best single match kept, so naming one of three actors scored 1.0.
# --------------------------------------------------------------------------

_ACTORS = ["Matt Damon", "Ben Affleck", "Robin Williams"]


def test_set_partial_credit_not_full():
    """The regression that motivated set scoring: one member != a full answer."""
    r = score_answer("Matt Damon", _ACTORS, "entity", _ACTORS)
    assert r["set_answer"] is True
    assert r["recall"] == pytest.approx(1 / 3)
    assert r["f1"] == pytest.approx(0.5)
    assert r["em"] == 0.0


@pytest.mark.parametrize("pred", [
    "Matt Damon, Ben Affleck and Robin Williams",
    "Robin Williams, Ben Affleck, Matt Damon",              # order-independent
    "The actors are Matt Damon, Ben Affleck, and Robin Williams.",  # prose carrier
    "- Matt Damon\n- Ben Affleck\n- Robin Williams",        # list markers
])
def test_set_complete_answer_scores_one(pred):
    r = score_answer(pred, _ACTORS, "entity", _ACTORS)
    assert r["f1"] == pytest.approx(1.0)
    assert r["em"] == 1.0


def test_set_repeated_member_cannot_satisfy_several_golds():
    """Greedy one-to-one matching: repetition must not fake coverage."""
    r = score_answer("Matt Damon, Matt Damon, Matt Damon", _ACTORS, "entity", _ACTORS)
    assert r["recall"] == pytest.approx(1 / 3)
    assert r["f1"] < 0.5


def test_set_spurious_members_cost_precision():
    r = score_answer("Matt Damon, Tom Hanks, Brad Pitt", _ACTORS, "entity", _ACTORS)
    assert r["precision"] == pytest.approx(1 / 3)
    assert r["recall"] == pytest.approx(1 / 3)


def test_set_all_wrong_scores_zero():
    r = score_answer("Tom Hanks, Brad Pitt", _ACTORS, "entity", _ACTORS)
    assert r["f1"] == 0.0


def test_set_abstention_still_wins():
    r = score_answer("The context does not mention the actors.", _ACTORS, "entity", _ACTORS)
    assert r["abstained"] is True
    assert r["score"] == 0.0


# --------------------------------------------------------------------------
# Single-answer path must be unchanged by set scoring
# --------------------------------------------------------------------------

def test_single_answer_surface_variants_still_max():
    """One entity with label+mention variants is NOT a set."""
    r = score_answer("Grammy Award", ["Grammy", "Grammy Award"], "entity", ["Grammy Award"])
    assert r["set_answer"] is False
    assert r["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("answer_type,pred,gold", [
    ("boolean", "yes", "Yes"),
    ("numerical", "7", "7"),
])
def test_boolean_and_numeric_never_set_scored(answer_type, pred, gold):
    """Even with several gold forms, these types must not go down the set path."""
    r = score_answer(pred, [gold, gold + " indeed"], answer_type, [gold, "decoy"])
    assert r["set_answer"] is False
    assert r["score"] == pytest.approx(1.0)


def test_set_scoring_requires_two_distinct_members():
    """Duplicate labels are one answer, not a two-member set."""
    r = score_answer("Grammy", ["Grammy"], "entity", ["Grammy", "grammy"])
    assert r["set_answer"] is False


# --------------------------------------------------------------------------
# Set-scoring hardening (2026-08-18 audit fixes)
# --------------------------------------------------------------------------

def test_set_duplicate_item_cannot_fuzzy_match_second_member():
    """"Iron Man, Iron Man" must not take exact 1.0 vs {Iron Man, Iron Man 2}.

    The second copy used to fuzzy-match "Iron Man 2" at F1 0.8, faking full
    coverage. A duplicate item can never match, but still costs precision.
    """
    r = set_match("Iron Man, Iron Man", ["Iron Man", "Iron Man 2"])
    assert r["exact"] == 0.0
    assert r["recall"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(0.5)


def test_set_member_containing_delimiter_matches_as_whole_span():
    """A member whose NAME contains a delimiter must not be shredded."""
    members = ["Bosnia and Herzegovina", "France"]
    r = set_match("Bosnia and Herzegovina, France", members)
    assert r["exact"] == 1.0
    assert r["f1"] == pytest.approx(1.0)


def test_set_ampersand_member_whole_answer_exact():
    members = ["Law & Order", "Homicide"]
    r = set_match("Law & Order and Homicide", members)
    assert r["exact"] == 1.0


def test_set_delimiter_member_fragment_accepted_as_partial_name():
    """"Bosnia" still matches "Bosnia and Herzegovina" via the 0.5 threshold.

    Partial-name acceptance ("Damon" -> "Matt Damon") is the documented design
    of _SET_MATCH_THRESHOLD; the span-protection fix must not remove it.
    """
    members = ["Bosnia and Herzegovina", "France"]
    r = set_match("Bosnia, France", members)
    assert r["recall"] == pytest.approx(1.0)


def test_set_repeated_span_member_counts_once():
    members = ["Law & Order", "Homicide"]
    r = set_match("Law & Order, Law & Order", members)
    assert r["recall"] == pytest.approx(0.5)
    assert r["exact"] == 0.0


# --------------------------------------------------------------------------
# numeric_match subset rule (2026-08-18): a multi-number gold needs ALL its
# numbers present, so 5'10" no longer matches gold 5'7" on the stray 5.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,expected", [
    ("5'10\"", "5'7\"", 0.0),                      # was 1.0: shared 5
    ("5'7\"", "5'7\"", 1.0),
    ("5 feet 7 inches", "5'7\"", 1.0),
    ("He is 5'7\" tall.", "5'7\"", 1.0),           # carrier prose
    ("7", "5'7\"", 0.0),                           # one of two gold numbers
    ("166", "166 pounds", 1.0),                    # single-number gold unchanged
])
def test_numeric_match_multi_number_gold_requires_all(pred, gold, expected):
    assert numeric_match(pred, gold) == expected


# --------------------------------------------------------------------------
# Infrastructure failures must be EXCLUDED, not graded
#
# API failures used to be stored as "ERROR: ..." in the answer string and then
# scored as attempted wrong answers. The 2026-07-25 sweep carried 9 such
# failures, all in the canonical k10_cap30 cell, depressing it and inflating
# every other cell's apparent gain.
# --------------------------------------------------------------------------

def _row(qid, answers, errors=None):
    r = {"id": qid, "expected": "Paris", "answer_entities": ["Paris"],
         "answer_type": "entity", "complexity": "generic", "answers": answers}
    if errors:
        r["errors"] = errors
    return r


def test_failed_configs_reads_errors_field():
    row = _row("q1", {"base_llm_abstain": "Paris"}, {"graph_rag": {"type": "APIError"}})
    assert failed_configs(row) == {"graph_rag"}


def test_failed_configs_reads_legacy_error_string():
    """Older result files encode the failure in the answer string."""
    row = _row("q1", {"base_llm_abstain": "Paris", "rag": "ERROR: Error code: 500"})
    assert failed_configs(row) == {"rag"}


def test_failed_configs_clean_row():
    assert failed_configs(_row("q1", {"base_llm_abstain": "Paris"})) == set()


def test_errored_question_excluded_from_every_config():
    """Matched question sets: one config's failure drops the question for all."""
    rows = [
        _row("good", {c: "Paris" for c in CONFIGS}),
        _row("bad", {c: "Paris" for c in CONFIGS}, {"graph_rag": {"type": "APIError"}}),
    ]
    sc = score_results(rows)
    for c in CONFIGS:
        assert sc["per_config"][c]["n"] == 1, f"{c} should have scored only the clean question"
    assert sc["errors"]["n_questions_excluded"] == 1
    assert sc["errors"]["excluded_ids"] == ["bad"]
    assert sc["errors"]["n_failures_by_config"] == {"graph_rag": 1}


def test_failure_is_not_scored_as_wrong_answer():
    """The regression: a failure must not depress F1 the way a wrong answer does."""
    clean = [_row("q1", {c: "Paris" for c in CONFIGS})]
    withfail = clean + [_row("q2", {c: "ERROR: Error code: 503" for c in CONFIGS})]
    assert (score_results(clean)["per_config"]["graph_rag"]["f1"]
            == score_results(withfail)["per_config"]["graph_rag"]["f1"] == 1.0)


def test_no_errors_reports_zero_excluded():
    sc = score_results([_row("q1", {c: "Paris" for c in CONFIGS})])
    assert sc["errors"]["n_questions_excluded"] == 0
    assert sc["errors"]["n_failures_by_config"] == {}


# --------------------------------------------------------------------------
# present_configs — result files predating a config must not gain a phantom row
# --------------------------------------------------------------------------

def test_present_configs_only_what_is_in_the_data():
    rows = [_row("q1", {"base_llm_abstain": "Paris", "rag": "Paris"})]
    assert present_configs(rows) == ["base_llm_abstain", "rag"]


def test_present_configs_preserves_canonical_order():
    rows = [_row("q1", {"rerank": "Paris", "base_llm_abstain": "Paris", "graph_rag": "Paris"})]
    assert present_configs(rows) == ["base_llm_abstain", "graph_rag", "rerank"]


def test_sweep_file_gets_no_phantom_baseline_row():
    """A retrieval-only sweep cell must not report C1 as an all-zero config."""
    rows = [_row("q1", {c: "Paris" for c in ("graph_rag", "rerank")})]
    assert "base_llm_abstain" not in score_results(rows)["per_config"]


def test_dropped_no_abstention_variant_is_ignored():
    """
    Pre-2026-08-04 files carry a `base_llm` column from the no-abstention variant
    that was dropped when C1b became C1. It must not be scored as the baseline.
    """
    rows = [_row("q1", {"base_llm": "Berlin", "base_llm_abstain": "Paris"})]
    assert present_configs(rows) == ["base_llm_abstain"]
    assert "base_llm" not in score_results(rows)["per_config"]


def test_config_known_only_via_errors_still_counted():
    rows = [_row("q1", {"base_llm_abstain": "Paris"}, {"rag": {"type": "APIError"}})]
    assert present_configs(rows) == ["base_llm_abstain", "rag"]


# --------------------------------------------------------------------------
# context_words — retrieved-context size reporting
#
# Equal top_k does not mean equal context (a Wikipedia chunk dwarfs a triple),
# so the size is reported per config. The distinction that matters here is
# None ("not measured") vs 0 ("measured, and it was empty") — collapsing them
# would make an old result file look like retrieval returned nothing.
# --------------------------------------------------------------------------

def test_context_words_reads_recorded_field():
    row = _row("q1", {"rag": "Paris"})
    row["context_words"] = {"rag": 2892}
    assert context_words(row, "rag") == 2892


def test_context_words_falls_back_to_captured_context():
    """Result files written before the field existed still report."""
    row = _row("q1", {"graph_rag": "Paris"})
    row["contexts"] = {"graph_rag": {"context": "a b c d e", "pool_size": 118}}
    assert context_words(row, "graph_rag") == 5


def test_context_words_recorded_field_wins_over_fallback():
    row = _row("q1", {"rag": "Paris"})
    row["context_words"] = {"rag": 7}
    row["contexts"] = {"rag": {"context": "a b c"}}
    assert context_words(row, "rag") == 7


def test_context_words_none_when_unmeasured():
    """C1 has no retrieval step: absent, not zero."""
    assert context_words(_row("q1", {"base_llm_abstain": "Paris"}), "base_llm_abstain") is None


def test_context_words_zero_is_preserved_not_treated_as_missing():
    row = _row("q1", {"graph_rag": "Paris"})
    row["context_words"] = {"graph_rag": 0}
    assert context_words(row, "graph_rag") == 0


def test_score_results_aggregates_mean_context_words():
    rows = []
    for qid, n in [("q1", 100), ("q2", 200)]:
        r = _row(qid, {"rag": "Paris", "base_llm_abstain": "Paris"})
        r["context_words"] = {"rag": n}
        rows.append(r)
    pc = score_results(rows)["per_config"]
    assert pc["rag"]["context_words"] == 150.0
    # No retrieval step -> no figure to report, rather than a misleading 0.
    assert pc["base_llm_abstain"]["context_words"] is None


def test_score_results_context_words_ignores_unmeasured_questions():
    """Mean is over questions that HAVE a measurement, not padded with zeros."""
    r1 = _row("q1", {"rag": "Paris"})
    r1["context_words"] = {"rag": 300}
    r2 = _row("q2", {"rag": "Paris"})  # same config, no measurement recorded
    pc = score_results([r1, r2])["per_config"]
    assert pc["rag"]["context_words"] == 300.0


# --------------------------------------------------------------------------
# THE ATTEMPT CONTRACT (2026-08-13).
#
# `score` was zeroed for ABSTENTION/OTHER but `em`/`f1` were not, while
# _aggregate averaged em/f1 over ALL questions. A refusal contains words, so it
# could carry nonzero overlap into the headline F1 column -- biased towards the
# configs that actually refuse (C2 51.8 -> 49.8, C3 42.3 -> 40.3 on
# statement-model-v1). f1_attempted never moved, and must not move now.
# --------------------------------------------------------------------------

class TestAttemptContract:
    def test_is_attempted_covers_exactly_correct_and_hallucination(self):
        assert ATTEMPTED_OUTCOMES == frozenset((CORRECT, HALLUCINATION))
        assert is_attempted({"outcome": CORRECT})
        assert is_attempted({"outcome": HALLUCINATION})
        assert not is_attempted({"outcome": ABSTENTION})
        assert not is_attempted({"outcome": OTHER})
        assert not is_attempted({})

    def test_refusal_matching_boolean_gold_scores_zero(self):
        """gold "No" vs a decline containing "not" -- the 54-answer defect."""
        row = {"id": "q1", "expected": "No", "answer_type": "boolean",
               "complexity": "yesno", "answer_entities": [],
               "answers": {"rag": "The context does not provide information "
                                  "about their heights."}}
        rec = per_question_scores([row])["rag"]["q1"]
        assert rec["outcome"] == OTHER          # leads with a refusal
        assert rec["raw_f1"] == 1.0             # the matcher DID accept it ...
        assert rec["f1"] == rec["em"] == rec["score"] == 0.0   # ... and it is zeroed

    def test_abstention_sentinel_is_zeroed(self):
        row = _row("q1", {"rag": ABSTAIN_SENTINEL})
        rec = per_question_scores([row])["rag"]["q1"]
        assert rec["outcome"] == ABSTENTION
        assert rec["abstained"] is True
        assert rec["f1"] == rec["em"] == rec["score"] == 0.0

    def test_raw_fields_preserve_the_matcher_verdict(self):
        """A correct answer keeps raw_* equal to the reported values."""
        rec = per_question_scores([_row("q1", {"rag": "Paris"})])["rag"]["q1"]
        assert rec["outcome"] == CORRECT
        assert rec["raw_f1"] == rec["f1"] == 1.0
        assert rec["raw_score"] == rec["score"]

    def test_aggregate_f1_excludes_non_attempts_but_attempted_does_not_move(self):
        rows = [
            _row("q1", {"rag": "Paris"}),                      # CORRECT, f1 1.0
            _row("q2", {"rag": "Berlin"}),                     # HALLUCINATION, 0.0
            _row("q3", {"rag": ABSTAIN_SENTINEL}),             # ABSTENTION
        ]
        pc = score_results(rows)["per_config"]["rag"]
        assert pc["n"] == 3 and pc["n_attempted"] == 2
        assert pc["f1"] == pytest.approx(1 / 3)        # over all three
        assert pc["f1_attempted"] == pytest.approx(1 / 2)   # unchanged by the fix

    def test_abstained_is_not_a_proxy_for_not_attempted(self):
        """OTHER is not an attempt, but it is not an abstention either."""
        row = _row("q1", {"rag": ""})           # empty reply -> OTHER
        rec = per_question_scores([row])["rag"]["q1"]
        assert rec["outcome"] == OTHER
        assert rec["abstained"] is False
        assert not is_attempted(rec)


# --------------------------------------------------------------------------
# entity_match — verbose-but-correct answers, and the either-or guard
#
# token_f1 divides by prediction length, so a correct entity inside a sentence
# scored as a hallucination. These pin the fix AND its limit: containment alone
# would credit a wrong answer that names the gold in passing, which the guard
# blocks for either-or questions.
# --------------------------------------------------------------------------

class TestTokenSpanIndex:
    def test_finds_contiguous_run(self):
        assert _token_span_index(["a", "b", "c"], ["b", "c"]) == 1

    def test_requires_contiguity(self):
        assert _token_span_index(["a", "b", "c"], ["a", "c"]) == -1

    def test_absent_and_empty(self):
        assert _token_span_index(["a"], ["b"]) == -1
        assert _token_span_index(["a"], []) == -1
        assert _token_span_index([], ["a"]) == -1


@pytest.mark.parametrize("pred,gold", [
    ("Bono", "Bono"),
    ("The answer is Bono", "Bono"),
    ("The singer of U2 born in Ireland is Bono", "Bono"),
    ("Camila Cabello left Fifth Harmony first, in December 2016.", "Camila Cabello"),
    ("The tallest president was Abraham Lincoln, who stood 6 feet 4.", "Abraham Lincoln"),
    ('The first video game played in space was "Tetris."', "Tetris"),
    ("The director of Metropolis, Fritz Lang, was born in Vienna, Austria.", "Vienna"),
    ("He was born in New York City in 1941", "New York"),
])
def test_entity_match_credits_verbose_correct_answers(pred, gold):
    assert entity_match(pred, gold) == 1.0
    # ...and raw token_f1 gave these at best borderline credit ("The answer is
    # Bono" lands exactly on the 0.5 cutoff; the longer ones fall well below).
    assert token_f1(pred, gold) <= 0.5 or pred == gold


@pytest.mark.parametrize("pred,gold", [
    ("Paris", "Berlin"),                       # simply wrong
    ("Paris", "Paris France"),                 # incomplete: gold not contained
    ("New York", "New York City"),             # incomplete
])
def test_entity_match_rejects_non_assertions(pred, gold):
    assert entity_match(pred, gold) == 0.0


class TestEitherOrGuard:
    """Containment alone over-credits when the gold is merely mentioned."""

    def test_distractor_named_first_scores_zero(self):
        pred = ("The Wheel of Time has more books. It has 14, while "
                "The Southern Vampire Mysteries has 13.")
        assert entity_match(pred, "The Southern Vampire Mysteries",
                            ["The Wheel of Time"]) == 0.0

    def test_gold_named_first_still_scores_one(self):
        pred = "The Magic Treehouse series has more books than The Boxcar Children."
        assert entity_match(pred, "The Magic Treehouse",
                            ["The Boxcar Children"]) == 1.0

    def test_wrong_comparative_scores_zero(self):
        assert entity_match("Juelz Santana is taller than Cam'ron.",
                            "Camron", ["Juelz Santana"]) == 0.0

    def test_guard_off_without_competitors(self):
        pred = "The Wheel of Time has more books than The Southern Vampire Mysteries."
        assert entity_match(pred, "The Southern Vampire Mysteries") == 1.0

    def test_a_gold_variant_is_never_its_own_rival(self):
        assert entity_match("Cam'ron is taller.", "Camron", ["Cam'ron"]) == 1.0


class TestRivalCandidates:
    def test_fires_only_when_gold_is_a_question_entity(self):
        # Either-or: gold IS one of the question's entities.
        assert rival_candidates(
            ["The Southern Vampire Mysteries"],
            ["The Southern Vampire Mysteries", "The Wheel of Time"],
        ) == ["The Wheel of Time"]

    def test_ordinary_lookup_gets_no_rivals(self):
        # "Where was the director of Metropolis born?" — gold Vienna is not a
        # question entity, so Metropolis must NOT suppress it.
        assert rival_candidates(["Vienna"], ["Metropolis"]) == []

    def test_no_question_entities(self):
        assert rival_candidates(["Vienna"], None) == []

    def test_all_gold_forms_are_excluded(self):
        assert rival_candidates(["Camron", "Cam'ron"],
                                ["Juelz Santana", "Cam'ron"]) == ["Juelz Santana"]


class TestQuestionEntityLabels:
    def test_prefers_the_row_then_falls_back_to_the_question_file(self):
        row = {"id": "q1"}
        qbi = {"q1": {"entity_names": ["Camron"], "entity_labels": ["Cam'ron"]}}
        assert question_entity_labels(row, qbi) == ["Camron", "Cam'ron"]

    def test_deduplicates_mention_and_label(self):
        qbi = {"q1": {"entity_names": ["U2"], "entity_labels": ["U2"]}}
        assert question_entity_labels({"id": "q1"}, qbi) == ["U2"]

    def test_empty_without_a_question_file(self):
        assert question_entity_labels({"id": "q1"}) == []


class TestScoreAnswerEntityIntegration:
    def test_verbose_correct_answer_is_no_longer_a_hallucination(self):
        # Verbatim from the 100-sample, where it scored 0.250 and was counted
        # as a confident error against gold "Abraham Lincoln".
        s = score_answer(
            "The tallest president was Abraham Lincoln, who stood at "
            "6 feet 4 inches (193 cm).", ["Abraham Lincoln"], "entity")
        assert s["score"] == 1.0
        # f1 keeps its old meaning so earlier result files stay comparable
        assert s["f1"] < 0.5

    def test_guard_applies_through_score_answer(self):
        s = score_answer("Juelz Santana is taller than Cam'ron.",
                         ["Camron", "Cam'ron"], "entity",
                         question_entities=["Juelz Santana", "Cam'ron"])
        assert s["score"] < 0.5

    def test_monotone_never_lowers_a_score(self):
        """The fix is max(token_f1, entity_match) — it can only raise."""
        for pred, gold in [("Paris", "Paris"), ("Paris", "Paris France"),
                           ("Berlin", "Paris"), ("New York City", "New York")]:
            s = score_answer(pred, [gold], "entity")
            assert s["score"] >= token_f1(pred, gold)

    def test_other_answer_types_are_untouched(self):
        assert score_answer("Yes, he did", ["Yes"], "boolean")["score"] == 1.0
        assert score_answer("They won 7 titles", ["7"], "numerical")["score"] == 1.0
        assert score_answer("Released in 2004", ["2004"], "date")["score"] == 1.0


class TestFourWayOutcome:
    """
    The three-way split forced every ambiguous answer into abstained/correct/
    wrong, and the paraphrase regex made BOTH possible errors on real data:
    it missed terse refusals (inflating hallucination) and it caught hedges
    inside real answers (deflating it, hiding one wrong C2 answer entirely).
    The fourth bucket reports that ambiguity instead of guessing at it.
    """

    def _scored(self, score):
        return {"abstained": False, "em": 0, "f1": score, "score": score}

    def test_exact_sentinel_is_an_abstention(self):
        assert metrics.classify_outcome(ABSTAIN_SENTINEL, self._scored(0.0)) == metrics.ABSTENTION

    def test_sentinel_plus_commentary_is_still_an_abstention(self):
        # Five C2 answers on the reference run gave the sentinel and then said
        # what the context DID cover. That is compliance, not evasion.
        a = ABSTAIN_SENTINEL + " The provided context discusses the band's history instead."
        assert metrics.classify_outcome(a, self._scored(0.0)) == metrics.ABSTENTION

    def test_refusal_in_the_wrong_words_is_other_not_abstention(self):
        # C4 produced ten of these. They are refusals, but they ignored the
        # required wording, so they are not evidence of a followed contract.
        for a in ["Not specified in the provided facts.", "Cannot be determined",
                  "Unable to determine from the provided facts."]:
            assert metrics.classify_outcome(a, self._scored(0.0)) == metrics.OTHER

    def test_correct_answer_wins_over_a_hedge(self):
        # REGRESSION: a C3 answer listed three Neil Breen films while noting the
        # context "does not specify 2021" and was graded a refusal. Scoring is
        # checked before the refusal-shape test precisely to stop that.
        a = 'Fateful Findings, Dire Duplicity (the context does not specify 2021)'
        assert metrics.classify_outcome(a, self._scored(0.9)) == metrics.CORRECT

    def test_wrong_answer_carrying_a_hedge_is_other_not_hallucination(self):
        # REGRESSION: a WRONG C2 answer ("Inglourious Basterds", gold Pulp
        # Fiction) was graded an abstention because its footnote said "not
        # mentioned in the context" -- so a hallucination vanished from the
        # count. It is not a confident error either; OTHER is the honest bucket.
        a = "Inglourious Basterds. (Note: other instances not mentioned in the context.)"
        assert metrics.classify_outcome(a, self._scored(0.1)) == metrics.OTHER

    def test_clean_wrong_answer_is_a_hallucination(self):
        assert metrics.classify_outcome("Warren Moon", self._scored(0.0)) == metrics.HALLUCINATION

    def test_empty_is_other_never_abstention(self):
        # An empty response is the PROVIDER returning nothing, not the model
        # declining. Nemotron with reasoning-on returned 8/8 empty; counting
        # those as calibrated refusals would have been a fabricated result.
        for a in ["", "   ", None]:
            assert metrics.classify_outcome(a, self._scored(0.0)) == metrics.OTHER

    def test_truncated_is_other_not_hallucination(self):
        # Cut off mid-answer is unusable, not fabricated -- the same distinction
        # the errors field was added to preserve.
        assert metrics.classify_outcome(
            "The tallest president was", self._scored(0.0), finish_reason="length"
        ) == metrics.OTHER

    def test_truncated_but_correct_still_counts_correct(self):
        assert metrics.classify_outcome(
            "Abraham Lincoln, who at 6 feet 4 inches was", self._scored(1.0),
            finish_reason="length") == metrics.CORRECT

    def test_buckets_are_exhaustive_and_named(self):
        assert set(metrics.OUTCOMES) == {
            metrics.CORRECT, metrics.HALLUCINATION, metrics.ABSTENTION, metrics.OTHER}

    # --- a refusal must not score CORRECT on its own vocabulary -------------
    #
    # Audited 2026-08-10 over 5,989 stored answers: 54 refusals were graded
    # CORRECT because the CORRECT test ran before the refusal-shape test, and a
    # refusal contains words a matcher will accept. ~40 are boolean questions
    # whose gold is "No" colliding with the "not" inside the decline.

    def test_refusal_does_not_score_correct_via_its_own_negation(self):
        """gold 'No' vs "...does not provide..." -- boolean_match reads the 'not'."""
        a = "The context does not provide information about their heights."
        assert metrics.classify_outcome(a, self._scored(1.0)) == metrics.OTHER

    def test_refusal_does_not_score_correct_by_quoting_the_question(self):
        """gold '3' vs a refusal that echoes 'August 3, 1492' back from the question."""
        a = ("The context does not provide information about the number of ships "
             "Columbus set sail with on August 3, 1492.")
        assert metrics.classify_outcome(a, self._scored(1.0)) == metrics.OTHER

    def test_a_real_answer_with_a_TRAILING_hedge_is_still_correct(self):
        """The cost of a plain reorder, and why the test is position-sensitive.

        Both of these ANSWER the question and then qualify. A blanket
        refusal-before-correct rule demoted them to OTHER; requiring the refusal
        to LEAD keeps them where they belong.
        """
        for a in ["**Answer:** Joe Biden **Fact (implied from context, though not "
                  "directly stated)**",
                  'All except possibly Ronald Reagan (insufficient data to confirm '
                  '"TV star" status)']:
            assert metrics.classify_outcome(a, self._scored(0.9)) == metrics.CORRECT

    def test_opens_with_refusal_is_about_position_not_presence(self):
        lead = "Not specified in the provided facts, though Bono sang for U2."
        trail = "Bono, though the context does not name the album he sang it on."
        assert metrics.opens_with_refusal(lead)
        assert not metrics.opens_with_refusal(trail)

    def test_the_documented_leak_is_pinned_so_it_is_not_mistaken_for_a_fix(self):
        """⚠️ KNOWN RESIDUAL, deliberate: a decline that starts late still leaks.

        Six stored refusals open with an apology and reach the decline only at
        offset ~55, past _REFUSAL_LEAD. They still score CORRECT if they happen
        to contain the gold. Widening the window to catch them costs real
        answers (measured: 40 -> 47 fixed but 1 genuine answer lost), so the
        leak is the chosen price. Pinned so a future reader sees it as a known
        limit rather than an oversight.
        """
        a = ("I'm sorry, but after reviewing the provided context, I couldn't find "
             "the information you asked for.")
        assert not metrics.opens_with_refusal(a)
        assert metrics.classify_outcome(a, self._scored(1.0)) == metrics.CORRECT
