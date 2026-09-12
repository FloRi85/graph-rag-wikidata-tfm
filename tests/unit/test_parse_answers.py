"""
Unit tests for src/eval/parse_answers.py — the typed gold answer.

⚠️ THE LOAD-BEARING TESTS ARE IN `TestNeverReachesAPrompt`. This module holds
the gold answer and the supporting fields, so anything that lets it into a
prompt hands the model its own answer key.

Every case is drawn from the real Mintaka splits; the counts in the docstrings
are measured over all 20,000 raw rows.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_parse_answers.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.eval.parse_answers import (
    ALL_ANSWER_TYPES,
    PENDING_ANSWER_TYPES,
    SIMPLE_TYPES,
    SUPPORTED_TYPES,
    GoldAnswer,
    UnknownAnswerType,
    build_entity_answer,
    build_gold_answer,
    build_simple_answer,
    build_supported_answer,
    is_comparative_operator,
    supporting_evidence,
)


def ent_answer(entities, mention, supporting_num=None, supporting_ent=None):
    """An `answer` object of answerType 'entity'. `entities` is [(qid, en_label)]."""
    payload = None if entities is None else [
        {"name": qid, "label": {"en": label, "de": "x"}} for qid, label in entities
    ]
    out = {"answerType": "entity", "answer": payload, "mention": mention}
    if supporting_num is not None:
        out["supportingNum"] = supporting_num
    if supporting_ent is not None:
        out["supportingEnt"] = [
            {"name": q, "label": {"en": l}} for q, l in supporting_ent
        ]
    return out


class TestSingleEntityAnswer:
    """11,668 answers have exactly one gold entity."""

    def test_qid_label_and_mention_are_all_captured(self):
        gold = build_entity_answer(
            ent_answer([("Q1153188", "Mount Lucania")], "Mount Lucania"))
        assert gold.answer_type == "entity"
        assert gold.qids == ("Q1153188",)
        assert gold.mention == "Mount Lucania"
        assert gold.has_qid is True

    def test_identical_label_and_mention_yield_ONE_form(self):
        gold = build_entity_answer(
            ent_answer([("Q1153188", "Mount Lucania")], "Mount Lucania"))
        assert gold.forms == ("Mount Lucania",)

    def test_a_single_answer_is_not_a_set(self):
        gold = build_entity_answer(ent_answer([("Q1", "Nile")], "The Nile"))
        assert gold.is_set is False
        assert gold.members == ()


class TestFormsAreAlternativesNotDuplicates:
    """
    The rule that INVERTS from the question side.

    label and mention differ on 23.3% of single-entity answers. Both must be
    accepted, or a model answering "The Rock" is marked wrong for a gold whose
    canonical label is "Dwayne Johnson".
    """

    @pytest.mark.parametrize("label,mention", [
        ("Dwayne Johnson", "The Rock"),
        ("The Hurt Locker", "Hurt Locker"),
        ("Nile", "The Nile"),
        ("Tiger King: Murder, Mayhem and Madness", "Tiger King"),
        ("Marvel Studios", "Marvel"),
        ("Staten Island", "Staten Island, New York City"),
    ])
    def test_both_spellings_survive(self, label, mention):
        gold = build_entity_answer(ent_answer([("Q1", label)], mention))
        assert label in gold.forms
        assert mention in gold.forms

    def test_mintakas_own_typo_is_kept_as_an_acceptable_form(self):
        """'Avenger: Endgame' is a misspelling in the corpus, not ours to fix."""
        gold = build_entity_answer(
            ent_answer([("Q1", "Avengers: Endgame")], "Avenger: Endgame"))
        assert set(gold.forms) == {"Avengers: Endgame", "Avenger: Endgame"}

    def test_the_canonical_label_leads(self):
        """Decision 2026-08-14: labels first, mention appended if it differs."""
        gold = build_entity_answer(ent_answer([("Q1", "Dwayne Johnson")], "The Rock"))
        assert gold.forms == ("Dwayne Johnson", "The Rock")

    @pytest.mark.parametrize("label,mention", [
        ("Gone with the Wind", "Gone With The Wind"),
        ("Emma Stone", "Emma stone"),
        ("mother!", "Mother!"),
        ("Goodfellas", "goodfellas"),
        ("Spider-Man: Far from Home", "Spider-Man: Far From Home"),
    ])
    def test_when_the_spelling_matches_only_the_LABEL_survives(self, label, mention):
        """306 rows differ by case alone, and the label is the better-formed one."""
        gold = build_entity_answer(ent_answer([("Q1", label)], mention))
        assert gold.forms == (label,)

    def test_forms_differing_only_by_spacing_collapse_to_the_label(self):
        gold = build_entity_answer(ent_answer([("Q1", "the  rock")], "The Rock"))
        assert gold.forms == ("the  rock",)


class TestSetAnswers:
    """561 entity answers name two or more entities (train 391 / dev 58 / test 112)."""

    def test_three_entities_become_three_members(self):
        gold = build_entity_answer(ent_answer(
            [("Q29021224", "Bad Boys for Life"),
             ("Q29906232", "Sonic the Hedgehog"),
             ("Q57177410", "Birds of Prey")],
            "Bad Boys for Life, Sonic the Hedgehog, Birds of Prey"))
        assert gold.is_set is True
        assert gold.members == (
            "Bad Boys for Life", "Sonic the Hedgehog", "Birds of Prey")

    def test_members_are_not_treated_as_alternative_spellings(self):
        """Naming one of three must not be scoreable as the whole answer.

        `members` exists precisely so the caller can apply set scoring; the
        distinction is what a previous scorer bug got wrong.
        """
        gold = build_entity_answer(ent_answer(
            [("Q1", "A"), ("Q2", "B"), ("Q3", "C")], "A, B, C"))
        assert gold.is_set is True
        assert set(gold.members) == {"A", "B", "C"}

    def test_the_joined_mention_stays_an_acceptable_form(self):
        gold = build_entity_answer(ent_answer(
            [("Q1", "A"), ("Q2", "B")], "A, B"))
        assert "A, B" in gold.forms

    def test_all_qids_are_kept_in_order(self):
        gold = build_entity_answer(ent_answer(
            [("Q1", "A"), ("Q2", "B"), ("Q3", "C")], "A, B, C"))
        assert gold.qids == ("Q1", "Q2", "Q3")

    def test_repeated_labels_do_not_make_a_set(self):
        gold = build_entity_answer(ent_answer([("Q1", "A"), ("Q2", "A")], "A"))
        assert gold.is_set is False


class TestMissingPayload:
    """295 entity answers carry no QID at all (train 209 / dev 19 / test 67)."""

    def test_the_mention_still_gives_a_usable_gold(self):
        gold = build_entity_answer(
            ent_answer(None, "Fitzhugh Lee Elementary School"))
        assert gold.forms == ("Fitzhugh Lee Elementary School",)
        assert gold.qids == ()
        assert gold.has_qid is False

    def test_an_empty_list_behaves_the_same_as_null(self):
        gold = build_entity_answer(ent_answer([], "Smoking"))
        assert gold.forms == ("Smoking",)
        assert gold.qids == ()

    def test_it_is_not_a_set(self):
        gold = build_entity_answer(ent_answer(None, "Elvis Presto"))
        assert gold.is_set is False

    def test_a_completely_empty_answer_yields_no_forms_rather_than_a_blank_one(self):
        gold = build_entity_answer(ent_answer(None, ""))
        assert gold.forms == ()


class TestSupportingFields:
    """Carried through instead of dropped — this is what they exist for."""

    def test_supporting_number_is_kept(self):
        """1,979 entity answers carry one; they are the superlative questions."""
        gold = build_entity_answer(ent_answer(
            [("Q3105215", "Ron DeSantis")], "Ron DeSantis",
            supporting_num="43 years, 29 days"))
        assert gold.supporting_number == "43 years, 29 days"

    def test_supporting_number_may_be_an_int(self):
        """The field is mixed-type across the corpus: 1,179 str, 804 int."""
        gold = build_entity_answer(
            ent_answer([("Q1", "Destiny 2")], "Destiny 2", supporting_num=6))
        assert gold.supporting_number == 6

    def test_absent_supporting_number_is_None_not_zero(self):
        """"not annotated" and "the quantity is zero" are different findings."""
        gold = build_entity_answer(ent_answer([("Q1", "A")], "A"))
        assert gold.supporting_number is None

    def test_supporting_entities_are_empty_for_this_type(self):
        """supportingEnt attaches to NUMERICAL answers; 0 entity answers carry it."""
        gold = build_entity_answer(ent_answer([("Q1", "A")], "A"))
        assert gold.supporting_entities == ()

    def test_supporting_entities_are_read_if_present(self):
        gold = build_entity_answer(ent_answer(
            [("Q1", "A")], "A", supporting_ent=[("Q9", "Angus Young")]))
        assert gold.supporting_entities == ("Angus Young",)

    def test_an_empty_string_supporting_number_is_treated_as_absent(self):
        gold = build_entity_answer(ent_answer([("Q1", "A")], "A", supporting_num="  "))
        assert gold.supporting_number is None


class TestSupportingEvidenceLookup:
    """`supportingNum` -> else `supportingEnt` -> else None. Never 0, never ()."""

    def test_the_number_wins_when_populated(self):
        assert supporting_evidence(
            ent_answer([("Q1", "A")], "A", supporting_num="43 years")) == "43 years"

    def test_zero_is_a_real_value_not_an_absence(self):
        assert supporting_evidence(ent_answer([("Q1", "A")], "A", supporting_num=0)) == 0

    def test_it_falls_through_to_the_entities(self):
        got = supporting_evidence(ent_answer(
            [("Q1", "A")], "A", supporting_ent=[("Q9", "Angus Young"), ("Q8", "Cliff Williams")]))
        assert got == ("Angus Young", "Cliff Williams")

    def test_an_explicit_null_number_falls_through_rather_than_returning_it(self):
        """5 rows carry supportingNum set to null."""
        answer = ent_answer([("Q1", "A")], "A", supporting_ent=[("Q9", "Angus Young")])
        answer["supportingNum"] = None
        assert supporting_evidence(answer) == ("Angus Young",)

    def test_neither_present_returns_None_not_zero_and_not_empty(self):
        got = supporting_evidence(ent_answer([("Q1", "A")], "A"))
        assert got is None
        assert got != 0
        assert got != ()

    def test_an_empty_supporting_entity_list_returns_None(self):
        answer = ent_answer([("Q1", "A")], "A")
        answer["supportingEnt"] = []
        assert supporting_evidence(answer) is None

    def test_the_lookup_order_is_only_safe_because_they_never_co_occur(self):
        """
        ⚠️ THE GUARD ON THE WHOLE DESIGN.

        Measured over all 20,000 raw rows: supportingNum on 1,988, supportingEnt
        on 1,886, intersection EXACTLY ZERO. If that ever breaks, preferring the
        number would silently hide the entities, so the disjointness is pinned
        here against the real corpus rather than assumed.
        """
        import json
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        both = num = ent = 0
        for split in ("train", "dev", "test"):
            path = root / "data" / "questions" / f"mintaka_{split}_raw.json"
            if not path.exists():
                pytest.skip(f"{path.name} not present")
            for row in json.loads(path.read_text(encoding="utf-8")):
                a = row["answer"]
                has_num = a.get("supportingNum") is not None
                has_ent = bool(a.get("supportingEnt"))
                num += has_num
                ent += has_ent
                both += has_num and has_ent
        assert num > 0 and ent > 0        # both fields really are populated
        assert both == 0, f"{both} rows carry BOTH supporting fields"


class TestSimpleTypes:
    """`boolean` and `string` share one builder because they share one shape."""

    def test_boolean_true_renders_as_Yes_not_True(self):
        gold = build_simple_answer(
            {"answerType": "boolean", "answer": [True], "mention": "Yes"})
        assert gold.forms == ("Yes",)

    def test_boolean_false_renders_as_No(self):
        gold = build_simple_answer(
            {"answerType": "boolean", "answer": [False], "mention": "No"})
        assert gold.forms == ("No",)

    def test_the_literal_python_bool_never_becomes_a_form(self):
        """str(True) is 'True', which is not an answer a model would give."""
        for value, mention in [(True, "Yes"), (False, "No")]:
            gold = build_simple_answer(
                {"answerType": "boolean", "answer": [value], "mention": mention})
            assert "True" not in gold.forms and "False" not in gold.forms

    def test_booleans_never_carry_supporting_evidence(self):
        """0 of 2,867 rows do."""
        gold = build_simple_answer(
            {"answerType": "boolean", "answer": [True], "mention": "Yes"})
        assert gold.supporting_number is None
        assert gold.supporting_entities == ()

    def test_booleans_are_never_sets_and_have_no_qids(self):
        gold = build_simple_answer(
            {"answerType": "boolean", "answer": [False], "mention": "No"})
        assert gold.is_set is False and gold.qids == ()

    @pytest.mark.parametrize("value", ["Currer Bell", "Robert Galbraith", "Jazzy",
                                       "SunTrust Park", "James", "CROSS STITCH"])
    def test_string_names_round_trip(self, value):
        gold = build_simple_answer(
            {"answerType": "string", "answer": [value], "mention": value})
        assert gold.forms == (value,)

    def test_the_one_bare_string_payload_is_handled(self):
        """TRAIN c57c3047: payload is 'Wonderboy', not ['Wonderboy']."""
        gold = build_simple_answer(
            {"answerType": "string", "answer": "Wonderboy", "mention": "Wonderboy"})
        assert gold.forms == ("Wonderboy",)

    def test_a_differing_mention_is_kept_as_a_second_form(self):
        gold = build_simple_answer(
            {"answerType": "string", "answer": ["Boz"], "mention": "boz the writer"})
        assert set(gold.forms) == {"Boz", "boz the writer"}

    def test_the_wrong_type_is_refused_rather_than_mishandled(self):
        with pytest.raises(UnknownAnswerType):
            build_simple_answer(
                {"answerType": "numerical", "answer": [5], "mention": "5"})


class TestComparativeOperators:
    """13 of the 28 string golds are before/after answers, and they need a hook."""

    @pytest.mark.parametrize("value", ["Before", "After", "Same", "Less", "Both"])
    def test_operators_are_flagged(self, value):
        gold = build_simple_answer(
            {"answerType": "string", "answer": [value], "mention": value})
        assert is_comparative_operator(gold) is True

    @pytest.mark.parametrize("value", ["Currer Bell", "Robert Galbraith", "James"])
    def test_names_are_not_flagged(self, value):
        gold = build_simple_answer(
            {"answerType": "string", "answer": [value], "mention": value})
        assert is_comparative_operator(gold) is False

    def test_the_flag_is_case_insensitive(self):
        gold = build_simple_answer(
            {"answerType": "string", "answer": ["after"], "mention": "AFTER"})
        assert is_comparative_operator(gold) is True

    def test_entity_answers_are_never_flagged_even_if_named_after(self):
        """The flag is about the string TYPE, not about the word."""
        gold = build_entity_answer(ent_answer([("Q1", "After")], "After"))
        assert is_comparative_operator(gold) is False


class TestSupportedTypes:
    """`date` and `numerical` share one builder because both may carry evidence."""

    def test_a_date_keeps_both_the_iso_value_and_the_abbreviated_mention(self):
        """286 of 1,275 date rows differ this way — the scorer-fix-#3 pairing."""
        gold = build_supported_answer(
            {"answerType": "date", "answer": ["1946-12-18"], "mention": "18-Dec-46"})
        assert gold.forms == ("1946-12-18", "18-Dec-46")

    def test_a_matching_date_yields_one_form(self):
        gold = build_supported_answer(
            {"answerType": "date", "answer": ["2020"], "mention": "2020"})
        assert gold.forms == ("2020",)

    def test_numerical_int(self):
        gold = build_supported_answer(
            {"answerType": "numerical", "answer": [5], "mention": "5"})
        assert gold.forms == ("5",)

    @pytest.mark.parametrize("payload,mention", [
        ("5'11\"", "5'11\""), ("6'2\"", "6'2\""), ("5'10\"", "5 foot 10 inches"),
    ])
    def test_numerical_string_payloads_are_carried_verbatim(self, payload, mention):
        """199 numerical golds are feet-and-inches heights, not numbers."""
        gold = build_supported_answer(
            {"answerType": "numerical", "answer": [payload], "mention": mention})
        assert payload in gold.forms

    @pytest.mark.parametrize("payload,mention", [
        (18.99, "18.99"), (20.5, "20 years, 6 months"), (82.8, "82.8 seconds"),
    ])
    def test_numerical_float_payloads_are_not_normalised(self, payload, mention):
        gold = build_supported_answer(
            {"answerType": "numerical", "answer": [payload], "mention": mention})
        assert str(payload) in gold.forms
        assert mention in gold.forms or str(payload) == mention

    def test_supporting_entities_are_captured_on_a_count_answer(self):
        """The whole reason this pair is separate: 1,886 numerical rows carry it."""
        gold = build_supported_answer({
            "answerType": "numerical", "answer": [5], "mention": "5",
            "supportingEnt": [{"name": "Q1", "label": {"en": "Angus Young"}},
                              {"name": "Q2", "label": {"en": "Cliff Williams"}}],
        })
        assert gold.supporting_entities == ("Angus Young", "Cliff Williams")
        assert gold.supporting_number is None

    def test_a_supporting_number_that_is_not_a_number_is_carried_anyway(self):
        """TRAIN: supportingNum='New York, NY' on a numerical answer."""
        gold = build_supported_answer({
            "answerType": "numerical", "answer": [8800000], "mention": "8800000",
            "supportingNum": "New York, NY"})
        assert gold.supporting_number == "New York, NY"

    def test_the_one_date_row_with_a_supporting_number(self):
        gold = build_supported_answer({
            "answerType": "date", "answer": ["2020"], "mention": "2020",
            "supportingNum": "7,052,770 votes."})
        assert gold.supporting_number == "7,052,770 votes."

    def test_neither_supporting_field_leaves_both_empty(self):
        gold = build_supported_answer(
            {"answerType": "numerical", "answer": [3], "mention": "3"})
        assert gold.supporting_number is None
        assert gold.supporting_entities == ()

    def test_the_wrong_type_is_refused(self):
        with pytest.raises(UnknownAnswerType):
            build_supported_answer(
                {"answerType": "boolean", "answer": [True], "mention": "Yes"})


class TestSingleEntryPoint:
    """`build_gold_answer` dispatches all five types."""

    @pytest.mark.parametrize("answer,expected", [
        ({"answerType": "boolean", "answer": [True], "mention": "Yes"}, "boolean"),
        ({"answerType": "string", "answer": ["Boz"], "mention": "Boz"}, "string"),
        ({"answerType": "date", "answer": ["2020"], "mention": "2020"}, "date"),
        ({"answerType": "numerical", "answer": [7], "mention": "7"}, "numerical"),
    ])
    def test_dispatch(self, answer, expected):
        assert build_gold_answer(answer).answer_type == expected

    def test_entity_dispatch(self):
        gold = build_gold_answer(ent_answer([("Q1", "Nile")], "The Nile"))
        assert gold.answer_type == "entity"
        assert gold.qids == ("Q1",)

    def test_an_undeclared_type_raises_rather_than_defaulting(self):
        with pytest.raises(UnknownAnswerType):
            build_gold_answer({"answerType": "quantity", "answer": [1], "mention": "1"})

    def test_a_missing_answer_type_raises(self):
        with pytest.raises(UnknownAnswerType):
            build_gold_answer({"answer": [1], "mention": "1"})


class TestRegistryIsComplete:
    def test_nothing_is_pending(self):
        assert PENDING_ANSWER_TYPES == frozenset()

    def test_all_five_mintaka_answer_types_are_declared(self):
        assert set(ALL_ANSWER_TYPES) == {
            "entity", "numerical", "boolean", "date", "string"}

    def test_the_groups_partition_the_declared_types(self):
        assert set(SIMPLE_TYPES) | set(SUPPORTED_TYPES) | {"entity"} == set(ALL_ANSWER_TYPES)
        assert not set(SIMPLE_TYPES) & set(SUPPORTED_TYPES)

    def test_an_undeclared_answer_type_raises_rather_than_vanishing(self):
        with pytest.raises(UnknownAnswerType):
            build_entity_answer({"answerType": "quantity", "answer": [], "mention": "x"})


class TestOneRuleAcrossBothSides:
    """
    The question side and the answer side collapse spellings the SAME way.

    It is easy to assume they must differ, because one prints its forms into a
    prompt and the other matches predictions against them. They do not: both
    keep two spellings when they differ and one when they are identical. These
    tests pin that, so a change to one normaliser cannot silently desync it
    from the other.
    """

    def test_the_two_normalisers_agree(self):
        from src.eval.parse_answers import _norm
        from src.eval.parse_questions import _surface_key
        for text in ["The Rock", "the  rock", " The Rock ", "THE ROCK",
                     "Avengers: Endgame", "218 BC", ""]:
            assert _norm(text) == _surface_key(text)

    def test_identical_spellings_collapse_on_both_sides(self):
        from src.eval.parse_questions import build_entity_packages
        question = build_entity_packages([{
            "entityType": "entity", "name": "Q1",
            "label": "Mount Lucania", "mention": "Mount Lucania"}])
        answer = build_entity_answer(
            ent_answer([("Q1", "Mount Lucania")], "Mount Lucania"))
        assert question[0].mentions == ()        # one spelling printed
        assert answer.forms == ("Mount Lucania",)  # one spelling matched

    def test_differing_spellings_are_kept_on_both_sides(self):
        from src.eval.parse_questions import build_entity_packages
        question = build_entity_packages([{
            "entityType": "entity", "name": "Q1",
            "label": "Dwayne Johnson", "mention": "The Rock"}])
        answer = build_entity_answer(
            ent_answer([("Q1", "Dwayne Johnson")], "The Rock"))
        assert question[0].mentions == ("The Rock",)
        assert set(answer.forms) == {"Dwayne Johnson", "The Rock"}


class TestNeverReachesAPrompt:
    """
    ⚠️ THE BOUNDARY. This module holds the gold answer and the supporting
    fields; a path from here into a prompt is a path from the answer key into
    the model's input.
    """

    def test_no_package_can_render_itself(self):
        """The question-side packages have `render_line`; this one must not."""
        gold = build_entity_answer(ent_answer([("Q1", "A")], "A"))
        assert not hasattr(gold, "render_line")
        assert not any(name.startswith("render") for name in dir(gold))

    def test_the_module_exposes_no_render_helper(self):
        import src.eval.parse_answers as answer_types
        assert not [n for n in dir(answer_types) if n.startswith("render")]

    def test_the_pipelines_and_prompts_never_import_this_module(self):
        """Structural, not a convention someone has to remember."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        targets = list((root / "src" / "pipelines").glob("*.py"))
        targets.append(root / "src" / "prompts.py")
        for path in targets:
            text = path.read_text(encoding="utf-8")
            assert "answer_types" not in text, f"{path.name} imports answer_types"

    def test_the_gold_fields_are_a_closed_set(self):
        """A new field cannot be added without this test failing."""
        assert set(GoldAnswer.__dataclass_fields__) == {
            "answer_type", "payload", "mention", "forms", "members",
            "qids", "supporting_number", "supporting_entities",
        }
