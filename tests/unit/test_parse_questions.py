"""
Unit tests for src/eval/parse_questions.py — the question-side metadata package.

Every case below was found by measuring the real Mintaka splits, not invented:
the ids in the docstrings are real question ids, and the counts are over all
20,000 raw rows. That matters because the behaviour being pinned here is the
handling of rare shapes (1 row with a QID and no label, 237 unlinked mentions,
13 TEST questions repeating a QID) which a hand-written fixture would miss.

⚠️ THE LOAD-BEARING TEST IS `TestAnswerSideNeverLeaks`. The package is fed into
the answering prompt of all four configs, so anything from `answer.*` reaching
it would be feeding the model its own gold.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_parse_questions.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.eval.parse_questions import (
    ALL_TYPES,
    CATEGORIES,
    COMPLEXITY_TYPES,
    QUESTION_ENTITY_TYPES,
    NotRawMintakaError,
    BLOCK_MODE_ALL,
    BLOCK_MODE_ENTITY,
    load_question_ids,
    load_questions,
    parse_question,
    raw_type,
    LITERAL_TYPES,
    PENDING_TYPES,
    CardinalPackage,
    DatePackage,
    EntityPackage,
    MoneyPackage,
    OrdinalPackage,
    PercentPackage,
    QuantityPackage,
    TimePackage,
    build_date_packages,
    build_money_packages,
    build_percent_packages,
    build_quantity_packages,
    build_time_packages,
    UnknownEntityType,
    as_number,
    build_cardinal_packages,
    build_entity_packages,
    build_ordinal_packages,
    build_packages,
    render_question_block,
    packages_to_rows,
    render_cardinal_line,
    render_entity_block,
    render_ordinal_line,
    rows_to_packages,
)


def ent(name, label, mention, entity_type="entity"):
    return {"entityType": entity_type, "name": name, "label": label, "mention": mention}


def card(name, mention):
    return {"entityType": "cardinal", "name": name, "label": None, "mention": mention}


def one_cardinal(name, mention):
    [pkg] = build_cardinal_packages([card(name, mention)])
    return pkg


def ordn(name, mention):
    return {"entityType": "ordinal", "name": name, "label": None, "mention": mention}


def one_ordinal(name, mention):
    [pkg] = build_ordinal_packages([ordn(name, mention)])
    return pkg


def render_line_of(pkg):
    return pkg.render_line()


def qty(name, mention):
    return {"entityType": "quantity", "name": name, "label": None, "mention": mention}


def one_quantity(name, mention):
    [pkg] = build_quantity_packages([qty(name, mention)])
    return pkg


def mon(name, mention):
    return {"entityType": "money", "name": name, "label": None, "mention": mention}


def one_money(name, mention):
    [pkg] = build_money_packages([mon(name, mention)])
    return pkg


def pct(name, mention):
    return {"entityType": "percent", "name": name, "label": None, "mention": mention}


def one_percent(name, mention):
    [pkg] = build_percent_packages([pct(name, mention)])
    return pkg


def tm(name, mention):
    return {"entityType": "time", "name": name, "label": None, "mention": mention}


def one_time(name, mention):
    [pkg] = build_time_packages([tm(name, mention)])
    return pkg


def dt(name, mention):
    return {"entityType": "date", "name": name, "label": None, "mention": mention}


def one_date(name, mention):
    [pkg] = build_date_packages([dt(name, mention)])
    return pkg


class TestBasicShape:
    def test_linked_entity_keeps_qid_label_and_type(self):
        [pkg] = build_entity_packages([ent("Q49", "North America", "North America")])
        assert pkg.name == "Q49"
        assert pkg.entity_type == "entity"
        assert pkg.label == "North America"
        assert pkg.linked is True

    def test_mention_equal_to_label_is_not_stored(self):
        """The specified rule: identical mention -> label only."""
        [pkg] = build_entity_packages([ent("Q49", "North America", "North America")])
        assert pkg.mentions == ()

    def test_mention_differing_from_label_is_stored(self):
        """Q65 is labelled 'Los Angeles' but mentioned as 'Los Angeles, California'."""
        [pkg] = build_entity_packages([ent("Q65", "Los Angeles", "Los Angeles, California")])
        assert pkg.mentions == ("Los Angeles, California",)

    def test_order_of_appearance_is_preserved(self):
        pkgs = build_entity_packages([
            ent("Q465469", "American League", "AL"),
            ent("Q634857", "Cy Young Award", "Cy Young Award"),
            ent("Q858082", "National League", "NL"),
        ])
        assert [p.name for p in pkgs] == ["Q465469", "Q634857", "Q858082"]

    def test_empty_and_none_input(self):
        assert build_entity_packages([]) == []
        assert build_entity_packages(None) == []


class TestCaseFolding:
    """Decision 2026-08-13: fold case. 107 TEST mentions differ by capitalisation only."""

    def test_case_only_difference_collapses_to_label(self):
        [pkg] = build_entity_packages([ent("Q1299", "The Beatles", "the Beatles")])
        assert pkg.mentions == ()

    def test_surrounding_whitespace_does_not_defeat_the_collapse(self):
        [pkg] = build_entity_packages([ent("Q1299", "The Beatles", "  The Beatles ")])
        assert pkg.mentions == ()

    def test_a_real_difference_still_survives(self):
        [pkg] = build_entity_packages([ent("Q41254", "Grammy Award", "Grammy Awards")])
        assert pkg.mentions == ("Grammy Awards",)


class TestRepeatedQid:
    """13 TEST / 9 DEV questions tag one QID twice. Merge, do not duplicate."""

    def test_identical_repeat_yields_one_package(self):
        """97034384: Q265538 'World Series' listed twice, same mention."""
        pkgs = build_entity_packages([
            ent("Q265538", "World Series", "World Series"),
            ent("Q265538", "World Series", "World Series"),
        ])
        assert len(pkgs) == 1
        assert pkgs[0].mentions == ()

    def test_two_surface_forms_merge_into_one_package(self):
        """8f75bf08: Q41254 as both 'Grammy Awards' and 'Grammy'."""
        pkgs = build_entity_packages([
            ent("Q41254", "Grammy Award", "Grammy Awards"),
            ent("Q41254", "Grammy Award", "Grammy"),
        ])
        assert len(pkgs) == 1
        assert pkgs[0].mentions == ("Grammy Awards", "Grammy")

    def test_repeat_does_not_disturb_the_position_of_later_entities(self):
        pkgs = build_entity_packages([
            ent("Q465469", "American League", "AL"),
            ent("Q634857", "Cy Young Award", "Cy Young Award"),
            ent("Q858082", "National League", "NL"),
            ent("Q634857", "Cy Young Award", "Cy Young Award"),
        ])
        assert [p.name for p in pkgs] == ["Q465469", "Q634857", "Q858082"]

    def test_duplicate_mention_string_is_not_stored_twice(self):
        pkgs = build_entity_packages([
            ent("Q41254", "Grammy Award", "Grammy"),
            ent("Q41254", "Grammy Award", "Grammy"),
        ])
        assert pkgs[0].mentions == ("Grammy",)


class TestUnlinkedMentions:
    """237 mentions are entityType='entity' with name=None. Kept, marked."""

    def test_unlinked_mention_is_kept(self):
        """d28e564f: 'Cooke Maroney' has no QID."""
        pkgs = build_entity_packages([
            ent("Q2006869", "X-Men", "X-Men movies"),
            ent(None, None, "Cooke Maroney"),
        ])
        assert len(pkgs) == 2
        assert pkgs[1].linked is False
        assert pkgs[1].name is None
        assert pkgs[1].mentions == ("Cooke Maroney",)

    def test_unlinked_mentions_sort_after_linked_ones(self):
        pkgs = build_entity_packages([
            ent(None, None, "Gerry Lane"),
            ent("Q49", "North America", "North America"),
        ])
        assert [p.linked for p in pkgs] == [True, False]

    def test_unlinked_duplicates_collapse(self):
        pkgs = build_entity_packages([
            ent(None, None, "Winston"),
            ent(None, None, "Winston"),
        ])
        assert len(pkgs) == 1

    def test_unlinked_with_empty_mention_is_dropped(self):
        """Nothing to say about it, so saying nothing beats an empty bullet."""
        assert build_entity_packages([ent(None, None, "")]) == []

    def test_a_question_of_only_unlinked_mentions_still_produces_a_package(self):
        """The 9 zero-anchor TEST questions previously produced nothing at all."""
        pkgs = build_entity_packages([ent(None, None, "Bruno Buckingham")])
        assert len(pkgs) == 1
        assert pkgs[0].linked is False


class TestMissingLabel:
    """Exactly one row in 20,000 has a QID and no label: TRAIN 7bb95643."""

    def test_mention_becomes_the_label(self):
        [pkg] = build_entity_packages([ent("Q3984816", None, "Terry Malloy")])
        assert pkg.label == "Terry Malloy"
        assert pkg.mentions == ()          # not repeated as a mention as well


class TestLiteralTypes:
    """The seven non-entity types are recognised and skipped, not silently unknown."""

    @pytest.mark.parametrize("literal_type", LITERAL_TYPES)
    def test_literal_types_produce_no_package(self, literal_type):
        assert build_entity_packages([ent(200000000, None, "$200 million", literal_type)]) == []

    def test_literals_do_not_disturb_entities_around_them(self):
        pkgs = build_entity_packages([
            ent("Q49", "North America", "North America"),
            ent(7, None, "seventh", "ordinal"),
            ent("Q65", "Los Angeles", "Los Angeles"),
        ])
        assert [p.name for p in pkgs] == ["Q49", "Q65"]

    def test_every_declared_type_is_accepted(self):
        for t in ALL_TYPES:
            build_entity_packages([ent("Q1", "x", "x", t)])   # must not raise

    def test_an_undeclared_type_raises_rather_than_vanishing(self):
        """The whole point of the registry: the old `in keep` test dropped these."""
        with pytest.raises(UnknownEntityType):
            build_entity_packages([ent("Q1", "x", "x", "duration")])

    def test_a_missing_entity_type_raises(self):
        with pytest.raises(UnknownEntityType):
            build_entity_packages([{"name": "Q1", "label": "x", "mention": "x"}])


class TestRendering:
    def test_empty_package_renders_to_empty_string_not_a_bare_header(self):
        assert render_entity_block([]) == ""

    def test_label_only_line(self):
        block = render_entity_block(build_entity_packages(
            [ent("Q634857", "Cy Young Award", "Cy Young Award")]))
        assert block == "Question entities:\n- Q634857 (entity): Cy Young Award"

    def test_line_with_one_surface_mention(self):
        block = render_entity_block(build_entity_packages(
            [ent("Q465469", "American League", "AL")]))
        assert block.endswith('- Q465469 (entity): American League — mentioned as "AL"')

    def test_line_with_two_surface_mentions(self):
        block = render_entity_block(build_entity_packages([
            ent("Q41254", "Grammy Award", "Grammy Awards"),
            ent("Q41254", "Grammy Award", "Grammy"),
        ]))
        assert block.endswith(
            '- Q41254 (entity): Grammy Award — mentioned as "Grammy Awards", "Grammy"')

    def test_unlinked_line_is_visibly_marked(self):
        block = render_entity_block(build_entity_packages([ent(None, None, "Cooke Maroney")]))
        assert block == 'Question entities:\n- (entity, no Wikidata id): "Cooke Maroney"'

    def test_mention_containing_a_quote_still_renders(self):
        """201 mentions carry the question's own quotation marks, e.g. '\"Chucky\"'."""
        block = render_entity_block(build_entity_packages(
            [ent("Q1", "Chucky", '"Chucky"')]))
        assert '"Chucky"' in block
        assert "\n" in block and block.count("\n") == 1     # still exactly one row

    def test_header_appears_once_for_many_entities(self):
        block = render_entity_block(build_entity_packages([
            ent("Q465469", "American League", "AL"),
            ent("Q858082", "National League", "NL"),
        ]))
        assert block.count("Question entities:") == 1
        assert len(block.splitlines()) == 3


class TestRoundTrip:
    """The package is stored on a flattened question row, so it must survive JSON."""

    def test_round_trip_preserves_every_field(self):
        original = build_entity_packages([
            ent("Q41254", "Grammy Award", "Grammy Awards"),
            ent("Q41254", "Grammy Award", "Grammy"),
            ent(None, None, "Cooke Maroney"),
        ])
        assert rows_to_packages(packages_to_rows(original)) == original

    def test_rows_are_json_serialisable(self):
        import json
        rows = packages_to_rows(build_entity_packages([ent("Q49", "North America", "NA")]))
        assert json.loads(json.dumps(rows)) == rows

    def test_round_trip_of_empty(self):
        assert rows_to_packages(packages_to_rows([])) == []
        assert rows_to_packages(None) == []


class TestRegistryIsComplete:
    """Every declared type has a handler, and the registry is exhaustive."""

    def test_nothing_is_pending(self):
        assert PENDING_TYPES == frozenset()

    def test_pending_is_a_subset_of_the_declared_types(self):
        assert PENDING_TYPES <= set(ALL_TYPES)

    def test_all_eight_mintaka_types_are_declared(self):
        assert set(ALL_TYPES) == {
            "entity", "ordinal", "date", "cardinal",
            "quantity", "money", "percent", "time",
        }

    @pytest.mark.parametrize("entity_type", [
        "entity", "ordinal", "date", "cardinal", "quantity", "money", "percent", "time",
    ])
    def test_every_type_produces_a_package_through_the_single_entry_point(self, entity_type):
        """`build_packages` must handle all eight — no type contributes nothing."""
        row = {"entityType": entity_type, "name": "Q1" if entity_type == "entity" else 5,
               "label": "thing" if entity_type == "entity" else None,
               "mention": "surface", "span": [0, 7]}
        assert len(build_packages([row])) == 1

    def test_every_package_renders_a_line_beginning_with_its_type(self):
        rows = [
            {"entityType": t, "name": "Q1" if t == "entity" else 5,
             "label": "thing" if t == "entity" else None,
             "mention": "surface", "span": [i, i + 1]}
            for i, t in enumerate(ALL_TYPES)
        ]
        block = render_question_block(rows)
        for t in ALL_TYPES:
            assert f"({t})" in block


class TestAsNumber:
    """Numeric equivalence, not string equivalence — the rule the collapse rests on."""

    @pytest.mark.parametrize("text,expected", [
        ("3", 3.0), (3, 3.0), ("7,000", 7000.0), ("60,000", 60000.0),
        ("1.000.000", 1000000.0), ("1,000,000", 1000000.0), ("0.5", 0.5),
        ("2.2", 2.2), (" 250 ", 250.0),
    ])
    def test_numeric_forms_parse(self, text, expected):
        assert as_number(text) == expected

    @pytest.mark.parametrize("text", [
        "two", "half", "No. 1", "four times", "fewer than 1 million",
        "40:49", "thirties", "", None, "   ",
    ])
    def test_non_numeric_forms_return_none(self, text):
        assert as_number(text) is None

    def test_a_single_period_stays_a_decimal_point(self):
        """'0.5' must not become 5 by having its period stripped as a separator."""
        assert as_number("0.5") == 0.5


class TestCardinalCollapse:
    """Show one value when both renderings denote the same number, else both."""

    @pytest.mark.parametrize("name,mention", [
        (3, "3"), (7000, "7,000"), (1000000, "1.000.000"),
        (1000000, "1,000,000"), (250, "250"), (60000, "60,000"),
    ])
    def test_same_number_drops_the_mention(self, name, mention):
        assert one_cardinal(name, mention).mention == ""

    @pytest.mark.parametrize("name,mention", [
        (2, "two"), (30, "thirties"), (21000000, "21 million"),
        ("0.5", "half"), (1, "No. 1"), (4, "four times"),
    ])
    def test_different_rendering_keeps_both(self, name, mention):
        pkg = one_cardinal(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_a_swallowed_operator_is_kept_verbatim(self):
        """Decision 2026-08-13: the mention is the literal surface text, untrimmed."""
        assert one_cardinal(1000000, "fewer than 1 million").mention == "fewer than 1 million"

    def test_mintakas_own_typo_survives_verbatim(self):
        """TRAIN carries 'more then 1000'. Not ours to correct."""
        assert one_cardinal(1000, "more then 1000").mention == "more then 1000"

    def test_the_normalised_value_is_what_is_carried_not_the_formatting(self):
        assert one_cardinal(7000, "7,000").name == 7000


class TestCardinalShape:
    def test_value_is_carried_verbatim_including_ranges(self):
        pkg = one_cardinal("40:49", "forties")
        assert pkg.name == "40:49"
        assert pkg.is_range is True

    def test_a_scalar_is_not_a_range(self):
        assert one_cardinal(30, "thirties").is_range is False

    def test_decades_have_two_encodings_and_both_survive(self):
        """Mintaka is inconsistent: 'thirties'->30 but 'forties'->'40:49'."""
        assert one_cardinal(30, "thirties").name == 30
        assert one_cardinal("40:49", "forties").name == "40:49"

    def test_order_is_preserved_and_nothing_is_merged(self):
        """27 questions carry two cardinals; no question repeats a VALUE."""
        pkgs = build_cardinal_packages([card(32, "32"), card(6, "6")])
        assert [p.name for p in pkgs] == [32, 6]

    def test_entities_and_other_literals_are_skipped(self):
        pkgs = build_cardinal_packages([
            ent("Q49", "North America", "North America"),
            card(3, "three"),
            ent(7, None, "seventh", "ordinal"),
        ])
        assert [p.name for p in pkgs] == [3]

    def test_a_missing_value_is_skipped_rather_than_rendered_empty(self):
        assert build_cardinal_packages([card(None, "some")]) == []

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_cardinal_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_cardinal_packages([]) == []
        assert build_cardinal_packages(None) == []


class TestCardinalRendering:
    def test_collapsed_line(self):
        assert render_cardinal_line(one_cardinal(7000, "7,000")) == "- (cardinal): 7000"

    def test_line_with_both_renderings(self):
        assert render_cardinal_line(one_cardinal(2, "two")) == '- (cardinal): 2 — mentioned as "two"'

    def test_the_word_cardinality_never_appears(self):
        """It means set size; a third of these are thresholds. See the module note."""
        line = render_cardinal_line(one_cardinal(4, "four"))
        assert "cardinality" not in line.lower()
        assert "(cardinal)" in line

    def test_range_renders_raw(self):
        assert render_cardinal_line(one_cardinal("40:49", "forties")) == (
            '- (cardinal): 40:49 — mentioned as "forties"')

    def test_it_uses_the_same_mentioned_as_convention_as_the_entity_lines(self):
        entity_line = render_entity_block(build_entity_packages(
            [ent("Q465469", "American League", "AL")])).splitlines()[1]
        cardinal_line = render_cardinal_line(one_cardinal(2, "two"))
        assert "— mentioned as " in entity_line
        assert "— mentioned as " in cardinal_line


class TestOrdinalPosition:
    """`-1` is a sentinel for "last", not a negative index."""

    def test_minus_one_renders_as_the_word_last(self):
        assert one_ordinal("-1", "last").position == "last"

    def test_the_sentinel_is_recognised_as_an_int_too(self):
        assert one_ordinal(-1, "last").position == "last"

    def test_the_raw_value_is_still_recoverable(self):
        """`position` is what gets printed; `name` keeps Mintaka's encoding."""
        pkg = one_ordinal("-1", "last")
        assert pkg.name == "-1"
        assert pkg.is_last is True

    def test_a_plain_position_is_its_own_number(self):
        pkg = one_ordinal(43, "43rd")
        assert pkg.position == "43"
        assert pkg.is_last is False


class TestOrdinalCollapse:
    """Compared against the RENDERED position, mirroring entity's label rule."""

    def test_sentinel_matching_last_collapses(self):
        assert one_ordinal("-1", "last").mention == ""

    def test_sentinel_is_case_insensitive(self):
        assert one_ordinal("-1", "Last").mention == ""

    def test_sentinel_meaning_final_keeps_both(self):
        """TRAIN b4367b5e: "When did the final Chinese dynasty end?"."""
        assert one_ordinal("-1", "final").mention == "final"

    @pytest.mark.parametrize("name,mention", [
        (7, "seventh"), (1, "first"), (1, "First"), (43, "43rd"),
        (121, "121st"), (2, "2nd"), (10, "tenth"),
    ])
    def test_every_plain_ordinal_keeps_both_forms(self, name, mention):
        """0 of 3,308 collapse: neither "first" nor "1st" parses as a number."""
        pkg = one_ordinal(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_a_bare_digit_mention_would_still_collapse(self):
        """Does not occur in the corpus, but the rule stays the same as cardinal's."""
        assert one_ordinal(7, "7").mention == ""


class TestOrdinalShape:
    def test_order_is_preserved_and_nothing_is_merged(self):
        pkgs = build_ordinal_packages([ordn(1, "first"), ordn(2, "second")])
        assert [p.name for p in pkgs] == [1, 2]

    def test_repeated_positions_are_both_kept(self):
        """Unlike entity, two ordinals are two constraints even at the same value."""
        assert len(build_ordinal_packages([ordn(1, "first"), ordn(1, "1st")])) == 2

    def test_other_types_are_skipped(self):
        pkgs = build_ordinal_packages([
            ent("Q49", "North America", "North America"),
            card(3, "three"),
            ordn(7, "seventh"),
        ])
        assert [p.name for p in pkgs] == [7]

    def test_the_mis_annotated_age_is_carried_not_corrected(self):
        """TRAIN: name=67 mention='the age of 67' is an age. Not ours to fix."""
        pkg = one_ordinal(67, "the age of 67")
        assert pkg.position == "67"
        assert pkg.mention == "the age of 67"

    def test_a_missing_value_is_skipped(self):
        assert build_ordinal_packages([ordn(None, "first")]) == []

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_ordinal_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_ordinal_packages([]) == []
        assert build_ordinal_packages(None) == []


class TestOrdinalRendering:
    def test_plain_ordinal_line(self):
        assert render_ordinal_line(one_ordinal(7, "seventh")) == (
            '- (ordinal): 7 — mentioned as "seventh"')

    def test_sentinel_line_collapsed(self):
        assert render_ordinal_line(one_ordinal("-1", "last")) == "- (ordinal): last"

    def test_sentinel_line_with_a_differing_mention(self):
        assert render_ordinal_line(one_ordinal("-1", "final")) == (
            '- (ordinal): last — mentioned as "final"')

    def test_the_raw_minus_one_never_reaches_the_prompt(self):
        """The whole reason the sentinel is decoded."""
        for mention in ("last", "final", "Last"):
            assert "-1" not in render_ordinal_line(one_ordinal("-1", mention))

    def test_it_shares_the_mentioned_as_convention(self):
        assert "— mentioned as " in render_ordinal_line(one_ordinal(43, "43rd"))


class TestDateShape:
    """`shape` classifies SYNTAX only — the bucket also holds ages and durations."""

    @pytest.mark.parametrize("name,expected", [
        (2020, "year"), ("2020", "year"),
        ("1990:1999", "range"), ("2003-05-15:2013-04-10", "range"),
        ("431 BC:404 BC", "range"),
        ("1974-10-11", "iso"), ("2020-08", "month"),
        ("XXXX-07-04", "recurring"), ("XXXX-07", "recurring"),
        ("218 BC", "bc"), ("2004 BC", "bc"),
        (53, "short"), (9, "short"), (814, "short"),
    ])
    def test_shape_classification(self, name, expected):
        assert one_date(name, "x").shape == expected

    def test_a_bc_range_is_a_range_not_a_bc(self):
        """Order matters: ':' is checked before the BC suffix."""
        assert one_date("2589 BC:2566 BC", "x").is_range is True

    def test_ages_and_years_share_the_short_shape_and_are_not_told_apart(self):
        """The point of not parsing: 11 is an age, 814 is a year, both 'short'."""
        assert one_date(11, "11").shape == one_date(814, "814").shape == "short"


class TestDateValue:
    """Raw, except the XXXX wildcard, which is spelled out."""

    @pytest.mark.parametrize("name", [
        2020, "1990:1999", "1974-10-11", "2020-08", "218 BC", 53,
    ])
    def test_everything_else_renders_raw(self, name):
        assert one_date(name, "x").value == str(name)

    def test_the_colon_range_is_carried_raw(self):
        """Decision 2026-08-14: `lo:hi` is Mintaka's convention, not ours to reformat."""
        assert one_date("1990:1999", "the 90s").value == "1990:1999"

    def test_wildcard_day_is_spelled_out(self):
        assert one_date("XXXX-07-04", "July 4th").value == "July 4 (any year)"

    def test_wildcard_month_only_is_spelled_out(self):
        assert one_date("XXXX-07", "July").value == "July (any year)"

    @pytest.mark.parametrize("name,expected", [
        ("XXXX-12-25", "December 25 (any year)"),
        ("XXXX-02-26", "February 26 (any year)"),
        ("XXXX-04-23", "April 23 (any year)"),
    ])
    def test_every_wildcard_in_the_corpus(self, name, expected):
        assert one_date(name, "x").value == expected

    def test_the_literal_xxxx_never_reaches_the_prompt(self):
        for name in ("XXXX-07-04", "XXXX-12-25", "XXXX-07"):
            assert "XXXX" not in one_date(name, "whatever").render_line()

    def test_the_raw_value_is_still_recoverable(self):
        pkg = one_date("XXXX-07-04", "July 4th")
        assert pkg.name == "XXXX-07-04"


class TestDateCollapse:
    """String OR numeric — '218 BC' matches its mention only as text."""

    @pytest.mark.parametrize("name,mention", [
        (2020, "2020"), ("2020", "2020"), ("218 BC", "218 BC"),
        ("480 BC", "480 BC"), ("1974-10-11", "1974-10-11"),
    ])
    def test_identical_renderings_collapse(self, name, mention):
        assert one_date(name, mention).mention == ""

    def test_bc_collapses_on_string_equality_which_numbers_cannot_do(self):
        """The reason this type needs a string test that `cardinal` did not."""
        assert as_number("218 BC") is None
        assert one_date("218 BC", "218 BC").mention == ""

    @pytest.mark.parametrize("name,mention", [
        ("1990:1999", "the 90s"), ("1974-10-11", "October 11, 1974"),
        ("2020-08", "August of 2020"), (21, "21st century"),
        ("1959:1980", "between 1959 and 1980"), (9, "nine"),
    ])
    def test_different_renderings_keep_both(self, name, mention):
        pkg = one_date(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_a_decoded_wildcard_never_collapses_against_its_surface_form(self):
        assert one_date("XXXX-07-04", "the Fourth of July").mention == "the Fourth of July"

    def test_case_is_folded(self):
        assert one_date("218 bc", "218 BC").mention == ""


class TestDateShapeQuirks:
    def test_the_mintaka_decade_bug_is_carried_not_corrected(self):
        """TRAIN c9674f96: '1980:1980' for "1980s" should be 1980:1989."""
        pkg = one_date("1980:1980", "1980s")
        assert pkg.name == "1980:1980"
        assert pkg.mention == "1980s"

    def test_two_dates_in_one_question_are_both_kept(self):
        """"president of Argentina from 1989 to 1999" — two ends of a span."""
        pkgs = build_date_packages([dt(1989, "1989"), dt(1999, "1999")])
        assert [p.name for p in pkgs] == [1989, 1999]

    def test_other_types_are_skipped(self):
        pkgs = build_date_packages([
            ent("Q49", "North America", "North America"),
            card(3, "three"), ordn(7, "seventh"), dt(2020, "2020"),
        ])
        assert [p.name for p in pkgs] == [2020]

    def test_a_missing_value_is_skipped(self):
        assert build_date_packages([dt(None, "2020")]) == []

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_date_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_date_packages([]) == []
        assert build_date_packages(None) == []


class TestDateRendering:
    def test_collapsed_year(self):
        assert render_line_of(one_date(2020, "2020")) == "- (date): 2020"

    def test_range_with_a_mention(self):
        assert render_line_of(one_date("1990:1999", "the 90s")) == (
            '- (date): 1990:1999 — mentioned as "the 90s"')

    def test_wildcard_with_a_mention(self):
        assert render_line_of(one_date("XXXX-12-25", "Christmas Day")) == (
            '- (date): December 25 (any year) — mentioned as "Christmas Day"')

    def test_bc_collapsed(self):
        assert render_line_of(one_date("480 BC", "480 BC")) == "- (date): 480 BC"


class TestQuantity:
    """The unit lives only in the mention, so the mention is never dropped."""

    @pytest.mark.parametrize("name,mention", [
        (20000, "20,000 feet"), (5000, "5,000 km"), (0.78, "just 0.78 square miles"),
        (1500000, "1.5-million-acre"), (1000, "one thousand yards"),
        (11.33, "11.33 meter"), (4, "4'"), (8, '8"'),
    ])
    def test_the_mention_is_always_kept(self, name, mention):
        pkg = one_quantity(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_the_compound_measurement_stays_two_lines(self):
        """TRAIN 9fcfcbda: "6 foot six inches" — same value, different units."""
        pkgs = build_quantity_packages([qty(6, "6 foot"), qty(6, "six inches")])
        assert len(pkgs) == 2
        assert [p.mention for p in pkgs] == ["6 foot", "six inches"]

    def test_feet_and_inches_of_one_height_are_not_merged(self):
        """TRAIN 36bed235: 4'8" arrives as two mentions."""
        pkgs = build_quantity_packages([qty(8, '8"'), qty(4, "4'")])
        assert [(p.name, p.mention) for p in pkgs] == [(8, '8"'), (4, "4'")]

    def test_the_scale_word_normalisation_is_preserved(self):
        """Unlike cardinal, the value adds information on ~15% of rows."""
        assert one_quantity(1500000, "1.5-million-acre").name == 1500000

    def test_decimal_values_survive_as_given(self):
        assert one_quantity("2.2", "2.2 meters").name == "2.2"

    def test_unit_kind_is_declared(self):
        assert QuantityPackage.unit_kind == "physical"

    def test_a_missing_value_is_skipped(self):
        assert build_quantity_packages([qty(None, "6 foot")]) == []

    def test_other_types_are_skipped(self):
        pkgs = build_quantity_packages([
            card(6, "six"), qty(6, "6 foot"), dt(2020, "2020"),
        ])
        assert [p.mention for p in pkgs] == ["6 foot"]

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_quantity_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_quantity_packages([]) == []
        assert build_quantity_packages(None) == []

    def test_the_collapse_rule_is_uniform_not_disabled(self):
        """Never satisfied by the corpus, but the rule is the same one."""
        assert one_quantity(6, "6").mention == ""


class TestQuantityRendering:
    def test_style_a_value_first(self):
        assert render_line_of(one_quantity(20000, "20,000 feet")) == (
            '- (quantity): 20000 — mentioned as "20,000 feet"')

    def test_it_matches_the_convention_of_every_other_type(self):
        lines = [
            render_line_of(one_quantity(6, "6 foot")),
            render_line_of(one_cardinal(2, "two")),
            render_line_of(one_ordinal(43, "43rd")),
            render_line_of(one_date("1990:1999", "the 90s")),
        ]
        for line in lines:
            assert line.startswith("- (")
            assert "— mentioned as " in line

    def test_a_mention_containing_a_double_quote_still_renders(self):
        """8" — the inches mark collides with the quoting."""
        assert '8"' in render_line_of(one_quantity(8, '8"'))


class TestMoney:
    """Every one of the 16 needs its value resolved AND its currency kept."""

    @pytest.mark.parametrize("name,mention", [
        (200000000, "$200 million"),
        (1400000000, "$1.4 billion"),
        (170000000, "US $170 million"),
        (2000000000, "2 billion USD"),
        (1000000000, "1 billion dollars"),
        (1000000, "a million dollars"),
        (100000000, "one hundred million dollars"),
        (40000000, "40 million dollar"),
    ])
    def test_six_surface_forms_all_resolve_and_all_keep_their_mention(self, name, mention):
        pkg = one_money(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_the_same_amount_written_two_ways_stays_two_mentions(self):
        """'2 billion USD' and '$2 billion' both -> 2000000000."""
        a = one_money(2000000000, "2 billion USD")
        b = one_money(2000000000, "$2 billion")
        assert a.name == b.name
        assert a.mention != b.mention

    def test_the_gold_piece_row_is_carried_like_any_other(self):
        """TRAIN: '12,000 gold piece' is not a modern currency. Not ours to fix."""
        pkg = one_money(12000, "12,000 gold piece")
        assert pkg.name == 12000
        assert pkg.mention == "12,000 gold piece"

    def test_unit_kind_is_declared(self):
        assert MoneyPackage.unit_kind == "currency"

    def test_the_currency_never_survives_in_the_value_alone(self):
        """Dropping the mention would leave a bare integer with no unit."""
        line = render_line_of(one_money(200000000, "$200 million"))
        assert "$" in line and "200000000" in line

    def test_a_missing_value_is_skipped(self):
        assert build_money_packages([mon(None, "$5")]) == []

    def test_other_types_are_skipped(self):
        pkgs = build_money_packages([
            qty(6, "6 foot"), mon(200000000, "$200 million"), card(3, "three"),
        ])
        assert [p.name for p in pkgs] == [200000000]

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_money_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_money_packages([]) == []
        assert build_money_packages(None) == []


class TestMoneyRendering:
    def test_style_a_value_first(self):
        assert render_line_of(one_money(1400000000, "$1.4 billion")) == (
            '- (money): 1400000000 — mentioned as "$1.4 billion"')

    def test_it_shares_the_family_rendering_with_quantity(self):
        money = render_line_of(one_money(200000000, "$200 million"))
        quantity = render_line_of(one_quantity(20000, "20,000 feet"))
        assert money.startswith("- (money): ") and quantity.startswith("- (quantity): ")
        assert "— mentioned as " in money and "— mentioned as " in quantity


class TestPercent:
    """Kept for the HEDGE, not for a unit — the unit here never varies."""

    @pytest.mark.parametrize("name,mention", [
        (100, "100%"), (70, "70%"), ("0.1", "0.1%"),
        (9, "nine per cent"), (2, "two percent"),
    ])
    def test_value_and_mention_both_survive(self, name, mention):
        pkg = one_percent(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    @pytest.mark.parametrize("name,mention,hedge", [
        (90, "approximately 90%", "approximately"),
        (98, "about 98%", "about"),
        (29, "approximately 29%", "approximately"),
    ])
    def test_the_hedge_survives_in_the_mention(self, name, mention, hedge):
        """"approximately 29%" is not the same claim as "29%"."""
        assert hedge in one_percent(name, mention).render_line()

    def test_dropping_the_mention_would_sharpen_a_vague_number(self):
        """The failure this type guards against, stated as a test."""
        pkg = one_percent(90, "approximately 90%")
        assert pkg.mention != ""

    def test_unit_kind_is_declared(self):
        assert PercentPackage.unit_kind == "proportion"

    def test_a_decimal_percentage_survives_as_given(self):
        assert one_percent("0.1", "0.1%").name == "0.1"

    def test_a_missing_value_is_skipped(self):
        assert build_percent_packages([pct(None, "50%")]) == []

    def test_other_types_are_skipped(self):
        pkgs = build_percent_packages([
            mon(200000000, "$200 million"), pct(70, "70%"), qty(6, "6 foot"),
        ])
        assert [p.name for p in pkgs] == [70]

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_percent_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_percent_packages([]) == []
        assert build_percent_packages(None) == []


class TestPercentRendering:
    def test_style_a_value_first(self):
        assert render_line_of(one_percent(29, "approximately 29%")) == (
            '- (percent): 29 — mentioned as "approximately 29%"')

    def test_bare_percentage(self):
        assert render_line_of(one_percent(100, "100%")) == (
            '- (percent): 100 — mentioned as "100%"')


class TestTime:
    """Six of nine rows share name=1, meaning two different things."""

    def test_one_hour_and_one_night_share_a_value_and_are_told_apart_only_by_mention(self):
        an_hour = one_time(1, "an hour")
        one_night = one_time(1, "one night")
        assert an_hour.name == one_night.name == 1
        assert an_hour.mention != one_night.mention

    @pytest.mark.parametrize("name,mention", [
        (1, "one night"), (1, "an hour"), (20, "20 minutes"),
        (24, "24-hour"), (24, "its first 24 hours"),
    ])
    def test_every_corpus_row_keeps_its_mention(self, name, mention):
        pkg = one_time(name, mention)
        assert pkg.name == name
        assert pkg.mention == mention

    def test_the_same_value_with_two_framings_stays_distinct(self):
        """24 is both '24-hour' (a period) and 'its first 24 hours' (a window)."""
        a, b = one_time(24, "24-hour"), one_time(24, "its first 24 hours")
        assert a.mention != b.mention

    def test_unit_kind_is_declared(self):
        assert TimePackage.unit_kind == "duration"

    def test_a_missing_value_is_skipped(self):
        assert build_time_packages([tm(None, "an hour")]) == []

    def test_other_types_are_skipped(self):
        pkgs = build_time_packages([pct(70, "70%"), tm(20, "20 minutes"), card(1, "one")])
        assert [p.mention for p in pkgs] == ["20 minutes"]

    def test_an_undeclared_type_raises_here_too(self):
        with pytest.raises(UnknownEntityType):
            build_time_packages([ent(1, None, "x", "duration")])

    def test_empty_and_none_input(self):
        assert build_time_packages([]) == []
        assert build_time_packages(None) == []


class TestTimeRendering:
    def test_style_a_value_first(self):
        assert render_line_of(one_time(20, "20 minutes")) == (
            '- (time): 20 — mentioned as "20 minutes"')

    def test_the_two_meanings_of_one_render_differently(self):
        assert render_line_of(one_time(1, "an hour")) != render_line_of(one_time(1, "one night"))


class TestSurfaceName:
    """`surface` / `Question.entity_mentions` — the name prepended to every
    rendered statement (`[Academy Award] winner: ...`).

    Decided 2026-08-16: the question's own wording, not the canonical label, for
    the reason `wikidata_pool.build_pool` documents — the prepend exists so the
    ranker sees the question's phrasing echoed in the candidate line. These pin
    the rule and the equivalence with the transform it replaces.
    """

    def _pkg(self, rows):
        return build_entity_packages(rows)[0]

    def test_surface_is_the_mention_when_it_differs_from_the_label(self):
        p = self._pkg([ent("Q19020", "Academy Awards", "Academy Award")])
        assert p.surface == "Academy Award"

    def test_surface_is_the_label_when_the_question_says_the_same_thing(self):
        p = self._pkg([ent("Q2263", "Tom Hanks", "Tom Hanks")])
        assert p.surface == "Tom Hanks"

    def test_surface_takes_the_earliest_mention_not_the_first_annotated(self):
        """Mintaka's order is not the question's — 59.3% of multi-mention rows."""
        rows = [
            {"entityType": "entity", "name": "Q30", "label": "United States of America",
             "mention": "Union", "span": [42, 47]},
            {"entityType": "entity", "name": "Q30", "label": "United States of America",
             "mention": "U.S.", "span": [9, 13]},
        ]
        assert self._pkg(rows).surface == "U.S."

    def test_a_label_identical_mention_still_counts_for_surface(self):
        """The Japan case, and the reason `surface` is not `mentions[0]`.

        "Did the United States declare war on Japan following the Japanese
        attack on Pearl Harbor?" — Q17 is mentioned as 'Japan' (span 37) and
        'Japanese' (57). 'Japan' equals the label so it is absent from the
        DISPLAY list, but it is what the question says first.
        """
        rows = [
            {"entityType": "entity", "name": "Q17", "label": "Japan",
             "mention": "Japanese", "span": [57, 65]},
            {"entityType": "entity", "name": "Q17", "label": "Japan",
             "mention": "Japan", "span": [37, 42]},
        ]
        p = self._pkg(rows)
        assert p.mentions == ("Japanese",)   # display: label-differing forms only
        assert p.surface == "Japan"          # prepend: what the question says first

    def test_question_exposes_mentions_and_labels_separately(self):
        q = parse_question({
            "id": "x", "question": "Did Gone With The Wind win an Academy Award?",
            "questionEntity": [
                {"entityType": "entity", "name": "Q2875",
                 "label": "Gone with the Wind", "mention": "Gone With The Wind",
                 "span": [4, 22]},
            ],
            "answer": {"answerType": "boolean", "answer": [True], "mention": "Yes"},
        })
        assert q.entity_mentions == ["Gone With The Wind"]   # -> the pipelines
        assert q.entity_names == ["Gone with the Wind"]      # -> the entity block
        assert q.qids == ["Q2875"]


class TestReadingOrder:
    """
    The block is sorted by `span`, not by Mintaka's annotation order.

    Measured: 7,225 of 12,192 multi-mention questions (59.3%) are annotated out
    of reading order, so without this the constraint and the thing it constrains
    end up in different halves of the block.
    """

    def test_mixed_types_interleave_by_position(self):
        """TRAIN a9011ddf: "the seventh tallest mountain in North America"."""
        qe = [
            {"entityType": "entity", "name": "Q49", "label": "North America",
             "mention": "North America", "span": [40, 53]},
            {"entityType": "ordinal", "name": 7, "label": None,
             "mention": "seventh", "span": [12, 19]},
        ]
        block = render_question_block(qe)
        assert block.splitlines()[1].startswith("- (ordinal)")
        assert block.splitlines()[2].startswith("- Q49")

    def test_annotation_order_does_not_survive(self):
        qe = [
            {"entityType": "cardinal", "name": 6, "label": None,
             "mention": "6", "span": [30, 31]},
            {"entityType": "cardinal", "name": 32, "label": None,
             "mention": "32", "span": [7, 9]},
        ]
        assert [p.name for p in build_packages(qe)] == [32, 6]

    def test_a_merged_entity_sorts_at_its_earliest_mention(self):
        """Q41254 appears twice; it belongs where the reader first meets it."""
        qe = [
            {"entityType": "entity", "name": "Q41254", "label": "Grammy Award",
             "mention": "Grammy", "span": [60, 66]},
            {"entityType": "ordinal", "name": 1, "label": None,
             "mention": "first", "span": [30, 35]},
            {"entityType": "entity", "name": "Q41254", "label": "Grammy Award",
             "mention": "Grammy Awards", "span": [4, 17]},
        ]
        pkgs = build_packages(qe)
        assert [p.name for p in pkgs] == ["Q41254", 1]
        assert pkgs[0].span == (4, 17)
        # ⚠️ SPAN ORDER, NOT ANNOTATION ORDER (changed 2026-08-16). This used to
        # assert ("Grammy", "Grammy Awards") -- Mintaka's order -- but the
        # question says "Grammy Awards" (span 4) before "Grammy" (span 60), and
        # the same reading-order argument that sorts the block sorts these.
        # It is load-bearing now: `surface` returns mentions[0] and that is the
        # name prepended to every rendered statement.
        assert pkgs[0].mentions == ("Grammy Awards", "Grammy")
        assert pkgs[0].surface == "Grammy Awards"

    def test_rows_without_a_usable_span_sort_last(self):
        """2 rows in 20,000 have no usable span; they must not sort to the front."""
        qe = [
            {"entityType": "cardinal", "name": 9, "label": None, "mention": "9"},
            {"entityType": "cardinal", "name": 1, "label": None,
             "mention": "1", "span": [5, 6]},
        ]
        assert [p.name for p in build_packages(qe)] == [1, 9]

    def test_a_malformed_span_is_treated_as_absent_rather_than_raising(self):
        qe = [{"entityType": "cardinal", "name": 3, "label": None,
               "mention": "3", "span": ["x", "y"]}]
        [pkg] = build_packages(qe)
        assert pkg.span is None

    def test_pending_types_contribute_nothing_to_the_block(self):
        """Picks whatever is still pending, so it does not need editing each round.

        Skips once the registry is complete — at that point there is no such
        thing as a recognised-but-unhandled type, which is the goal.
        """
        if not PENDING_TYPES:
            pytest.skip("every type has a handler; nothing can be pending")
        pending = sorted(PENDING_TYPES)[0]
        qe = [
            {"entityType": pending, "name": 4242, "label": None,
             "mention": "4242 somethings", "span": [64, 81]},
            {"entityType": "entity", "name": "Q24871", "label": "Avatar",
             "mention": "Avatar's", "span": [4, 12]},
        ]
        block = render_question_block(qe)
        assert "4242" not in block
        assert "Q24871" in block

    def test_empty_input_renders_to_empty_string(self):
        assert render_question_block([]) == ""
        assert render_question_block(None) == ""

    def test_header_appears_exactly_once(self):
        qe = [
            {"entityType": "entity", "name": "Q49", "label": "North America",
             "mention": "North America", "span": [40, 53]},
            {"entityType": "ordinal", "name": 7, "label": None,
             "mention": "seventh", "span": [12, 19]},
        ]
        assert render_question_block(qe).count("Question entities:") == 1


class TestLoader:
    """`load_questions` reads RAW Mintaka and returns the question side only."""

    RAW = [{
        "id": "q1",
        "question": "What is the seventh tallest mountain in North America?",
        "category": "geography",
        "complexityType": "ordinal",
        "questionEntity": [
            {"entityType": "entity", "name": "Q49", "label": "North America",
             "mention": "North America", "span": [40, 53]},
            {"entityType": "ordinal", "name": 7, "label": None,
             "mention": "seventh", "span": [12, 19]},
        ],
        "answer": {"answerType": "entity", "mention": "Mount Lucania",
                   "answer": [{"name": "Q1153188", "label": {"en": "Mount Lucania"}}]},
    }]

    def _write(self, tmp_path, rows, name="q.json"):
        import json
        p = tmp_path / name
        p.write_text(json.dumps(rows), encoding="utf-8")
        return p

    def test_it_parses_a_raw_record(self, tmp_path):
        [q] = load_questions(self._write(tmp_path, self.RAW))
        assert q.id == "q1"
        assert q.text.startswith("What is the seventh")
        assert q.category == "geography"
        assert q.complexity == "ordinal"

    def test_entities_are_reading_ordered(self, tmp_path):
        [q] = load_questions(self._write(tmp_path, self.RAW))
        assert q.entities[0].entity_type == "ordinal"
        assert q.entities[1].entity_type == "entity"

    def test_qids_and_names_are_parallel_and_linked_only(self, tmp_path):
        [q] = load_questions(self._write(tmp_path, self.RAW))
        assert q.qids == ["Q49"]
        assert q.entity_names == ["North America"]
        assert len(q.qids) == len(q.entity_names)

    def test_the_prompt_block_is_the_rendered_prompt_section(self, tmp_path):
        [q] = load_questions(self._write(tmp_path, self.RAW))
        block = q.prompt_block(BLOCK_MODE_ALL)
        assert block.startswith("Question entities:")
        assert "(ordinal)" in block and "Q49" in block

    def test_entity_mode_drops_the_literals(self, tmp_path):
        """The second declared prompt arm — see prompts.ENTITY_BLOCK_MODE."""
        [q] = load_questions(self._write(tmp_path, self.RAW))
        block = q.prompt_block(BLOCK_MODE_ENTITY)
        assert "Q49" in block
        assert "(ordinal)" not in block

    def test_an_unlinked_mention_reaches_NEITHER_the_block_nor_the_qids(self, tmp_path):
        """Decision 3b, 2026-08-16.

        A mention with no QID has no label and nothing fetched for it. A line
        naming it would report the pipeline's coverage gap rather than describe
        the question, and the effect would land on abstention — a headline
        metric. It stays available on `q.entities` for analysis.
        """
        rows = [dict(self.RAW[0], questionEntity=[
            {"entityType": "entity", "name": None, "label": None,
             "mention": "Cooke Maroney", "span": [0, 13]}])]
        [q] = load_questions(self._write(tmp_path, rows))
        assert q.qids == []
        for mode in (BLOCK_MODE_ALL, BLOCK_MODE_ENTITY):
            assert q.prompt_block(mode) == ""
        # Still parsed and reachable — excluded from the PROMPT, not dropped.
        assert [p.surface for p in q.entities] == ["Cooke Maroney"]

    def test_an_unknown_mode_raises(self, tmp_path):
        [q] = load_questions(self._write(tmp_path, self.RAW))
        with pytest.raises(ValueError, match="unknown block mode"):
            q.prompt_block("everything")

    def test_ids_filter_preserves_file_order(self, tmp_path):
        rows = [dict(self.RAW[0], id=f"q{i}") for i in range(5)]
        got = load_questions(self._write(tmp_path, rows), ids={"q3", "q1"})
        assert [q.id for q in got] == ["q1", "q3"]

    def test_an_empty_file_loads_to_nothing(self, tmp_path):
        assert load_questions(self._write(tmp_path, [])) == []

    def test_a_non_mintaka_file_raises_rather_than_yielding_empty_questions(self, tmp_path):
        """
        The failure this replaces was silent: `scientific_figures.json` uses
        `entity_qid` (singular), was read as raw Mintaka, and produced twelve
        rows with no entities and no gold. `run_eval`'s usage line names it.
        """
        rows = [{"id": "sf_001", "question": "x", "entity_qid": "Q1",
                 "expected_answer": "y"}]
        with pytest.raises(NotRawMintakaError):
            load_questions(self._write(tmp_path, rows))

    def test_an_old_flattened_sample_also_raises(self, tmp_path):
        rows = [{"id": "a", "question": "x", "entity_qids": ["Q1"],
                 "answer_type": "entity", "expected_answer": "y"}]
        with pytest.raises(NotRawMintakaError):
            load_questions(self._write(tmp_path, rows))

    def test_load_question_ids_accepts_a_bare_id_list(self, tmp_path):
        assert load_question_ids(self._write(tmp_path, ["a", "b"])) == {"a", "b"}

    def test_load_question_ids_accepts_an_old_flattened_sample(self, tmp_path):
        """A stale sample can still say WHICH questions it selected."""
        rows = [{"id": "a", "entity_qids": []}, {"id": "b", "entity_qids": []}]
        assert load_question_ids(self._write(tmp_path, rows)) == {"a", "b"}


class TestStratificationVocabularies:
    """Moved here from the deleted load_questions.py — they describe a QUESTION."""

    def test_categories(self):
        assert set(CATEGORIES) == {
            "history", "movies", "music", "videogames",
            "sports", "books", "geography", "politics"}

    def test_complexity_types_are_qualified(self):
        assert all(c.startswith("c_") for c in COMPLEXITY_TYPES)
        assert len(COMPLEXITY_TYPES) == 9

    def test_question_entity_types_are_qualified_and_cover_all_eight(self):
        assert all(t.startswith("q_") for t in QUESTION_ENTITY_TYPES)
        assert {raw_type(t) for t in QUESTION_ENTITY_TYPES} == set(ALL_TYPES)

    @pytest.mark.parametrize("qualified,bare", [
        ("q_entity", "entity"), ("c_count", "count"), ("a_date", "date"),
    ])
    def test_raw_type_strips_the_qualifier(self, qualified, bare):
        assert raw_type(qualified) == bare

    def test_raw_type_is_idempotent(self):
        assert raw_type(raw_type("q_entity")) == "entity"
        assert raw_type("entity") == "entity"


class TestAnswerSideNeverLeaks:
    """
    ⚠️ THE BOUNDARY THIS MODULE EXISTS TO HOLD.

    `answer.supportingEnt` on a count question IS the answer's member list, and
    `answer.mention` is the gold string. The package is rendered into the
    answering prompt of all four configs, so either reaching it would hand the
    model its own gold and every retrieval number would become meaningless.

    The defence is structural -- `build_entity_packages` accepts only the
    `questionEntity` list -- and these tests pin that it stays structural.
    """

    def test_builder_ignores_an_answer_object_smuggled_in_beside_the_mentions(self):
        rows = [
            ent("Q49", "North America", "North America"),
            {"entityType": "entity", "name": "Q999", "label": "Quincy Jones",
             "mention": "Quincy Jones", "supportingEnt": [{"name": "Q1"}],
             "supportingNum": 27, "answerType": "entity"},
        ]
        pkgs = build_entity_packages(rows)
        rendered = render_entity_block(pkgs)
        assert "supportingEnt" not in rendered
        assert "supportingNum" not in rendered
        assert "27" not in rendered
        assert "answerType" not in rendered

    def test_package_fields_are_a_closed_set(self):
        """A new field cannot be added to the dataclass without this test failing."""
        assert set(EntityPackage.__dataclass_fields__) == {
            "name", "entity_type", "label", "mentions", "span", "first_mention"
        }

    def test_literal_package_fields_are_closed_sets_too(self):
        for cls in (CardinalPackage, OrdinalPackage):
            assert set(cls.__dataclass_fields__) == {
                "name", "entity_type", "mention", "span"
            }

    def test_serialised_rows_carry_no_other_keys(self):
        rows = packages_to_rows(build_entity_packages([ent("Q49", "North America", "NA")]))
        assert set(rows[0]) == {"name", "entity_type", "label", "mentions",
                                "span", "first_mention"}
