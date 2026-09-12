"""
Unit tests for src/retrieval/statement.py — the statement representation.

These pin the properties the rebuild depends on, not the exact wording:

1. **Hashability.** `_clean_statements()` dedups via `dict.fromkeys`, so an
   unhashable field anywhere in the tree silently breaks deduplication. A dict
   in `SnakValue.raw` would do it, which is why `raw` is a `str`.
2. **Qualifier order.** SPARQL guarantees no row order. Unsorted qualifiers make
   the same statement render two ways, so both survive dedup and the top-k
   spends a slot on a duplicate.
3. **`somevalue` / `novalue` are visible.** Keeping `snaktype` buys nothing if
   the renderer collapses it to an empty string — "we know there is none" would
   look exactly like "we have no data", which is the distinction the field
   exists to preserve.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_statement.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval.statement import (
    NO_VALUE, UNKNOWN_VALUE, Qualifier, SnakValue, Statement, sort_qualifiers,
)


def _v(label, **kw):
    return SnakValue(snaktype="value", datatype="wikibase-item", label=label, **kw)


def _oscar(qualifiers=()):
    """The worked example throughout the docs: one of Hanks's two Best Actor wins."""
    return Statement(
        subject_id="Q2263", subject_label="Tom Hanks",
        property_id="P166", property_label="award received",
        value=_v("Academy Award for Best Actor", id="Q103916"),
        rank="normal", qualifiers=tuple(qualifiers),
        direction="outgoing", source_entity_id="Q2263",
        statement_id="Q2263-8303B10F-909B-4CE9-B369-60A07E5091E9",
    )


class TestRender:
    def test_bare_statement_matches_the_established_line_format(self):
        """`[Subject] property: value` — the shape the old pipeline produced."""
        assert _oscar().render() == "[Tom Hanks] award received: Academy Award for Best Actor"

    def test_qualifiers_are_labelled_and_appended(self):
        s = _oscar([
            Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994-03-21")),
            Qualifier("P1686", "for work", _v("Philadelphia", id="Q204057")),
        ])
        assert s.render() == (
            "[Tom Hanks] award received: Academy Award for Best Actor "
            "(point in time: 1994-03-21; for work: Philadelphia)"
        )

    def test_the_two_oscars_render_differently(self):
        """The whole point: one truthy edge became two distinguishable facts."""
        a = _oscar([Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994-03-21"))])
        b = _oscar([Qualifier("P585", "point in time", SnakValue("value", "time", raw="1995-03-27"))])
        assert a.render() != b.render()

    def test_subject_prefix_is_present(self):
        """Bare `pred: obj` strings caused mass abstention — the model could not attribute."""
        assert _oscar().render().startswith("[Tom Hanks] ")

    def test_qualifier_property_names_are_kept(self):
        """A bare "(1994)" is ambiguous, and the ranker embeds this string."""
        s = _oscar([Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994"))])
        assert "point in time: 1994" in s.render()


class TestNonValues:
    def test_somevalue_renders_visibly(self):
        v = SnakValue(snaktype="somevalue", datatype="wikibase-item")
        assert v.render() == UNKNOWN_VALUE
        assert v.render().strip()

    def test_novalue_renders_visibly(self):
        """'We know there is none' must not look like 'we have no data'."""
        v = SnakValue(snaktype="novalue", datatype="wikibase-item")
        assert v.render() == NO_VALUE
        assert v.render() != UNKNOWN_VALUE

    def test_a_value_is_never_rendered_empty(self):
        """An empty render would read as a missing field in the context block."""
        for v in (SnakValue("value", "string"),
                  SnakValue("somevalue", "time"),
                  SnakValue("novalue", "quantity")):
            assert v.render().strip()

    def test_raw_is_used_when_no_label_resolved(self):
        assert SnakValue("value", "time", raw="1994-03-21").render() == "1994-03-21"

    def test_label_wins_over_raw(self):
        assert SnakValue("value", "wikibase-item", label="Philadelphia",
                         raw="Q204057").render() == "Philadelphia"

    def test_non_values_survive_into_a_statement_line(self):
        s = Statement("Q42", "Douglas Adams", "P22", "father",
                      SnakValue("somevalue", "wikibase-item"), "normal")
        assert s.render() == f"[Douglas Adams] father: {UNKNOWN_VALUE}"


class TestHashability:
    def test_statement_is_hashable(self):
        """_clean_statements() dedups via dict.fromkeys; an unhashable field breaks it silently."""
        assert len({_oscar(), _oscar()}) == 1

    def test_hashable_with_qualifiers(self):
        q = (Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994")),)
        assert len({_oscar(q), _oscar(q)}) == 1

    def test_dict_fromkeys_dedup_works(self):
        """The exact mechanism _clean uses."""
        assert len(list(dict.fromkeys([_oscar(), _oscar(), _oscar()]))) == 1

    def test_statements_differing_only_in_qualifiers_are_distinct(self):
        a = _oscar([Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994"))])
        b = _oscar([Qualifier("P585", "point in time", SnakValue("value", "time", raw="1995"))])
        assert len({a, b}) == 2, "the two Oscars must not collapse"


class TestQualifierOrder:
    def test_sort_is_deterministic_across_permutations(self):
        a = Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994"))
        b = Qualifier("P1686", "for work", _v("Philadelphia"))
        assert sort_qualifiers([a, b]) == sort_qualifiers([b, a])

    def test_permuted_qualifiers_render_identically_once_sorted(self):
        """Otherwise the same fact survives dedup twice and wastes a top-k slot."""
        a = Qualifier("P585", "point in time", SnakValue("value", "time", raw="1994"))
        b = Qualifier("P1686", "for work", _v("Philadelphia"))
        assert _oscar(sort_qualifiers([a, b])).render() == _oscar(sort_qualifiers([b, a])).render()

    def test_repeated_qualifier_property_is_kept_not_collapsed(self):
        """Real case: place of birth Concord carries TWO 'located in' qualifiers."""
        q = sort_qualifiers([
            Qualifier("P131", "located in", _v("California")),
            Qualifier("P131", "located in", _v("Contra Costa County")),
        ])
        assert len(q) == 2
        rendered = Statement("Q2263", "Tom Hanks", "P19", "place of birth",
                             _v("Concord"), "normal", q).render()
        assert "California" in rendered and "Contra Costa County" in rendered

    def test_sorting_tolerates_non_values(self):
        q = sort_qualifiers([
            Qualifier("P585", "point in time", SnakValue("somevalue", "time")),
            Qualifier("P1686", "for work", _v("Philadelphia")),
        ])
        assert len(q) == 2


class TestRankAndMetadata:
    def test_deprecated_is_flagged(self):
        assert Statement("Q1", "x", "P1", "p", _v("v"), "deprecated").is_deprecated
        assert not Statement("Q1", "x", "P1", "p", _v("v"), "normal").is_deprecated

    def test_statement_id_is_never_rendered(self):
        """A GUID in the prompt is noise; it is metadata for drift detection."""
        assert "8303B10F" not in _oscar().render()

    def test_source_entity_and_direction_are_carried_not_inferred(self):
        s = _oscar()
        assert s.source_entity_id == "Q2263"
        assert s.direction == "outgoing"

    def test_same_fact_from_two_paths_differs_by_direction(self):
        """Content dedup will NOT collapse these — statement_id identifies them."""
        fwd = _oscar()
        rev = Statement(**{**_oscar().__dict__, "direction": "incoming",
                           "source_entity_id": "Q204057"})
        assert fwd != rev
        assert fwd.statement_id == rev.statement_id


class TestUnits:
    """
    A bare quantity is not the fact. Wikidata says *600,000,000 United States
    dollars*; "600000000" alone is a number the reader must guess at. Measured:
    23 of 108 quantity statements in the DEV sample carry a real unit.
    """

    def test_unit_is_appended(self):
        v = SnakValue("value", "quantity", raw="600000000", unit="United States dollar")
        assert v.render() == "600000000 United States dollar"

    def test_no_unit_renders_the_bare_number(self):
        """Counts and ordinals are genuinely unitless — most quantities are."""
        assert SnakValue("value", "quantity", raw="7").render() == "7"

    def test_unit_survives_into_a_statement_line(self):
        s = Statement("Q1", "Titanic", "P2130", "cost",
                      SnakValue("value", "quantity", raw="200000000",
                                unit="United States dollar"), "normal")
        assert s.render().endswith("cost: 200000000 United States dollar")

    def test_a_unit_on_a_qualifier_renders_too(self):
        q = Qualifier("P2130", "cost", SnakValue("value", "quantity",
                                                 raw="5", unit="kilogram"))
        assert q.render() == "cost: 5 kilogram"


class TestTimePrecision:
    """
    🔴 WITHOUT PRECISION WE ASSERT A DAY WIKIDATA DOES NOT CLAIM. A year-precision
    date is stored as 1 January, so rendering the stored string turns "1994" into
    "1994-01-01". Measured: 16 of 36 time values in the DEV sample are year
    precision — roughly 44% of dates were being over-specified.
    """

    def _t(self, raw, precision):
        return SnakValue("value", "time", raw=raw, precision=precision).render()

    def test_day_precision_keeps_the_day(self):
        assert self._t("1994-03-21T00:00:00Z", 11) == "1994-03-21"

    def test_month_precision_drops_the_day(self):
        assert self._t("1994-03-01T00:00:00Z", 10) == "1994-03"

    def test_year_precision_drops_the_fabricated_january_first(self):
        assert self._t("1994-01-01T00:00:00Z", 9) == "1994"

    def test_decade_precision_says_decade(self):
        assert self._t("1990-01-01T00:00:00Z", 8) == "1990s"

    def test_coarser_than_decade_is_hedged_not_invented(self):
        """
        The stored year is the START of a century, not a claim about that year.
        Naming it plainly would be the same fabrication precision prevents.
        """
        assert self._t("1901-01-01T00:00:00Z", 7) == "circa 1901"

    def test_missing_precision_keeps_the_old_behaviour(self):
        """A value cached before the precision join must still render."""
        assert self._t("1994-01-01T00:00:00Z", None) == "1994-01-01"

    # BCE dates (2026-08-18): astronomical year numbering carries a leading
    # "-", which used to fail the digits gate so truncation never applied —
    # a year-precision BCE date asserted "1 January". 66 cached snaks.

    def test_bce_year_precision_truncates_with_the_sign(self):
        assert self._t("-0500-01-01T00:00:00Z", 9) == "-0500"

    def test_bce_day_precision_keeps_the_full_date(self):
        assert self._t("-0044-03-15T00:00:00Z", 11) == "-0044-03-15"

    def test_bce_month_precision(self):
        assert self._t("-0044-03-01T00:00:00Z", 10) == "-0044-03"

    def test_bce_decade_precision(self):
        assert self._t("-0510-01-01T00:00:00Z", 8) == "-0510s"

    def test_bce_coarser_than_decade_is_hedged(self):
        assert self._t("-0500-01-01T00:00:00Z", 7) == "circa -0500"

    def test_precision_applies_inside_a_qualifier(self):
        s = Statement("Q2263", "Tom Hanks", "P166", "award received",
                      _v("Academy Award for Best Actor"), "normal",
                      (Qualifier("P585", "point in time",
                                 SnakValue("value", "time",
                                           raw="+1994-01-01T00:00:00Z", precision=9)),))
        assert s.render().endswith("(point in time: 1994)")

    def test_non_dates_are_not_truncated_by_a_stray_precision(self):
        assert SnakValue("value", "string", raw="Philadelphia",
                         precision=9).render() == "Philadelphia"


class TestTimeTidying:
    """
    Wikidata pads every date to midnight: a plain day arrives as
    "+1994-03-21T00:00:00Z". With qualifiers, a `point in time` rides on many
    statements, so eleven characters of padding each is real context budget.
    """

    def test_midnight_padding_is_stripped(self):
        assert SnakValue("value", "time", raw="1994-03-21T00:00:00Z").render() == "1994-03-21"

    def test_leading_sign_is_stripped(self):
        assert SnakValue("value", "time", raw="+1994-03-21T00:00:00Z").render() == "1994-03-21"

    def test_a_real_time_of_day_is_left_alone(self):
        """Only exact midnight is padding; anything else is data."""
        assert SnakValue("value", "time", raw="1994-03-21T14:30:00Z").render() \
            == "1994-03-21T14:30:00Z"

    def test_precision_is_not_guessed(self):
        """
        A YEAR-precision date is padded to 1 January. Shortening 1994-01-01 to
        1994 without wikibase:timePrecision would invent a day Wikidata does not
        claim -- and the fetch does not request the psv: value node.
        """
        assert SnakValue("value", "time", raw="1994-01-01T00:00:00Z").render() == "1994-01-01"

    def test_non_dates_are_untouched(self):
        for raw in ("Philadelphia", "600000000", "tom-hanks", ""):
            v = SnakValue("value", "string", raw=raw or None)
            assert v.render() == (raw or UNKNOWN_VALUE)

    def test_it_applies_inside_a_rendered_statement(self):
        s = Statement("Q2263", "Tom Hanks", "P166", "award received",
                      _v("Academy Award for Best Actor"), "normal",
                      (Qualifier("P585", "point in time",
                                 SnakValue("value", "time", raw="+1994-03-21T00:00:00Z")),))
        assert s.render().endswith("(point in time: 1994-03-21)")
