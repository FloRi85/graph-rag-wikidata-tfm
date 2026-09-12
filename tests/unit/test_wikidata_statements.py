"""
Unit tests for the STATEMENT fetch layer in src/retrieval/wikidata.py.

⚠️ WHY THIS FILE EXISTS AS A SEPARATE FILE, AND WHY IT IS NOT OPTIONAL.
The statement fetch shipped with `{p}` missing from the `str.format()` call, so
every live call raised `KeyError: 'p'` — the layer had never once executed. It
was caught by running it by hand against Wikidata, which is the slowest and
least repeatable way to find a bug that a mocked call would have surfaced in
milliseconds. `TestQueryTemplates` below is the regression for exactly that.

The tests mock `_sparql_query` and redirect `STATEMENT_CACHE_DIR` at a tmp_path,
so they need no network and no key. Row dicts mirror the SPARQL JSON shape:
one row per (statement x qualifier), which is why folding is tested explicitly.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_wikidata_statements.py -v
"""

import json as _json
import re as _re
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval import wikidata
from src.retrieval.wikidata import fetch_statements

_ENT = "http://www.wikidata.org/entity/"
_ONT = "http://wikiba.se/ontology#"


def _row(st, prop, prop_label, value, value_label, rank="Normal", dt="WikibaseItem",
         qual_prop=None, qual_prop_label=None, qual_value=None, qual_value_label=None,
         qual_dt="WikibaseItem"):
    """One forward SPARQL row. `value` may be a QID or a bare literal."""
    r = {
        "st": {"value": f"{_ENT}statement/{st}"},
        "prop": {"value": f"{_ENT}{prop}"},
        "propLabel": {"value": prop_label},
        "dt": {"value": f"{_ONT}{dt}"},
        "value": {"value": f"{_ENT}{value}" if str(value).startswith("Q") else value},
        "valueLabel": {"value": value_label},
        "rank": {"value": f"{_ONT}{rank}Rank"},
    }
    if qual_prop:
        r["qualProp"] = {"value": f"{_ENT}{qual_prop}"}
        r["qualPropLabel"] = {"value": qual_prop_label}
        r["qualDt"] = {"value": f"{_ONT}{qual_dt}"}
        r["qualValue"] = {"value": f"{_ENT}{qual_value}" if str(qual_value).startswith("Q")
                          else qual_value}
        r["qualValueLabel"] = {"value": qual_value_label}
    return r


def _rev_row(st, subj, subj_label, prop, prop_label, rank="Normal", dt="WikibaseItem"):
    return {
        "st": {"value": f"{_ENT}statement/{st}"},
        "subj": {"value": f"{_ENT}{subj}"},
        "subjLabel": {"value": subj_label},
        "prop": {"value": f"{_ENT}{prop}"},
        "propLabel": {"value": prop_label},
        "dt": {"value": f"{_ONT}{dt}"},
        "rank": {"value": f"{_ONT}{rank}Rank"},
    }


_LABEL = [{"l": {"value": "Tom Hanks"}}]


def _fetch(tmp_path, forward, reverse=None, label=None, reverse_exc=None,
           reverse_truncated=False, strict=True):
    """Drive fetch_statements with mocked SPARQL: label, then forward.

    ⚠️ The REVERSE side is patched at `_reverse_rows`, not at `_sparql_query`.
    Since 2026-08-14 the incoming fetch is several requests -- the entity-valued
    property list, four discovery probes, then one request per batch of
    properties -- so a fixed `side_effect` sequence no longer lines up, and
    pinning the exact call count here would make these tests break every time
    the batch size is tuned. What they are about is what comes BACK.
    """
    calls = [label if label is not None else _LABEL, forward]

    def fake_reverse(qid):
        if reverse_exc is not None:
            raise reverse_exc
        return list(reverse or []), reverse_truncated

    with patch("src.retrieval.wikidata._sparql_query", side_effect=calls), \
         patch("src.retrieval.wikidata._reverse_rows", side_effect=fake_reverse), \
         patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
        return fetch_statements("Q2263", strict=strict)


# ---------------------------------------------------------------------------

class TestQueryTemplates:
    """
    Regression for the shipped `KeyError: 'p'`.

    The templates carry three substitution points and are filled at CALL time,
    long after import, so a missing key is invisible until a live fetch runs.
    """

    # ⚠️ Keep in sync with _statement_rows. This list existing is the point: when
    # `{lang}` was added for the `mul` fix, these tests failed instantly rather
    # than the query raising KeyError on the next live fetch.
    FIELDS = {"qid": "Q2263", "limit": 4000, "p": wikidata._p, "lang": "en,mul"}
    # The reverse query is assembled from per-property subqueries, so it takes
    # `blocks` instead of `qid`/`limit` — those are baked into each block.
    REVERSE_FIELDS = {"blocks": "  { }", "p": wikidata._p, "lang": "en,mul"}

    def test_forward_template_formats_with_the_arguments_the_caller_passes(self):
        out = wikidata._FORWARD_STATEMENTS.format(**self.FIELDS)
        # NOT "no braces remain" — SPARQL's own doubled braces legitimately
        # collapse to single ones here. What must not remain is a named field.
        for field in self.FIELDS:
            assert "{" + field + "}" not in out

    def test_reverse_template_formats_with_the_arguments_the_caller_passes(self):
        out = wikidata._REVERSE_BY_PROPERTIES.format(**self.REVERSE_FIELDS)
        for field in self.REVERSE_FIELDS:
            assert "{" + field + "}" not in out

    @pytest.mark.parametrize("omit", ["qid", "limit", "p", "lang"])
    def test_every_forward_field_is_required(self, omit):
        """Each one is a live KeyError if the caller forgets it — as `p` was."""
        args = {k: v for k, v in self.FIELDS.items() if k != omit}
        with pytest.raises(KeyError):
            wikidata._FORWARD_STATEMENTS.format(**args)

    @pytest.mark.parametrize("omit", ["blocks", "p", "lang"])
    def test_every_reverse_field_is_required(self, omit):
        args = {k: v for k, v in self.REVERSE_FIELDS.items() if k != omit}
        with pytest.raises(KeyError):
            wikidata._REVERSE_BY_PROPERTIES.format(**args)

    @pytest.mark.parametrize("omit", ["pid", "qid", "cap"])
    def test_every_reverse_block_field_is_required(self, omit):
        """The per-property block carries the bound predicate and the cap."""
        args = {k: v for k, v in
                {"pid": "P166", "qid": "Q2263", "cap": 30}.items() if k != omit}
        with pytest.raises(KeyError):
            wikidata._REVERSE_BLOCK.format(**args)

    def test_statement_rows_supplies_every_field(self):
        """Pins the CALLER, not the template: the bug was a missing kwarg."""
        with patch("src.retrieval.wikidata._sparql_query", return_value=[]) as m:
            wikidata._statement_rows(wikidata._FORWARD_STATEMENTS, "Q2263")
        sent = m.call_args[0][0]
        assert "Q2263" in sent and str(wikidata.STATEMENT_ROW_LIMIT) in sent
        assert wikidata._p in sent


class TestTheTwoOscars:
    """The worked example the whole rebuild exists for."""

    def test_same_property_and_value_survive_as_two_statements(self, tmp_path):
        rows = [
            _row("A", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1994-03-21T00:00:00Z", qual_value_label="1994-03-21T00:00:00Z"),
            _row("B", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1995-03-27T00:00:00Z", qual_value_label="1995-03-27T00:00:00Z"),
        ]
        out = _fetch(tmp_path, rows)
        assert len(out) == 2, "the truthy view collapsed these into one edge"
        assert len({s.render() for s in out}) == 2

    def test_genid_value_becomes_somevalue_and_never_renders_the_url(self, tmp_path):
        """A SOMEVALUE snak arrives as a skolem /.well-known/genid/ IRI.

        Until 2026-08-18 every snak was stamped "value", so the renderer's raw
        fallback printed the literal genid URL into the context (one context
        in every k30 DEV run). It must survive cleaning — "unknown
        value" is Wikidata's own claim — but render as the UNKNOWN_VALUE token.
        """
        genid = "http://www.wikidata.org/.well-known/genid/8f81b31bf9d3c8e1a6c8"
        rows = [_row("A", "P22", "father", genid, genid)]
        out = _fetch(tmp_path, rows)
        assert len(out) == 1
        assert out[0].value.snaktype == "somevalue"
        assert "genid" not in out[0].render()
        from src.retrieval.statement import UNKNOWN_VALUE
        assert UNKNOWN_VALUE in out[0].render()

    def test_qualifiers_are_folded_across_rows(self, tmp_path):
        """One row per (statement x qualifier) — three rows are ONE statement."""
        rows = [
            _row("A", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1994", qual_value_label="1994"),
            _row("A", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P1686", qual_prop_label="for work",
                 qual_value="Q204057", qual_value_label="Philadelphia"),
        ]
        out = _fetch(tmp_path, rows)
        assert len(out) == 1
        assert len(out[0].qualifiers) == 2

    def test_qualifier_order_is_stable_across_row_permutations(self, tmp_path):
        """SPARQL guarantees no row order; unsorted tuples would defeat dedup."""
        a = _row("A", "P166", "award received", "Q103916", "Academy Award",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1994", qual_value_label="1994")
        b = _row("A", "P166", "award received", "Q103916", "Academy Award",
                 qual_prop="P1686", qual_prop_label="for work",
                 qual_value="Q204057", qual_value_label="Philadelphia")
        first = _fetch(tmp_path, [a, b])[0].render()
        second = _fetch(tmp_path / "other", [b, a])[0].render()
        assert first == second


class TestRankPolicy:
    def test_deprecated_is_dropped(self, tmp_path):
        """Serving a value Wikidata itself judges wrong would be worse than truthy."""
        rows = [_row("A", "P1082", "population", "1000", "1000",
                     rank="Deprecated", dt="Quantity"),
                _row("B", "P1082", "population", "2000", "2000", dt="Quantity")]
        out = _fetch(tmp_path, rows)
        assert [s.value.render() for s in out] == ["2000"]

    def test_normal_is_kept_alongside_preferred(self, tmp_path):
        """The deliberate change: truthy drops ALL normal once any preferred exists."""
        rows = [_row("A", "P1082", "population", "1000", "1000",
                     rank="Normal", dt="Quantity"),
                _row("B", "P1082", "population", "2000", "2000",
                     rank="Preferred", dt="Quantity")]
        assert len(_fetch(tmp_path, rows)) == 2

    def test_policy_is_a_parameter_not_a_constant(self, tmp_path):
        """An ablation on the rank axis must cost a flag flip, not a rebuild."""
        rows = [_row("A", "P1082", "population", "1000", "1000",
                     rank="Normal", dt="Quantity"),
                _row("B", "P1082", "population", "2000", "2000",
                     rank="Preferred", dt="Quantity")]
        with patch("src.retrieval.wikidata.RANK_POLICY", ("preferred",)):
            out = _fetch(tmp_path, rows)
        assert [s.rank for s in out] == ["preferred"]

    def test_missing_rank_defaults_to_normal(self, tmp_path):
        row = _row("A", "P166", "award received", "Q1", "Oscar")
        del row["rank"]
        assert _fetch(tmp_path, [row])[0].rank == "normal"


class TestLabelResolution:
    def test_unresolved_value_label_drops_the_statement(self, tmp_path):
        """A bare QID in the prompt is noise the answering model cannot read."""
        rows = [_row("A", "P1686", "for work", "Q134773", "Q134773")]
        assert _fetch(tmp_path, rows) == []

    def test_a_label_merely_starting_with_Q_is_kept(self, tmp_path):
        """
        Replaces a documented latent bug. The truthy path dropped any label
        matching startswith("Q") — which also discards Quebec, Queen Victoria,
        Quentin Tarantino. The test is now exact.
        """
        rows = [_row("A", "P19", "place of birth", "Q1946", "Quebec")]
        out = _fetch(tmp_path, rows)
        assert [s.value.label for s in out] == ["Quebec"]

    def test_unresolved_qualifier_is_dropped_but_the_statement_survives(self, tmp_path):
        """
        Losing a qualifier must not lose the fact.

        ⚠️ The case that motivated this test — Hanks's 1995 Oscar losing
        `for work: Forrest Gump` — turned out to be a QUERY bug, not missing
        data: Q134773's name lives under Wikidata's `mul` label code and the
        query asked only for "en". Fixed via LABEL_LANGS. The behaviour pinned
        here still matters for values that really are unresolvable, and the
        example is kept because it is the one that exposed it.
        """
        rows = [
            _row("A", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1995", qual_value_label="1995"),
            _row("A", "P166", "award received", "Q103916", "Academy Award for Best Actor",
                 qual_prop="P1686", qual_prop_label="for work",
                 qual_value="Q134773", qual_value_label="Q134773"),
        ]
        out = _fetch(tmp_path, rows)
        assert len(out) == 1
        assert [q.property_label for q in out[0].qualifiers] == ["point in time"]

    def test_missing_property_label_drops_the_statement(self, tmp_path):
        rows = [_row("A", "P166", "", "Q1", "Oscar")]
        assert _fetch(tmp_path, rows) == []


class TestMulLabels:
    """
    Wikidata's `mul` label code (2024) holds names identical across languages,
    and editors REMOVE the per-language duplicates when they migrate. Proper
    nouns are exactly that case, so asking for "en" alone returns the QID and
    the value is then dropped as unresolved. Verified live:

        language "en"      -> ['Q134773', 'Q8337', 'Tom Hanks']
        language "en,mul"  -> ['Forrest Gump', 'Harry Potter', 'Tom Hanks']
    """

    def test_label_langs_requests_mul(self):
        assert "mul" in wikidata.LABEL_LANGS

    def test_english_is_still_preferred_over_the_multilingual_default(self):
        """Where both exist, a genuine en label must win."""
        assert wikidata.LABEL_LANGS.split(",")[0] == "en"

    def test_the_forward_query_uses_it(self):
        with patch("src.retrieval.wikidata._sparql_query", return_value=[]) as m:
            wikidata._statement_rows(wikidata._FORWARD_STATEMENTS, "Q2263")
        assert f'wikibase:language "{wikidata.LABEL_LANGS}"' in m.call_args[0][0]

    def test_the_reverse_query_uses_it(self):
        with patch("src.retrieval.wikidata._sparql_query", return_value=[]) as m, \
             patch("src.retrieval.wikidata.incoming_properties", return_value=["P166"]):
            wikidata._reverse_rows("Q2263")
        assert f'wikibase:language "{wikidata.LABEL_LANGS}"' in m.call_args[0][0]

    def test_the_own_label_query_asks_for_mul_too(self, tmp_path):
        """
        The own label is what every INCOMING statement renders as its value, so
        losing it degrades the whole reverse direction of an entity at once.
        """
        with patch("src.retrieval.wikidata._sparql_query",
                   side_effect=[[], [], []]) as m, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q8337")
        assert "mul" in m.call_args_list[0][0][0]


class TestRedirects:
    """
    🔴 A MERGED ENTITY IS TOTAL, SILENT DATA LOSS FOR THAT QUESTION.

    Mintaka was annotated against a 2021 Wikidata snapshot; Wikidata is live and
    items get merged. The stale QID stays syntactically valid, resolves to
    nothing and carries NO statements, so the question gets an empty context
    while its gold QID looks perfectly well-formed.

    Verified live: DEV-200 gold Q4439148 -> Q315625 ("intern"), 0 statements at
    the stale QID against 92 at the target.
    """

    def _meta(self, **kw):
        base = {"label": "x", "description": None, "aliases": [], "redirect": None}
        return base | kw

    def test_the_fetch_follows_the_redirect(self, tmp_path):
        rows = [_row("A", "P31", "instance of", "Q1", "occupation")]
        with patch("src.retrieval.wikidata._meta_for",
                   side_effect=[self._meta(label="Q4439148", redirect="Q315625"),
                                self._meta(label="intern")]), \
             patch("src.retrieval.wikidata._statement_rows",
                   return_value=(rows, False)) as m, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            out = fetch_statements("Q4439148")
        assert m.call_args_list[0][0][1] == "Q315625", "queried the stale QID"
        assert out and out[0].subject_label == "intern"

    def test_provenance_stays_the_gold_qid(self, tmp_path):
        """
        `groups` keys the per-entity z-normalisation on the QID the QUESTION
        named. A redirect must not leak into it.
        """
        rows = [_row("A", "P31", "instance of", "Q1", "occupation")]
        with patch("src.retrieval.wikidata._meta_for",
                   side_effect=[self._meta(label="Q4439148", redirect="Q315625"),
                                self._meta(label="intern")]), \
             patch("src.retrieval.wikidata._statement_rows",
                   return_value=(rows, False)), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            out = fetch_statements("Q4439148")
        assert out[0].source_entity_id == "Q4439148"

    def test_a_redirect_is_followed_only_once(self, tmp_path):
        """A chain would turn a data error into a hang; merges are not chained."""
        with patch("src.retrieval.wikidata._meta_for",
                   side_effect=[self._meta(label="A", redirect="Q2"),
                                self._meta(label="B", redirect="Q3")]) as m, \
             patch("src.retrieval.wikidata._statement_rows", return_value=([], False)), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q1")
        assert m.call_count == 2

    def test_no_redirect_leaves_the_fetch_alone(self, tmp_path):
        with patch("src.retrieval.wikidata._meta_for",
                   return_value=self._meta(label="Tom Hanks")) as m, \
             patch("src.retrieval.wikidata._statement_rows",
                   return_value=([], False)) as rows, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q2263")
        assert m.call_count == 1
        assert rows.call_args_list[0][0][1] == "Q2263"

    def test_entity_meta_also_follows_it(self, tmp_path):
        """The alias/description lines must describe the TARGET, not the tombstone."""
        with patch("src.retrieval.wikidata._meta_for",
                   side_effect=[self._meta(label="Q4439148", redirect="Q315625"),
                                self._meta(label="intern", aliases=["Intern"])]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            meta = wikidata.fetch_entity_meta("Q4439148")
        assert meta["label"] == "intern"
        assert meta["redirect"] == "Q315625"


class TestOwnLabelFailure:
    """
    ⚠️ A silent cache-poisoning bug, observed live on Q8337 (Harry Potter).

    `own_label` falls back to the QID when the label query fails. An incoming
    statement's VALUE is this entity, so the fallback renders ~90 statements as
    "[Fred Weasley] present in work: Q8337" — and the result was then CACHED, so
    nothing would ever ask again.
    """

    def test_a_failed_own_label_is_not_cached(self, tmp_path):
        rows = [_rev_row("R1", "Q3111", "Fred Weasley", "P1441", "present in work")]
        with patch("src.retrieval.wikidata._sparql_query", side_effect=[[], [], rows]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q8337")
        assert not (tmp_path / "Q8337.json").exists(), \
            "a poisoned entity was cached — nothing would ever refetch it"

    def test_the_failure_is_announced(self, tmp_path, capsys):
        with patch("src.retrieval.wikidata._sparql_query", side_effect=[[], [], []]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q8337")
        assert "label unresolved" in capsys.readouterr().out

    def test_a_resolved_label_still_caches(self, tmp_path):
        with patch("src.retrieval.wikidata._sparql_query",
                   side_effect=[_LABEL, [_row("A", "P166", "award received", "Q1", "Oscar")], []]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q2263")
        assert (tmp_path / "Q2263.json").exists()


class TestNoiseFiltering:
    def test_external_id_dropped_by_datatype(self, tmp_path):
        rows = [_row("A", "P345", "IMDb ID", "nm0000158", "nm0000158", dt="ExternalId"),
                _row("B", "P166", "award received", "Q1", "Oscar")]
        out = _fetch(tmp_path, rows)
        assert [s.property_label for s in out] == ["award received"]

    def test_noise_qualifier_property_is_dropped_from_a_kept_statement(self, tmp_path):
        """
        ⚠️ Dropped by DATATYPE since 2026-08-12. "Instagram numeric ID" happens to
        end in " id" so a label rule would also catch it, but the qualifier's own
        `wikibase:propertyType` is now resolved in-query — which is what let the
        static label→datatype map be deleted.
        """
        rows = [
            _row("A", "P166", "award received", "Q1", "Oscar",
                 qual_prop="P585", qual_prop_label="point in time",
                 qual_value="1994", qual_value_label="1994", qual_dt="Time"),
            _row("A", "P166", "award received", "Q1", "Oscar",
                 qual_prop="P2003", qual_prop_label="Instagram numeric ID",
                 qual_value="12345", qual_value_label="12345", qual_dt="ExternalId"),
        ]
        out = _fetch(tmp_path, rows)
        assert len(out) == 1
        assert [q.property_label for q in out[0].qualifiers] == ["point in time"]

    def test_a_qualifier_is_dropped_on_datatype_ALONE(self, tmp_path):
        """
        The case the map used to be needed for: a qualifier property whose label
        matches no rule at all. Only its datatype identifies it as media.
        """
        rows = [
            _row("A", "P166", "award received", "Q1", "Oscar",
                 qual_prop="P18", qual_prop_label="image",
                 qual_value="x.jpg", qual_value_label="x.jpg", qual_dt="CommonsMedia"),
        ]
        out = _fetch(tmp_path, rows)
        assert out[0].qualifiers == ()

    def test_the_qualifier_datatype_is_actually_requested(self):
        """Without ?qualDt in the SELECT there is nothing to filter on."""
        for tpl in (wikidata._FORWARD_STATEMENTS, wikidata._REVERSE_BY_PROPERTIES):
            assert "?qualDt" in tpl
            assert "?qualProp wikibase:propertyType ?qualDt" in tpl

    def test_filtering_runs_at_return_time_so_cached_data_is_refiltered(self, tmp_path):
        """
        Same contract as the truthy path: the cache stores RAW statements, so a
        filter or rank-policy change takes effect with no refetch.
        """
        rows = [_row("A", "P1082", "population", "1000", "1000",
                     rank="Normal", dt="Quantity"),
                _row("B", "P1082", "population", "2000", "2000",
                     rank="Preferred", dt="Quantity")]
        assert len(_fetch(tmp_path, rows)) == 2
        with patch("src.retrieval.wikidata._sparql_query") as m, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata.RANK_POLICY", ("preferred",)):
            out = fetch_statements("Q2263")
        m.assert_not_called()
        assert len(out) == 1


class TestReverse:
    def test_reverse_statement_is_oriented_subject_first(self, tmp_path):
        rows = [_rev_row("R1", "Q3111", "Ronnie Wood", "P463", "member of")]
        out = _fetch(tmp_path, [], reverse=rows)
        assert out[0].direction == "incoming"
        assert out[0].subject_label == "Ronnie Wood"
        assert out[0].value.label == "Tom Hanks"

    def test_direction_is_carried_not_inferred_from_the_subject_shape(self, tmp_path):
        """Retires the `re.fullmatch(r"Q\\d+", subj)` hack."""
        fwd = _fetch(tmp_path, [_row("A", "P166", "award received", "Q1", "Oscar")])
        assert fwd[0].direction == "outgoing"
        assert fwd[0].source_entity_id == "Q2263"

    def test_unresolved_reverse_subject_is_dropped(self, tmp_path):
        rows = [_rev_row("R1", "Q999", "Q999", "P463", "member of")]
        assert _fetch(tmp_path, [], reverse=rows) == []

    def test_reverse_per_prop_cap_applied(self, tmp_path):
        rows = [_rev_row(f"R{i}", f"Q{i}", f"Person {i}", "P161", "cast member")
                for i in range(wikidata.REVERSE_PER_PROP_CAP + 10)]
        out = _fetch(tmp_path, [], reverse=rows)
        assert len(out) == wikidata.REVERSE_PER_PROP_CAP

    # -- reverse-failure contract (rewritten 2026-08-13) ---------------------
    #
    # This used to assert "degrades to forward-only" and pass, which is what let
    # the bug live: the forward-only result was ALSO cached, so one exhausted
    # reverse query permanently cost the entity every incoming statement, with
    # nothing on disk to show it. Forward-only is now an explicit demo mode,
    # never a silent default, and it is never written to the cache.

    def test_reverse_failure_raises_under_strict_retrieval(self, tmp_path):
        """A canonical run must not answer from a half-fetched graph."""
        fwd = [_row("A", "P166", "award received", "Q1", "Oscar")]
        with pytest.raises(wikidata.IncompleteRetrievalError, match="reverse"):
            _fetch(tmp_path, fwd, reverse_exc=TimeoutError("reverse timed out"))

    def test_reverse_failure_is_never_cached(self, tmp_path):
        """🔴 The poisoning itself: a forward-only entity must not persist."""
        fwd = [_row("A", "P166", "award received", "Q1", "Oscar")]
        _fetch(tmp_path, fwd, reverse_exc=TimeoutError("reverse timed out"),
               strict=False)
        assert not (tmp_path / "Q2263.json").exists(), \
            "a forward-only entity was cached — every incoming statement is " \
            "lost permanently and nothing would ever refetch it"

    def test_reverse_failure_degrades_only_when_explicitly_asked(self, tmp_path):
        """The demo path: partial beats nothing, but only on request."""
        fwd = [_row("A", "P166", "award received", "Q1", "Oscar")]
        out = _fetch(tmp_path, fwd, reverse_exc=TimeoutError("reverse timed out"),
                     strict=False)
        assert len(out) == 1

    def test_strict_default_is_on(self):
        """If this flips, every eval silently accepts partial graphs again."""
        assert wikidata.STRICT_RETRIEVAL is True


class TestCache:
    def test_cache_prevents_a_second_fetch(self, tmp_path):
        rows = [_row("A", "P166", "award received", "Q1", "Oscar")]
        _fetch(tmp_path, rows)
        with patch("src.retrieval.wikidata._sparql_query") as m, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            out = fetch_statements("Q2263")
        m.assert_not_called()
        assert len(out) == 1

    def test_round_trip_preserves_every_field(self, tmp_path):
        rows = [_row("A", "P166", "award received", "Q103916", "Academy Award",
                     qual_prop="P585", qual_prop_label="point in time",
                     qual_value="1994", qual_value_label="1994")]
        first = _fetch(tmp_path, rows)
        with patch("src.retrieval.wikidata._sparql_query"), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            second = fetch_statements("Q2263")
        assert first == second

    def test_wrong_version_cache_is_refetched(self, tmp_path):
        """A v2 truthy cache must never be read as statements."""
        (tmp_path / "Q2263.json").write_text(_json.dumps(
            {"v": 2, "triples": [["Q2263", "award received", "Oscar"]]}))
        out = _fetch(tmp_path, [_row("A", "P166", "award received", "Q1", "Oscar")])
        assert len(out) == 1 and out[0].subject_label == "Tom Hanks"

    def test_statement_cache_is_a_separate_namespace_from_the_truthy_one(self):
        """
        The truthy `data/cache/wikidata/` may still exist on disk from before the
        rebuild. Its entries are 3-tuples, not statements, so reading one as a
        statement entry would be a silent shape error — hence a distinct
        directory rather than a version bump inside the same one.
        """
        truthy_dir = (Path(wikidata.__file__).resolve().parent.parent.parent
                      / "data" / "cache" / "wikidata")
        assert wikidata.STATEMENT_CACHE_DIR != truthy_dir
        assert wikidata.STATEMENT_CACHE_DIR.name == "wikidata_statements"


class TestRowLimit:
    def test_binding_is_reported_not_swallowed(self, tmp_path, capsys):
        """
        There is still no ORDER BY, so a bound limit means an ARBITRARY slice —
        exactly what the old LIMIT 500 hid.
        """
        rows = [_row(f"S{i}", "P166", "award received", f"Q{i}", f"Award {i}")
                for i in range(wikidata.STATEMENT_ROW_LIMIT)]
        _fetch(tmp_path, rows)
        out = capsys.readouterr().out
        assert "truncated for Q2263" in out
        # The two directions bind for different reasons and the message says
        # which: forward on the global row limit, reverse on the per-property
        # cap. Reporting both as "row limit" made the reverse count look like a
        # regression when the cap had simply become visible.
        assert f"forward (row limit {wikidata.STATEMENT_ROW_LIMIT})" in out
        assert "reverse" not in out

    def test_reverse_truncation_names_the_per_property_cap(self, tmp_path, capsys):
        _fetch(tmp_path, [_row("A", "P166", "award received", "Q1", "Oscar")],
               reverse_truncated=True)
        out = capsys.readouterr().out
        assert f"reverse (per-property cap {wikidata.REVERSE_PER_PROP_CAP})" in out

    def test_no_warning_when_the_limit_does_not_bind(self, tmp_path, capsys):
        _fetch(tmp_path, [_row("A", "P166", "award received", "Q1", "Oscar")])
        assert "truncated for" not in capsys.readouterr().out


class TestSparqlRetry:
    """
    The endpoint's latency is erratic under shared load: the same Q2263 query
    measured 6.7 s, a 30 s timeout, then 3.0 s on three consecutive calls. Query
    SHAPE was ruled out — a no-datatype-join variant and an ontology-joined
    variant showed the same spread. Without retry, a 350-entity refetch fails on
    load rather than on anything this code controls.
    """

    def test_truthy_path_still_does_not_retry(self):
        """Every result on record was produced with attempts=1. Keep it that way."""
        with patch("src.retrieval.wikidata.urllib.request.urlopen",
                   side_effect=TimeoutError()) as m:
            with pytest.raises(TimeoutError):
                wikidata._sparql_query("SELECT * {}")
        assert m.call_count == 1

    def test_statement_path_retries_and_succeeds(self):
        with patch("src.retrieval.wikidata._sparql_query", return_value=[]) as m:
            wikidata._statement_rows(wikidata._FORWARD_STATEMENTS, "Q2263")
        assert m.call_args.kwargs["attempts"] > 1

    def test_retry_gives_up_and_raises_rather_than_returning_empty(self):
        """An empty list would read as 'this entity has no facts'."""
        with patch("src.retrieval.wikidata.urllib.request.urlopen",
                   side_effect=TimeoutError()), \
             patch("src.retrieval.wikidata.time.sleep"):
            with pytest.raises(TimeoutError):
                wikidata._sparql_query("SELECT * {}", attempts=3)


# ---------------------------------------------------------------------------
# Cache completeness (2026-08-13).
#
# ONE VALIDATOR, THREE READERS. fetch_statements, fetch_entity_meta and
# tools/prewarm_statement_cache.py used to disagree: prewarm tested only
# Path.exists(), so after a version bump every stale entry still existed and it
# reported "nothing to do" -- pushing the live fetches it exists to prevent
# straight into the eval run.
# ---------------------------------------------------------------------------

class TestCacheCompleteness:
    def _write_good(self, tmp_path):
        calls = [_LABEL, [_row("A", "P166", "award received", "Q1", "Oscar")], []]
        with patch("src.retrieval.wikidata._sparql_query", side_effect=calls), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q2263")
        return tmp_path / "Q2263.json"

    def test_a_complete_entry_is_ok_and_records_its_provenance(self, tmp_path):
        path = self._write_good(tmp_path)
        entry = _json.loads(path.read_text(encoding="utf-8"))
        assert entry["query_succeeded"] == {"label": True, "forward": True,
                                            "reverse": True}
        assert entry["truncated"] == {"forward": False, "reverse": False}
        assert entry["fingerprint"] == wikidata.cache_fingerprint()
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            assert wikidata.cache_entry_status("Q2263")["state"] == "ok"

    def test_missing_and_malformed_are_distinguished(self, tmp_path):
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            assert wikidata.cache_entry_status("Q1")["state"] == "missing"
            (tmp_path / "Q2.json").write_text("{not json", encoding="utf-8")
            assert wikidata.cache_entry_status("Q2")["state"] == "malformed"
            (tmp_path / "Q3.json").write_text('{"v": 6}', encoding="utf-8")
            assert wikidata.cache_entry_status("Q3")["state"] == "malformed"

    def test_an_older_cache_version_is_stale_not_ok(self, tmp_path):
        """🔴 v5 entries may be silently forward-only — they must be refetched."""
        path = self._write_good(tmp_path)
        entry = _json.loads(path.read_text(encoding="utf-8"))
        entry["v"] = 5
        path.write_text(_json.dumps(entry), encoding="utf-8")
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            st = wikidata.cache_entry_status("Q2263")
        assert st["state"] == "stale" and "v5" in st["reason"]

    def test_changed_query_limits_make_an_entry_stale(self, tmp_path):
        path = self._write_good(tmp_path)
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata.STATEMENT_ROW_LIMIT", 999):
            assert wikidata.cache_entry_status("Q2263")["state"] == "stale"

    def test_read_time_settings_do_not_invalidate_the_cache(self, tmp_path):
        """RANK_POLICY is applied by _clean_statements at RETURN time."""
        self._write_good(tmp_path)
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata.RANK_POLICY", ("preferred",)):
            assert wikidata.cache_entry_status("Q2263")["state"] == "ok"

    def test_an_entry_with_a_failed_query_is_incomplete(self, tmp_path):
        path = self._write_good(tmp_path)
        entry = _json.loads(path.read_text(encoding="utf-8"))
        entry["query_succeeded"]["reverse"] = False
        path.write_text(_json.dumps(entry), encoding="utf-8")
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            st = wikidata.cache_entry_status("Q2263")
        assert st["state"] == "incomplete" and "reverse" in st["reason"]

    def test_truncation_is_persisted_not_just_printed(self, tmp_path):
        """A console warning scrolls away; an arbitrary slice must stay visible."""
        rows = [_row(f"A{i}", "P166", "award received", f"Q{i}", f"Oscar {i}")
                for i in range(4)]
        with patch("src.retrieval.wikidata._sparql_query",
                   side_effect=[_LABEL, rows, []]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata.STATEMENT_ROW_LIMIT", 4):
            fetch_statements("Q2263")
            entry = _json.loads((tmp_path / "Q2263.json").read_text(encoding="utf-8"))
            assert entry["truncated"] == {"forward": True, "reverse": False}
            assert wikidata.truncated_entities(["Q2263"]) == {"Q2263": ["forward"]}

    def test_fetch_entity_meta_uses_the_same_validator(self, tmp_path):
        """A stale entry must not serve metadata that fetch_statements rejects."""
        path = self._write_good(tmp_path)
        entry = _json.loads(path.read_text(encoding="utf-8"))
        entry["v"] = 5
        path.write_text(_json.dumps(entry), encoding="utf-8")
        with patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata._meta_for",
                   return_value={"label": "refetched"}) as m:
            assert wikidata.fetch_entity_meta("Q2263")["label"] == "refetched"
        m.assert_called_once()


class TestRedirectLabelValidation:
    """
    ⚠️ `own_label != query_qid` is not sufficient. When a redirect is followed
    and the TARGET's metadata fetch also fails, `meta` stays the stale entity's
    -- whose label may itself be the stale QID. "Q4439148" != "Q315625" passes
    that test and caches a bare-QID label, which is the exact poisoning the
    guard exists to prevent.
    """

    def test_unresolved_redirect_target_is_not_cached(self, tmp_path):
        with patch("src.retrieval.wikidata._meta_for",
                   side_effect=[{"label": "Q4439148", "redirect": "Q315625"},
                                {"label": "Q315625"}]), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=[[], []]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q4439148")
        assert not (tmp_path / "Q4439148.json").exists(), \
            "a bare-QID label was cached — every incoming statement is unreadable"

    def test_a_qid_shaped_label_is_never_accepted(self, tmp_path):
        with patch("src.retrieval.wikidata._meta_for", return_value={"label": "Q99"}), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=[[], []]), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path):
            fetch_statements("Q2263")
        assert not (tmp_path / "Q2263.json").exists()


# ---------------------------------------------------------------------------
# PER-PROPERTY INCOMING FETCH (2026-08-14).
#
# The single-query reverse fetch cannot serve hub entities: Q30 has 2,246,467
# subjects on `country` alone. It failed for the seven highest-fan-in DEV gold
# entities, and the old code cached the forward-only remainder — six were stored
# with ZERO incoming statements. Measured after this change: Q30 1,823 incoming,
# France 1,475, UK 1,404, Japan 1,333, Brazil 1,204, film 484, actor 322.
#
# ⚠️ And it must not cost the entities that already worked. Measured old vs new,
# both after _clean_statements: Q2263 151 -> 151, Q23 244 -> 242, Q76 422 -> 417.
# ---------------------------------------------------------------------------

class TestIncomingPropertyDiscovery:
    def test_candidate_set_is_entity_valued_properties_only(self, tmp_path):
        """An ExternalId can never be the property of an INCOMING statement."""
        rows = [{"p": {"value": f"{_ENT}P{n}"}} for n in (17, 27, 161)]
        with patch("src.retrieval.wikidata._sparql_query", return_value=rows), \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata._item_properties_memo", None):
            assert wikidata.item_properties() == ["P17", "P27", "P161"]

    def test_the_property_list_is_cached_and_fetched_once(self, tmp_path):
        rows = [{"p": {"value": f"{_ENT}P17"}}]
        with patch("src.retrieval.wikidata._sparql_query", return_value=rows) as m, \
             patch("src.retrieval.wikidata.STATEMENT_CACHE_DIR", tmp_path), \
             patch("src.retrieval.wikidata._item_properties_memo", None):
            wikidata.item_properties()
            wikidata.item_properties()
        assert m.call_count == 1
        assert (tmp_path / "_item_properties.json").exists()

    def test_the_cache_file_cannot_collide_with_a_qid(self, tmp_path):
        """It lives beside {qid}.json, so its name must not look like one."""
        assert not _re.fullmatch(r"Q\d+", "_item_properties")

    def test_discovery_probes_every_candidate_in_chunks(self, tmp_path):
        pids = [f"P{i}" for i in range(1, 1201)]
        seen = []

        def fake(query, **kw):
            seen.append(query)
            return [{"psP": {"value": f"{_ENT}P17"}}] if "P17 " in query or \
                   query.rstrip().endswith("ps:P17 }") else []

        with patch("src.retrieval.wikidata.item_properties", return_value=pids), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=fake):
            found = wikidata.incoming_properties("Q30")
        # 1200 candidates / 500 per chunk = 3 probes, every candidate covered.
        assert len(seen) == 3
        assert all(f"ps:{p}" in "".join(seen) for p in ("P1", "P600", "P1200"))
        assert found == ["P17"]

    def test_chunk_probe_failure_falls_back_to_per_property_limit1(self):
        """A chunk whose EXISTS probe times out is re-probed one property at a
        time with the LIMIT-1 form (the Q5/P31 materialization case). Same
        exact answer, so a property findable only that way is still found."""
        pids = [f"P{i}" for i in range(1, 1201)]
        queries = []

        def fake(query, **kw):
            queries.append(query)
            if "VALUES" in query and "ps:P1 " in query:
                raise OSError("504 on the EXISTS materialization")  # chunk 0
            if "LIMIT 1" in query:
                return [{"st": {"value": "s"}}] if "ps:P31 " in query else []
            return ([{"psP": {"value": f"{_ENT}P600"}}]
                    if "ps:P600" in query else [])

        with patch("src.retrieval.wikidata.item_properties", return_value=pids), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=fake):
            found = wikidata.incoming_properties("Q5")

        assert found == ["P31", "P600"]
        singles = [q for q in queries if "LIMIT 1" in q]
        # Exactly the failed chunk's 500 properties, no others.
        assert len(singles) == 500
        assert any("ps:P31 " in q for q in singles)
        assert not any("ps:P600" in q for q in singles)

    def test_a_failing_single_probe_still_raises(self):
        """Discovery stays exact: a property that cannot be probed raises
        rather than being silently skipped (the forward-only-refusal
        argument — a skip would silently lose its incoming statements)."""
        pids = [f"P{i}" for i in range(1, 3)]

        def fake(query, **kw):
            if "VALUES" in query:
                raise OSError("chunk probe down")
            if "LIMIT 1" in query and "ps:P2 " in query:
                raise OSError("single probe down too")
            return []

        with patch("src.retrieval.wikidata.item_properties", return_value=pids), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=fake), \
             pytest.raises(OSError, match="single probe down"):
            wikidata.incoming_properties("Q5")


class TestPerPropertyReverseFetch:
    def _rows_for(self, pid, n):
        return [_rev_row(f"{pid}-{i}", f"Q{i}", f"Subject {i}", pid, "prop")
                for i in range(n)]

    def test_each_property_is_capped_independently(self):
        """The cap bounds the QUERY now, not a filter over a global slice."""
        cap = wikidata.REVERSE_PER_PROP_CAP
        calls = []

        def fake(query, **kw):
            calls.append(query)
            return self._rows_for("P161", 3) + self._rows_for("P57", 2)

        with patch("src.retrieval.wikidata.incoming_properties",
                   return_value=["P161", "P57"]), \
             patch("src.retrieval.wikidata._sparql_query", side_effect=fake):
            rows, truncated = wikidata._reverse_rows("Q2263")
        assert len(calls) == 1                      # both fit in one batch
        assert f"LIMIT {cap}" in calls[0]
        assert "ps:P161" in calls[0] and "ps:P57" in calls[0]
        assert 'hint:Query hint:optimizer "None"' in calls[0], \
            "without the optimizer hint a hub fetch times out instead of " \
            "returning in 0.5s — the hint is the whole fix"
        assert truncated is False

    def test_properties_are_batched_not_one_request_each(self):
        props = [f"P{i}" for i in range(1, 61)]     # 60 properties
        with patch("src.retrieval.wikidata.incoming_properties", return_value=props), \
             patch("src.retrieval.wikidata._sparql_query", return_value=[]) as m:
            wikidata._reverse_rows("Q30")
        # 60 / REVERSE_FETCH_BATCH(20) = 3 requests, not 60 round trips.
        assert m.call_count == 3

    def test_hitting_the_cap_is_reported_as_truncation(self):
        cap = wikidata.REVERSE_PER_PROP_CAP
        with patch("src.retrieval.wikidata.incoming_properties", return_value=["P161"]), \
             patch("src.retrieval.wikidata._sparql_query",
                   return_value=self._rows_for("P161", cap)):
            _, truncated = wikidata._reverse_rows("Q2263")
        assert truncated is True

    def test_truncation_counts_statements_not_rows(self):
        """The qualifier join multiplies rows; counting them would over-report."""
        dupes = [_rev_row("S1", "Q1", "One", "P161", "cast member")] * 100
        with patch("src.retrieval.wikidata.incoming_properties", return_value=["P161"]), \
             patch("src.retrieval.wikidata._sparql_query", return_value=dupes):
            _, truncated = wikidata._reverse_rows("Q2263")
        assert truncated is False, "100 rows of ONE statement is not truncation"

    def test_no_incoming_properties_means_no_fetch_at_all(self):
        with patch("src.retrieval.wikidata.incoming_properties", return_value=[]), \
             patch("src.retrieval.wikidata._sparql_query") as m:
            rows, truncated = wikidata._reverse_rows("Q999")
        assert rows == [] and truncated is False
        m.assert_not_called()
