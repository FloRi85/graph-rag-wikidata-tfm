"""
Unit tests for src/retrieval/wikidata_pool.py — the ranking pool shared by Configs 3 & 4.

⚠️ THE LOAD-BEARING TEST IN THIS FILE IS `TestConfigsShareOnePool`. The thesis
compares C3 and C4 on the premise that they differ in exactly ONE step, the
condensing call. Until 2026-08-12 that premise rested on two verbatim-duplicated
loops staying in sync by hand; now it rests on one function, and this file pins
that they still reach the ranker with byte-identical input.

Run from repo root:
    python -m pytest tests/unit/test_wikidata_pool.py -v
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.pipelines.graph_rag as graph_rag
import src.pipelines.graph_rag_rerank as graph_rag_rerank
from src.retrieval.wikidata_pool import build_pool
from src.retrieval.statement import Qualifier, SnakValue, Statement

_POOL_FETCH = "src.retrieval.wikidata_pool.fetch_statements"
_POOL_META = "src.retrieval.wikidata_pool.fetch_entity_meta"
_CLIENT = "src.llm_config.get_client"

_NO_META = {"label": None, "description": None, "aliases": []}


@pytest.fixture(autouse=True)
def _no_entity_meta():
    """
    Entity metadata off by default, so each test states what it exercises.

    ⚠️ Autouse because `build_pool` calls `fetch_entity_meta` unconditionally —
    without this the suite makes live SPARQL calls, which it did once and turned
    an 8-second run into 87.

    The value-description memo is cleared on BOTH sides of every test: it
    persists per process, so a description mocked in one test would otherwise
    be served to the next from the memo, bypassing that test's own mock.
    """
    import src.retrieval.wikidata_pool as pool_mod
    pool_mod._VALUE_DESC_MEMO.clear()
    with patch(_POOL_META, return_value=dict(_NO_META)):
        yield
    pool_mod._VALUE_DESC_MEMO.clear()


def _v(label, datatype="wikibase-item", **kw):
    return SnakValue("value", datatype, label=label, **kw)


def _oscar(sid, *quals, subject_label="Tom Hanks", qid="Q2263"):
    return Statement(
        subject_id=qid, subject_label=subject_label,
        property_id="P166", property_label="award received",
        value=_v("Academy Award for Best Actor", id="Q103916"),
        rank="normal", qualifiers=tuple(quals),
        direction="outgoing", source_entity_id=qid, statement_id=sid,
    )


def _when(year):
    return Qualifier("P585", "point in time", SnakValue("value", "time", raw=str(year)))


class TestTheTwoOscars:
    """The case the rebuild exists for, at the pool level rather than the type level."""

    def test_statements_differing_only_in_qualifiers_both_survive(self):
        stmts = [_oscar("A", _when(1994)), _oscar("B", _when(1995))]
        with patch(_POOL_FETCH, return_value=stmts):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert len(lines) == 2, "the truthy pool showed these as one line"

    def test_a_genuinely_identical_line_is_deduped(self):
        """A duplicate would waste a top-k slot."""
        stmts = [_oscar("A", _when(1994)), _oscar("B", _when(1994))]
        with patch(_POOL_FETCH, return_value=stmts):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert len(lines) == 1


class TestParallelLists:
    def test_all_three_returns_stay_index_aligned(self):
        stmts = [_oscar("A", _when(1994)), _oscar("B", _when(1995))]
        with patch(_POOL_FETCH, return_value=stmts):
            lines, groups, statements = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert len(lines) == len(groups) == len(statements)
        for line, st in zip(lines, statements):
            assert line == st.render()

    def test_groups_name_the_source_entity_not_the_subject(self):
        """The ranker z-normalises per QUERY entity, which is what `groups` is."""
        with patch(_POOL_FETCH, side_effect=[[_oscar("A")], [_oscar("B", qid="Q76",
                                                                   subject_label="Barack Obama")]]):
            _, groups, _ = build_pool(["Q2263", "Q76"],
                                      {"Q2263": "Tom Hanks", "Q76": "Barack Obama"})
        assert groups == ["Q2263", "Q76"]

    def test_structure_is_reachable_without_reparsing_the_string(self):
        """`value.id` is what unlocks matching against Mintaka's unused answer_qids."""
        with patch(_POOL_FETCH, return_value=[_oscar("A")]):
            _, _, statements = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert statements[0].value.id == "Q103916"
        assert statements[0].rank == "normal"


class TestEntityNaming:
    def test_dataset_name_wins_over_the_wikidata_label(self):
        """
        Preserves the truthy pool's behaviour (`qid_names.get(subj, subj)`). The
        question's own wording is what the prepend exists to match.
        """
        with patch(_POOL_FETCH, return_value=[_oscar("A", subject_label="Thomas J. Hanks")]):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert lines[0].startswith("[Tom Hanks] ")

    def test_wikidata_label_is_the_fallback_when_the_dataset_names_nothing(self):
        with patch(_POOL_FETCH, return_value=[_oscar("A")]):
            lines, _, _ = build_pool(["Q2263"], {})
        assert lines[0].startswith("[Tom Hanks] ")

    def test_incoming_value_takes_the_dataset_name(self):
        """
        An incoming statement's VALUE is the question entity, so the dataset's
        name belongs there. Also the second of two defences against the observed
        Q8337 failure, where a failed own-label query rendered every incoming
        statement of an entity as "... : Q8337".
        """
        incoming = Statement(
            subject_id="Q3111", subject_label="Fred Weasley",
            property_id="P1441", property_label="present in work",
            value=_v("Q8337", id="Q8337"), rank="normal",
            direction="incoming", source_entity_id="Q8337", statement_id="R1",
        )
        with patch(_POOL_FETCH, return_value=[incoming]):
            lines, _, _ = build_pool(["Q8337"], {"Q8337": "Harry Potter"})
        assert lines[0] == "[Fred Weasley] present in work: Harry Potter"

    def test_incoming_subject_label_is_never_overridden(self):
        """
        An incoming statement's subject is a DIFFERENT entity. Renaming it to the
        question entity would assert something false — "[Tom Hanks] member of:
        Tom Hanks" instead of "[Ronnie Wood] member of: The Rolling Stones".
        """
        incoming = Statement(
            subject_id="Q3111", subject_label="Ronnie Wood",
            property_id="P463", property_label="member of",
            value=_v("The Rolling Stones", id="Q11036"), rank="normal",
            direction="incoming", source_entity_id="Q11036", statement_id="R1",
        )
        with patch(_POOL_FETCH, return_value=[incoming]):
            lines, _, _ = build_pool(["Q11036"], {"Q11036": "The Rolling Stones"})
        assert lines[0].startswith("[Ronnie Wood] ")


class TestEntityMetadata:
    """
    Aliases and the one-line description, rendered as synthetic statements so
    they are RANKED rather than force-fed into the context.
    """

    def test_aliases_become_one_ranked_line(self):
        meta = {"label": "Caligula", "description": None,
                "aliases": ["Gaius Julius Caesar Germanicus Major", "Gaius Caesar"]}
        with patch(_POOL_META, return_value=meta), \
             patch(_POOL_FETCH, return_value=[]):
            lines, groups, statements = build_pool(["Q1409"], {"Q1409": "Caligula"})
        assert lines == ["[Caligula] also known as: "
                         "Gaius Julius Caesar Germanicus Major; Gaius Caesar"]
        assert groups == ["Q1409"]
        assert len(statements) == 1, "must stay index-parallel"

    def test_one_line_not_one_per_alias(self):
        """N alias lines would crowd the top-k with near-duplicates of each other."""
        meta = {"label": "X", "description": None,
                "aliases": [f"alias {i}" for i in range(20)]}
        with patch(_POOL_META, return_value=meta), patch(_POOL_FETCH, return_value=[]):
            lines, _, _ = build_pool(["Q1"], {"Q1": "X"})
        assert len(lines) == 1

    def test_alias_count_is_capped(self):
        meta = {"label": "X", "description": None,
                "aliases": [f"a{i}" for i in range(20)]}
        with patch(_POOL_META, return_value=meta), patch(_POOL_FETCH, return_value=[]):
            lines, _, _ = build_pool(["Q1"], {"Q1": "X"})
        from src.retrieval.wikidata_pool import MAX_ALIASES
        assert lines[0].count(";") == MAX_ALIASES - 1

    def test_an_alias_equal_to_the_name_is_dropped(self):
        meta = {"label": "X", "description": None, "aliases": ["X"]}
        with patch(_POOL_META, return_value=meta), patch(_POOL_FETCH, return_value=[]):
            assert build_pool(["Q1"], {"Q1": "X"}) == ([], [], [])

    def test_description_is_labelled_as_such(self):
        """It is PROSE, not a Wikidata claim, and must not read as one."""
        meta = {"label": "Tom Hanks", "description": "American actor", "aliases": []}
        with patch(_POOL_META, return_value=meta), patch(_POOL_FETCH, return_value=[]):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert lines == ["[Tom Hanks] description: American actor"]

    def test_metadata_lines_carry_no_statement_id(self):
        """Wikidata does not model these as statements; a fake GUID would corrupt
        the drift signal that `statement_id` exists to carry."""
        meta = {"label": "X", "description": "d", "aliases": ["y"]}
        with patch(_POOL_META, return_value=meta), patch(_POOL_FETCH, return_value=[]):
            _, _, statements = build_pool(["Q1"], {"Q1": "X"})
        assert all(s.statement_id == "" for s in statements)

    def test_metadata_precedes_the_statements(self):
        meta = {"label": "Tom Hanks", "description": "American actor", "aliases": []}
        with patch(_POOL_META, return_value=meta), \
             patch(_POOL_FETCH, return_value=[_oscar("A")]):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert "description" in lines[0]
        assert len(lines) == 2


class TestValueDescriptionEnrichment:
    """
    The named post-TEST option (VALUE_DESC_MODE="collision"): on a label
    collision the value entity's description is appended, so the answering
    model and the encoder can finally tell the two same-named entities apart.

    ⚠️ "off" is the FROZEN configuration — the first test pins that the default
    changes nothing, because every result of record ran without enrichment.
    """

    @staticmethod
    def _gwtw(desc_by_qid=None):
        """`[Gone With The Wind] based on: Gone with the Wind` — the audit's case."""
        film_to_novel = Statement(
            subject_id="Q2875", subject_label="Gone With The Wind",
            property_id="P144", property_label="based on",
            value=_v("Gone with the Wind", id="Q192724"),
            rank="normal", direction="outgoing", source_entity_id="Q2875",
            statement_id="S1",
        )

        def meta(qid):
            return {"label": None, "aliases": [],
                    "description": (desc_by_qid or {}).get(qid)}

        return film_to_novel, meta

    def _pool_lines(self, mode, statement, meta_side_effect):
        with patch("src.retrieval.wikidata_pool.VALUE_DESC_MODE", mode), \
             patch(_POOL_META, side_effect=meta_side_effect), \
             patch(_POOL_FETCH, return_value=[statement]):
            lines, _, _ = build_pool(["Q2875"], {"Q2875": "Gone With The Wind"})
        return lines

    def test_off_mode_is_a_strict_no_op(self):
        st, meta = self._gwtw({"Q192724": "1936 novel by Margaret Mitchell"})
        lines = self._pool_lines("off", st, meta)
        assert lines == ["[Gone With The Wind] based on: Gone with the Wind"]

    def test_collision_appends_the_value_entitys_description(self):
        st, meta = self._gwtw({"Q192724": "1936 novel by Margaret Mitchell"})
        lines = self._pool_lines("collision", st, meta)
        assert lines == ["[Gone With The Wind] based on: "
                        "Gone with the Wind (1936 novel by Margaret Mitchell)"]

    def test_collision_is_case_insensitive(self):
        """The encoder is uncased, so a case-only difference is still ambiguous."""
        st, meta = self._gwtw({"Q192724": "1936 novel"})
        assert st.subject_label != st.value.label  # differs only in case
        lines = self._pool_lines("collision", st, meta)
        assert "(1936 novel)" in lines[0]

    def test_non_colliding_lines_are_untouched(self):
        with patch("src.retrieval.wikidata_pool.VALUE_DESC_MODE", "collision"), \
             patch(_POOL_FETCH, return_value=[_oscar("A")]):
            lines, _, _ = build_pool(["Q2263"], {"Q2263": "Tom Hanks"})
        assert lines == ["[Tom Hanks] award received: Academy Award for Best Actor"]

    def test_missing_description_leaves_the_line_alone(self):
        """A collision with nothing to append must not render '(None)'."""
        st, meta = self._gwtw()   # no description for the novel
        lines = self._pool_lines("collision", st, meta)
        assert lines == ["[Gone With The Wind] based on: Gone with the Wind"]

    def test_string_valued_collisions_are_skipped(self):
        """No QID, nothing to describe — `title: X` self-description stays as is."""
        st = Statement(
            subject_id="Q1", subject_label="X",
            property_id="P1476", property_label="title",
            value=SnakValue("value", "monolingualtext", label="X"),
            rank="normal", direction="outgoing", source_entity_id="Q1",
            statement_id="S1",
        )
        with patch("src.retrieval.wikidata_pool.VALUE_DESC_MODE", "collision"), \
             patch(_POOL_FETCH, return_value=[st]), \
             patch(_POOL_META) as meta_mock:
            meta_mock.return_value = dict(_NO_META)
            lines, _, _ = build_pool(["Q1"], {"Q1": "X"})
        assert lines == ["[X] title: X"]

    def test_descriptions_are_memoized_per_qid(self):
        """Collision values recur across questions; each distinct QID may cost a
        live query, so the second occurrence must come from the memo."""
        st, _ = self._gwtw()
        calls = []

        def meta(qid):
            calls.append(qid)
            return {"label": None, "aliases": [], "description": "1936 novel"}

        with patch("src.retrieval.wikidata_pool.VALUE_DESC_MODE", "collision"), \
             patch(_POOL_META, side_effect=meta), \
             patch(_POOL_FETCH, return_value=[st, _oscar("A", qid="Q2875",
                                                         subject_label="Gone With The Wind")]):
            build_pool(["Q2875"], {"Q2875": "Gone With The Wind"})
            build_pool(["Q2875"], {"Q2875": "Gone With The Wind"})
        assert calls.count("Q192724") == 1

    def test_the_frozen_default_is_off(self):
        """The env default ships as the frozen configuration."""
        from src.retrieval.wikidata_pool import VALUE_DESC_MODE, VALUE_DESC_MODES
        assert VALUE_DESC_MODES == ("off", "collision")
        assert VALUE_DESC_MODE == "off"


class TestEmptyPool:
    def test_no_statements_yields_three_empty_lists(self):
        """83 dataset questions have no Wikidata anchor at all."""
        with patch(_POOL_FETCH, return_value=[]):
            assert build_pool(["Q1"], {}) == ([], [], [])

    def test_no_entities_does_not_call_the_fetcher(self):
        with patch(_POOL_FETCH) as m:
            assert build_pool([], {}) == ([], [], [])
        m.assert_not_called()


class TestConfigsShareOnePool:
    """
    ⚠️ The premise of the C3-vs-C4 comparison, pinned in code.

    If these two ever reach the ranker with different input, the condensing step
    is no longer the only difference between the configs and every C4-vs-C3
    number becomes uninterpretable — without anything failing loudly.
    """

    @staticmethod
    def _ranker_input(module, mock_client, stmts):
        seen = {}

        def spy(question, chunks, top_k=3, groups=None):
            seen["chunks"], seen["groups"], seen["top_k"] = chunks, groups, top_k
            return "\n\n".join(chunks[:top_k])

        with patch(_CLIENT, return_value=mock_client), \
             patch(_POOL_FETCH, return_value=stmts), \
             patch(f"{module.__name__}.embed_retrieve", side_effect=spy):
            module.answer("Which Oscars did he win?", ["Tom Hanks"], qids=["Q2263"])
        return seen

    def test_c3_and_c4_hand_the_ranker_identical_input(self):
        from tests.unit.test_pipelines import _openai_mock
        stmts = [_oscar("A", _when(1994)), _oscar("B", _when(1995))]

        c3 = self._ranker_input(graph_rag, _openai_mock("Philadelphia"), stmts)

        c4_mock = _openai_mock("Philadelphia")
        c4_mock.chat.completions.create.side_effect = None
        c4 = self._ranker_input(graph_rag_rerank, c4_mock, stmts)

        assert c3["chunks"] == c4["chunks"]
        assert c3["groups"] == c4["groups"]
        assert c3["top_k"] == c4["top_k"], "retrieval DEPTH must stay aligned too"

    def test_both_configs_call_the_same_builder(self):
        """A future edit that inlines the loop back into one config would fail here."""
        assert graph_rag.build_pool is graph_rag_rerank.build_pool is build_pool
