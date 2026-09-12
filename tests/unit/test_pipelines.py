"""
Unit tests for all four pipeline configs (Configs 1–4).
LLM and retrieval calls are mocked — no network or API key required.

The LLM client is shared via src.llm_config.get_client(), so a single
patch target ("src.llm_config.get_client") mocks the client for every config.

The literal tests verify that numbers and dates (Wikidata objects with no QID)
flow through the pipeline and appear verbatim in the prompt sent to the LLM.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_pipelines.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import src.pipelines.base_llm as base_llm
import src.pipelines.rag as rag
import src.pipelines.graph_rag as graph_rag
import src.pipelines.graph_rag_rerank as graph_rag_rerank
from src import llm_config, token_counter
from src.prompts import ABSTAIN_CLAUSE, ABSTAIN_SENTINEL
from src.eval import metrics
from src.retrieval.statement import Qualifier, SnakValue, Statement

# Single patch target for the shared client across all four configs.
_CLIENT = "src.llm_config.get_client"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# ⚠️ ONE patch target for both C3 and C4, because since 2026-08-12 they call the
# SAME pool builder (src/retrieval/wikidata_pool.py). Patching `fetch_statements` inside
# that module rather than mocking `build_pool` keeps the real rendering,
# deduplication and grouping under test — the pool builder is where a C3/C4
# divergence would now have to occur, so mocking it out would blind exactly the
# code that guarantees it cannot.
_POOL_FETCH = "src.retrieval.wikidata_pool.fetch_statements"
# The text arm's equivalent seam. Patched at the FETCH, not at
# `rag.build_pool`, so the pool's own work — group labelling and line dedup — is
# still exercised, exactly as _POOL_FETCH does for the KG arm.
#
# ⚠️ PATCHED IN THE POOL MODULE, NOT IN `wikipedia`. Both pools do
# `from ... import fetch_*`, which binds the name at import time, so patching
# the source module leaves the already-bound reference untouched — and the test
# then makes a LIVE network call and passes or fails on whatever the cache
# happens to hold. Caught exactly that way on 2026-08-16.
_WIKI_FETCH = "src.retrieval.wikipedia_pool.fetch_chunks"


@pytest.fixture(autouse=True)
def _no_entity_meta():
    """`build_pool` always asks for aliases/description; keep the suite offline."""
    # Mock usage fields must not leak into other files' token-log tests.
    token_counter.reset()
    try:
        with patch("src.retrieval.wikidata_pool.fetch_entity_meta",
                   return_value={"label": None, "description": None, "aliases": []}):
            yield
    finally:
        token_counter.reset()


def _stmt(prop_label: str, value: str, *, datatype: str = "string",
          qualifiers=(), subject_label: str = "Albert Einstein",
          subject_id: str = "Q937") -> Statement:
    return Statement(
        subject_id=subject_id, subject_label=subject_label,
        property_id="P0", property_label=prop_label,
        value=SnakValue("value", datatype, raw=value, label=value),
        rank="normal", qualifiers=tuple(qualifiers),
        direction="outgoing", source_entity_id=subject_id,
        statement_id=f"{subject_id}-{prop_label}-{value}",
    )


# Literals — values with no QID — are the case that must survive the whole chain
# and reach the prompt verbatim.
LITERAL_STATEMENTS = [
    _stmt("date of birth", "14 March 1879", datatype="time"),
    _stmt("number of children", "5", datatype="quantity"),
]


def wiki_chunks(qid: str, *texts: str) -> list:
    """`Chunk`s for one entity, as `wikipedia.fetch_chunks` would return them."""
    from src.retrieval.chunk import Chunk
    return [Chunk(text=t, source_entity_id=qid, title=f"{qid} article",
                  index=i, n_chunks=len(texts), revision_id=1000 + i)
            for i, t in enumerate(texts)]


def _openai_mock(reply: str = "14 March 1879") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=reply))]
    )
    return client


def _prompt_text(mock_client: MagicMock) -> str:
    """Return the full concatenated prompt text from the last LLM call."""
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    return " ".join(m["content"] for m in messages)


# ---------------------------------------------------------------------------
# Config 1 — Base LLM
# ---------------------------------------------------------------------------

class TestBaseLLM:
    def test_returns_string(self):
        with patch(_CLIENT, return_value=_openai_mock("42")):
            result = base_llm.answer("What is 6 times 7?")
        assert isinstance(result, str) and len(result) > 0

    def test_question_in_prompt(self):
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock):
            base_llm.answer("When was Einstein born?")
        assert "When was Einstein born?" in _prompt_text(mock)

    def test_prompt_permits_abstention(self):
        """
        C1 carries the abstention clause, so an abstention gap between C1 and the
        retrieval configs cannot be explained by mere permission to decline.
        Dropping it silently reintroduces that confound.
        """
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock):
            base_llm.answer("When was Einstein born?")
        assert ABSTAIN_CLAUSE in _prompt_text(mock)


class TestAbstentionClauseParity:
    """
    The clause must be BYTE-IDENTICAL across all four configs, and must name the
    exact string the scorer looks for.

    Two separate regressions are pinned here. (a) C1 carried a reworded variant
    ("if you do not know the answer") until 2026-08-06, which left prompt wording
    as a live confound alongside retrieval. (b) The clause used to say "say so",
    letting each model invent its own refusal phrasing -- Nemotron's terse "Not
    in context." escaped the detector 25 times on DEV-200, concentrated in C3/C4,
    turning refusals into apparent confident errors.
    """

    def test_all_four_configs_share_one_clause(self):
        templates = {
            "C1": base_llm.PROMPT_TEMPLATE,
            "C2": rag.PROMPT_TEMPLATE,
            "C3": graph_rag.PROMPT_TEMPLATE,
            "C4": graph_rag_rerank.ANSWER_PROMPT,
        }
        for name, tpl in templates.items():
            assert ABSTAIN_CLAUSE in tpl, f"{name} lost the shared abstention clause"

    def test_clause_names_the_sentinel_the_scorer_matches(self):
        # If these drift, models are asked for one string while the scorer waits
        # for another, and every abstention silently becomes a confident error.
        assert ABSTAIN_SENTINEL in ABSTAIN_CLAUSE
        assert metrics.is_abstention(ABSTAIN_SENTINEL)

    def test_condense_prompt_has_no_abstention_clause(self):
        # C4's step-2 condenser summarises facts; it is not an answerer, and
        # giving it permission to decline would let it refuse before the
        # answering call ever sees the triples.
        assert ABSTAIN_CLAUSE not in graph_rag_rerank.CONDENSE_PROMPT


class TestEntityBlockParity:
    """
    The question-entity block: same section, same place, in C2, C3 and BOTH of
    C4's calls — and never in C1 (decisions 3a-3d, 2026-08-16).

    Pinned rather than trusted for the same reason the abstention clause above
    is: a section that reaches three configs through four templates is three
    chances to drift, and a difference between them would be indistinguishable
    from a retrieval effect in the results.
    """

    BLOCK = "Question entities:\n- Q2263 (entity): Tom Hanks"

    def test_the_three_retrieval_configs_carry_the_placeholder(self):
        for name, tpl in (("C2", rag.PROMPT_TEMPLATE),
                          ("C3", graph_rag.PROMPT_TEMPLATE),
                          ("C4 answer", graph_rag_rerank.ANSWER_PROMPT),
                          ("C4 condense", graph_rag_rerank.CONDENSE_PROMPT)):
            assert "{entities}" in tpl, f"{name} lost the entity-block slot"

    def test_c1_has_no_slot_and_no_parameter(self):
        """C1 is the config the headline delta is measured against; if it ever
        gains the block, §5's C1 comparison changes meaning."""
        assert "{entities}" not in base_llm.PROMPT_TEMPLATE
        import inspect
        assert "entity_block" not in inspect.signature(base_llm.answer).parameters

    def test_the_block_precedes_the_context_in_every_config(self):
        for name, tpl, section in (("C2", rag.PROMPT_TEMPLATE, "Context:"),
                                   ("C3", graph_rag.PROMPT_TEMPLATE, "Context:"),
                                   ("C4 answer", graph_rag_rerank.ANSWER_PROMPT, "Context:"),
                                   ("C4 condense", graph_rag_rerank.CONDENSE_PROMPT, "Facts:")):
            assert tpl.index("{entities}") < tpl.index(section), \
                f"{name} renders the block after its evidence section"

    def test_an_absent_block_leaves_no_trace(self):
        """83 questions have no linked entity. Their prompt must be exactly the
        prompt from before this section existed — no header, no blank line."""
        from src.prompts import entity_section
        assert entity_section("") == ""
        filled = rag.PROMPT_TEMPLATE.format(entities=entity_section(""),
                                            context="ctx", question="q?")
        assert "Question entities:" not in filled
        assert "\n\n\n" not in filled

    def test_a_present_block_is_followed_by_a_blank_line(self):
        from src.prompts import entity_section
        filled = rag.PROMPT_TEMPLATE.format(entities=entity_section(self.BLOCK),
                                            context="ctx", question="q?")
        assert f"{self.BLOCK}\n\nContext:" in filled

    def test_both_of_c4s_calls_render_the_same_block(self):
        """C4-vs-C3 rests on the condensing call being the ONE step that
        differs; a block in only one of C4's calls would add a second."""
        from src.prompts import entity_section
        section = entity_section(self.BLOCK)
        condense = graph_rag_rerank.CONDENSE_PROMPT.format(
            entities=section, top_k_facts="- f", question="q?")
        answer = graph_rag_rerank.ANSWER_PROMPT.format(
            entities=section, context="ctx", question="q?")
        assert self.BLOCK in condense and self.BLOCK in answer


# ---------------------------------------------------------------------------
# Config 2 — RAG (Wikipedia text)
# ---------------------------------------------------------------------------

class TestRankingSymmetry:
    """
    ⭐ All three retrieval configs must share the SELECTION RULE, not just the
    encoder and the depth.

    Until 2026-08-12, `rag.py` called `embed_retrieve` without `groups` while
    graph_rag.py and graph_rag_rerank.py both passed it, so C2 ranked on plain
    global cosine and C3/C4 z-normalized within each source entity. Two project
    documents asserted "one identical ranking stage", which was true of the model
    and the depth and false of the selection rule — in the one comparison the
    research question names explicitly.
    """

    @staticmethod
    def _groups_passed_by(module, patches):
        seen = {}

        def spy(question, chunks, top_k=3, groups=None):
            seen["groups"] = groups
            return "ctx"

        with ExitStack() as stack:
            stack.enter_context(patch(_CLIENT, return_value=_openai_mock("x")))
            stack.enter_context(patch(f"{module}.embed_retrieve", side_effect=spy))
            for p in patches:
                stack.enter_context(p)
            sys.modules[module].answer("q?", ["A", "B"], qids=["Q1", "Q2"])
        return seen["groups"]

    def test_c2_passes_groups(self):
        groups = self._groups_passed_by(
            "src.pipelines.rag",
            [patch(_WIKI_FETCH, side_effect=[wiki_chunks("Q1", "a1", "a2"),
                                            wiki_chunks("Q2", "b1")])],
        )
        assert groups == ["Q1", "Q1", "Q2"], "C2 must z-normalize per source entity"

    def test_groups_stay_parallel_to_the_chunks(self):
        """
        A misaligned `groups` list would normalize by the WRONG entity — silently
        producing a different selection with no error anywhere.
        """
        seen = {}

        def spy(question, chunks, top_k=3, groups=None):
            seen["n_chunks"], seen["n_groups"] = len(chunks), len(groups)
            return "ctx"

        with patch(_CLIENT, return_value=_openai_mock("x")), \
             patch(_WIKI_FETCH,
                   side_effect=[wiki_chunks("Q1", "a1", "a2", "a3"), [],
                                wiki_chunks("Q3", "c1")]), \
             patch("src.pipelines.rag.embed_retrieve", side_effect=spy):
            rag.answer("q?", ["A", "B", "C"], qids=["Q1", "Q2", "Q3"])
        assert seen["n_chunks"] == seen["n_groups"] == 4

    @pytest.mark.parametrize("module,patches_factory", [
        ("src.pipelines.graph_rag", lambda: [patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS)]),
        ("src.pipelines.graph_rag_rerank", lambda: [patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS)]),
    ])
    def test_c3_and_c4_still_pass_groups(self, module, patches_factory):
        assert self._groups_passed_by(module, patches_factory()) is not None


class TestRAG:
    def test_returns_string(self):
        mock = _openai_mock("physicist")
        with patch(_CLIENT, return_value=mock), \
             patch(_WIKI_FETCH, return_value=wiki_chunks("Q1", "Albert Einstein was a physicist.")), \
             patch("src.pipelines.rag.embed_retrieve", return_value="Albert Einstein was a physicist."):
            result = rag.answer("Who was Einstein?", ["Albert Einstein"], qids=["Q937"])
        assert isinstance(result, str)

    def test_wikipedia_chunk_in_llm_prompt(self):
        """Text retrieved from Wikipedia must reach the LLM context."""
        chunk = "Albert Einstein was born in Ulm on 14 March 1879."
        mock = _openai_mock("14 March 1879")
        with patch(_CLIENT, return_value=mock), \
             patch(_WIKI_FETCH, return_value=wiki_chunks("Q1", chunk)), \
             patch("src.pipelines.rag.embed_retrieve", return_value=chunk):
            rag.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])
        assert "14 March 1879" in _prompt_text(mock)

    def test_no_chunks_uses_fallback_context(self):
        mock = _openai_mock("I don't know.")
        with patch(_CLIENT, return_value=mock), \
             patch(_WIKI_FETCH, return_value=[]), \
             patch("src.pipelines.rag.embed_retrieve", return_value=""):
            result = rag.answer("Who was Einstein?", ["Albert Einstein"], qids=["Q937"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Config 3 — Graph-RAG (literal triples)
# ---------------------------------------------------------------------------

class TestGraphRAG:
    def test_returns_string(self):
        mock = _openai_mock("14 March 1879")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag.embed_retrieve", return_value="date of birth: 14 March 1879"):
            result = graph_rag.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])
        assert isinstance(result, str)

    def test_date_literal_reaches_llm_prompt(self):
        """A date literal (no QID) from Wikidata must appear verbatim in the LLM prompt."""
        mock = _openai_mock("14 March 1879")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag.embed_retrieve", return_value="date of birth: 14 March 1879"):
            graph_rag.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])
        assert "14 March 1879" in _prompt_text(mock)

    def test_number_literal_reaches_llm_prompt(self):
        """A number literal (no QID) from Wikidata must appear verbatim in the LLM prompt."""
        mock = _openai_mock("5")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag.embed_retrieve", return_value="number of children: 5"):
            graph_rag.answer("How many children did Einstein have?", ["Albert Einstein"], qids=["Q937"])
        assert "number of children: 5" in _prompt_text(mock)

    def test_no_triples_returns_string(self):
        """Pipeline must not crash when the statement pool comes back empty."""
        mock = _openai_mock("I don't know.")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=[]), \
             patch("src.pipelines.graph_rag.embed_retrieve", return_value=""):
            result = graph_rag.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Config 4 — Graph-RAG + LLM condensing
# ---------------------------------------------------------------------------

class TestGraphRAGRerank:
    def _two_call_mock(self, condense_reply: str, answer_reply: str) -> MagicMock:
        """Returns a mock that gives different responses for the two LLM calls."""
        client = MagicMock()
        replies = iter([condense_reply, answer_reply])
        client.chat.completions.create.side_effect = lambda **kw: MagicMock(
            choices=[MagicMock(message=MagicMock(content=next(replies)))]
        )
        return client

    def test_date_literal_in_condensing_prompt(self):
        """Date literal must appear in the first (condensing) LLM call."""
        calls = []

        def capture(**kw):
            calls.append(kw["messages"])
            reply = "Einstein was born on 14 March 1879." if len(calls) == 1 else "14 March 1879"
            return MagicMock(choices=[MagicMock(message=MagicMock(content=reply))])

        mock = MagicMock()
        mock.chat.completions.create.side_effect = capture

        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve",
                   return_value="date of birth: 14 March 1879"):
            graph_rag_rerank.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])

        condense_prompt = " ".join(m["content"] for m in calls[0])
        assert "14 March 1879" in condense_prompt

    def test_condensed_prose_reaches_answering_llm(self):
        """The prose output of the condensing call must become the context for the answering call."""
        calls = []
        condensed = "Einstein was born on 14 March 1879 in Ulm."

        def capture(**kw):
            calls.append(kw["messages"])
            reply = condensed if len(calls) == 1 else "14 March 1879"
            return MagicMock(choices=[MagicMock(message=MagicMock(content=reply))])

        mock = MagicMock()
        mock.chat.completions.create.side_effect = capture

        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve",
                   return_value="date of birth: 14 March 1879"):
            graph_rag_rerank.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])

        assert len(calls) == 2, "Config 4 must make exactly two LLM calls per question"
        answer_prompt = " ".join(m["content"] for m in calls[1])
        assert condensed in answer_prompt

    def test_no_triples_does_not_crash(self):
        mock = _openai_mock("I don't know.")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=[]), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve", return_value=""):
            result = graph_rag_rerank.answer("When was Einstein born?", ["Albert Einstein"], qids=["Q937"])
        assert isinstance(result, str)

    def test_empty_condensation_fallback_is_RECORDED(self):
        """An empty condensation silently answered under C3-style raw facts —
        a treatment change, not a cosmetic one (audit finding 2026-08-19). The
        capture flag is what lets a run say how often C4 was actually C3."""
        capture = {}
        mock = self._two_call_mock(condense_reply="", answer_reply="14 March 1879")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve",
                   return_value="date of birth: 14 March 1879"):
            graph_rag_rerank.answer("When was Einstein born?", ["Albert Einstein"],
                                    qids=["Q937"], capture=capture)
        assert capture["condense_fallback"] is True
        # The fallback context is C3's rendering, so the answering call still
        # received the ranked facts rather than an empty block.
        assert "14 March 1879" in capture["context"]

    def test_successful_condensation_records_fallback_False(self):
        """Explicit False, not absent: False marks the row as INSTRUMENTED, so
        a reader can tell "no fallback happened" from "a file written before
        the flag existed" (2026-08-19 audit refinement)."""
        capture = {}
        mock = self._two_call_mock(condense_reply="Einstein was born 14 March 1879.",
                                   answer_reply="14 March 1879")
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve",
                   return_value="date of birth: 14 March 1879"):
            graph_rag_rerank.answer("When was Einstein born?", ["Albert Einstein"],
                                    qids=["Q937"], capture=capture)
        assert capture["condense_fallback"] is False


# ---------------------------------------------------------------------------
# Retrieval depth (top_k) — shared contract across Configs 2, 3 and 4
#
# Two properties are pinned here:
#   1. All three retrieval configs default to the SAME depth. Equal top_k is the
#      reason depth is not a confound between them; a drift in one module would
#      quietly reintroduce it and nothing else would notice.
#   2. The default is resolved at CALL time from the module global, not frozen
#      as a default argument at import. The sweep tools set those globals between
#      batches, so a `top_k: int = TOP_K` signature would silently ignore them
#      and score every cell at the import-time depth — producing a sweep whose
#      cells differ in label only.
# ---------------------------------------------------------------------------

class TestRetrievalDepth:
    def test_all_retrieval_configs_share_one_default_depth(self):
        assert rag.TOP_K == graph_rag.TOP_K == graph_rag_rerank.EMBED_TOP_K

    @pytest.mark.parametrize("module, attr", [
        (rag, "TOP_K"),
        (graph_rag, "TOP_K"),
        (graph_rag_rerank, "EMBED_TOP_K"),
    ])
    def test_depth_is_thirty(self, module, attr):
        # Selected on the held-out DEV-200 split; see graph_rag.TOP_K.
        assert getattr(module, attr) == 30

    def test_graph_rag_reads_module_global_at_call_time(self):
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag.embed_retrieve", return_value="x") as embed, \
             patch("src.pipelines.graph_rag.TOP_K", 7):
            graph_rag.answer("q", ["Albert Einstein"], qids=["Q937"])
        assert embed.call_args.kwargs["top_k"] == 7

    def test_rerank_reads_module_global_at_call_time(self):
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve", return_value="x") as embed, \
             patch("src.pipelines.graph_rag_rerank.EMBED_TOP_K", 7):
            graph_rag_rerank.answer("q", ["Albert Einstein"], qids=["Q937"])
        assert embed.call_args.kwargs["top_k"] == 7

    def test_rag_reads_module_global_at_call_time(self):
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock), \
             patch(_WIKI_FETCH, return_value=wiki_chunks("Q1", "chunk")), \
             patch("src.pipelines.rag.embed_retrieve", return_value="x") as embed, \
             patch("src.pipelines.rag.TOP_K", 7):
            rag.answer("q", ["Albert Einstein"], qids=["Q937"])
        assert embed.call_args.kwargs["top_k"] == 7

    @pytest.mark.parametrize("module, patches", [
        (graph_rag, "src.pipelines.graph_rag"),
        (graph_rag_rerank, "src.pipelines.graph_rag_rerank"),
    ])
    def test_explicit_top_k_overrides_the_global(self, module, patches):
        # Config 4 gained this parameter on 2026-07-27; before that, run_eval.py
        # could not vary its depth at all.
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=LITERAL_STATEMENTS), \
             patch(f"{patches}.embed_retrieve", return_value="x") as embed:
            module.answer("q", ["Albert Einstein"], qids=["Q937"], top_k=3)
        assert embed.call_args.kwargs["top_k"] == 3

    def test_rag_explicit_top_k_overrides_the_global(self):
        mock = _openai_mock()
        with patch(_CLIENT, return_value=mock), \
             patch(_WIKI_FETCH, return_value=wiki_chunks("Q1", "chunk")), \
             patch("src.pipelines.rag.embed_retrieve", return_value="x") as embed:
            rag.answer("q", ["Albert Einstein"], qids=["Q937"], top_k=3)
        assert embed.call_args.kwargs["top_k"] == 3


class TestCondenseFinishReasonRecorded:
    """
    Config 4 is the only config that calls the LLM twice, and llm_config keeps
    ONE thread-local slot for finish_reason. The answering call overwrote the
    condensing call's value, so C4's first-call truncation was invisible:
    measured 71.5% of condensed contexts ending mid-sentence on the 2026-08-07
    reference run against a recorded finish_reason of 1/200.
    """

    def test_condense_finish_reason_survives_the_answering_call(self):
        # call 1 truncates, call 2 does not -- the naive read returns call 2's
        # value, so the capture must have taken call 1's at the time.
        mock = _openai_mock()
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="prose"),
                                         finish_reason="length")], usage=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Bono"),
                                         finish_reason="stop")], usage=None),
        ]
        mock.chat.completions.create.side_effect = responses
        capture: dict = {}
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=[
                 _stmt("member of", "U2", subject_label="Bono", subject_id="Q1")]), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve",
                   return_value="[U2] member of: Bono"):
            graph_rag_rerank.answer("Who sings for U2?", ["U2"], qids=["Q1"], capture=capture)

        assert capture["condense_finish_reason"] == "length"
        # and the slot itself now holds the ANSWERING call's value -- which is
        # exactly why the capture had to be taken early.
        assert llm_config.last_finish_reason() == "stop"

    def test_condense_words_recorded(self):
        mock = _openai_mock()
        mock.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="one two three"),
                                         finish_reason="stop")], usage=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content="x"),
                                         finish_reason="stop")], usage=None),
        ]
        capture: dict = {}
        with patch(_CLIENT, return_value=mock), \
             patch(_POOL_FETCH, return_value=[
                 _stmt("p", "o", subject_id="Q1")]), \
             patch("src.pipelines.graph_rag_rerank.embed_retrieve", return_value="[E] p: o"):
            graph_rag_rerank.answer("q?", ["E"], qids=["Q1"], capture=capture)
        assert capture["condense_words"] == 3
