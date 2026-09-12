"""
Unit tests for tools/demo_server.py — the demonstration interface.

Covers the import-safe helpers only: input validation, manual question
construction, the frozen-wins cache overlay, capture→evidence mapping,
exception→error-row mapping (strict failure path via mocks, never a live
endpoint), badge logic, and reference-configuration pinning.

No network, no API key, no server start: importing demo_server performs no
side effects beyond sys.path insertion (its contract).

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_demo_server.py -v
"""

import importlib.util
import json
import os
import sys
import threading

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _ROOT)

from src import llm_config  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.prompts import ABSTAIN_SENTINEL  # noqa: E402
from src.retrieval.wikidata import IncompleteRetrievalError  # noqa: E402


def _load_demo_server():
    """tools/ is not a package, so load the module by path."""
    spec = importlib.util.spec_from_file_location(
        "demo_server", os.path.join(_ROOT, "tools", "demo_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ds = _load_demo_server()


# ---------------------------------------------------------------------------
# validate_run_request
# ---------------------------------------------------------------------------

class TestValidateRunRequest:
    def _manual(self, **overrides):
        body = {"config": "graph_rag", "question": "Who directed Blade Runner?",
                "entities": [{"qid": "Q184843", "label": "Blade Runner",
                              "phrase": "blade runner"}]}
        body.update(overrides)
        return body

    def test_valid_manual_request(self):
        parsed, err = ds.validate_run_request(self._manual(), presets={})
        assert err is None
        assert parsed["config"] == "graph_rag"
        assert parsed["preset_id"] is None
        assert parsed["entities"][0]["qid"] == "Q184843"

    def test_valid_preset_request_ignores_question_fields(self):
        parsed, err = ds.validate_run_request(
            {"config": "rag", "preset_id": "abc"}, presets={"abc": object()})
        assert err is None
        assert parsed == {"config": "rag", "preset_id": "abc"}

    def test_unknown_preset_rejected(self):
        parsed, err = ds.validate_run_request(
            {"config": "rag", "preset_id": "nope"}, presets={})
        assert parsed is None and "preset" in err

    def test_unknown_config_rejected(self):
        _, err = ds.validate_run_request(self._manual(config="c5"), presets={})
        assert "config" in err

    @pytest.mark.parametrize("qid", ["Q0", "Q1x", "q5", "5", "Q01", "", None,
                                     "Q1/../../etc"])
    def test_invalid_qids_rejected_before_any_path_use(self, qid):
        body = self._manual(entities=[{"qid": qid, "label": "x", "phrase": ""}])
        parsed, err = ds.validate_run_request(body, presets={})
        assert parsed is None and err is not None

    def test_too_many_entities_rejected(self):
        ents = [{"qid": f"Q{i+1}", "label": "x", "phrase": ""} for i in range(21)]
        _, err = ds.validate_run_request(self._manual(entities=ents), presets={})
        assert "entities" in err

    def test_oversized_label_rejected(self):
        body = self._manual(entities=[{"qid": "Q5", "label": "x" * 301, "phrase": ""}])
        _, err = ds.validate_run_request(body, presets={})
        assert "label" in err

    @pytest.mark.parametrize("question", ["", "   ", "x" * 2001, None])
    def test_bad_question_rejected(self, question):
        _, err = ds.validate_run_request(self._manual(question=question), presets={})
        assert "question" in err

    def test_non_dict_body_rejected(self):
        parsed, err = ds.validate_run_request(["not", "a", "dict"], presets={})
        assert parsed is None and err is not None


# ---------------------------------------------------------------------------
# build_manual_question
# ---------------------------------------------------------------------------

class TestBuildManualQuestion:
    def test_phrase_differing_from_label_becomes_mention(self):
        q = ds.build_manual_question("Who directed it?", [
            {"qid": "Q184843", "label": "Blade Runner", "phrase": "the blade runner film"}])
        assert q.qids == ["Q184843"]
        # first_mention drives ranking: the pipelines receive the phrase.
        assert q.entity_mentions == ["the blade runner film"]
        # The display list carries the differing form, so the entity block
        # renders the mapping it exists to explain.
        block = q.prompt_block("all")
        assert "mentioned as" in block and "the blade runner film" in block

    def test_phrase_equal_to_label_yields_no_display_mention(self):
        q = ds.build_manual_question("Who directed it?", [
            {"qid": "Q184843", "label": "Blade Runner", "phrase": "blade runner"}])
        # Same rule as the parser: `mentions` holds ONLY label-differing forms
        # (case-insensitive), so the rendered line does not repeat itself...
        assert "mentioned as" not in q.prompt_block("all")
        # ...but the ranking input still preserves the user's own casing.
        assert q.entity_mentions == ["blade runner"]

    def test_empty_phrase_falls_back_to_label(self):
        q = ds.build_manual_question("Who?", [
            {"qid": "Q5", "label": "human", "phrase": ""}])
        assert q.entity_mentions == ["human"]

    def test_demo_id_and_no_entities(self):
        q = ds.build_manual_question("What is the capital of France?", [])
        assert q.id.startswith("demo-")
        assert q.qids == [] and q.prompt_block("all") == ""


# ---------------------------------------------------------------------------
# seed_cache_for — frozen always wins, seeded per configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dirs(tmp_path):
    dirs = {name: tmp_path / name for name in
            ("frozen_wd", "frozen_wp", "demo_wd", "demo_wp")}
    for d in dirs.values():
        d.mkdir()
    return dirs


class TestSeedCacheFor:
    def test_frozen_overwrites_stale_demo_entry(self, cache_dirs):
        (cache_dirs["frozen_wd"] / "Q42.json").write_text('{"v": "frozen"}')
        (cache_dirs["demo_wd"] / "Q42.json").write_text('{"v": "stale-demo"}')
        ds.seed_cache_for("graph_rag", ["Q42"], **cache_dirs)
        assert (cache_dirs["demo_wd"] / "Q42.json").read_text() == '{"v": "frozen"}'

    def test_demo_only_entry_survives_when_frozen_lacks_qid(self, cache_dirs):
        (cache_dirs["demo_wd"] / "Q7.json").write_text('{"v": "demo-only"}')
        ds.seed_cache_for("graph_rag", ["Q7"], **cache_dirs)
        assert (cache_dirs["demo_wd"] / "Q7.json").read_text() == '{"v": "demo-only"}'

    def test_rag_seeds_only_wikipedia(self, cache_dirs):
        (cache_dirs["frozen_wd"] / "Q42.json").write_text("wd")
        (cache_dirs["frozen_wp"] / "Q42.json").write_text("wp")
        ds.seed_cache_for("rag", ["Q42"], **cache_dirs)
        assert (cache_dirs["demo_wp"] / "Q42.json").exists()
        assert not (cache_dirs["demo_wd"] / "Q42.json").exists()

    @pytest.mark.parametrize("config", ["graph_rag", "rerank"])
    def test_graph_configs_seed_only_wikidata(self, cache_dirs, config):
        (cache_dirs["frozen_wd"] / "Q42.json").write_text("wd")
        (cache_dirs["frozen_wp"] / "Q42.json").write_text("wp")
        ds.seed_cache_for(config, ["Q42"], **cache_dirs)
        assert (cache_dirs["demo_wd"] / "Q42.json").exists()
        assert not (cache_dirs["demo_wp"] / "Q42.json").exists()

    def test_base_llm_touches_nothing(self, cache_dirs):
        (cache_dirs["frozen_wd"] / "Q42.json").write_text("wd")
        (cache_dirs["frozen_wp"] / "Q42.json").write_text("wp")
        ds.seed_cache_for("base_llm_abstain", ["Q42"], **cache_dirs)
        assert not any(cache_dirs["demo_wd"].iterdir())
        assert not any(cache_dirs["demo_wp"].iterdir())

    def test_copy_failure_raises_seed_error(self, cache_dirs, monkeypatch):
        (cache_dirs["frozen_wd"] / "Q42.json").write_text("wd")

        def boom(src_path, dst_path):
            raise OSError("disk on fire")
        monkeypatch.setattr(ds, "_atomic_copy", boom)
        with pytest.raises(ds.CacheSeedError):
            ds.seed_cache_for("graph_rag", ["Q42"], **cache_dirs)

    def test_invalid_qid_refused_defense_in_depth(self, cache_dirs):
        with pytest.raises(ds.CacheSeedError):
            ds.seed_cache_for("graph_rag", ["Q1/../evil"], **cache_dirs)


# ---------------------------------------------------------------------------
# evidence_from_capture
# ---------------------------------------------------------------------------

class TestEvidenceFromCapture:
    def test_c1_has_no_evidence(self):
        assert ds.evidence_from_capture("base_llm_abstain", {}) == {}

    def test_c2_splits_chunks_on_double_newline(self):
        cap = {"context": "chunk one text\n\nchunk two text",
               "articles": [{"qid": "Q1", "title": "T", "revision_id": 5, "n_chunks": 2}]}
        ev = ds.evidence_from_capture("rag", cap)
        assert ev["chunks"] == ["chunk one text", "chunk two text"]
        assert ev["articles"][0]["title"] == "T"

    def test_c3_statement_bullets_split_per_line_not_per_blob(self):
        # format_ranked joins bullets with SINGLE newlines — a "\n\n" split
        # would return one blob, which is exactly the bug this pins against.
        cap = {"context": "Wikidata facts about X:\n  • a: b\n  • c: d"}
        ev = ds.evidence_from_capture("graph_rag", cap)
        assert len(ev["statements"]) == 3
        assert ev["statements"][1] == "  • a: b"

    def test_c4_exposes_facts_and_condensed_prose(self):
        cap = {"context": "One condensed sentence.", "top_facts": ["a: b", "c: d"],
               "condense_finish_reason": "stop", "condense_words": 3,
               "condense_fallback": False}
        ev = ds.evidence_from_capture("rerank", cap)
        assert ev["top_facts"] == ["a: b", "c: d"]
        assert ev["condensed"] == "One condensed sentence."
        assert ev["condense_fallback"] is False


# ---------------------------------------------------------------------------
# error_payload — exception-specific messaging
# ---------------------------------------------------------------------------

class TestErrorPayload:
    def test_incomplete_retrieval_gets_endpoint_hint(self):
        p = ds.error_payload(IncompleteRetrievalError("reverse query failed"))
        assert p["type"] == "IncompleteRetrievalError"
        assert p["hint"] == ds.INCOMPLETE_HINT

    def test_generic_exception_gets_no_endpoint_gloss(self):
        # An LLM/embedding/programming failure must not be blamed on the
        # Wikidata endpoint.
        p = ds.error_payload(ValueError("bad input"))
        assert p["type"] == "ValueError" and "hint" not in p

    def test_seed_error_names_the_cache(self):
        p = ds.error_payload(ds.CacheSeedError("could not seed Q42"))
        assert "cache" in p["hint"]


# ---------------------------------------------------------------------------
# badges
# ---------------------------------------------------------------------------

class TestBadges:
    def test_sentinel_is_abstained_not_refusal_shaped(self):
        # Precondition making the invariant non-vacuous: the sentinel itself
        # opens with refusal language, so the grey badge must exclude it.
        assert metrics.opens_with_refusal(ABSTAIN_SENTINEL)
        flags = ds.badges(ABSTAIN_SENTINEL)
        assert flags["abstained"] is True
        assert flags["refusal_shaped"] is False

    def test_plain_answer_gets_no_badges(self):
        flags = ds.badges("Ridley Scott")
        assert flags == {"abstained": False, "refusal_shaped": False}

    def test_none_answer_is_safe(self):
        flags = ds.badges(None)
        assert flags == {"abstained": False, "refusal_shaped": False}


# ---------------------------------------------------------------------------
# token_delta
# ---------------------------------------------------------------------------

def test_token_delta_fieldwise():
    before = {"calls": 2, "prompt_tokens": 100, "completion_tokens": 10}
    after = {"calls": 4, "prompt_tokens": 350, "completion_tokens": 25}
    assert ds.token_delta(before, after) == {"calls": 2, "prompt": 250, "completion": 15}


# ---------------------------------------------------------------------------
# pin_reference_configuration
# ---------------------------------------------------------------------------

class TestPinReferenceConfiguration:
    def _snapshot(self):
        import src.pipelines.graph_rag as graph_rag
        import src.pipelines.graph_rag_rerank as rrk
        import src.pipelines.rag as rag
        import src.prompts as prompts
        import src.retrieval.embedding_retriever as emb
        import src.retrieval.wikidata as wikidata
        import src.retrieval.wikidata_pool as pool
        return {
            "model": llm_config.MODEL, "max": llm_config.MAX_TOKENS,
            "amax": rrk.ANSWER_MAX_TOKENS, "cmax": rrk.CONDENSE_MAX_TOKENS,
            "k2": rag.TOP_K, "k3": graph_rag.TOP_K, "k4": rrk.EMBED_TOP_K,
            "alloc": emb.ALLOCATION, "block": prompts.ENTITY_BLOCK_MODE,
            "vdesc": pool.VALUE_DESC_MODE, "noise": wikidata.NOISE_FILTER_MODE,
        }

    def _restore(self, snap):
        import src.pipelines.graph_rag as graph_rag
        import src.pipelines.graph_rag_rerank as rrk
        import src.pipelines.rag as rag
        import src.prompts as prompts
        import src.retrieval.embedding_retriever as emb
        import src.retrieval.wikidata as wikidata
        import src.retrieval.wikidata_pool as pool
        llm_config.MODEL = snap["model"]; llm_config.MAX_TOKENS = snap["max"]
        rrk.ANSWER_MAX_TOKENS = snap["amax"]; rrk.CONDENSE_MAX_TOKENS = snap["cmax"]
        rag.TOP_K = snap["k2"]; graph_rag.TOP_K = snap["k3"]; rrk.EMBED_TOP_K = snap["k4"]
        emb.ALLOCATION = snap["alloc"]; prompts.ENTITY_BLOCK_MODE = snap["block"]
        pool.VALUE_DESC_MODE = snap["vdesc"]; wikidata.NOISE_FILTER_MODE = snap["noise"]

    def test_reference_model_pins_all_axes(self, monkeypatch):
        import src.pipelines.graph_rag_rerank as rrk
        monkeypatch.setenv("LLM_BASE_URL", ds.REFERENCE_BASE_URL)
        snap = self._snapshot()
        try:
            llm_config.MAX_TOKENS = 999          # simulate a drifted .env
            rrk.ANSWER_MAX_TOKENS = 999          # the import-latched copy
            non_ref = ds.pin_reference_configuration(ds.REFERENCE_MODEL)
            assert non_ref == []
            assert llm_config.MODEL == ds.REFERENCE_MODEL
            assert llm_config.MAX_TOKENS == 128
            assert rrk.ANSWER_MAX_TOKENS == 128
            assert rrk.CONDENSE_MAX_TOKENS == 384
        finally:
            self._restore(snap)

    def test_non_reference_model_is_reported(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", ds.REFERENCE_BASE_URL)
        snap = self._snapshot()
        try:
            non_ref = ds.pin_reference_configuration("mistralai/mistral-nemotron")
            assert any(a.startswith("model=") for a in non_ref)
        finally:
            self._restore(snap)

    def test_non_reference_base_url_is_reported(self, monkeypatch):
        """The reference model name served by another provider is NOT reference."""
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        snap = self._snapshot()
        try:
            non_ref = ds.pin_reference_configuration(ds.REFERENCE_MODEL)
            assert any(a.startswith("base_url=") for a in non_ref)
            assert not any(a.startswith("model=") for a in non_ref)
        finally:
            self._restore(snap)

    def test_unset_base_url_is_reported(self, monkeypatch):
        """Unset resolves to OpenAI's default endpoint, which is not reference."""
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        snap = self._snapshot()
        try:
            non_ref = ds.pin_reference_configuration(ds.REFERENCE_MODEL)
            assert "base_url=(unset)" in non_ref
        finally:
            self._restore(snap)

    def test_base_url_compare_ignores_trailing_slash_and_case(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", ds.REFERENCE_BASE_URL.upper() + "/")
        snap = self._snapshot()
        try:
            assert ds.pin_reference_configuration(ds.REFERENCE_MODEL) == []
        finally:
            self._restore(snap)


# ---------------------------------------------------------------------------
# execute_run — strict failure path via mocks (never a live endpoint)
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_state(tmp_path, monkeypatch):
    # Other files exercise mocked LLM usage; isolate this fixture's token log
    # from any counter state left by those tests, regardless of test order.
    monkeypatch.setattr(ds.token_counter, "_usage", {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
    })
    dirs = {}
    for key in ("frozen_wd", "frozen_wp", "demo_wd", "demo_wp"):
        dirs[key] = tmp_path / key
        dirs[key].mkdir()
    return {**dirs, "presets": {}, "golds": {}, "preset_order": [],
            "embedder_state": "ready", "embedder_error": None,
            "non_reference_axes": [], "model_is_reference": True,
            "run_lock": threading.Lock()}


def _manual_parsed(config):
    return {"config": config, "preset_id": None,
            "question": "Who directed Blade Runner?",
            "entities": [{"qid": "Q184843", "label": "Blade Runner", "phrase": ""}]}


class TestExecuteRun:
    def test_incomplete_retrieval_becomes_error_row_with_hint(self, demo_state, monkeypatch):
        def raising_dispatch(config, q, capture):
            raise IncompleteRetrievalError("reverse query failed for Q184843")
        monkeypatch.setattr(ds, "dispatch", raising_dispatch)
        row = ds.execute_run(_manual_parsed("graph_rag"), demo_state)
        assert row["answer"] is None
        assert row["error"]["type"] == "IncompleteRetrievalError"
        assert row["error"]["hint"] == ds.INCOMPLETE_HINT
        # A failed call must not report a finish reason (it may retain the
        # PRECEDING call's thread-local value).
        assert row["finish_reason"] is None

    def test_generic_failure_gets_no_endpoint_gloss(self, demo_state, monkeypatch):
        def raising_dispatch(config, q, capture):
            raise RuntimeError("provider 500")
        monkeypatch.setattr(ds, "dispatch", raising_dispatch)
        row = ds.execute_run(_manual_parsed("rag"), demo_state)
        assert row["error"]["type"] == "RuntimeError"
        assert "hint" not in row["error"]

    def test_successful_run_row_shape(self, demo_state, monkeypatch):
        def fake_dispatch(config, q, capture):
            capture["context"] = "Wikidata facts about X:\n  • a: b"
            capture["pool_size"] = 12
            return "Ridley Scott"
        monkeypatch.setattr(ds, "dispatch", fake_dispatch)
        row = ds.execute_run(_manual_parsed("graph_rag"), demo_state)
        assert row["error"] is None
        assert row["answer"] == "Ridley Scott"
        assert row["abstained"] is False
        assert row["pool_size"] == 12
        assert row["context_words"] == 7
        assert row["evidence"]["statements"]
        # The mocked dispatch made no LLM call, so the token delta is zero
        # and the row must say so rather than display 0/0 as a measurement.
        assert row["usage_unavailable"] is True

    def test_abstention_row_flags_sentinel(self, demo_state, monkeypatch):
        def fake_dispatch(config, q, capture):
            capture["context"] = "ctx"
            capture["pool_size"] = 0
            return ABSTAIN_SENTINEL
        monkeypatch.setattr(ds, "dispatch", fake_dispatch)
        row = ds.execute_run(_manual_parsed("rag"), demo_state)
        assert row["abstained"] is True and row["refusal_shaped"] is False
        assert row["pool_size"] == 0

    def test_c1_has_null_context_words(self, demo_state, monkeypatch):
        monkeypatch.setattr(ds, "dispatch", lambda config, q, capture: "Paris")
        row = ds.execute_run(_manual_parsed("base_llm_abstain"), demo_state)
        assert row["context_words"] is None

    def test_seed_failure_is_an_error_row_not_a_live_fetch(self, demo_state, monkeypatch):
        (demo_state["frozen_wd"] / "Q184843.json").write_text("{}")

        def boom(src_path, dst_path):
            raise OSError("copy failed")
        monkeypatch.setattr(ds, "_atomic_copy", boom)
        dispatched = []
        monkeypatch.setattr(ds, "dispatch",
                            lambda config, q, capture: dispatched.append(config))
        row = ds.execute_run(_manual_parsed("graph_rag"), demo_state)
        assert row["error"]["type"] == "CacheSeedError"
        assert dispatched == []  # the pipeline never ran


# ---------------------------------------------------------------------------
# log_call — the per-call record (question, QIDs, stats)
# ---------------------------------------------------------------------------

def _read_log(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestCallLog:
    def test_absent_call_log_key_writes_nothing(self, demo_state, monkeypatch, tmp_path):
        """The unit fixtures omit the key; logging must then be a no-op."""
        assert "call_log" not in demo_state
        monkeypatch.setattr(ds, "dispatch", lambda config, q, capture: "Ridley Scott")
        ds.execute_run(_manual_parsed("base_llm_abstain"), demo_state)
        assert list(tmp_path.glob("**/*.jsonl")) == []

    def test_records_question_qids_and_stats(self, demo_state, monkeypatch, tmp_path):
        log = tmp_path / "calls.jsonl"
        demo_state["call_log"] = log

        def fake_dispatch(config, q, capture):
            capture["context"] = "Wikidata facts about X:\n  • a: b"
            capture["pool_size"] = 12
            return "Ridley Scott"
        monkeypatch.setattr(ds, "dispatch", fake_dispatch)
        ds.execute_run(_manual_parsed("graph_rag"), demo_state)

        (rec,) = _read_log(log)
        assert rec["question"] == "Who directed Blade Runner?"
        assert rec["qids"] == ["Q184843"]
        assert rec["config"] == "graph_rag"
        assert rec["mode"] == "manual"
        assert rec["answer"] == "Ridley Scott"
        assert rec["pool_size"] == 12
        assert rec["context_words"] == 7
        assert rec["abstained"] is False
        assert isinstance(rec["latency_s"], float)
        assert rec["error"] is None

    def test_record_is_self_describing_about_the_model(self, demo_state, monkeypatch,
                                                       tmp_path):
        """A line logged under a non-reference endpoint must say so itself."""
        log = tmp_path / "calls.jsonl"
        demo_state["call_log"] = log
        demo_state["model_is_reference"] = False
        demo_state["non_reference_axes"] = ["base_url=https://openrouter.ai/api/v1"]
        monkeypatch.setattr(ds, "dispatch", lambda config, q, capture: "x")
        ds.execute_run(_manual_parsed("base_llm_abstain"), demo_state)

        (rec,) = _read_log(log)
        assert rec["model_is_reference"] is False
        assert rec["non_reference_axes"] == ["base_url=https://openrouter.ai/api/v1"]
        assert rec["model"] == llm_config.MODEL

    def test_appends_one_line_per_call(self, demo_state, monkeypatch, tmp_path):
        log = tmp_path / "calls.jsonl"
        demo_state["call_log"] = log
        monkeypatch.setattr(ds, "dispatch", lambda config, q, capture: "a")
        for config in ("base_llm_abstain", "rag", "graph_rag", "rerank"):
            ds.execute_run(_manual_parsed(config), demo_state)

        records = _read_log(log)
        assert [r["config"] for r in records] == list(ds.CONFIGS)
        # Four lines for ONE question: they share it and differ in config.
        assert {r["question"] for r in records} == {"Who directed Blade Runner?"}

    def test_error_rows_are_logged_too(self, demo_state, monkeypatch, tmp_path):
        """A failed demonstration is exactly the one worth having a record of."""
        log = tmp_path / "calls.jsonl"
        demo_state["call_log"] = log

        def raising_dispatch(config, q, capture):
            raise RuntimeError("provider 500")
        monkeypatch.setattr(ds, "dispatch", raising_dispatch)
        ds.execute_run(_manual_parsed("rag"), demo_state)

        (rec,) = _read_log(log)
        assert rec["answer"] is None
        assert rec["error"]["type"] == "RuntimeError"

    def test_write_failure_does_not_fail_the_run(self, demo_state, monkeypatch, tmp_path):
        """Logging is telemetry: it must never break a run that succeeded."""
        demo_state["call_log"] = tmp_path        # a directory, not a file
        monkeypatch.setattr(ds, "dispatch", lambda config, q, capture: "Ridley Scott")
        row = ds.execute_run(_manual_parsed("base_llm_abstain"), demo_state)
        assert row["answer"] == "Ridley Scott"
        assert row["error"] is None
