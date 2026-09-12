"""
Unit tests for tools/document_run.py — the run-provenance layer.

WHY THESE EXIST. `collect_specs` called `prompt_style(role)` positionally, but
that function's first parameter is `model`. Python raised nothing: "answer" was
looked up as a model name, missed `_PROMPT_STYLES`, and fell back to the default
OpenAI style — for BOTH roles, since the role argument then took its own
default. Every Nemotron run documented before 2026-08-10 therefore recorded

    system: "You are a factual question-answering assistant. ..."

when the calls that produced those answers actually sent `"detailed thinking
off"` with the persona and the brevity line in the user preamble. In a project
where one line of prompt wording is worth tens of points of abstention, that is
not a cosmetic metadata slip — it is provenance that says the run was something
it was not, and it is invisible unless something asserts on it.

So the tests below pin the property that matters (the recorded framing is the
framing this model would actually be called with), not the call syntax.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_document_run.py -v
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import llm_config

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_NEMOTRON = "nvidia/llama-3.3-nemotron-super-49b-v1"


def _load_document_run():
    """tools/ is not a package, so load the module by path."""
    spec = importlib.util.spec_from_file_location(
        "document_run", os.path.join(_ROOT, "tools", "document_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


document_run = _load_document_run()


class TestPromptFramingIsRecorded:
    """The framing in meta.json must match what build_messages would send."""

    def test_nemotron_framing_is_recorded_not_the_openai_default(self, monkeypatch):
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        framing = document_run.collect_specs()["prompt_framing"]

        assert framing["answer"]["system"] == "detailed thinking off"
        assert framing["condense"]["system"] == "detailed thinking off"
        # The exact failure this guards: the persona belongs in the user
        # preamble on this model, never in the system slot.
        assert llm_config._PERSONA not in (framing["answer"]["system"] or "")
        assert llm_config._PERSONA in framing["answer"]["preamble"]

    def test_the_brevity_line_is_visible_on_the_answering_role(self, monkeypatch):
        """~54 points of C3 abstention ride on this line; it must be in the record."""
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        framing = document_run.collect_specs()["prompt_framing"]

        assert llm_config._BREVITY in framing["answer"]["preamble"]
        # Config 4's condensing call is asked for prose and deliberately does
        # NOT carry it. Recording one framing for both roles hid that too.
        assert llm_config._BREVITY not in framing["condense"]["preamble"]

    def test_the_two_roles_are_not_recorded_as_identical(self, monkeypatch):
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        framing = document_run.collect_specs()["prompt_framing"]
        assert framing["answer"] != framing["condense"]

    def test_default_model_still_records_the_default_style(self, monkeypatch):
        """Every gpt-4o-mini result was produced on this framing — it must not move."""
        monkeypatch.setattr(llm_config, "MODEL", "gpt-4o-mini")
        framing = document_run.collect_specs()["prompt_framing"]

        assert framing["answer"] == {"system": llm_config._PERSONA, "preamble": ""}
        # system=None means "send no system message", which is what C4's
        # condensing call has always done. Distinct from the empty string.
        assert framing["condense"] == {"system": None, "preamble": ""}

    def test_recorded_framing_matches_what_build_messages_sends(self, monkeypatch):
        """The record is checked against the caller, not against a copied literal."""
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        framing = document_run.collect_specs()["prompt_framing"]

        for role in ("answer", "condense"):
            msgs = llm_config.build_messages("BODY", role=role)
            system = next((m["content"] for m in msgs if m["role"] == "system"), None)
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assert framing[role]["system"] == system
            assert user == framing[role]["preamble"] + "BODY"

    def test_passing_the_role_positionally_is_now_an_error(self):
        """
        The original miscall must raise rather than return a plausible value.

        Note the keyword-only `*` is NOT sufficient by itself: a single
        positional argument still binds legally to `model`, which is exactly
        the shape document_run.py used. The explicit role-name rejection in
        prompt_style is what closes it, so both shapes are pinned here.
        """
        import pytest

        for bad in ("answer", "condense"):
            with pytest.raises(TypeError):
                llm_config.prompt_style(bad)               # the original miscall
            with pytest.raises(TypeError):
                llm_config.prompt_style(None, bad)         # two-positional form

    def test_legitimate_calls_still_work(self):
        """The guard must not reject a real model name or the no-arg form."""
        assert llm_config.prompt_style()["system"] is not None or True
        assert set(llm_config.prompt_style(_NEMOTRON, role="condense")) == {"system", "preamble"}

    def test_collect_specs_does_not_mutate_the_live_style(self, monkeypatch):
        """meta.json is a snapshot; editing it must not reach back into config."""
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        framing = document_run.collect_specs()["prompt_framing"]
        framing["answer"]["system"] = "TAMPERED"

        assert llm_config.prompt_style(role="answer")["system"] == "detailed thinking off"


class TestRetrievalSpecNamesExist:
    """
    A spec field that reads a renamed constant must fail, not record a null.

    `getattr(wikidata, "FILTER_MODE", None)` shipped for months against a live
    name of NOISE_FILTER_MODE, so every archived meta.json said
    `"noise_filter_mode": null` while the runs used `property_type`. The
    getattr default is the whole bug: it converts a wrong name into a
    plausible-looking value. These read the same attributes the tool reads.
    """

    def test_no_retrieval_field_is_silently_none(self):
        retrieval = document_run.collect_specs()["retrieval"]
        missing = [f"{section}.{k}" for section, fields in retrieval.items()
                   for k, v in fields.items() if v is None]
        assert not missing, f"recorded as null instead of raising: {missing}"

    def test_noise_filter_mode_is_a_real_declared_mode(self):
        from src.retrieval import wikidata

        mode = document_run.collect_specs()["retrieval"]["active"]["noise_filter_mode"]
        assert mode in wikidata._NOISE_FILTER_MODES

    def test_every_retrieval_constant_is_read_from_a_name_that_exists(self):
        """Guards renames in either direction — the tool and the module must agree."""
        from src.retrieval import wikidata, embedding_retriever

        for mod, name in [(wikidata, "NOISE_FILTER_MODE"),
                          (wikidata, "REVERSE_PER_PROP_CAP"),
                          (wikidata, "STATEMENT_CACHE_VERSION"),
                          (wikidata, "STATEMENT_ROW_LIMIT"),
                          (wikidata, "RANK_POLICY"),
                          (wikidata, "LABEL_LANGS"),
                          (wikidata, "STRICT_RETRIEVAL"),
                          (wikidata, "REVERSE_DISCOVERY_CHUNK"),
                          (wikidata, "REVERSE_FETCH_BATCH"),
                          (wikidata, "HOPS"),
                          (embedding_retriever, "MODEL_NAME")]:
            assert hasattr(mod, name), f"{mod.__name__}.{name} no longer exists"

    def test_top_k_is_recorded_for_all_three_retrieval_configs(self):
        top_k = document_run.collect_specs()["retrieval"]["active"]["top_k"]
        assert set(top_k) == {"C2", "C3", "C4"}
        assert all(isinstance(v, int) for v in top_k.values())

    def test_the_statement_model_settings_are_recorded(self):
        """🔴 None of these were captured until 2026-08-13, so a meta.json could
        not distinguish a pre-`mul`-fix run from a post-fix one."""
        active = document_run.collect_specs()["retrieval"]["active"]
        for k in ("statement_cache_version", "statement_row_limit", "rank_policy",
                  "label_langs", "strict_retrieval", "follows_redirects",
                  "statement_query_attempts", "cache_fingerprint"):
            assert k in active, f"{k} is not recorded in the run provenance"

    def test_dormant_settings_are_separated_from_active(self):
        """A flat list made `HOPS = 1` read as a live decision, not a parked axis."""
        retrieval = document_run.collect_specs()["retrieval"]
        assert set(retrieval) == {"active", "dormant"}
        assert "hops" in retrieval["dormant"]

    def test_the_retired_truthy_settings_are_not_recorded_as_active(self):
        """
        `reverse_limit` sat in `active` until 2026-08-16 and was NEVER active: it
        capped the truthy reverse query, which the statement path never ran. A
        setting recorded as active but read by nothing is worse than an absent
        one — it reads as provenance.
        """
        retrieval = document_run.collect_specs()["retrieval"]
        assert "reverse_limit" not in retrieval["active"]
        assert "legacy_truthy" not in retrieval

    def test_generation_caps_are_recorded(self):
        gen = document_run.collect_specs()["generation"]
        assert gen["c4_condense_max_tokens"] == 384
        assert gen["answer_max_tokens"] > 0


class TestArmOverrides:
    """A prompt-variant arm rewrites the framing; the record must say which."""

    def test_per_arm_brevity_line_is_captured(self):
        raw = {"arms": {
            "C_prime": {"brevity_line": "LINE C", "rows": []},
            "E": {"brevity_line": "LINE E", "rows": []},
        }}
        assert document_run.arm_overrides(raw) == {
            "C_prime": {"brevity_line": "LINE C"},
            "E": {"brevity_line": "LINE E"},
        }

    def test_an_arm_that_removed_the_line_records_none_not_absence(self):
        """Variant B ran with NO brevity line. That is a value, not a gap."""
        raw = {"arms": {"B_no_brevity": {"brevity_line": None, "rows": []}}}
        assert document_run.arm_overrides(raw) == {"B_no_brevity": {"brevity_line": None}}

    def test_four_config_runs_have_no_overrides(self):
        """run_eval output shares one framing across configs — nothing to override."""
        raw = [{"id": "q1", "answers": {"base_llm_abstain": "Bono"}}]
        assert document_run.arm_overrides(raw) == {}


class TestDirtyFlagIgnoresRunOutputs:
    """
    The dirty flag must mean "the code differs from the SHA", not "a run wrote
    its own output". Every post-rebuild run through 2026-08-17 recorded
    scorer_tree_dirty=true because its own untracked directory under
    data/results/ sat in porcelain at scoring time — a flag that is always on
    carries no information, which defeats the reason it is recorded.
    """

    def test_run_outputs_are_filtered(self, monkeypatch):
        from src import experiment_spec

        monkeypatch.setattr(
            experiment_spec, "_git_status_porcelain",
            lambda strict=False: ["data/results/dev_runs/20260817_2040_x/meta.json",
                     "data\\results\\dev_runs\\20260817_2040_x\\report.md"])
        assert experiment_spec.code_dirty_paths() == []

    def test_code_paths_survive_the_filter(self, monkeypatch):
        from src import experiment_spec

        monkeypatch.setattr(
            experiment_spec, "_git_status_porcelain",
            lambda strict=False: ["src/eval/metrics.py",
                     "data/results/dev_runs/20260817_2040_x/meta.json"])
        assert experiment_spec.code_dirty_paths() == ["src/eval/metrics.py"]

    def test_data_outside_results_still_counts_as_dirty(self, monkeypatch):
        """Only RUN OUTPUTS are excluded — an edited questions file is real dirt."""
        from src import experiment_spec

        monkeypatch.setattr(
            experiment_spec, "_git_status_porcelain",
            lambda strict=False: ["data/questions/mintaka_sample_dev_200.json"])
        assert experiment_spec.code_dirty_paths() == [
            "data/questions/mintaka_sample_dev_200.json"]

    def test_collect_specs_uses_the_filtered_view(self, monkeypatch):
        from src import experiment_spec

        monkeypatch.setattr(
            experiment_spec, "_git_status_porcelain",
            lambda strict=False: ["data/results/dev_runs/20260817_2040_x/raw.json"])
        git = experiment_spec.collect_specs()["git"]
        assert git["dirty"] is False
        assert git["dirty_files"] == []


class TestCondenseFallbackDistinction:
    """
    An empty-pool fallback is an EMPTY C3-style context (header line alone),
    not "answered under raw facts" — all nine TEST-4000 fallbacks were
    empty-pool, and the old report wording called them "effectively answered
    as Configuration 3", which was wrong. The summary must count the two
    kinds apart so the rendered report can say which happened.
    """

    def _row(self, fallback, pool_size):
        return {"context": {"condense_fallback": fallback,
                            "pool_size": pool_size,
                            "condense_words": 10,
                            "condense_finish_reason": "stop"}}

    def test_empty_pool_fallbacks_are_counted_apart(self):
        rows = [self._row(True, 0), self._row(True, 12), self._row(False, 30)]
        c = document_run._condense_stats(rows)
        assert c["fallback_instrumented"] == 3
        assert c["fallbacks"] == 2
        assert c["fallbacks_empty_pool"] == 1

    def test_no_fallbacks_reports_zero_for_both(self):
        rows = [self._row(False, 30), self._row(False, 25)]
        c = document_run._condense_stats(rows)
        assert c["fallbacks"] == 0 and c["fallbacks_empty_pool"] == 0

    def test_context_words_column_is_labeled_as_a_mean(self):
        # The report prints the MEAN context size; the thesis quotes medians
        # from compare_context_sizes.py. The label must say which it is.
        headers = [h for h, key, _ in document_run.OUTCOME_COLUMNS
                   if key == "context_words"]
        assert headers == ["Mean CtxW"]
