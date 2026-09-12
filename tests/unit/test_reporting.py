"""
Unit tests for the reporting layer: outcome classification, hallucination rate,
per-question scores, and stratification (src/eval/metrics.py additions).

The invariant that matters most is that the three outcome buckets PARTITION:
every answer is abstained, correct or wrong — never two, never none. If that
breaks, the headline table stops summing to 100 and every comparison derived
from it is quietly wrong.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.eval import metrics

from src.eval.metrics import (
    ABSTAINED, CORRECT, WRONG, WRONG_THRESHOLD,
    classify, per_question_scores, score_results,
)

ABSTENTION_TEXT = "The answer is not in the context."


def _row(qid, pred_by_config, expected="Paris", answer_type="entity",
         complexity="generic", **extra):
    row = {
        "id": qid,
        "question": "q?",
        "expected": expected,
        "answer_entities": [expected],
        "answer_type": answer_type,
        "complexity": complexity,
        "answers": dict(pred_by_config),
    }
    row.update(extra)
    return row


class TestClassify:
    def test_abstention_is_never_a_hallucination(self):
        assert classify({"abstained": True, "score": 0.0}) == ABSTAINED

    def test_confident_error_is_wrong(self):
        assert classify({"abstained": False, "score": 0.0}) == WRONG

    def test_good_answer_is_correct(self):
        assert classify({"abstained": False, "score": 1.0}) == CORRECT

    def test_threshold_is_an_inclusive_lower_bound(self):
        assert classify({"abstained": False, "score": WRONG_THRESHOLD}) == CORRECT

    def test_just_below_threshold_is_wrong(self):
        assert classify({"abstained": False, "score": WRONG_THRESHOLD - 0.01}) == WRONG

    def test_threshold_is_configurable(self):
        s = {"abstained": False, "score": 0.6}
        assert classify(s, threshold=0.5) == CORRECT
        assert classify(s, threshold=0.7) == WRONG


class TestHallucinationRate:
    def _cfg(self, rows, **kw):
        return score_results(rows, **kw)["per_config"]

    def test_buckets_partition_every_question(self):
        rows = [
            _row("q1", {"base_llm_abstain": "Paris"}),            # correct
            _row("q2", {"base_llm_abstain": "Berlin"}),           # confident error
            _row("q3", {"base_llm_abstain": ABSTENTION_TEXT}),    # declined
        ]
        d = self._cfg(rows)["base_llm_abstain"]
        assert d["abstention_rate"] + d["correct_rate"] + d["hallucination_rate"] \
            == pytest.approx(1.0)
        assert d["correct_rate"] == pytest.approx(1 / 3)
        assert d["hallucination_rate"] == pytest.approx(1 / 3)
        assert d["abstention_rate"] == pytest.approx(1 / 3)

    def test_abstaining_lowers_hallucination_where_f1_cannot_tell(self):
        # The exact behaviour the thesis attributes to Graph-RAG: declining
        # instead of guessing must show as a LOWER hallucination rate at the same
        # correct rate. F1 scores both cases 0 and cannot express the difference —
        # which is the reason this metric had to be added at all.
        guesser = [_row(f"q{i}", {"base_llm_abstain": "Berlin"}) for i in range(4)]
        abstainer = [_row(f"q{i}", {"base_llm_abstain": ABSTENTION_TEXT}) for i in range(4)]
        g = self._cfg(guesser)["base_llm_abstain"]
        a = self._cfg(abstainer)["base_llm_abstain"]
        assert g["hallucination_rate"] == 1.0
        assert a["hallucination_rate"] == 0.0
        assert g["correct_rate"] == a["correct_rate"] == 0.0
        assert g["f1"] == a["f1"] == 0.0

    def test_threshold_moves_a_partial_answer_between_buckets(self):
        # Gold longer than the prediction, so the answer is genuinely INCOMPLETE
        # and stays on token_f1 (0.667). The reverse ("Paris France" for gold
        # "Paris") is a complete answer inside a longer string and is now scored
        # 1.0 by entity_match at any threshold — see test_entity_match.
        rows = [_row("q1", {"base_llm_abstain": "Paris"}, expected="Paris France")]
        assert self._cfg(rows, threshold=0.3)["base_llm_abstain"]["correct_rate"] == 1.0
        assert self._cfg(rows, threshold=0.9)["base_llm_abstain"]["hallucination_rate"] == 1.0

    def test_threshold_is_recorded_in_the_output(self):
        assert score_results([_row("q1", {"base_llm_abstain": "Paris"})],
                             threshold=0.42)["threshold"] == 0.42

    def test_api_failure_is_not_counted_as_a_hallucination(self):
        # An API 500 is not a confident error. Grading it as one was the
        # 2026-07-25 sweep bug; it must stay fixed in the outcome view too.
        rows = [
            _row("q1", {"base_llm_abstain": "Paris", "graph_rag": "Paris"}),
            _row("q2", {"base_llm_abstain": "Berlin"}, errors={"graph_rag": {"type": "APIError"}}),
        ]
        d = self._cfg(rows)
        assert d["graph_rag"]["n"] == 1
        assert d["base_llm_abstain"]["n"] == 1, "the question is dropped for EVERY config"
        assert d["base_llm_abstain"]["hallucination_rate"] == 0.0


class TestPerQuestionScores:
    def test_returns_a_record_per_config_and_question(self):
        rows = [_row("q1", {"base_llm_abstain": "Paris", "graph_rag": "Berlin"})]
        s = per_question_scores(rows)
        assert set(s) == {"base_llm_abstain", "graph_rag"}
        assert s["base_llm_abstain"]["q1"]["outcome"] == CORRECT
        assert s["graph_rag"]["q1"]["outcome"] == metrics.HALLUCINATION

    def test_carries_the_stratification_fields(self):
        rows = [_row("q1", {"base_llm_abstain": "yes"}, expected="yes",
                     answer_type="boolean", complexity="yesno")]
        r = per_question_scores(rows)["base_llm_abstain"]["q1"]
        assert r["answer_type"] == "boolean"
        assert r["complexity"] == "yesno"

    def test_excluded_questions_absent_from_every_config(self):
        rows = [
            _row("q1", {"base_llm_abstain": "Paris", "rag": "Paris"}),
            _row("q2", {"base_llm_abstain": "Paris"}, errors={"rag": {"type": "APIError"}}),
        ]
        s = per_question_scores(rows)
        assert "q2" not in s["base_llm_abstain"]
        assert "q2" not in s["rag"]

    def test_agrees_with_the_aggregate_table(self):
        # The per-question view and the headline are computed from one source
        # precisely so they cannot drift apart; pin that they agree.
        rows = [
            _row("q1", {"base_llm_abstain": "Paris"}),
            _row("q2", {"base_llm_abstain": "Berlin"}),
            _row("q3", {"base_llm_abstain": ABSTENTION_TEXT}),
        ]
        s = per_question_scores(rows)["base_llm_abstain"]
        agg = score_results(rows)["per_config"]["base_llm_abstain"]
        n = len(s)
        assert sum(r["outcome"] == metrics.HALLUCINATION for r in s.values()) / n == \
            pytest.approx(agg["hallucination_rate"])
        assert sum(r["f1"] for r in s.values()) / n == pytest.approx(agg["f1"])

    def test_honours_the_threshold(self):
        # Incomplete answer (gold longer than pred) — stays on token_f1 at 0.667.
        rows = [_row("q1", {"base_llm_abstain": "Paris"}, expected="Paris France")]
        assert per_question_scores(rows, threshold=0.3)["base_llm_abstain"]["q1"]["outcome"] == CORRECT
        assert per_question_scores(rows, threshold=0.9)["base_llm_abstain"]["q1"]["outcome"] == metrics.HALLUCINATION


class TestStratification:
    def test_groups_by_answer_type(self):
        rows = [
            _row("q1", {"base_llm_abstain": "yes"}, expected="yes", answer_type="boolean"),
            _row("q2", {"base_llm_abstain": "Paris"}, expected="Paris", answer_type="entity"),
        ]
        by_type = score_results(rows)["per_answer_type"]
        assert set(by_type) == {"boolean", "entity"}
        assert by_type["boolean"]["base_llm_abstain"]["n"] == 1
        assert by_type["entity"]["base_llm_abstain"]["correct_rate"] == 1.0

    def test_answer_type_separates_what_aggregate_f1_hides(self):
        # Two configs with identical aggregate F1 and opposite per-type profiles.
        # This is why the thesis reports the split rather than one averaged number.
        rows = [
            _row("q1", {"base_llm_abstain": "yes", "graph_rag": "no"},
                 expected="yes", answer_type="boolean"),
            _row("q2", {"base_llm_abstain": "Berlin", "graph_rag": "Paris"},
                 expected="Paris", answer_type="entity"),
        ]
        scored = score_results(rows)
        cfg = scored["per_config"]
        assert cfg["base_llm_abstain"]["f1"] == pytest.approx(cfg["graph_rag"]["f1"])

        by_type = scored["per_answer_type"]
        assert by_type["boolean"]["base_llm_abstain"]["correct_rate"] == 1.0
        assert by_type["boolean"]["graph_rag"]["correct_rate"] == 0.0
        assert by_type["entity"]["base_llm_abstain"]["correct_rate"] == 0.0
        assert by_type["entity"]["graph_rag"]["correct_rate"] == 1.0


class TestConfigWhitelist:
    """
    Reporting only covers config ids listed in metrics.CONFIGS.

    That keeps result files predating a config from growing a phantom all-zero
    row for it — but it also means a NEW config added to run_eval.py and not to
    metrics.CONFIGS is silently absent from every table, with no error. Pinned
    here so the behaviour is discoverable rather than surprising.
    """

    def test_unknown_config_id_is_dropped_silently(self):
        rows = [_row("q1", {"base_llm_abstain": "Paris", "config_99": "Paris"})]
        assert set(score_results(rows)["per_config"]) == {"base_llm_abstain"}

    def test_known_configs_are_reported_in_a_stable_order(self):
        rows = [_row("q1", {"rerank": "Paris", "base_llm_abstain": "Paris", "rag": "Paris"})]
        assert list(score_results(rows)["per_config"]) == ["base_llm_abstain", "rag", "rerank"]

    def test_complexity_grouping_is_still_produced(self):
        rows = [_row("q1", {"base_llm_abstain": "Paris"}, complexity="multihop")]
        assert "multihop" in score_results(rows)["per_complexity"]

    def test_group_totals_match_the_overall_n(self):
        rows = [
            _row("q1", {"base_llm_abstain": "Paris"}, answer_type="entity"),
            _row("q2", {"base_llm_abstain": "yes"}, expected="yes", answer_type="boolean"),
            _row("q3", {"base_llm_abstain": "3"}, expected="3", answer_type="numerical"),
        ]
        scored = score_results(rows)
        total = sum(g["base_llm_abstain"]["n"] for g in scored["per_answer_type"].values())
        assert total == scored["per_config"]["base_llm_abstain"]["n"] == 3


class TestSealedSplitGuard:
    """
    dev_runs/ is defined as "runs you were allowed to look at and act on". A test
    run is spent once, so filing one there erases the distinction the dev-tunes/
    test-sealed protocol depends on -- and the mistake is silent and expensive.
    The guard is two-way: it also refuses to file a repeatable dev sample in
    test_runs/, where its presence would imply it is not repeatable.
    """

    def _run(self, argv):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        return subprocess.run([sys.executable, "-m", "src.eval.run_eval", *argv],
                              cwd=root, capture_output=True, text=True)

    def test_test_split_without_the_flag_is_refused(self):
        r = self._run(["data/questions/mintaka_test.json", "--limit", "1"])
        assert r.returncode != 0
        assert "looks like the TEST split" in r.stderr

    def test_dev_sample_with_the_flag_is_refused(self):
        """Refused by QUESTION ID since 2026-08-18, not by filename.

        --workers 2 --rpm 33 and no --limit: the frozen-configuration guard
        (below) runs first, and this test is about the id check, not that one.
        --rpm 33 pins the resolved rpm so the frozen check passes regardless of
        this machine's endpoint default.
        """
        r = self._run(["data/questions/mintaka_sample_dev_200.json", "--test-run",
                       "--workers", "2", "--rpm", "33"])
        assert r.returncode != 0
        assert "not in the test split" in r.stderr

    def test_test_sample_under_neutral_name_is_refused(self, tmp_path):
        """The sampler bypass: a TEST-split sample with no 'test' in its name.

        `mintaka_sample_100.json` was exactly this file, and the filename
        heuristic waved it into dev_runs/. Question ids are the authority now.
        """
        import json, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        raw = json.loads((root / "data/questions/mintaka_test_raw.json")
                         .read_text(encoding="utf-8"))
        sneaky = tmp_path / "mintaka_sample_sneaky.json"
        sneaky.write_text(json.dumps(raw[:2]), encoding="utf-8")
        r = self._run([str(sneaky), "--limit", "1"])
        assert r.returncode != 0
        assert "SEALED TEST split" in r.stderr

    def test_nonpositive_limit_is_refused(self):
        r = self._run(["data/questions/mintaka_sample_dev_200.json", "--limit", "0"])
        assert r.returncode != 0
        assert "--limit must be a positive integer" in r.stderr

    def test_the_two_directories_are_distinct(self):
        from src.eval import run_eval
        assert run_eval.TEST_RUNS_DIR != run_eval.DEV_RUNS_DIR
        assert run_eval.TEST_RUNS_DIR.name == "test_runs"


class TestTestRunHardening:
    """
    The 2026-08-19 guards: a --test-run runs the FROZEN configuration on the
    canonical file, once — or it does not run. The frozen values are the module
    defaults, so the guard refuses override FLAGS rather than duplicating the
    values (a copy could drift from the constants it duplicates). None of these
    tests may pass every guard: an invocation that did would start a real run.
    """

    def _run(self, argv):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        return subprocess.run([sys.executable, "-m", "src.eval.run_eval", *argv],
                              cwd=root, capture_output=True, text=True)

    def test_limit_under_test_run_is_refused(self):
        """The audit's exact bypass: membership was checked before slicing, so
        `--test-run --limit 1` used to spend one question of the sealed split."""
        r = self._run(["data/questions/mintaka_test_raw.json", "--test-run",
                       "--limit", "1", "--workers", "2"])
        assert r.returncode != 0
        assert "non-frozen configuration" in r.stderr
        assert "--limit" in r.stderr

    def test_sweep_flags_under_test_run_are_refused_and_named(self):
        r = self._run(["data/questions/mintaka_test_raw.json", "--test-run",
                       "--top-k", "10", "--workers", "2"])
        assert r.returncode != 0
        assert "--top-k" in r.stderr

    def test_default_workers_is_refused_because_frozen_is_2(self):
        """--workers is the ONE frozen value that is not the module default
        (DEFAULT_WORKERS=8 suits dev samples), so it must be passed explicitly."""
        r = self._run(["data/questions/mintaka_test_raw.json", "--test-run"])
        assert r.returncode != 0
        assert "--workers" in r.stderr

    def test_test_derived_subset_is_refused_by_hash(self, tmp_path):
        """A subset of TEST questions passes the id-membership check — the file
        hash is what enforces 'the whole split, from the canonical file'."""
        import json, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        raw = json.loads((root / "data/questions/mintaka_test_raw.json")
                         .read_text(encoding="utf-8"))
        subset = tmp_path / "mintaka_sample_sneaky.json"
        subset.write_text(json.dumps(raw[:2]), encoding="utf-8")
        r = self._run([str(subset), "--test-run", "--workers", "2", "--rpm", "33"])
        assert r.returncode != 0
        assert "canonical test file" in r.stderr

    def test_allow_override_without_test_run_is_refused(self):
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--allow-override", "--limit", "1"])
        assert r.returncode != 0
        assert "--allow-override" in r.stderr

    def test_allow_override_requires_a_purpose(self):
        """A bypassed guard with no stated reason is the provenance hole the
        guards exist to close."""
        r = self._run(["data/questions/mintaka_test_raw.json", "--test-run",
                       "--workers", "2", "--rpm", "33", "--allow-override"])
        assert r.returncode != 0
        assert "non-empty --purpose" in r.stderr

    def test_nonfrozen_rpm_under_test_run_is_refused(self):
        """--rpm is checked by RESOLVED VALUE (env + endpoint default + flag),
        frozen at 33 — rpm 0 (pacing off) burned retries on every dev run that
        tried it."""
        r = self._run(["data/questions/mintaka_test_raw.json", "--test-run",
                       "--workers", "2", "--rpm", "0"])
        assert r.returncode != 0
        assert "resolved rpm 0" in r.stderr

    def test_negative_rpm_is_refused_everywhere(self):
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--rpm", "-5"])
        assert r.returncode != 0
        assert "--rpm must be >= 0" in r.stderr

    def test_frozen_invocation_has_no_violations(self):
        """The canonical command's flag set is clean — if this fails, the
        canonical command in test_runs/README.md no longer launches."""
        import argparse
        from src.eval import run_eval
        args = argparse.Namespace(limit=None, top_k=None, allocation=None,
                                  entity_block=None, prompt_variant=None,
                                  value_desc=None, no_document=False,
                                  workers=run_eval._FROZEN_TEST_WORKERS)
        assert run_eval.frozen_config_violations(
            args, run_eval._FROZEN_TEST_RPM) == []

    def test_every_override_flag_is_named_in_the_violations(self):
        import argparse
        from src.eval import run_eval
        args = argparse.Namespace(limit=1, top_k=10, allocation="floor",
                                  entity_block="entity", prompt_variant="c-prime",
                                  value_desc="collision",
                                  no_document=True, workers=8)
        joined = "\n".join(run_eval.frozen_config_violations(args, 0))
        for flag in ("--limit", "--top-k", "--allocation", "--entity-block",
                     "--prompt-variant", "--value-desc", "--no-document",
                     "--workers", "resolved rpm"):
            assert flag in joined

    def test_dirty_tree_guard_fails_closed(self, monkeypatch):
        """git failing must read as 'cannot verify', never as 'clean'."""
        from src import experiment_spec
        import subprocess as sp

        def boom(*a, **kw):
            raise OSError("git not found")
        monkeypatch.setattr(sp, "run", boom)
        try:
            experiment_spec.code_dirty_paths(strict=True)
            assert False, "strict mode must raise when git fails"
        except experiment_spec.GitUnavailableError:
            pass
        # Non-strict (snapshot recording) keeps the old swallow-and-continue.
        assert experiment_spec.code_dirty_paths() == []

    def test_one_spend_counts_run_directories_not_files(self, tmp_path, monkeypatch):
        """test_runs/ legitimately holds README.md; only DIRECTORIES are spends."""
        from src.eval import run_eval
        monkeypatch.setattr(run_eval, "TEST_RUNS_DIR", tmp_path)
        (tmp_path / "README.md").write_text("docs", encoding="utf-8")
        assert run_eval.prior_test_runs() == []
        (tmp_path / "20260901_0900_model_test-4000").mkdir()
        assert len(run_eval.prior_test_runs()) == 1

    def test_missing_test_runs_dir_is_no_prior_spend(self, tmp_path, monkeypatch):
        from src.eval import run_eval
        monkeypatch.setattr(run_eval, "TEST_RUNS_DIR", tmp_path / "absent")
        assert run_eval.prior_test_runs() == []

    def test_pinned_hash_matches_the_canonical_file(self):
        """If the download is ever refreshed, this fails before the guard lies.

        Either line-ending form of the checkout is the canonical file (CRLF on a
        Windows autocrlf checkout, LF on any other clone or on the export).
        """
        from src.eval import run_eval
        assert run_eval._sha256(run_eval._TEST_RAW_FILE) in run_eval._TEST_RAW_SHA256_FORMS

    def test_the_two_pinned_forms_are_the_same_content(self):
        """The LF pin must be exactly the CRLF file with its line endings normalised."""
        import hashlib
        from src.eval import run_eval
        raw = run_eval._TEST_RAW_FILE.read_bytes()
        lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        assert lf == run_eval._TEST_RAW_SHA256_LF


class TestResumeFrom:
    """
    --resume-from (2026-08-19): recovery for an interrupted run via its
    .partial.jsonl sidecar. Structural refusals are testable via subprocess
    (they all fire before any API call); the experiment-state signature is
    tested in-process below. NONE of these checks accept --allow-override.
    """

    def _run(self, argv):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        return subprocess.run([sys.executable, "-m", "src.eval.run_eval", *argv],
                              cwd=root, capture_output=True, text=True)

    @staticmethod
    def _sidecar(tmp_path, rows, meta: dict | str = None):
        """A sidecar plus (by default) a minimal 'started' meta beside it."""
        import json
        sidecar = tmp_path / "x_raw.partial.jsonl"
        sidecar.write_text("".join(json.dumps(r) + "\n" for r in rows),
                           encoding="utf-8")
        if meta is None:
            meta = {"status": "started"}
        mp = tmp_path / "meta.json"
        mp.write_text(meta if isinstance(meta, str) else json.dumps(meta),
                      encoding="utf-8")
        return sidecar

    def test_missing_sidecar_is_refused(self):
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", "no_such.partial.jsonl"])
        assert r.returncode != 0
        assert "does not exist" in r.stderr

    def test_sidecar_without_adjacent_meta_is_refused(self, tmp_path):
        """A sidecar copied out of its run directory has no start snapshot to
        verify against."""
        import json
        sidecar = tmp_path / "orphan.partial.jsonl"
        sidecar.write_text(json.dumps({"id": "x", "answers": {}}) + "\n",
                           encoding="utf-8")
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "no meta.json beside" in r.stderr

    def test_corrupt_meta_is_refused(self, tmp_path):
        sidecar = self._sidecar(tmp_path, [{"id": "x", "answers": {}}],
                                meta="{not json")
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "corrupt" in r.stderr

    def test_completed_run_cannot_be_resumed(self, tmp_path):
        sidecar = self._sidecar(tmp_path, [{"id": "x", "answers": {}}],
                                meta={"status": "documented"})
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "not 'started'" in r.stderr

    def test_incomplete_row_is_refused(self, tmp_path):
        sidecar = self._sidecar(tmp_path, [{"id": "x"}])   # no "answers"
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "not a complete result row" in r.stderr

    def test_duplicate_ids_are_refused(self, tmp_path):
        sidecar = self._sidecar(tmp_path, [{"id": "x", "answers": {}},
                                           {"id": "x", "answers": {}}])
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "duplicate" in r.stderr.lower()

    def test_sidecar_from_another_question_set_is_refused(self, tmp_path):
        """A sidecar whose rows are not in the questions file would silently
        smuggle foreign rows into the raw file — refused by id."""
        sidecar = self._sidecar(tmp_path,
                                [{"id": "not-a-real-question", "answers": {}}])
        r = self._run(["data/questions/mintaka_sample_dev_200.json",
                       "--limit", "1", "--resume-from", str(sidecar)])
        assert r.returncode != 0
        assert "different question set" in r.stderr


class TestResumeSignature:
    """The experiment-state match a resume must pass — no override exists."""

    def _meta(self, **overrides):
        base = {
            "status": "started",
            "sample": {"sha256": "abc", "n": 4000},
            "start_settings": {"workers": 2, "rpm": 33},
            "preflight": {"n_entities": 2530},
            "article_preflight": {"n_entities": 2530, "no_article": {}},
            "specs": {
                "git": {"sha": "deadbeef", "dirty": False},
                "model": {"name": "m", "base_url": "u",
                          "temperature": 0, "max_tokens": 128},
                "prompt_framing": {"answer": {}},
                "prompt_fragments": {"PROMPT_VARIANT": "f"},
                "prompt_templates": {"C1_base_llm": "t"},
                "retrieval": {"active": {"statement_cache_version": 8}},
                "context_construction": {},
                "generation": {"answer_max_tokens": 128},
            },
        }
        base.update(overrides)
        return base

    def test_identical_metas_pass(self):
        from src.eval import run_eval
        assert run_eval._resume_mismatches(self._meta(), self._meta()) == []

    def test_different_commit_is_named(self):
        from src.eval import run_eval
        other = self._meta()
        other["specs"] = dict(other["specs"],
                              git={"sha": "cafebabe", "dirty": False})
        problems = run_eval._resume_mismatches(self._meta(), other)
        assert any("git_sha" in p for p in problems)

    def test_dirty_source_tree_is_refused(self):
        from src.eval import run_eval
        src = self._meta()
        src["specs"] = dict(src["specs"], git={"sha": "deadbeef", "dirty": True})
        problems = run_eval._resume_mismatches(src, self._meta())
        assert any("dirty tree" in p for p in problems)

    def test_missing_rpm_in_source_is_a_mismatch(self):
        """A source meta too old to record rpm cannot be verified — None must
        mismatch a real value, not silently pass."""
        from src.eval import run_eval
        src = self._meta(start_settings={"workers": 2})   # no rpm key
        problems = run_eval._resume_mismatches(src, self._meta())
        assert any(p.startswith("rpm") for p in problems)

    def test_changed_cache_version_is_named(self):
        from src.eval import run_eval
        other = self._meta()
        other["specs"] = dict(other["specs"],
                              retrieval={"active": {"statement_cache_version": 9}})
        problems = run_eval._resume_mismatches(self._meta(), other)
        assert any("retrieval" in p for p in problems)
