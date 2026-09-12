"""
Unit tests for tools/compare_faithfulness.py — the sanctioned source for
faithfulness statistics.

Offline by construction (the tool reads a stored JSON and makes no calls), so
the tests run it end-to-end on synthetic files: what is pinned is that the
marginal means and the paired common-set contrasts are DIFFERENT constructs
computed on different sets, that a duplicated (config, id) record is fatal
rather than silently double-weighted, and that the persisted JSON carries the
provenance fields (sha, seed, resamples) that make it quotable.
"""

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "compare_faithfulness",
        os.path.join(ROOT, "tools", "compare_faithfulness.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(config, qid, score, outcome="correct"):
    return {"id": qid, "config": config, "outcome": outcome, "score": score,
            "n_claims": 1, "status": "ok" if score is not None else "error"}


def _run(tool, tmp_path, records, monkeypatch):
    src = tmp_path / "faithfulness_synthetic.json"
    src.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(tool, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["compare_faithfulness.py", str(src), "--resamples", "200"])
    assert tool.main() == 0
    out = tmp_path / "faithfulness_comparison_synthetic.json"
    return json.loads(out.read_text(encoding="utf-8"))


class TestDuplicateRejection:
    def test_duplicate_config_id_is_fatal(self, tmp_path, monkeypatch):
        tool = _load_tool()
        records = [_rec("rag", "q1", 1.0), _rec("rag", "q1", 0.0)]
        src = tmp_path / "faithfulness_dup.json"
        src.write_text(json.dumps(records), encoding="utf-8")
        with pytest.raises(SystemExit):
            tool.load_records(src)

    def test_same_id_across_configs_is_fine(self, tmp_path):
        tool = _load_tool()
        records = [_rec("rag", "q1", 1.0), _rec("graph_rag", "q1", 0.0)]
        src = tmp_path / "faithfulness_ok.json"
        src.write_text(json.dumps(records), encoding="utf-8")
        assert len(tool.load_records(src)) == 2


class TestComputation:
    def test_marginals_and_common_set_are_different_constructs(
            self, tmp_path, monkeypatch):
        tool = _load_tool()
        # graph_rag scores q1, q2; rag scores q1, q3 and has q2 unscoreable
        # (None). The common scoreable set is exactly {q1}: q2 is one-sided-
        # None, q3 is absent from graph_rag.
        records = [
            _rec("graph_rag", "q1", 1.0), _rec("graph_rag", "q2", 0.0),
            _rec("rag", "q1", 0.5), _rec("rag", "q2", None), _rec("rag", "q3", 1.0),
            _rec("rerank", "q1", 1.0),
        ]
        payload = _run(tool, tmp_path, records, monkeypatch)

        marg = payload["operating_point_means"]
        assert marg["graph_rag"] == {"n": 2, "mean": 0.5}
        assert marg["rag"]["n"] == 2 and marg["rag"]["mean"] == pytest.approx(0.75)

        pair = next(p for p in payload["paired_common_scoreable"]
                    if p["pair"] == ["graph_rag", "rag"])
        assert pair["bootstrap"]["n"] == 1, "q2 (one-sided None) and q3 must drop"
        assert pair["bootstrap"]["delta"] == pytest.approx(0.5)

    def test_correct_vs_wrong_groups_are_disjoint_by_outcome(
            self, tmp_path, monkeypatch):
        tool = _load_tool()
        records = [
            _rec("rag", "q1", 0.9, outcome="correct"),
            _rec("rag", "q2", 0.7, outcome="correct"),
            _rec("rag", "q3", 0.2, outcome="hallucination"),
            _rec("rag", "q4", 0.5, outcome="abstention"),   # in neither group
        ]
        payload = _run(tool, tmp_path, records, monkeypatch)
        row = next(r for r in payload["correct_vs_wrong_unpaired"]
                   if r["config"] == "rag")
        assert row["n_correct"] == 2 and row["n_wrong"] == 1
        assert row["mean_correct"] == pytest.approx(0.8)
        assert row["mean_wrong"] == pytest.approx(0.2)

    def test_payload_carries_provenance(self, tmp_path, monkeypatch):
        tool = _load_tool()
        payload = _run(tool, tmp_path, [_rec("rag", "q1", 1.0)], monkeypatch)
        assert len(payload["input_sha256"]) == 64
        assert payload["seed"] == 0
        assert payload["n_resamples"] == 200
        assert "percentile-bootstrap" in payload["p_value_kind"]
