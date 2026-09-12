"""
Unit tests for the statistical sections of tools/report.py.

Pinned here: the p-value display floor (<.001, never a false exact zero), the
selection diagnostic in section 5 (the not-attempted complement INCLUDES Other
rows and the baseline's F1 on it is reported beside the attempted-subset F1 —
the section is a diagnostic, not proof of no selection), and section 6's joint
subset (both configs attempted; the marginal F1@attempted pair lives on
different sets and must not be what gets compared).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_report():
    spec = importlib.util.spec_from_file_location(
        "report", os.path.join(ROOT, "tools", "report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(f1=0.0, outcome="correct"):
    return {"f1": f1, "score": f1, "outcome": outcome,
            "abstained": outcome == "abstention", "em": 0.0}


class TestFmtP:
    def test_below_resolution_prints_floor(self):
        report = _load_report()
        assert report.fmt_p(0.0) == "<.001"
        assert report.fmt_p(0.0009) == "<.001"

    def test_at_and_above_threshold_prints_three_decimals(self):
        report = _load_report()
        assert report.fmt_p(0.001) == "0.001"
        assert report.fmt_p(0.323) == "0.323"


class TestWithinAttemptedSubset:
    def _scores(self):
        base = {"q1": _rec(0.8), "q2": _rec(0.6),
                "q3": _rec(0.2), "q4": _rec(0.4)}
        rag = {"q1": _rec(1.0, "correct"), "q2": _rec(0.0, "hallucination"),
               "q3": _rec(0.0, "abstention"), "q4": _rec(0.0, "other")}
        return {"base_llm_abstain": base, "rag": rag}

    def test_not_attempted_complement_includes_other_rows(self, capsys):
        report = _load_report()
        rows = report.section_attempted_subset(self._scores(),
                                               "base_llm_abstain", 200)
        row = rows[0]
        assert row["n_attempted_shared"] == 2
        assert row["baseline_f1_attempted"] == pytest.approx(0.7)
        # q3 (abstention) AND q4 (other) form the complement: (0.2+0.4)/2.
        # Testing only abstentions would give 0.2 — the wrong construct.
        assert row["baseline_f1_not_attempted"] == pytest.approx(0.3)

    def test_header_names_the_construct_not_a_verdict(self, capsys):
        report = _load_report()
        report.section_attempted_subset(self._scores(), "base_llm_abstain", 200)
        out = capsys.readouterr().out
        assert "WITHIN-ATTEMPTED-SUBSET" in out
        assert "selection diagnostic" in out
        assert "SELECTION-EFFECT CHECK" not in out

    def test_missing_baseline_returns_empty(self, capsys):
        report = _load_report()
        assert report.section_attempted_subset({"rag": {}}, "nope", 200) == []


class TestJointlyAttemptedPairs:
    def _scores(self):
        # graph_rag attempts q1,q2,q3; rag attempts q1,q2,q4 -> joint {q1,q2}.
        rag = {"q1": _rec(0.5, "correct"), "q2": _rec(0.0, "hallucination"),
               "q3": _rec(0.0, "abstention"), "q4": _rec(1.0, "correct")}
        graph = {"q1": _rec(1.0, "correct"), "q2": _rec(0.0, "hallucination"),
                 "q3": _rec(1.0, "correct"), "q4": _rec(0.0, "abstention")}
        return {"rag": rag, "graph_rag": graph}

    def test_joint_subset_is_the_intersection_of_attempts(self, capsys):
        report = _load_report()
        rows = report.section_pairwise_attempted(self._scores(), 200)
        row = next(r for r in rows if r["pair"] == ["graph_rag", "rag"])
        assert row["n_joint"] == 2, "q3/q4 are attempted by only one config"
        assert row["f1_a_joint"] == pytest.approx(0.5)   # (1.0 + 0.0) / 2
        assert row["f1_b_joint"] == pytest.approx(0.25)  # (0.5 + 0.0) / 2
        assert "mcnemar_hallucination" in row

    def test_prints_the_construct_name(self, capsys):
        report = _load_report()
        report.section_pairwise_attempted(self._scores(), 200)
        out = capsys.readouterr().out
        assert "JOINTLY ATTEMPTED" in out
        assert "NOT the marginal F1@attempted" in out

    def test_no_retrieval_pair_present_returns_empty(self):
        report = _load_report()
        assert report.section_pairwise_attempted(
            {"base_llm_abstain": {"q1": _rec()}}, 200) == []
