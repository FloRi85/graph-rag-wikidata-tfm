"""
Unit tests for tools/compare_runs.py — one config, two runs, paired by id.

The contract worth pinning is the tool's own, not the bootstrap's (that is
tests/unit/test_stats.py): pairing is by question id rather than by row
position, non-overlap is REPORTED rather than silently dropped, scoring goes
through `gold_source` and the shared scorer rather than reading anything out of
the run rows, a malformed input refuses loudly, and the output is deterministic
so two invocations of the same comparison cannot disagree.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools import compare_runs

ABSTENTION_TEXT = "The answer is not in the context."

# One gold label per question id. The run rows built below carry NO gold fields
# at all — if scoring ever falls back to the rows instead of the questions
# file, every answer scores wrong and the deltas in these tests change.
GOLDS = {
    "cr-q1": ("Q90", "Paris"),
    "cr-q2": ("Q64", "Berlin"),
    "cr-q3": ("Q220", "Rome"),
    "cr-q4": ("Q2807", "Madrid"),
}


def _mintaka_record(qid: str, gold_qid: str, gold_label: str) -> dict:
    """A minimal but structurally complete RAW Mintaka record."""
    return {
        "id": qid,
        "question": f"Which city is {gold_label}?",
        "translations": {},
        "questionEntity": [
            {
                "name": "Q142",
                "entityType": "entity",
                "label": "France",
                "mention": "France",
                "span": [0, 6],
            }
        ],
        "answer": {
            "answerType": "entity",
            "answer": [{"name": gold_qid, "label": {"en": gold_label}}],
            "mention": gold_label,
        },
        "category": "geography",
        "complexityType": "generic",
    }


@pytest.fixture()
def questions_file(tmp_path):
    path = tmp_path / "questions_raw.json"
    records = [_mintaka_record(qid, g[0], g[1]) for qid, g in GOLDS.items()]
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _run_file(tmp_path, name: str, answers_by_qid: dict, config: str = "graph_rag"):
    """A run_eval-shaped raw file: a list of rows, one per question."""
    rows = [{"id": qid, "question": "q?", "answers": {config: pred}}
            for qid, pred in answers_by_qid.items()]
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _main(capsys, *argv) -> str:
    old = sys.argv
    sys.argv = ["compare_runs.py", *[str(a) for a in argv], "--resamples", "2000"]
    try:
        rc = compare_runs.main()
    finally:
        sys.argv = old
    assert rc == 0
    return capsys.readouterr().out


class TestPairing:
    def test_delta_is_id_matched_not_positional(self, tmp_path, questions_file, capsys):
        # Run A answers every question correctly; run B gets exactly q3 and q4
        # wrong — and B's rows are stored in REVERSE order, so a positional
        # pairing would compare different questions and land elsewhere.
        a = _run_file(tmp_path, "a.json",
                      {q: GOLDS[q][1] for q in GOLDS})
        b_answers = {"cr-q4": "Oslo", "cr-q3": "Oslo",
                     "cr-q2": GOLDS["cr-q2"][1], "cr-q1": GOLDS["cr-q1"][1]}
        b = _run_file(tmp_path, "b.json", b_answers)
        out = _main(capsys, a, b, "--questions", questions_file)
        assert "+50.0" in out              # 100% F1 vs 50% F1, id-matched
        assert "F1" in out

    def test_missing_questions_are_reported_not_silently_dropped(
            self, tmp_path, questions_file, capsys):
        a = _run_file(tmp_path, "a.json", {q: GOLDS[q][1] for q in GOLDS})
        b = _run_file(tmp_path, "b.json",
                      {q: GOLDS[q][1] for q in list(GOLDS)[:3]})  # cr-q4 absent
        out = _main(capsys, a, b, "--questions", questions_file)
        assert "1 question(s) not in both runs" in out

    def test_unrelated_configs_failure_does_not_drop_a_healthy_row(
            self, tmp_path, questions_file, capsys):
        """Per-config exclusion (2026-08-18): a C4 error must not shrink C3's cohort.

        Under the all-or-nothing rule (right for within-run tables, wrong
        here), cr-q4's rerank error would remove the graph_rag row too — the
        pairing would silently run over 3 questions instead of 4.
        """
        def rows_with_c4_error(answers):
            rows = [{"id": qid, "question": "q?",
                     "answers": {"graph_rag": pred}}
                    for qid, pred in answers.items()]
            rows[-1]["answers"]["rerank"] = ""
            rows[-1]["errors"] = {"rerank": "HTTP 500"}
            return rows

        a = tmp_path / "a.json"
        a.write_text(json.dumps(rows_with_c4_error(
            {q: GOLDS[q][1] for q in GOLDS})), encoding="utf-8")
        b = _run_file(tmp_path, "b.json", {q: "Oslo" for q in GOLDS})
        out = _main(capsys, a, b, "--questions", questions_file)
        # All 4 graph_rag rows pair: the +100 delta needs cr-q4 on both sides.
        assert "not in both runs" not in out
        assert "+100.0" in out

    def test_own_config_failure_is_excluded_and_surfaced(
            self, tmp_path, questions_file, capsys):
        """A failure of the COMPARED config drops that row and says so."""
        rows = [{"id": qid, "question": "q?",
                 "answers": {"graph_rag": GOLDS[qid][1]}}
                for qid in GOLDS]
        rows[-1]["answers"]["graph_rag"] = ""
        rows[-1]["errors"] = {"graph_rag": "HTTP 500"}
        a = tmp_path / "a.json"
        a.write_text(json.dumps(rows), encoding="utf-8")
        b = _run_file(tmp_path, "b.json", {q: GOLDS[q][1] for q in GOLDS})
        out = _main(capsys, a, b, "--questions", questions_file)
        assert "1 question(s) not in both runs" in out
        assert "infrastructure failures for this config" in out

    def test_scoring_goes_through_the_questions_file(
            self, tmp_path, questions_file, capsys):
        # The rows carry no gold fields, so a +100.0 all-correct-vs-all-wrong
        # delta is only reachable if the gold came from `gold_source`.
        a = _run_file(tmp_path, "a.json", {q: GOLDS[q][1] for q in GOLDS})
        b = _run_file(tmp_path, "b.json", {q: "Oslo" for q in GOLDS})
        out = _main(capsys, a, b, "--questions", questions_file)
        assert "+100.0" in out


class TestMetricSelection:
    def test_hallucination_metric_counts_confident_errors_only(
            self, tmp_path, questions_file, capsys):
        # A guesses wrong everywhere (hallucination 100%); B declines
        # everywhere (hallucination 0%). On --metric hallucination the delta is
        # +100; on the default F1 both score 0.0 and the delta is 0 — the exact
        # distinction the thesis's primary metric exists to make.
        a = _run_file(tmp_path, "a.json", {q: "Oslo" for q in GOLDS})
        b = _run_file(tmp_path, "b.json", {q: ABSTENTION_TEXT for q in GOLDS})
        out = _main(capsys, a, b, "--questions", questions_file,
                    "--metric", "hallucination")
        assert "hallucination %" in out
        assert "+100.0" in out
        out_f1 = _main(capsys, a, b, "--questions", questions_file)
        assert "+0.0" in out_f1


class TestRefusals:
    def test_a_non_list_raw_file_is_refused(self, tmp_path, questions_file):
        bad = tmp_path / "meta.json"
        bad.write_text(json.dumps({"purpose": "not a raw file"}), encoding="utf-8")
        ok = _run_file(tmp_path, "ok.json", {q: GOLDS[q][1] for q in GOLDS})
        old = sys.argv
        sys.argv = ["compare_runs.py", str(bad), str(ok),
                    "--questions", str(questions_file)]
        try:
            with pytest.raises(SystemExit, match="not a run_eval raw file"):
                compare_runs.main()
        finally:
            sys.argv = old

    def test_runs_sharing_no_config_are_refused(self, tmp_path, questions_file):
        a = _run_file(tmp_path, "a.json", {q: GOLDS[q][1] for q in GOLDS},
                      config="graph_rag")
        b = _run_file(tmp_path, "b.json", {q: GOLDS[q][1] for q in GOLDS},
                      config="rag")
        old = sys.argv
        sys.argv = ["compare_runs.py", str(a), str(b),
                    "--questions", str(questions_file)]
        try:
            with pytest.raises(SystemExit, match="share no config keys"):
                compare_runs.main()
        finally:
            sys.argv = old


class TestDeterminism:
    def test_the_same_comparison_prints_the_same_report_twice(
            self, tmp_path, questions_file, capsys):
        # The bootstrap is seeded; two invocations of one comparison must agree
        # to the last digit, or "re-run it" becomes a way to shop for a p-value.
        a = _run_file(tmp_path, "a.json",
                      {"cr-q1": GOLDS["cr-q1"][1], "cr-q2": "Oslo",
                       "cr-q3": GOLDS["cr-q3"][1], "cr-q4": "Oslo"})
        b = _run_file(tmp_path, "b.json",
                      {"cr-q1": "Oslo", "cr-q2": GOLDS["cr-q2"][1],
                       "cr-q3": GOLDS["cr-q3"][1], "cr-q4": GOLDS["cr-q4"][1]})
        first = _main(capsys, a, b, "--questions", questions_file)
        second = _main(capsys, a, b, "--questions", questions_file)
        assert first == second
