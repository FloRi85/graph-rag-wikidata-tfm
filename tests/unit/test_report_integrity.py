"""
`report.md` reports every metric, from the one scorer, in the right row.

⚠️ THESE TEST THE THREE WAYS A REPORT CAN LIE WHILE LOOKING FINE, each of which
has actually happened in this project:

  OMISSION    a metric is computed and never printed. F1 and F1@attempted were
              computed on every run since July and appeared in no report.md.
  DIVERGENCE  a second implementation of scoring drifts from the first.
              `score_arm` built its gold from flattened keys after the raw-record
              migration and reported `correct 0.0` beside report.py's 44.2.
  MISALIGNMENT a value printed against the wrong arm. This has not happened, and
              is the one that would be both invisible and fatal, so it is pinned
              with deliberately distinct per-arm values rather than trusted.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_report_integrity.py -v
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.eval.metrics import score_results
from src.eval.parse_answers import build_gold_answer
from tools import document_run
from tools.document_run import OUTCOME_COLUMNS, RUN_COLUMNS


# --------------------------------------------------------------------------
# A tiny run whose four arms are deliberately DIFFERENT from each other.
#
# Every arm gets a distinct outcome profile so a value cannot be swapped
# between rows without a test failing. Gold is "yes" throughout; the arms
# differ only in what they answered.
# --------------------------------------------------------------------------

GOLD = {"answerType": "boolean", "answer": [True], "mention": "Yes"}

# 4 questions. base=all right · rag=3 right · graph=2 right · rerank=1 right,
# with the remainder split between refusals and wrong answers so abstention,
# correct and hallucination all differ per arm.
ANSWERS = {
    "base_llm_abstain": ["Yes", "Yes", "Yes", "Yes"],
    "rag":              ["Yes", "Yes", "Yes", "No"],
    "graph_rag":        ["Yes", "Yes", "The answer is not in the context.", "No"],
    "rerank":           ["Yes", "The answer is not in the context.",
                         "The answer is not in the context.", "No"],
}


@pytest.fixture
def run():
    rows = []
    for i in range(4):
        rows.append({
            "id": f"q{i}",
            "question": f"question {i}?",
            "answers": {cfg: ANSWERS[cfg][i] for cfg in ANSWERS},
            "contexts": {}, "context_words": {}, "finish_reasons": {}, "errors": {},
        })
    golds = {f"q{i}": build_gold_answer(GOLD) for i in range(4)}
    entities = {f"q{i}": {"entity_names": [], "entity_labels": []} for i in range(4)}
    return rows, golds, entities


def build_summaries(run):
    """Exactly what `document()` builds, without touching the filesystem."""
    rows, golds, entities = run
    arms = document_run.normalise(rows)
    summaries = {name: document_run.score_arm(arm_rows, golds, entities, set(), 0)
                 for name, arm_rows in arms.items()}
    scored = score_results(rows, entities, golds=golds)
    for name, per_cfg in scored["per_config"].items():
        summaries[name] = {**summaries[name], **per_cfg}
    return summaries, scored


class TestNothingIsOmitted:

    def test_every_scorer_metric_is_reported(self, run):
        """🔴 THE OMISSION GUARD. If `score_results` grows a per-config key that
        `OUTCOME_COLUMNS` does not render, this fails — because the failure mode
        is silence, and silence is how F1 went unreported for a month."""
        summaries, scored = build_summaries(run)
        produced = set(next(iter(scored["per_config"].values())))
        rendered = {key for _, key, _ in OUTCOME_COLUMNS}
        missing = produced - rendered
        assert not missing, (
            f"score_results computes {sorted(missing)} and report.md never "
            f"prints them. Add to OUTCOME_COLUMNS or justify the omission here.")

    def test_the_run_columns_are_present_too(self, run):
        summaries, _ = build_summaries(run)
        for _, key in RUN_COLUMNS:
            assert key in next(iter(summaries.values())), \
                f"{key} is rendered but never computed"

    def test_the_rendered_report_contains_every_column_heading(self, run):
        summaries, scored = build_summaries(run)
        meta = _meta()
        text = document_run.render(meta, summaries, scored)
        for heading, _, _ in OUTCOME_COLUMNS:
            assert heading in text, f"{heading} missing from report.md"
        for heading, _ in RUN_COLUMNS:
            assert heading in text


class TestOneSourceOfTruth:

    def test_report_md_values_come_from_score_results(self, run):
        """🔴 THE DIVERGENCE GUARD. Every scoring number in the summary must be
        the value `score_results` produced — not a second computation that
        happens to agree today."""
        summaries, scored = build_summaries(run)
        for arm, per_cfg in scored["per_config"].items():
            for _, key, _ in OUTCOME_COLUMNS:
                assert summaries[arm][key] == per_cfg[key], (
                    f"{arm}.{key}: report has {summaries[arm][key]}, "
                    f"score_results says {per_cfg[key]}")

    def test_root_metrics_win_over_any_local_computation(self, run):
        """`score_arm` also counts outcomes. Where the two overlap, the root
        must be the one that survives the merge."""
        rows, golds, entities = run
        arms = document_run.normalise(rows)
        local = document_run.score_arm(arms["rag"], golds, entities, set(), 0)
        merged, scored = build_summaries(run)
        # score_arm reports percentages under its own key names; the root
        # reports rates. The merged record must carry the ROOT's rate.
        assert merged["rag"]["correct_rate"] == scored["per_config"]["rag"]["correct_rate"]
        assert "correct_pct" in local        # the local one still exists...
        assert merged["rag"]["correct_rate"] != local["correct_pct"]  # ...and is not used


class TestNoRowIsConfusedWithAnother:
    """🔴 THE MISALIGNMENT GUARD — the failure that would be invisible.

    The four arms are constructed to have DIFFERENT values on every metric that
    can differ, so any transposition between rows changes a number and fails.
    """

    def test_the_four_arms_really_do_differ(self, run):
        """If the fixture ever became uniform, the tests below would pass
        vacuously. Pin that they do not."""
        _, scored = build_summaries(run)
        correct = [scored["per_config"][a]["correct_rate"] for a in ANSWERS]
        assert len(set(correct)) == 4, "fixture no longer discriminates arms"

    def test_each_row_carries_its_own_arms_numbers(self, run):
        summaries, scored = build_summaries(run)
        meta = _meta()
        text = document_run.render(meta, summaries, scored)

        # ⚠️ ANCHOR ON THE SCORING HEADER, do not just match arm-named rows.
        # BOTH tables begin their rows with the arm name, so a naive selector
        # picks up 8 rows and can read a Generation cell as a Scoring one --
        # which is the very confusion these tests exist to rule out.
        rows = _scoring_rows(text)
        assert len(rows) == 4, f"expected one scoring row per arm, got {len(rows)}"

        n_index = [k for _, k, _ in OUTCOME_COLUMNS].index("correct_rate") + 2
        for line in rows:
            cells = [c.strip() for c in line.split("|")]
            arm = cells[1]
            printed = float(cells[n_index])
            expected = scored["per_config"][arm]["correct_rate"] * 100
            assert abs(printed - expected) < 0.05, (
                f"row '{arm}' prints Correct% {printed}, but {arm} scored "
                f"{expected:.1f} — a value from another row")

    def test_column_headings_and_values_stay_in_step(self, run):
        """A column added to OUTCOME_COLUMNS without a matching cell would shift
        every value one place left — the classic off-by-one that reads as a
        plausible number."""
        summaries, scored = build_summaries(run)
        text = document_run.render(_meta(), summaries, scored)
        header = next(ln for ln in text.splitlines()
                      if ln.startswith("| Arm |") and "Abst%" in ln)
        n_headings = len([c for c in header.split("|") if c.strip()])
        n_cells = len([c for c in _scoring_rows(text)[0].split("|") if c.strip()])
        assert n_headings == n_cells, (
            f"{n_headings} headings against {n_cells} cells — the table is "
            f"misaligned and every value is under the wrong name")


class TestOutcomePartitionAndRounding:
    """🔴 THE ROUNDING-DIVERGENCE GUARD, added 2026-08-20.

    On the TEST-4000 run of record, meta.json said C3 hallucination 8.4 while
    report.md said 8.5 — same count (338/4000, exactly 8.45%), two rounding
    paths: `round(100 * count / n, 1)` and `f"{(count / n) * 100:.1f}"` build
    different floats at exact percentage ties. And the partition itself
    (Abst + Correct + Halluc + Other = n) held only by construction: a graded
    row whose gold was missing was silently skipped, shrinking the buckets but
    not the denominator.
    """

    def test_meta_and_table_agree_on_every_possible_count(self):
        """Sweep EVERY count at the standard split sizes: the number meta.json
        stores must render to the digits the report.md table prints."""
        for n in (200, 4000):
            for c in range(n + 1):
                stored = document_run._pct1(c, n)
                printed = document_run._PCT(c / n)
                assert f"{stored:.1f}" == printed, (
                    f"{c}/{n}: meta.json stores {stored}, report.md prints "
                    f"{printed} — the run record disagrees with its own report")

    def test_score_arm_refuses_to_drop_a_graded_row(self):
        """A graded row with no gold must fail loudly, not silently leave the
        four outcome percentages summing below 100."""
        rows = [{"id": "q-nogold", "answer": "Paris", "finish_reason": "stop"}]
        with pytest.raises(ValueError, match="no gold"):
            document_run.score_arm(rows, golds={}, entities_by_id={})

    def test_score_results_rejects_an_outcome_outside_the_four_buckets(self, run, monkeypatch):
        """A fifth outcome added to classify_outcome without extending the
        aggregation would silently leak out of every table."""
        from src.eval import metrics
        monkeypatch.setattr(metrics, "classify_outcome",
                            lambda *a, **k: "confabulation")
        rows, golds, entities = run
        with pytest.raises(ValueError, match="not in OUTCOMES"):
            score_results(rows, entities, golds=golds)


def _scoring_rows(text: str) -> list[str]:
    """The body rows of the SCORING table only.

    Both tables in the Outcomes section start their rows with the arm name, so
    the scoring one is identified by the header above it ("Abst%") and read
    until the block ends.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("| Arm |") and "Abst%" in ln)
    out = []
    for ln in lines[start + 2:]:               # skip the |---| separator
        if not ln.startswith("| "):
            break
        out.append(ln)
    return out


def _meta() -> dict:
    """What `render` needs, with REAL specs and no run on disk.

    ⚠️ The specs come from `collect_specs()` rather than a hand-built stub.
    `render` reads spec keys DIRECTLY -- the project's convention, so a renamed
    constant raises instead of printing None -- which means a stub inevitably
    drifts from what the renderer expects. Using the real thing also makes this
    test fail if a spec key a report prints ever disappears.
    """
    return {
        "run_id": "test", "slug": "test", "purpose": "unit test",
        "sample": {"file": "f.json", "n": 4, "sha256": "0" * 64},
        "specs": document_run.collect_specs(),
    }


def _provenance_meta(*, dirty=False):
    meta = _meta()
    meta["sample"]["sha256"] = "a" * 64
    meta["specs"]["git"].update(
        sha="b" * 40, branch="private/provenance-test", dirty=dirty)
    meta["completion"] = {
        "documented_at": "2026-08-20T12:00:00", "raw_file": "answers.json",
        "raw_sha256": "e" * 64,
    }
    meta["scoring"] = [{
        "scored_at": "2026-08-20T12:00:00", "scorer_git_sha": "c" * 40,
        "scorer_tree_dirty": dirty, "scorer_manifest_sha256": "d" * 64,
    }]
    return meta


class TestCompactProvenance:
    def test_identifiers_are_hidden_but_metadata_is_linked(self, run):
        summaries, scored = build_summaries(run)
        text = document_run.render(_provenance_meta(), summaries, scored)
        for marker in ("a" * 12, "b" * 12, "c" * 12, "d" * 12,
                       "e" * 12, "private/provenance-test", "raw_sha256",
                       "Code at RUN START", "## Scoring history", "| scorer SHA |"):
            assert marker not in text
        assert "- Sample: `f.json` (n=4)" in text
        assert "## Provenance" in text
        assert ("File checksums and available scoring history are retained in "
                "[meta.json](meta.json). Historical scorer hashes before and "
                "after 17 August 2026 are not directly comparable.") in text
        assert ("- [meta.json](meta.json) — recorded settings, file checksums "
                "and available scoring history.") in text

    @pytest.mark.parametrize("dirty", [False, True])
    def test_run_start_dirty_disclosure_survives_without_git_ids(self, run, dirty):
        summaries, scored = build_summaries(run)
        text = document_run.render(_provenance_meta(dirty=dirty), summaries, scored)
        notice = "- Recorded code state: uncommitted changes."
        assert (notice in text) is dirty
        if not dirty:
            assert "Recorded code state:" not in text

    @pytest.mark.parametrize("with_scoring", [False, True])
    def test_corrections_and_reconstructed_notes_survive_independently(self, run, with_scoring):
        summaries, scored = build_summaries(run)
        meta = _provenance_meta()
        if not with_scoring:
            meta.pop("scoring")
        meta["notes"] = (
            "Prompt framing was corrected without changing answers.\n\n"
            "The noise-filter setting is RECONSTRUCTED, not recovered or measured.")
        meta["provenance_corrections"] = [{
            "corrected_at": "2026-08-17T18:19:52", "field": "completion.raw_sha256",
            "old_value": None, "new_value": "pinned-value",
            "reason": "Added after execution; not run-start proof.",
            "evidence": "archived file",
        }]
        before = copy.deepcopy(meta)
        text = document_run.render(meta, summaries, scored)
        assert meta["notes"] in text
        assert ("> ⚠️ **Provenance correction** (2026-08-17T18:19:52): "
                "`completion.raw_sha256` recorded `None`, actually `pinned-value`. "
                "Added after execution; not run-start proof. Evidence: archived file") in text
        assert ("## Provenance" in text) is with_scoring
        assert "[meta.json](meta.json)" in text
        assert meta == before

    def test_provenance_changes_leave_inputs_and_numeric_section_unchanged(self, run):
        summaries, scored = build_summaries(run)
        plain = document_run.render(_meta(), summaries, scored)
        meta = _provenance_meta(dirty=True)
        before = copy.deepcopy((meta, summaries, scored))
        rendered = document_run.render(meta, summaries, scored)
        assert (meta, summaries, scored) == before
        def outcomes(text):
            return text.split("## Outcomes", 1)[1].split("## Prompt", 1)[0]
        assert outcomes(rendered) == outcomes(plain)

    @pytest.mark.parametrize("old,new", [
        (" per CLAUDE.md Current State", ""),
        ("Pre-registered in docs/reference_run_plan.md D5:",
         "Declared in advance in the development protocol:"),
        ("(pre-registered, docs/prompt_abstention_study.md §3)",
         "(declared in advance in the development protocol)"),
        ("(docs/reference_run_plan.md)", "(development protocol)"),
        ("pre-registration in docs/reference_run_plan.md",
         "advance declaration in the development protocol"),
    ])
    def test_only_known_purpose_text_is_replaced_for_display(self, run, old, new):
        summaries, scored = build_summaries(run)
        meta = _meta()
        meta["purpose"] = "Retain before " + old + " retain after."
        original = meta["purpose"]
        text = document_run.render(meta, summaries, scored)
        assert ("**Retain before " + new + " retain after.**") in text
        assert original not in text
        assert meta["purpose"] == original

    @pytest.mark.parametrize("run_id,slug,notice", [
        ("20260817_2226", "reference",
         "Historical reference: superseded by the 18 August 2026 `reference-v8` run."),
        ("20260812_1738", "statement-model-v1",
         "The raw-file checksum was recorded on 17 August 2026, after execution; "
         "it does not establish integrity before that date."),
    ])
    def test_historical_notice_requires_the_matching_run_and_slug(self, run, run_id, slug, notice):
        summaries, scored = build_summaries(run)
        meta = _meta()
        meta.update(run_id=run_id, slug=slug)
        assert notice in document_run.render(meta, summaries, scored)
        meta["slug"] = "unrelated"
        assert notice not in document_run.render(meta, summaries, scored)
