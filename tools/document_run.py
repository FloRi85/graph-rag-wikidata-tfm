"""
Document a development run so it is reproducible, comparable and never lost.

Every DEV run gets a directory under `data/results/dev_runs/`:

    <timestamp>_<model-slug>_<run-slug>/
        <run_id>_<slug>_raw.json   the answers, verbatim and untruncated (git-tracked)
        meta.json     machine-readable specs
        report.md     human-readable specs + outcomes

Raw answers are preserved because `temperature=0` does not guarantee identical
provider responses. Re-running an experiment is not a substitute for its saved
outputs. See docs/reproduction.md for the retained runs and provenance.

THREE INVARIANTS, each learned from a specific failure:

1. The raw answers file is IMMUTABLE. Answers cost API calls and cannot be reproduced;
   scores are cheap and change whenever the scorer improves (five scorer fixes
   so far, several of which moved published numbers). So raw is written once and
   `report.md` is regenerated from it -- never the other way round.

2. Specs are captured from the LIVE CODE at documentation time, including the
   git SHA and whether the tree was dirty. Prompt wording is an experimental
   variable here, not boilerplate: on Nemotron the abstention clause is worth
   ~33 points of C3 abstention and the brevity line ~54. A run whose prompt is
   not recorded cannot be interpreted, only guessed at.

3. Runs are ARMS, not configs. The C'/E comparison was one config under two
   prompt variants; a four-config eval is four configs under one. Both shapes
   normalise to {arm_name: rows}, so prompt-variant runs fit the same layout
   instead of being improvised somewhere else.

Usage:
    python tools/document_run.py RAW.json --slug c3-prompt-variants \
        --purpose "Does an abstention-preserving brevity line beat the shipped one?" \
        --questions data/questions/mintaka_sample_dev_200.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEV_RUNS = ROOT / "data" / "results" / "dev_runs"


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

# Relocated to src/experiment_spec.py so run_eval can snapshot the specs BEFORE
# the first API call without importing from tools/. Re-exported here because
# every existing caller and test refers to `document_run.collect_specs`.
from src.experiment_spec import (          # noqa: E402
    SPEC_SCHEMA_VERSION, collect_specs, _git, _git_status_porcelain,
    code_dirty_paths,
)
from src.eval.metrics import (                   # noqa: E402
    WRONG_THRESHOLD, score_results as metrics_score_results,
)


def _scoring_manifest_hash() -> str:
    """SHA256 over every file that can change a SCORE.

    ⚠️ A GIT SHA IS NOT ENOUGH and neither is hashing metrics.py alone. A dirty
    tree means the SHA describes something other than what ran, and scoring
    depends on more than one module: the gold-form loader decides what an answer
    is compared against, and `prompts.ABSTAIN_SENTINEL` IS the abstention
    detector. A change in any of them moves numbers.
    """
    h = hashlib.sha256()
    # ⚠️ `src/eval/load_questions.py` was replaced here on 2026-08-16 by the two
    # modules that took over its job. It is a SCORING input: `parse_answers`
    # builds the gold an answer is compared against, and `parse_questions`
    # supplies the question-entity surface forms the either-or guard reads.
    # `gold_source` decides WHICH file those come from, so it belongs here too.
    #
    # ⚠️ The hash therefore changes for every run documented from today, and
    # that is correct rather than unfortunate: the scoring inputs did change.
    # 🔴 `tools/document_run.py` IS IN THIS LIST, AND ITS ABSENCE WAS A REAL
    # FAILURE (fixed 2026-08-17). `score_arm` lives in this file and calls
    # `classify_outcome` itself, so this file decides the table -- yet it was not
    # among the files it fingerprints. When `score_arm` was fixed on 2026-08-17,
    # every number in three reports moved (`correct 0.0` -> `44.2`) while the
    # manifest hash stayed identical, reporting "same scorer" across the largest
    # scoring change of the day. A fingerprint that omits the code doing the work
    # is worse than none, because it licenses trust.
    for rel in ("src/eval/metrics.py", "src/eval/stats.py",
                "src/eval/parse_questions.py", "src/eval/parse_answers.py",
                "src/eval/gold_source.py", "src/prompts.py",
                "tools/document_run.py"):
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Raw shapes -> arms
# ---------------------------------------------------------------------------

def normalise(raw) -> dict[str, list[dict]]:
    """
    Return {arm_name: [row, ...]} where each row has id/answer/finish_reason/error.

    Two shapes are accepted deliberately: `run_eval.py` output (one row per
    question carrying every config's answer) and prompt-variant output (one
    block per arm). Anything else should be converted rather than special-cased
    here, so the layout stays one thing.
    """
    if isinstance(raw, dict) and "arms" in raw:                 # variant run
        return {
            name: [
                {"id": r["id"], "answer": r.get("answer", ""),
                 "finish_reason": r.get("finish_reason"), "error": r.get("error"),
                 "context": r.get("context"), "secs": r.get("secs")}
                for r in arm["rows"]
            ]
            for name, arm in raw["arms"].items()
        }

    rows = raw if isinstance(raw, list) else raw.get("results", [])
    if rows and "answers" in rows[0]:                            # run_eval output
        arms: dict[str, list[dict]] = {}
        for r in rows:
            for cfg, ans in (r.get("answers") or {}).items():
                arms.setdefault(cfg, []).append({
                    "id": r["id"], "answer": ans,
                    "finish_reason": (r.get("finish_reasons") or {}).get(cfg),
                    "error": (r.get("errors") or {}).get(cfg),
                    "context": (r.get("contexts") or {}).get(cfg),
                    "context_words": (r.get("context_words") or {}).get(cfg),
                })
        return arms
    raise SystemExit("Unrecognised raw shape: expected an 'arms' dict or run_eval rows.")


def arm_overrides(raw) -> dict[str, dict]:
    """
    Per-arm deviations from the framing in `collect_specs`, read off the raw file.

    A prompt-variant runner rewrites `_NEMOTRON_STYLE["answer"]["preamble"]` in
    place between passes, so by the time a run is documented the live style holds
    only the LAST arm's wording -- and recording that one line as the run's
    framing describes exactly one of the arms and misdescribes the rest. The
    runners already store the line they used per arm; this surfaces it so the
    arm that differs says so instead of inheriting a neighbour's prompt.
    """
    if not (isinstance(raw, dict) and "arms" in raw):
        return {}
    return {
        name: {"brevity_line": arm["brevity_line"]}
        for name, arm in raw["arms"].items()
        if "brevity_line" in arm
    }


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


# Every per-config metric `metrics.score_results` produces, in the order the
# report prints them, with how to render each.
#
# ⚠️ THIS LIST IS CLOSED AND A TEST PINS IT. `test_every_scorer_metric_is_reported`
# fails if `score_results` grows a key that is not rendered here -- because the
# failure mode is silence: a new metric is computed, nobody prints it, and the
# run record is quietly incomplete. That is exactly how F1 and F1@attempted went
# unreported until 2026-08-17 despite being computed on every run since July.
_PCT = lambda v: f"{v * 100:.1f}"                      # noqa: E731
_NUM = lambda v: f"{v:.0f}" if v is not None else "—"  # noqa: E731

# ⚠️ THE NUMERIC TWIN OF `_PCT`, for percentages meta.json stores as numbers.
# Both MUST divide first and multiply by 100 after, because the two orderings
# build different floats at exact percentage ties: 338/4000 is exactly 8.45%,
# and `round(100 * 338 / 4000, 1)` gives 8.4 while `f"{(338/4000) * 100:.1f}"`
# prints 8.5 — meta.json and report.md disagreed by 0.1 pt about the SAME
# counts on the TEST-4000 run of record. tools/report.py also multiplies the
# fraction, so division-first is the ordering every printed number already
# used; `test_report_integrity` sweeps every count at the standard split sizes
# to pin the agreement.
_pct1 = lambda num, den: round(num / den * 100, 1)     # noqa: E731

OUTCOME_COLUMNS = [
    # (report heading, key in score_results per_config, renderer)
    ("n scored",   "n",                  _NUM),
    ("Attempted",  "n_attempted",        _NUM),
    ("Abst%",      "abstention_rate",    _PCT),
    ("Correct%",   "correct_rate",       _PCT),
    ("Halluc%",    "hallucination_rate", _PCT),
    # The same two over the ATTEMPTED subset. Reported beside the all-questions
    # rates, never instead of them -- see metrics._aggregate for why this
    # denominator is gameable in the direction the thesis argues.
    ("Correct%@att", "correct_rate_attempted",       _PCT),
    ("Halluc%@att",  "hallucination_rate_attempted", _PCT),
    ("Other%",     "other_rate",         _PCT),
    ("Coverage%",  "coverage",           _PCT),
    ("EM",         "em",                 _PCT),
    ("F1",         "f1",                 _PCT),
    ("EM@att",     "em_attempted",       _PCT),
    ("F1@att",     "f1_attempted",       _PCT),
    ("Score",      "score",              _PCT),
    ("Mean CtxW",  "context_words",      _NUM),   # a MEAN — the thesis quotes medians from compare_context_sizes.py
]

# Facts about an arm that `score_results` does not model: how the generation
# went, rather than how it scored.
RUN_COLUMNS = [
    ("Truncated",    "truncated"),
    ("Errors",       "errors"),
    ("Excluded",     "excluded"),
    ("Median words", "median_answer_words"),
]


def _outcome_tables(summaries: dict) -> list[str]:
    """The outcomes table, split scoring metrics from generation facts.

    ⚠️ TWO TABLES, NOT ONE WIDE ONE. Thirteen scoring columns beside four
    bookkeeping ones wraps in any terminal and in GitHub's markdown, and a
    wrapped table is where a value gets read against the wrong row -- the one
    error that would be both invisible and fatal here.
    """
    L = ["**Scoring** — every metric `metrics.score_results` computes.", ""]
    L.append("| Arm | " + " | ".join(h for h, _, _ in OUTCOME_COLUMNS) + " |")
    L.append("|---" + "|---:" * len(OUTCOME_COLUMNS) + "|")
    for name, v in summaries.items():
        cells = []
        for _, key, render in OUTCOME_COLUMNS:
            cells.append(render(v[key]) if v.get(key) is not None else "—")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    L += ["", "**Generation** — how the run behaved, not how it scored.", ""]
    L.append("| Arm | " + " | ".join(h for h, _ in RUN_COLUMNS) + " |")
    L.append("|---" + "|---:" * len(RUN_COLUMNS) + "|")
    for name, v in summaries.items():
        L.append(f"| {name} | "
                 + " | ".join(str(v.get(k, 0)) for _, k in RUN_COLUMNS) + " |")
    return L


def _stratified_tables(scored: dict) -> list[str]:
    """F1 per answer type and per complexity, from `score_results`.

    Reported because the failure modes separate on answer TYPE in a way the
    aggregate hides -- C3 collapses on numerical and nearly holds on boolean --
    and `tools/report.py` has printed this since July while the run's own record
    did not.
    """
    L: list[str] = []
    for title, key in (("answer type", "per_answer_type"),
                       ("complexity", "per_complexity")):
        groups = scored.get(key) or {}
        if not groups:
            continue
        arms = sorted({a for g in groups.values() for a in g})
        L += [f"**F1 by {title}**", "",
              "| " + title + " | n | " + " | ".join(arms) + " |",
              "|---|---:" + "|---:" * len(arms) + "|"]
        for name in sorted(groups):
            per_arm = groups[name]
            n = max((per_arm[a]["n"] for a in per_arm), default=0)
            cells = [f"{per_arm[a]['f1'] * 100:.1f}" if a in per_arm else "—"
                     for a in arms]
            L.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def _condense_stats(rows: list[dict]) -> dict | None:
    """Configuration 4's condensing step: how often it overran, and how long it ran.

    🔴 THIS EXISTED NOWHERE IN THE RUN RECORD UNTIL 2026-08-17, AND IT IS THE
    MEASUREMENT THAT EXPOSED C4's WORST DEFECT. `run_eval` computes the same
    figure for its `!! C4 CONDENSE step truncated on N/200` console warning and
    then discards it, so `sweep-k30`'s report gave no hint that 35 of its 197 C4
    contexts were cut off mid-sentence. That is the "a warning that scrolls
    away" failure this project has now fixed in four separate places.

    ⚠️ COMPUTED FROM THE STORED CAPTURE, NOT INSTRUMENTED AT RUN TIME, so
    re-documenting BACKFILLS every run already on disk rather than only
    describing runs made from today. `_condense` writes
    `condense_finish_reason` / `condense_words` into the capture dict per
    question, and `normalise` carries that through as `row["context"]`.

    Returns None for any arm that has no condensing step -- which is every
    config except C4, and C4 itself on files written before `_condense` began
    capturing its own finish_reason.
    """
    seen = 0
    truncated = 0
    lengths: list[int] = []
    # `condense_fallback` (2026-08-19) is counted INDEPENDENTLY of
    # `condense_words`: a fallback row is answered under the C3-style RENDERING
    # of its ranked facts, and some fallback paths never populate a word count.
    # With an EMPTY fact pool that rendering is the header line alone — an
    # empty C3-style context, which is a different fact from "raw facts", so
    # the two are counted apart. Rows carrying the key at all are
    # "instrumented" (the pipeline writes it True AND False since 2026-08-19);
    # rows without it predate the flag, and absence there means "not
    # measured", never "no fallback".
    fallback_instrumented = 0
    fallbacks = 0
    fallbacks_empty_pool = 0
    for r in rows:
        cap = r.get("context")
        if not isinstance(cap, dict):
            continue
        if "condense_fallback" in cap:
            fallback_instrumented += 1
            if cap["condense_fallback"] is True:
                fallbacks += 1
                if cap.get("pool_size") == 0:
                    fallbacks_empty_pool += 1
        if "condense_words" not in cap:
            continue
        seen += 1
        lengths.append(cap.get("condense_words") or 0)
        if cap.get("condense_finish_reason") == "length":
            truncated += 1
    if not seen and not fallback_instrumented:
        return None
    lengths.sort()
    return {
        "n": seen,
        "truncated": truncated,
        "truncated_pct": _pct1(truncated, seen) if seen else 0.0,
        "median_words": lengths[len(lengths) // 2] if lengths else 0,
        "max_words": lengths[-1] if lengths else 0,
        "fallback_instrumented": fallback_instrumented,
        "fallbacks": fallbacks,
        "fallbacks_empty_pool": fallbacks_empty_pool,
    }


def score_arm(rows: list[dict], golds: dict, entities_by_id: dict,
              excluded: set[str] | None = None, own_errors: int = 0) -> dict:
    """Outcome counts for one arm, scored the way `tools/report.py` scores.

    🔴 THIS SCORED EVERYTHING AS A HALLUCINATION UNTIL 2026-08-17. It built its
    gold from FLATTENED question keys (`expected_answer`, `answer_type`), and the
    raw-record migration (5dd56e1) made the questions file raw Mintaka — where
    those keys do not exist. So gold came back empty, every answer scored 0, and
    everything that was not an abstention was classified a confident error. The
    three k-sweep reports written before the fix said `correct 0.0` while the
    run's own summary said 34.9%.

    ⚠️ IT NOW TAKES THE SAME `GoldAnswer` OBJECTS THE SCORER USES, via
    `gold_source.resolve_gold_source`. That is the point: this report and
    `tools/report.py` must not be able to disagree about the same answers, and
    the only way to guarantee it is to give them one gold definition rather than
    two readers of one file.
    """
    from src.eval import metrics

    # ⚠️ ALL-OR-NOTHING EXCLUSION, matching `metrics.score_results`. This used to
    # filter PER ARM (`if not r.get("error")`), so a config that hit three API
    # failures lost three questions while the others kept them -- giving the four
    # columns of this report different denominators and making them
    # incomparable with each other. On the k=30 sweep arm that read
    # 200/200/200/197 here against a flat 197 from tools/report.py, i.e. the two
    # tools disagreed about the same file, which the comment below has always
    # said must not happen. An API failure is not a prediction, and grading it
    # for some configs but not others breaks the paired comparison.
    excluded = excluded or set()
    graded = [r for r in rows if not r.get("error") and r["id"] not in excluded]
    counts = {k: 0 for k in metrics.OUTCOMES}
    for r in graded:
        gold_answer = golds.get(r["id"])
        if gold_answer is None:
            continue
        gold, answer_entities = metrics.parsed_gold_forms(gold_answer)
        q = entities_by_id.get(r["id"], {})
        q_entities = metrics.question_entity_labels({"id": r["id"]},
                                                    {r["id"]: q})
        # Scored as an attempt, then classified four ways — the same order
        # report.py uses, so a run's own report cannot disagree with the
        # headline tool run over the same file.
        # answer_type comes from the GoldAnswer, not from the question row: the
        # matcher and the strings it dispatches on must come from one place.
        sc = metrics.score_answer(r["answer"] or "", gold, gold_answer.answer_type,
                                  answer_entities, q_entities, assume_attempted=True)
        counts[metrics.classify_outcome(r["answer"], sc, r.get("finish_reason"))] += 1
    # ⚠️ THE FOUR OUTCOMES MUST PARTITION THE GRADED ROWS. `classify_outcome`
    # can only return the four buckets, so the one way a graded row can vanish
    # from the counts is the `gold_answer is None` skip above — and a skipped
    # row would make the four percentages silently sum below 100 (the
    # denominator keeps the row, the buckets lose it). `resolve_gold_source`
    # is supposed to refuse a gold file that does not cover the run, so this
    # firing means that contract broke; fail loudly rather than publish a
    # table that leaks questions.
    counted = sum(counts.values())
    if counted != len(graded):
        raise ValueError(
            f"outcome counts sum to {counted} but {len(graded)} rows were "
            f"graded — {len(graded) - counted} graded row(s) have no gold and "
            f"were skipped; Abst+Correct+Halluc+Other would not sum to 100%")
    abst, corr, wrong = counts[metrics.ABSTENTION], counts[metrics.CORRECT], counts[metrics.HALLUCINATION]
    n = len(graded) or 1
    words = sorted(len((r["answer"] or "").split()) for r in graded)
    condense = _condense_stats(rows)
    # ⚠️ `own_errors` IS PASSED IN, NOT COUNTED HERE. `normalise` only creates an
    # arm row when a config produced an ANSWER, so a config that failed on a
    # question has no row for it at all -- counting `r.get("error")` over the arm
    # therefore reported 0 errors for the one config that actually had them. It
    # is counted from the raw rows, where the `errors` field lives.
    return {
        # ⚠️ `n` IS THE DENOMINATOR THE RATES USE, not the number of rows the arm
        # has. Reporting 200 beside percentages computed over 197 invited exactly
        # the wrong reading. `errors` is now this arm's OWN failures, and
        # `excluded` the questions it lost because a DIFFERENT arm failed there --
        # C1 showed "3 errors" while having none, which was the same confusion in
        # a different column.
        "n": len(graded), "rows": len(rows), "graded": len(graded),
        "errors": own_errors,
        # Questions this arm lost because ANOTHER arm failed there. Its own
        # failures are already absent from `rows` (see above), so they are not
        # subtracted again.
        "excluded": len(rows) - len(graded),
        # Present only for Configuration 4; None everywhere else. See
        # `_condense_stats` for why this belongs in the run record at all.
        **({"condense": condense} if condense else {}),
        # `_pct1`, not `round(100 * x / n, 1)`: same digits as the report.md
        # table and tools/report.py, or the record disagrees with its own
        # report at exact ties (see the `_pct1` definition).
        "abstention_pct": _pct1(abst, n),
        "correct_pct": _pct1(corr, n),
        "hallucination_pct": _pct1(wrong, n),
        "other_pct": _pct1(counts[metrics.OTHER], n),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "median_answer_words": words[len(words) // 2] if words else 0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _display_purpose(purpose: str) -> str:
    """Replace known private-file references without altering the saved purpose."""
    replacements = (
        (" per CLAUDE.md Current State", ""),
        ("Pre-registered in docs/reference_run_plan.md D5:",
         "Declared in advance in the development protocol:"),
        ("(pre-registered, docs/prompt_abstention_study.md §3)",
         "(declared in advance in the development protocol)"),
        ("(docs/reference_run_plan.md)", "(development protocol)"),
        ("pre-registration in docs/reference_run_plan.md",
         "advance declaration in the development protocol"),
    )
    for old, new in replacements:
        purpose = purpose.replace(old, new)
    return purpose


def _historical_report_notice(meta: dict) -> str | None:
    """Display known limitations of archived runs without rewriting metadata."""
    key = (meta["run_id"], meta["slug"])
    if key == ("20260817_2226", "reference"):
        return "Historical reference: superseded by the 18 August 2026 `reference-v8` run."
    if key == ("20260812_1738", "statement-model-v1"):
        return ("The raw-file checksum was recorded on 17 August 2026, after execution; "
                "it does not establish integrity before that date.")
    return None


def render(meta: dict, summaries: dict, scored: dict | None = None) -> str:
    s, m = meta["specs"], meta["specs"]["model"]
    notice = _historical_report_notice(meta)
    L = [
        f"# {meta['slug']}", "",
        f"**{_display_purpose(meta['purpose'])}**", "",
        *([f"> {notice}", ""] if notice else []),
        f"- Run: `{meta['run_id']}`  ·  documented "
        f"{(meta.get('completion') or {}).get('documented_at', meta.get('documented_at', '?'))}",
        f"- Sample: `{meta['sample']['file']}` (n={meta['sample']['n']})",
        f"- Model: `{m['name']}` @ `{m['base_url']}`  ·  temperature {m['temperature']}  ·  max_tokens {m['max_tokens']}",
        # Preserve the recorded limitation; checkout identifiers remain in meta.json.
        *(["- Recorded code state: uncommitted changes."] if s["git"]["dirty"] else []),
        "",
        "## Outcomes", "",
    ] + _outcome_tables(summaries)
    L += ["",
          "> `hallucination` = a clean attempted answer that disagrees with the Mintaka",
          "> benchmark gold (scored < 0.50) — a factuality proxy against the 2021",
          "> annotation, not independently verified world-factuality: a live-graph answer",
          "> more current than the key is charged as wrong. `abstention` = the exact",
          "> instructed refusal string. `other` = empty, truncated, or a refusal in the",
          "> model's own words — reported rather than guessed at.",
          "> Abst + Correct + Halluc + Other = 100 — exact in the underlying counts",
          "> (a guard in `score_arm` enforces the partition); each printed column is",
          "> rounded to one decimal independently, so a displayed row may sum to",
          "> anywhere in 99.8–100.2. Coverage = the attempted share,",
          "> Correct + Halluc = 1 - Abst - Other (an OTHER row is not an attempt).",
          "> EM/F1 are over ALL questions (a non-attempt scores 0); the @att pair is",
          "> over the attempted subset only, which is why it is far higher.",
          "> Scores are regenerable from the raw file; re-run this tool after any",
          "> scorer change rather than editing this file.", ""]
    if scored:
        L += _stratified_tables(scored)

    # Configuration 4's condensing step, which no other config has. Reported
    # because a truncated condensation hands the answering call a context that
    # stops mid-sentence, and that is invisible in every column above.
    for name, v in summaries.items():
        c = v.get("condense")
        if not c:
            continue
        L += [f"**{name} — condensing step (Configuration 4 only)**", "",
              f"- condensations: {c['n']}",
              f"- truncated at CONDENSE_MAX_TOKENS: {c['truncated']} "
              f"({c['truncated_pct']}%)"
              + ("  ⚠️ **those answering calls received a context cut off "
                 "mid-sentence**" if c["truncated"] else ""),
              f"- condensation length: median {c['median_words']} words, "
              f"max {c['max_words']}"]
        # A fallback question was answered under the C3-style RENDERING of its
        # ranked facts — with an empty fact pool that is a header-only, empty
        # C3-style context, NOT raw facts, and saying "effectively answered as
        # Configuration 3" for those rows was wrong (all nine TEST fallbacks
        # were empty-pool). Instrumented per row only since 2026-08-19; on
        # older files absence means "not measured", so the denominator names
        # the instrumented rows and a legacy run says so instead of claiming 0.
        fi = c.get("fallback_instrumented", 0)
        if fi:
            fb = c.get("fallbacks", 0)
            fe = c.get("fallbacks_empty_pool", 0)
            line = f"- empty-condensation fallback: {fb} of {fi} instrumented rows"
            if fb:
                detail = []
                if fe:
                    detail.append(f"{fe} with an EMPTY fact pool — the fallback "
                                  "context is the header line alone (an empty "
                                  "C3-style context, not raw facts)")
                if fb - fe:
                    detail.append(f"{fb - fe} answered under the C3-style "
                                  "rendering of their ranked facts — a "
                                  "treatment change")
                line += "  ⚠️ **" + "; ".join(detail) + "**"
            L += [line]
        else:
            L += ["- empty-condensation fallback: not instrumented (run "
                  "predates the 2026-08-19 `condense_fallback` flag)"]
        L += [""]

    if meta.get("notes"):
        L += ["## Notes", "", meta["notes"], ""]

    L += ["## Prompt", "",
          "⚠️ Prompt wording is an experimental variable in this project, not boilerplate.", ""]
    for role, v in (s.get("prompt_framing") or {}).items():
        L += [f"**Framing — `{role}` role**", "```",
              f"system:   {v['system']!r}", f"preamble: {v['preamble']!r}", "```", ""]
    # An arm that rewrote the framing says so here. Without this the framing
    # block above reads as if it applied to every arm, which is false for any
    # run whose whole purpose was to vary one line of it.
    per_arm = {n: a["prompt_override"] for n, a in (meta.get("arms") or {}).items()
               if a.get("prompt_override")}
    if per_arm:
        L += ["**Per-arm overrides** — these arms replaced part of the framing above.", "",
              "| Arm | brevity line |", "|---|---|",
              *(f"| {n} | `{o.get('brevity_line')!r}` |" for n, o in per_arm.items()), ""]
    for k, v in s["prompt_fragments"].items():
        L.append(f"- `{k}` = `{v}`")
    L.append("")
    for name, tpl in s["prompt_templates"].items():
        L += [f"**{name}**", "```", tpl, "```", ""]

    # `retrieval` gained active/dormant/legacy sections on 2026-08-13; older
    # meta.json files are flat. Read both rather than crashing on an archived run.
    r = s["retrieval"]
    active = r.get("active", r)
    L += ["## Retrieval settings", "",
          f"- top_k: C2 {active['top_k']['C2']} · C3 {active['top_k']['C3']}"
          f" · C4 {active['top_k']['C4']}",
          f"- embedding model: `{active['embedding_model']}`"
          + (f" · top-k allocation: `{active['topk_allocation']}`"
             if "topk_allocation" in active else ""),
          # 🔴 `reverse_limit` WAS READ HERE AND KILLED EVERY RUN'S REPORT. The
          # truthy-path deletion (b756a52) removed it from `collect_specs` --
          # correctly: it capped the retired truthy reverse query and the
          # statement path never read it -- but left this reader in place. No run
          # was attempted between that commit and 2026-08-17, so the first three
          # sweep arms all completed and then failed to document with
          # `KeyError: 'reverse_limit'`. The answers survived (raw is written
          # before documentation, by design); the reports had to be regenerated.
          #
          # ⚠️ Every key here is read DIRECTLY, and that is the convention --
          # `.get()` with a default would have turned this into a report quietly
          # printing "reverse limit None" instead of an error, which is the exact
          # failure mode `experiment_spec`'s docstring warns about. The lesson is
          # not "use .get()", it is that removing a spec key means grepping for
          # its readers.
          f"- noise filter: `{active['noise_filter_mode']}` · per-property cap "
          f"{active['reverse_per_prop_cap']}"]
    if "statement_cache_version" in active:
        L += [f"- statements: cache v{active['statement_cache_version']} · row limit "
              f"{active['statement_row_limit']} · rank policy "
              f"{active['rank_policy']} · labels `{active['label_langs']}`",
              f"- strict retrieval: {active['strict_retrieval']} · follows "
              f"redirects: {active['follows_redirects']}"]
    if "article_cache_version" in active:
        # Labelled "articles", so `cache v1` here next to `cache v7` above reads
        # as two independent counters rather than one arm lagging the other.
        fp = active.get("article_cache_fingerprint") or {}
        L += [f"- articles: cache v{active['article_cache_version']} · "
              f"{fp.get('chunk_size')}-word windows, {fp.get('overlap')} overlap · "
              f"strict retrieval: {active['article_strict_retrieval']} · "
              f"{active['article_fetch_attempts']} fetch attempts"]
    if r.get("dormant") or r.get("legacy_truthy"):
        L += [f"- dormant: {r.get('dormant')} · legacy: {r.get('legacy_truthy')}"]
    L.append("")

    if s.get("generation"):
        g = s["generation"]
        L += ["## Generation", "",
              f"- answer max_tokens {g['answer_max_tokens']} · C4 condense "
              f"max_tokens {g['c4_condense_max_tokens']} · temperature "
              f"{g['temperature']}", ""]

    pf = meta.get("preflight") or {}
    apf = meta.get("article_preflight") or {}
    if pf or apf:
        L += ["## Retrieval preflight", ""]
    if pf:
        # ⚠️ SPLIT BY DIRECTION, NAMING THE CONSTANT THAT BOUND. This said
        # "row limit bound" until 2026-08-17, which is STATEMENT_ROW_LIMIT on the
        # OUTGOING query -- a limit that fired zero times on DEV-200. Every one of
        # the 154 is the INCOMING per-property cap. The two have different fixes,
        # so collapsing them made the line unactionable as well as wrong.
        trunc = pf.get("truncated") or {}
        directions: dict[str, list[str]] = {}
        for qid, dirs in sorted(trunc.items()):
            for d in (dirs or ["unrecorded"]):
                directions.setdefault(d, []).append(qid)
        WHY = {"reverse": "a property hit REVERSE_PER_PROP_CAP",
               "forward": "STATEMENT_ROW_LIMIT bound"}
        L += [f"- gold entities checked: {pf.get('n_entities', '?')}",
              f"- truncated (served an arbitrary slice): {len(trunc)}"
              + (f" of {pf['n_entities']}" if pf.get("n_entities") else "")]
        for d in ("reverse", "forward"):
            ids = directions.get(d, [])
            L.append(f"    - {d} {len(ids)} — {WHY[d]}"
                     + (f": {', '.join(ids)}" if ids else " (never bound)"))
        for d, ids in directions.items():
            if d not in ("reverse", "forward"):
                L.append(f"    - {d} {len(ids)}: {', '.join(ids)}")
        L.append("")
    if apf:
        # The TEXT arm's half. Captured in meta.json since 2026-08-16 but not
        # printed until 2026-08-17, so C2's corpus coverage was invisible while
        # the KG arm's was listed in full.
        empty = apf.get("no_article") or {}
        L += [f"- articles checked: {apf.get('n_entities', '?')}",
              f"- no English article: {len(empty)}"
              + (f" — contributing no chunks to Configuration 2's pool: "
                 f"{', '.join(sorted(empty))}" if empty else " (all have one)"),
              ""]

    # 🔴 MERGED, NOT `or`. This read `completion or operational`, and on a
    # RE-documentation `completion` holds only {documented_at, raw_file,
    # raw_sha256} -- truthy, so it short-circuited and the run's telemetry
    # (workers, call counts, answers_recorded) silently vanished from the report
    # while surviving in meta.json. Every re-document erased it. `completion`
    # still wins on key collisions, since its documented_at is the current one.
    o = {**(meta.get("operational") or {}), **(meta.get("completion") or {})}
    if o:
        L += ["## Operational", ""]
        # ⚠️ LOGICAL CALLS FIRST, THEN THE COUNTERS. `completion_responses` counts
        # every call that reached the API, retries included -- which is what makes
        # retry pressure measurable (sweep-k10 carried 189 of them) but means it
        # does not reconcile with the obvious arithmetic. On a 200-question run it
        # read 1002 against 200 x 5 = 1000, because `run_config_with_retry`
        # retries the WHOLE configuration: a transient failure on C4's answering
        # call re-runs its condense too, and that second condensation succeeds.
        #
        # The logical count is derived from the raw file rather than instrumented,
        # so it backfills for runs already on disk.
        answers = o.get("answers_recorded")
        condensations = sum((v.get("condense") or {}).get("n", 0)
                            for v in summaries.values())
        if answers:
            logical = answers + condensations
            L += [f"- LLM calls (logical): {logical}",
                  f"    {answers} answers + {condensations} C4 condensations"]
            attempts = o.get("completion_attempts")
            responses = o.get("completion_responses")
            if attempts and responses:
                L += [f"- API attempts: {attempts}",
                      f"    {responses} succeeded · "
                      f"{o.get('completion_exceptions', 0)} failed",
                      f"    {attempts - logical} extra from retries "
                      f"(a retry re-runs C4's condense as well as its answer)"]
        skip = {"answers_recorded", "completion_attempts", "completion_responses",
                "completion_exceptions"} if answers else set()
        skip.add("raw_sha256")  # Kept in metadata and the raw-file restoration instructions.
        L += [*(f"- {k}: {v}" for k, v in o.items() if k not in skip), ""]

    sc = meta.get("scoring") or []
    if sc:
        L += ["## Provenance", "",
              "File checksums and available scoring history are retained in [meta.json](meta.json). "
              "Historical scorer hashes before and after 17 August 2026 are not directly comparable.", ""]

    for c in (meta.get("provenance_corrections") or []):
        L += [f"> ⚠️ **Provenance correction** ({c['corrected_at']}): `{c['field']}` "
              f"recorded `{c['old_value']}`, actually `{c['new_value']}`. "
              f"{c['reason']} Evidence: {c['evidence']}", ""]

    L += ["## Files", "",
          "- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**",
          "- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.",
          "- `report.md` — this file, regenerated by `tools/document_run.py`.", ""]
    return "\n".join(L)


def write_start_snapshot(run_dir: Path, *, run_id: str, slug: str, purpose: str,
                         questions_file: str, n_questions: int,
                         workers: int, started: str, rpm: int | None = None,
                         preflight: dict | None = None,
                         article_preflight: dict | None = None,
                         resume: dict | None = None) -> dict:
    """
    The IMMUTABLE record of what a run was configured with, written BEFORE the
    first API call.

    ⚠️ THIS IS THE FIX FOR RE-DOCUMENTING OVERWRITING PROVENANCE. `document()`
    used to call `collect_specs()` itself and rewrite `meta.json` wholesale, so
    re-generating a report after a code change stamped the run with the CURRENT
    model, prompts and retrieval settings. Re-documenting the truthy-era
    reference run today would have relabelled it as post-rebuild -- silently
    turning an archived result into a false one. For a sealed TEST run that is
    unrecoverable.

    So: runtime specs are captured here, once, and `document()` preserves them.
    Everything it learns later (outcomes, telemetry, scoring) is written into
    separate keys.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "meta_schema_version": SPEC_SCHEMA_VERSION,
        "run_id": run_id, "slug": slug, "purpose": purpose,
        "status": "started",
        "started_at": started,
        "sample": {
            "file": questions_file, "n": n_questions,
            "sha256": hashlib.sha256(Path(questions_file).read_bytes()).hexdigest(),
        },
        # The RESOLVED rpm (env + endpoint default + flag), not the CLI input:
        # what the run will actually pace at. None only for legacy callers.
        "start_settings": {"workers": workers, "rpm": rpm},
        # BOTH arms. Named separately rather than merged: truncation (KG) and
        # an absent article (text) are different events with different meanings,
        # and one "preflight" blob holding both would need a reader to know
        # which keys belong to which arm.
        "preflight": preflight or {},
        "article_preflight": article_preflight or {},
        # Captured before anything is spent. Never rewritten.
        "specs": collect_specs(),
    }
    if resume:
        # A resumed run: which rows were carried verbatim from the interrupted
        # run's sidecar, hashes of the source meta/sidecar, and the telemetry
        # scope flag ("fresh-segment-only") — see run_eval._resume_mismatches
        # for the verification that gated this.
        meta["resume"] = resume
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def _load_existing_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def new_run_dir(base: Path, run_id: str, model_slug: str, slug: str) -> Path:
    """A run directory that cannot collide with an existing run.

    ⚠️ `run_id` is MINUTE resolution and the old code used `mkdir(exist_ok=True)`,
    so two runs started in the same minute with the same slug wrote into one
    directory -- the second silently overwriting the first's meta and report
    while both raw files sat there. This repo already came within one minute of
    it (`..._1737_smoke-statement-model` and `..._1738_statement-model-v1`),
    saved only by the slugs differing.
    """
    out = base / f"{run_id}_{model_slug}_{slug}"
    if not out.exists():
        return out
    for n in range(2, 100):
        alt = base / f"{run_id}_{model_slug}_{slug}-{n}"
        if not alt.exists():
            print(f"  [document_run] {out.name} exists — using {alt.name}")
            return alt
    raise RuntimeError(f"cannot find a free run directory beside {out}")


def document(raw_path: Path, slug: str, purpose: str,
             questions_file: str = "data/questions/mintaka_sample_dev_200.json",
             notes: str = "", operational: dict | None = None,
             run_id: str | None = None, run_dir: Path | None = None,
             backfill: bool = False) -> Path:
    """
    Finalise a run: score it, append a scoring record, regenerate report.md.

    `run_dir` is passed by run_eval, which has already created the directory,
    written the start snapshot and put the raw file there. Use `backfill=True`
    for a detached raw file produced by a one-off script -- that path creates a
    NEW directory and, having no start snapshot to preserve, records the specs
    as captured at documentation time rather than pretending they are the
    run's.

    ⚠️ RUNTIME SPECS ARE NEVER REWRITTEN when a start snapshot exists. Re-run
    this after a scorer change and you get a new scoring record and a fresh
    report; you do not get a run relabelled with today's code.
    """
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    arms = normalise(raw)
    overrides = arm_overrides(raw)

    # One gold definition, shared with tools/report.py and the scorer. Coverage
    # is CHECKED: a questions file that does not hold every id being scored
    # fails here rather than producing a report with silently missing gold.
    from src.eval.gold_source import resolve_gold_source
    row_ids = {r["id"] for rows in arms.values() for r in rows}
    golds, entities_by_id, _ = resolve_gold_source(
        row_ids, explicit=questions_file, verbose=False)

    # Computed over the RAW rows, so it sees every config's failures at once --
    # `arms` has already split them apart by config.
    from collections import Counter
    from src.eval.metrics import excluded_question_ids
    if isinstance(raw, list):
        excluded = excluded_question_ids(raw)
        errors_by_config = Counter(
            cfg for r in raw for cfg in (r.get("errors") or {}))
    else:
        excluded, errors_by_config = set(), Counter()
    summaries = {name: score_arm(rows, golds, entities_by_id, excluded,
                                 errors_by_config.get(name, 0))
                 for name, rows in arms.items()}

    # 🔴 THE METRICS COME FROM `metrics.score_results` — THE SAME FUNCTION
    # `tools/report.py` CALLS. Until 2026-08-17 this file computed its own
    # outcome counts by looping rows and calling `classify_outcome` itself, i.e.
    # a SECOND implementation of scoring. Two implementations of one thing is
    # how they came to disagree: the score_arm gold bug produced `correct 0.0`
    # here beside a correct 44.2 there, over the same answers.
    #
    # `score_arm` survives for the per-arm facts `score_results` does not model
    # -- truncation, median answer length, C4's condensing step, this arm's own
    # errors -- and, for prompt-variant files, as the only path available:
    # `score_results` understands run_eval rows, not an {arm: rows} dict.
    if isinstance(raw, list):
        scored = metrics_score_results(raw, entities_by_id, golds=golds)
        for name, per_cfg in scored["per_config"].items():
            if name in summaries:
                # Root metrics win on any key collision, so a stale local
                # computation can never shadow the sanctioned one.
                summaries[name] = {**summaries[name], **per_cfg}
    else:
        scored = None

    if run_dir is None:
        if not backfill:
            raise ValueError(
                "document() needs run_dir. A detached raw file must be filed "
                "explicitly with backfill=True (or --backfill), because the "
                "alternative -- inventing a NEW timestamped directory -- forks "
                "the run instead of finalising it.")
        slug_model = re.sub(r"[^a-z0-9]+", "-",
                            collect_specs()["model"]["name"].lower()).strip("-")
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M")
        out = new_run_dir(DEV_RUNS, run_id, slug_model, slug)
    else:
        out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M")

    meta = _load_existing_meta(out)
    raw_bytes = Path(raw_path).read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()

    # ⚠️ FATAL, not a warning. A meta.json describing different answers than the
    # raw file beside it is worse than no meta.json: every number in the report
    # would be attributed to a run that did not produce them.
    recorded_sha = ((meta.get("completion") or {}).get("raw_sha256"))
    if recorded_sha and recorded_sha != raw_sha:
        raise RuntimeError(
            f"raw file hash mismatch in {out.name}:\n"
            f"  meta.json records {recorded_sha}\n"
            f"  {Path(raw_path).name} hashes to {raw_sha}\n"
            f"The raw answers file is immutable. Refusing to re-document.")

    if not meta.get("specs"):
        # Backfill / legacy path: no start snapshot exists, so say so rather
        # than presenting documentation-time specs as the run's own.
        meta.setdefault("meta_schema_version", SPEC_SCHEMA_VERSION)
        meta["specs"] = collect_specs()
        meta["specs"]["captured"] = (
            "AT DOCUMENTATION TIME, not at run start — this run predates the "
            "start-snapshot lifecycle, so these specs describe the code as of "
            "documentation and may not be what ran.")

    meta.update({
        "run_id": meta.get("run_id", run_id),
        "slug": meta.get("slug", slug),
        "purpose": meta.get("purpose") or purpose,
        "status": "complete",
        # `n` is the distinct question count of the RUN (same meaning as the
        # normal path's n_questions). This line shipped as `len(qs)` — a name
        # that never existed in this function — so every backfill of a meta.json
        # lacking a `sample` key died with a NameError (found 2026-08-18).
        "sample": meta.get("sample") or {
            "file": questions_file, "n": len(row_ids),
            "sha256": hashlib.sha256(Path(questions_file).read_bytes()).hexdigest(),
        },
        "arms": {
            k: {"summary": v, **({"prompt_override": overrides[k]} if k in overrides else {})}
            for k, v in summaries.items()
        },
        "notes": notes or meta.get("notes", ""),
    })
    # Completion telemetry is a SEPARATE key from start settings: one describes
    # the configuration, the other what actually happened under it.
    meta["completion"] = {
        "documented_at": datetime.now().isoformat(timespec="seconds"),
        "raw_file": Path(raw_path).name,
        "raw_sha256": raw_sha,
        **(operational or {}),
    }
    # Kept for readers (and older tooling) that expect the flat key.
    meta["operational"] = operational or meta.get("operational") or {}

    # APPEND-ONLY. Every re-score is its own record, so "which scorer produced
    # this table" is answerable from the file instead of from git archaeology.
    scoring_record = {
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "scorer_git_sha": _git("rev-parse", "HEAD"),
        # code_dirty_paths, not raw porcelain: the run's own untracked output
        # directory is not evidence about the scorer (see experiment_spec).
        "scorer_tree_dirty": bool(code_dirty_paths()),
        "scorer_manifest_sha256": _scoring_manifest_hash(),
        "questions_file": questions_file,
        "questions_sha256": hashlib.sha256(
            Path(questions_file).read_bytes()).hexdigest(),
        "raw_sha256": raw_sha,
        "threshold": WRONG_THRESHOLD,
        "summaries": summaries,
        "n_excluded_by_arm": {k: v.get("errors", 0) for k, v in summaries.items()},
    }
    # ⚠️ APPEND ONLY WHEN THE SCORER STATE CHANGED. It appended on every run,
    # so re-documenting five runs four times produced eight near-identical rows
    # per run and buried the two that recorded a real change. A record is kept
    # when the commit, the tree-dirty flag or the manifest differs from the last
    # one -- which is exactly when the table could have moved.
    #
    # ⚠️ STILL APPEND-ONLY: nothing is ever rewritten or removed, and a genuinely
    # new scorer state always lands. This drops repeats, not history.
    history = meta.setdefault("scoring", [])
    prev = history[-1] if history else None
    changed = prev is None or any(
        prev.get(k) != scoring_record.get(k)
        for k in ("scorer_git_sha", "scorer_tree_dirty", "scorer_manifest_sha256"))
    if changed:
        history.append(scoring_record)
    else:
        # The summaries are refreshed in place so the record still describes the
        # table above it, without claiming a new scoring event happened.
        prev["scored_at"] = scoring_record["scored_at"]
        prev["summaries"] = scoring_record["summaries"]
        prev["n_excluded_by_arm"] = scoring_record["n_excluded_by_arm"]

    # Only copy when raw lives elsewhere. run_eval writes the raw file into
    # run_dir itself, and copying a file onto itself raises SameFileError.
    dest = out / f"{run_id}_{slug}_raw.json"
    if Path(raw_path).resolve() != dest.resolve():
        shutil.copyfile(raw_path, dest)                  # verbatim, untruncated
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render(meta, summaries, scored), encoding="utf-8")
    for name, v in summaries.items():
        print(f"  {name:22} abst {v['abstention_pct']:5} correct {v['correct_pct']:5} "
              f"halluc {v['hallucination_pct']:5} trunc {v['truncated']:3} err {v['errors']}")
    return out


def add_provenance_correction(run_dir: Path, field: str, old_value, new_value,
                              reason: str, evidence: str) -> dict:
    """
    Record that a stored provenance value is wrong, WITHOUT overwriting it.

    ⚠️ PERMITTED UNDER test_runs/ TOO, and deliberately so. Prohibiting
    corrections there sounds safer but is not: it makes a KNOWN-WRONG metadata
    value permanent and undocumented. What must never happen is mutation of the
    original assertion, and append-only already guarantees that -- the original
    stays exactly where it was, with a dated, evidenced note beside it.
    """
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.setdefault("provenance_corrections", []).append({
        "corrected_at": datetime.now().isoformat(timespec="seconds"),
        "field": field, "old_value": old_value, "new_value": new_value,
        "reason": reason, "evidence": evidence,
    })
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--questions", default="data/questions/mintaka_sample_dev_200.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--operational", default="{}", help="JSON dict of run facts")
    ap.add_argument("--run-id", default=None, help="Override; default = now")
    ap.add_argument("--run-dir", default=None,
                    help="The run's existing directory. Finalises it in place, "
                         "preserving its start-snapshot specs.")
    ap.add_argument("--backfill", action="store_true",
                    help="Detached raw file with no run directory: create a NEW "
                         "one. Required explicitly, because doing it implicitly "
                         "forks a run instead of finalising it.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None and not args.backfill:
        # Infer the run directory from the raw file's own location before
        # asking. A raw file already sitting in a run directory is the common
        # case, and creating a second directory for it was the bug.
        parent = Path(args.raw).resolve().parent
        if (parent / "meta.json").exists():
            run_dir = parent
            print(f"  [document_run] finalising {parent.name} in place")

    out = document(Path(args.raw), slug=args.slug, purpose=args.purpose,
                   questions_file=args.questions, notes=args.notes,
                   operational=json.loads(args.operational), run_id=args.run_id,
                   run_dir=run_dir, backfill=args.backfill)
    print(f"documented -> {out}")


if __name__ == "__main__":
    main()
