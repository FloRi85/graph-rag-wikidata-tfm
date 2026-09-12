r"""
Evaluation runner — runs all four configs on a question set and prints results.

Usage (from repo root):
    venv\Scripts\python src/eval/run_eval.py data/questions/mintaka_sample_dev_200.json

⚠️ It takes RAW MINTAKA ONLY (the verbatim `mintaka_*_raw.json` shape or a
sample of raw records), and anything else fails loudly (`NotRawMintakaError`):
a non-Mintaka file was once mis-detected as raw and produced rows with no
entities and an empty gold without raising. Writing a run on the SEALED test
split needs `--test-run`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import datetime
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the box-drawing
# glyphs used below or accented characters in Mintaka answers. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.pipelines.base_llm as base_llm
import src.pipelines.rag as rag
import src.pipelines.graph_rag as graph_rag
import src.pipelines.graph_rag_rerank as graph_rag_rerank
from src.eval.metrics import score_results, print_summary
# 🔴 THE TWO HALVES ARE LOADED SEPARATELY ON PURPOSE. `parse_questions` gives
# the question side and `parse_answers` the gold; nothing under `src/pipelines/`
# imports the latter, so a pipeline cannot reach the answer key -- a property of
# the import graph rather than a convention someone has to keep. Keeping that
# true here means the golds never travel with the questions into `run_question`.
from src.eval.parse_questions import load_questions, load_question_ids, BLOCK_MODES
from src.eval.parse_answers import load_answers
from src.eval.parallel import map_questions, prewarm, DEFAULT_WORKERS
from src.retrieval.wikidata import (
    IncompleteRetrievalError, cache_entry_status, truncated_entities,
)
from src.retrieval import wikipedia, embedding_retriever, wikidata_pool
from src import llm_config, token_counter, prompts
from src.experiment_spec import code_dirty_paths, collect_specs, GitUnavailableError

DEV_RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "dev_runs"
# The sealed split lives somewhere else ON PURPOSE. dev_runs/ is defined as
# "runs you were allowed to look at and act on"; a test run is spent once and
# must not be mistakable for one of those in a directory listing, a git diff or
# a citation. Separating them keeps the dev-tunes/test-sealed protocol visible
# in the file layout rather than only in prose.
TEST_RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "test_runs"

# The authority for "is this question from the sealed split" — the verbatim
# test download, checked by QUESTION ID. A filename substring is not provenance:
# `mintaka_sample_100.json` was drawn FROM test and carried nothing in its name
# to say so, which is how the sealed split leaked into scorer development.
_TEST_RAW_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "questions" / "mintaka_test_raw.json"

# The verbatim TEST download, byte for byte (pinned 2026-08-19). Membership by
# question id proves every question BELONGS to the split, but a test-derived
# subset file also passes that test — the pin is what enforces "the whole
# split, from the canonical file". Recompute if the file is ever legitimately
# re-downloaded: certutil -hashfile data\questions\mintaka_test_raw.json SHA256
#
# TWO FORMS OF ONE FILE (found 2026-09-09 validating the hand-in export on a
# cold checkout). Git stores the download with LF line endings; a Windows
# checkout with core.autocrlf=true materialises it with CRLF, and that is the
# form every sealed-run record hashed (`sample.sha256` / `questions_sha256` in
# the TEST meta.json = the CRLF value). A Linux/macOS clone -- or the snapshot
# export, which is byte-identical to the blob -- yields the LF value. Both are
# the same verbatim download, so the guard accepts either; it still refuses any
# file whose CONTENT differs.
_TEST_RAW_SHA256 = "2784aa9a047d86d23bb78645dad79c971478486f9520489915cd0a2d22ae547f"  # CRLF checkout
_TEST_RAW_SHA256_LF = "b1eb231d0929fa387faaf2b043765a1ccbe348ae98475a498435d878185c6acd"  # LF (git blob)
_TEST_RAW_SHA256_FORMS = frozenset({_TEST_RAW_SHA256, _TEST_RAW_SHA256_LF})

# The frozen configuration's worker count (reference-v8 ran --workers 2; it cut
# 429 retries from 141–189 to 13 and exclusions to zero). Unlike the other
# frozen values this is NOT the module default (DEFAULT_WORKERS=8 suits dev
# samples), so the canonical TEST command states it explicitly.
_FROZEN_TEST_WORKERS = 2
# The frozen RESOLVED rpm — NVIDIA Build's paced rate, the endpoint default, so
# the canonical command omits --rpm. Checked as the resolved value because env
# (LLM_RPM_LIMIT), endpoint default and flag collapse into one number.
_FROZEN_TEST_RPM = 33


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_config_violations(args, resolved_rpm: int) -> list[str]:
    """Every way this invocation deviates from the frozen TEST configuration.

    The frozen values (k=30, global allocation, entity block `all`, prompt
    variant F) are the module defaults, so the check is "no override flag was
    passed" rather than a second copy of the values — a copy could drift from
    the constants it duplicates. Two exceptions checked by VALUE: `--workers`
    (its default serves dev samples, so the frozen 2 is required explicitly)
    and the RESOLVED rpm (env + endpoint default + flag collapse into one
    number; the frozen configuration is 33, NVIDIA Build's paced rate).
    """
    violations = []
    if args.limit is not None:
        violations.append("--limit (the split is spent whole: all 4,000 questions)")
    if args.top_k is not None:
        violations.append("--top-k (frozen: 30, the module default — omit the flag)")
    if args.allocation is not None:
        violations.append("--allocation (frozen: global, the module default — omit the flag)")
    if args.entity_block is not None:
        violations.append("--entity-block (frozen: all, the module default — omit the flag)")
    if args.prompt_variant is not None:
        violations.append("--prompt-variant (frozen: shipped variant F — omit the flag)")
    if args.value_desc is not None:
        violations.append("--value-desc (frozen: off — an exploratory retrieval "
                          "arm, never the sealed split)")
    if args.no_document:
        violations.append("--no-document (a sealed run without provenance cannot be cited)")
    if args.workers != _FROZEN_TEST_WORKERS:
        violations.append(f"--workers {args.workers} (frozen: {_FROZEN_TEST_WORKERS} — "
                          f"pass --workers {_FROZEN_TEST_WORKERS} explicitly)")
    if resolved_rpm != _FROZEN_TEST_RPM:
        violations.append(f"resolved rpm {resolved_rpm} (frozen: {_FROZEN_TEST_RPM}, "
                          f"NVIDIA Build's paced rate — omit --rpm and LLM_RPM_LIMIT "
                          f"on an NVIDIA endpoint)")
    return violations


def prior_test_runs() -> list[Path]:
    """Run directories already in test_runs/ — the split is spent ONCE."""
    if not TEST_RUNS_DIR.exists():
        return []
    return sorted(p for p in TEST_RUNS_DIR.iterdir() if p.is_dir())


def _short(value) -> str:
    r = repr(value)
    return r if len(r) <= 70 else r[:67] + "..."


def _resume_signature(meta: dict) -> dict:
    """The fields a resume must match EXACTLY, normalised from a run's meta.

    Timestamps (captured_at, started_at) and the branch name are excluded —
    they legitimately differ across the interruption. Everything that shapes
    an answer is included: code SHA, sample identity, model/endpoint/limits,
    every prompt fragment/template/framing, retrieval settings with cache
    versions and fingerprints, workers, the resolved rpm, and both preflight
    summaries. Missing keys surface as None and mismatch — a source meta too
    old to carry a field cannot be verified and must not be resumed.
    """
    specs = meta.get("specs") or {}
    git = specs.get("git") or {}
    return {
        "git_sha": git.get("sha"),
        "git_dirty": git.get("dirty"),
        "sample_sha256": (meta.get("sample") or {}).get("sha256"),
        "sample_n": (meta.get("sample") or {}).get("n"),
        "model": specs.get("model"),
        "prompt_framing": specs.get("prompt_framing"),
        "prompt_fragments": specs.get("prompt_fragments"),
        "prompt_templates": specs.get("prompt_templates"),
        "retrieval": specs.get("retrieval"),
        "context_construction": specs.get("context_construction"),
        "generation": specs.get("generation"),
        "workers": (meta.get("start_settings") or {}).get("workers"),
        "rpm": (meta.get("start_settings") or {}).get("rpm"),
        "preflight": meta.get("preflight"),
        "article_preflight": meta.get("article_preflight"),
    }


def _resume_mismatches(source_meta: dict, current_meta: dict) -> list[str]:
    """Every way the interrupted run and this invocation differ. Empty = safe.

    ⚠️ NO OVERRIDE EXISTS FOR THESE. Merging two experiment states into one
    raw file is never legitimate; if code or corpus had to change mid-run, the
    honest outcome is a fresh run, not a blend.
    """
    src, cur = _resume_signature(source_meta), _resume_signature(current_meta)
    problems = []
    if src["git_dirty"] is not False:
        problems.append("the interrupted run started on a dirty tree (or its "
                        "meta predates the dirty flag) — its SHA does not "
                        "fully describe what it ran")
    if cur["git_dirty"] is not False:
        problems.append("the CURRENT tree is dirty — commit before resuming")
    for key, src_val in src.items():
        if key == "git_dirty":
            continue
        if src_val != cur[key]:
            problems.append(f"{key}: interrupted run {_short(src_val)} "
                            f"!= current {_short(cur[key])}")
    return problems


def _sealed_test_ids() -> set[str]:
    """Ids of the sealed TEST split, or empty when the raw file is absent.

    Empty is only tolerated for DEV runs (no overlap is then provable-false
    rather than verified, but a dev sample cannot be harmed by that); a
    --test-run REFUSES to proceed without the file — see the guard in main().
    """
    if not _TEST_RAW_FILE.exists():
        return set()
    return load_question_ids(_TEST_RAW_FILE)

# The base-LLM key keeps its historical `_abstain` suffix: result files written
# before 2026-08-04 carry BOTH an abstention-permitted `base_llm_abstain` column
# and a `base_llm` column from a since-dropped no-abstention variant. Reusing the
# bare `base_llm` key here would make those old columns re-score as the current
# baseline, which they are not.
CONFIGS = [
    ("base_llm_abstain", "Config 1 — Base LLM"),
    ("rag",              "Config 2 — RAG"),
    ("graph_rag",        "Config 3 — Graph-RAG"),
    # Key stays `rerank` (stored result files); label says condensing, which is
    # what C4 actually adds over C3 — both rerank by embedding.
    ("rerank",           "Config 4 — Graph-RAG + condensing"),
]


def run_config(config_id: str, question: str, entity_names: list[str], entity_qids: list[str],
               capture: dict | None = None, entity_block: str = "") -> str:
    """Dispatch one config.

    ⚠️ `entity_block` REACHES C2, C3 AND C4 BUT NOT C1 (decision 2026-08-16).
    `base_llm.answer` takes no such argument at all, so the asymmetry is a
    property of the signature rather than a call site someone has to remember.
    Its consequence belongs in §5: the C1-vs-retrieval delta now measures
    retrieval PLUS an entity annotation, so C1 is not a clean control.
    """
    if config_id == "base_llm_abstain":
        return base_llm.answer(question)
    if config_id == "rag":
        return rag.answer(question, entity_names, qids=entity_qids, capture=capture,
                          entity_block=entity_block)
    if config_id == "graph_rag":
        return graph_rag.answer(question, entity_names, qids=entity_qids, capture=capture,
                                entity_block=entity_block)
    if config_id == "rerank":
        return graph_rag_rerank.answer(question, entity_names, qids=entity_qids,
                                       capture=capture, entity_block=entity_block)
    raise ValueError(f"Unknown config: {config_id}")


# ---------------------------------------------------------------------------
# Transient-failure handling
#
# API failures used to be written into the answer string as "ERROR: ..." and
# were then scored as attempted WRONG answers — inflating the error'd config's
# apparent hallucination rate. The 2026-07-25 sweep carried 9 such failures
# (7x HTTP 500, 2x 503), all in the canonical k10_cap30 cell, which depressed
# that cell and inflated every other cell's apparent gain.
#
# Two-part fix: retry transient failures here (most are recoverable), and
# record whatever still fails in a separate `errors` field so the scorer can
# exclude it instead of grading it. See metrics.score_results.
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = (
    "500", "502", "503", "504", "429",
    "timeout", "timed out", "too many concurrent", "rate limit",
    "overloaded", "connection", "temporarily unavailable", "bad gateway",
)

MAX_ATTEMPTS = 4
BACKOFF_BASE = 2.0  # seconds: 2, 4, 8

# 🔴 EXHAUSTED ELSEWHERE — DO NOT RETRY HERE. Retrieval owns its own retry
# (`wikidata._STATEMENT_QUERY_ATTEMPTS` = 3 endpoint attempts per query), so an
# IncompleteRetrievalError has already failed three times when it arrives.
# Retrying it here would make 4 x 3 = 12 endpoint hits per entity, four of them
# identical in every respect, and would hammer an endpoint that is evidently
# already struggling.
#
# ⚠️ It needs an explicit exemption rather than relying on the marker list: the
# message embeds the underlying exception type, so "(TimeoutError)" matches the
# "timeout" marker and the error would be classified transient by accident.
_NON_TRANSIENT_TYPES = (IncompleteRetrievalError,)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, _NON_TRANSIENT_TYPES):
        return False
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def preflight_statement_cache(questions: list, questions_file: str) -> dict:
    """
    Abort before spending budget if any gold entity is not usably cached.

    Returns a record for the run's provenance: how many entities were checked,
    and which ones are TRUNCATED (served an arbitrary slice because a limit
    bound). Truncation is permitted -- forbidding it would mean not running at
    all -- but it is recorded rather than left in scrolled-away console output.

    ⚠️ IT IS COMMON, NOT RARE, AND THE FIGURE HERE USED TO SAY OTHERWISE. This
    docstring claimed "34 of 238" until 2026-08-16; that was the pre-2026-08-14
    meaning, when `truncated` marked the global 4,000-row limit binding.
    Measured on the live v7 cache it is **154 of 238, all `reverse`** -- i.e. at
    least one property hit `REVERSE_PER_PROP_CAP`. Same loss as before, finally
    visible: `_clean_statements` always applied that cap and nothing recorded
    it. Read 154 as "now measured", not as a regression.
    """
    qids = sorted({qid for question in questions for qid in question.qids})
    if not qids:
        return {"n_entities": 0, "truncated": {}}

    bad: dict[str, list[str]] = {}
    for qid in qids:
        st = cache_entry_status(qid)
        if st["state"] != "ok":
            bad.setdefault(st["state"], []).append(qid)

    if bad:
        lines = [f"{n} {state}" for state, ids in sorted(bad.items())
                 for n in [len(ids)]]
        sample = {s: ids[:5] for s, ids in sorted(bad.items())}
        raise SystemExit(
            f"\n🔴 STATEMENT CACHE PREFLIGHT FAILED — {', '.join(lines)} of "
            f"{len(qids)} gold entities are not usable.\n"
            f"   {sample}\n\n"
            f"   Nothing was spent. Warm the cache first:\n"
            f"     venv\\Scripts\\python tools/prewarm_statement_cache.py "
            f"--questions {questions_file}\n")

    trunc = truncated_entities(qids)
    print(f"cache preflight: {len(qids)} gold entities usable"
          + (f", {len(trunc)} truncated — at least one property hit its cap; "
             f"recorded per entity in meta.json" if trunc else ""))
    return {"n_entities": len(qids), "truncated": trunc}


def preflight_article_cache(questions: list, questions_file: str) -> dict:
    """
    Abort before spending budget if any gold entity's article is not usably cached.

    The text arm's half of the preflight, and it exists because C2 used to be
    the arm that failed QUIETLY: a missing article, an empty extract and a
    network error all reached the pipeline as "No Wikipedia articles found".

    Returns a record for the run's provenance: how many entities were checked,
    and which have NO English article. That is permitted -- many Wikidata items
    have no enwiki page, and forbidding it would mean not running -- but it is
    recorded, because those entities contribute nothing to C2's pool and a
    reader of the run should be able to see how many there were.

    ⚠️ MIRRORS `preflight_statement_cache` DELIBERATELY, down to the message
    shape. The two arms are checked the same way and fail the same way, so a
    reader does not have to learn two conventions to interpret one abort.
    """
    qids = sorted({qid for question in questions for qid in question.qids})
    if not qids:
        return {"n_entities": 0, "no_article": {}}

    bad: dict[str, list[str]] = {}
    for qid in qids:
        st = wikipedia.cache_entry_status(qid)
        if st["state"] != "ok":
            bad.setdefault(st["state"], []).append(qid)

    if bad:
        lines = [f"{len(ids)} {state}" for state, ids in sorted(bad.items())]
        sample = {s: ids[:5] for s, ids in sorted(bad.items())}
        raise SystemExit(
            f"\n🔴 ARTICLE CACHE PREFLIGHT FAILED — {', '.join(lines)} of "
            f"{len(qids)} gold entities are not usable.\n"
            f"   {sample}\n\n"
            f"   Nothing was spent. Warm the cache first:\n"
            f"     venv\\Scripts\\python tools/prewarm_article_cache.py "
            f"--questions {questions_file}\n")

    empty = wikipedia.empty_articles(qids)
    print(f"article preflight: {len(qids)} gold entities usable"
          + (f", {len(empty)} with no English article — they contribute nothing "
             f"to C2's pool; recorded per entity in meta.json" if empty else ""))
    return {"n_entities": len(qids), "no_article": empty}


def run_config_with_retry(config_id, question, entity_names, entity_qids, capture,
                          entity_block=""):
    """
    Returns (answer, error) — exactly one is None.

    Transient API failures are retried with exponential backoff. A non-transient
    exception fails immediately; there is no point burning quota on a bug.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return run_config(config_id, question, entity_names, entity_qids, capture,
                              entity_block), None
        except Exception as e:                                    # noqa: BLE001
            last = e
            if not _is_transient(e) or attempt == MAX_ATTEMPTS:
                break
            delay = BACKOFF_BASE ** attempt
            print(f"    ! transient failure ({type(e).__name__}), "
                  f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s: {str(e)[:90]}")
            time.sleep(delay)
            capture.clear()  # don't mix a failed attempt's partial capture in

    return None, {
        "type": type(last).__name__,
        "message": str(last)[:500],
        "transient": _is_transient(last) if last else False,
        "attempts": MAX_ATTEMPTS if _is_transient(last) else 1,
    }


# Serializes each question's output block. Workers finish out of order, so
# without this the per-config lines of different questions interleave into
# unreadable soup. Blocks still ARRIVE in completion order; `results` is
# separately kept in input order by map_questions.
_print_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Crash safety
#
# Results used to be serialized ONCE, after every question had finished. An
# unattended overnight run that died — or a laptop that slept — at question 190
# of 200 lost all 190, and on a rate-limited or slow provider a full run is
# hours, not minutes. Each completed row is therefore appended to a JSONL
# sidecar as it lands, so a killed run can be recovered from the partial file
# instead of repeated. The sidecar is deleted once the real results file is
# written; its presence afterwards means the run did not finish.
# ---------------------------------------------------------------------------

_partial_lock = threading.Lock()
_partial_path: Path | None = None


def _append_partial(row: dict) -> None:
    if _partial_path is None:
        return
    line = json.dumps(row, ensure_ascii=False)
    with _partial_lock:
        with _partial_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            # Losing the buffer is the exact failure this exists to prevent.
            fh.flush()


def _row_answer_value(gold):
    """Mintaka's typed literal, unwrapped, for non-entity answers; else None.

    Matches the shape every stored result file uses, so a row written today and
    one written in July describe their gold the same way. Entity answers keep
    None: their payload is a list of {QID, labels}, which `answer_qids` and
    `answer_entities` already carry in a readable form.
    """
    if gold is None or gold.answer_type == "entity":
        return None
    payload = gold.payload
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload


def run_question(i: int, q, total: int, gold=None) -> dict:
    """Run all configs for one question and return its result row.

    `q` is a `parse_questions.Question`; `gold` its `parse_answers.GoldAnswer`.

    ⚠️ THE GOLD IS WRITTEN INTO THE ROW BUT IS NO LONGER AUTHORITATIVE. Scoring
    resolves it from raw Mintaka by id (`score_results(..., golds=...)`), so
    these fields are DESCRIPTIVE — they keep the raw file readable on its own,
    which matters because it is the evidence. Freezing the gold into the row is
    what let a run carry whatever definition was current the day it was written;
    that is exactly what the `golds` argument exists to stop. Do not add a
    scoring rule that reads them back.

    ⚠️ `gold` is passed in rather than looked up here so the answer key never
    enters the same scope as the pipelines' inputs by accident.
    """
    entity_qids  = q.qids
    # The name the QUESTION used, not the canonical label — see
    # parse_questions.Question.entity_mentions for the decision and the measured
    # 122/301 anchors where they differ.
    entity_names = q.entity_mentions
    question     = q.text
    expected     = gold.mention if gold else ""
    # Built ONCE per question and handed to C2/C3/C4 (and to both of C4's
    # calls). Rendering it per config would let the four prompts drift apart in
    # the one section they are supposed to share.
    entity_block = q.prompt_block(prompts.ENTITY_BLOCK_MODE)

    row = {
        "id":              q.id or str(i),
        "question":        question,
        "expected":        expected,                      # answer mention (surface)
        # Mintaka's typed literal for non-entity answers, in the scalar shape
        # older result files use.
        "answer_value":    _row_answer_value(gold),
        # Every string that would have counted as right. Written because the raw
        # file is the evidence and "what was this graded against" should be
        # answerable without re-deriving it from a split file.
        "answer_forms":    list(gold.forms) if gold else [],
        "answer_entities": list(gold.members) if gold else [],   # set members only
        "answer_type":     gold.answer_type if gold else "",     # matcher dispatch
        "type":            q.category,
        "complexity":      q.complexity,
        "answer_qids":     list(gold.qids) if gold else [],
        "entity_qids":     entity_qids,
        "answers":         {},
        "contexts":        {},  # per-config retrieved context (error analysis)
        # Words of context actually handed to the answering LLM, per config.
        # Equal top_k does NOT mean equal context: a Wikipedia chunk is orders
        # of magnitude longer than a statement. Measured on the DEV-200 sweep, C2
        # receives ~33x more text than C3 at k=10 (2774 vs 85 words) and ~31x
        # at k=30 (7469 vs 243). Recorded per question so the summary can state
        # that asymmetry instead of leaving it implicit (see rag.py TOP_K).
        "context_words":   {},
        # config_id -> finish_reason, recorded ONLY when it is not "stop".
        # Absent means the answer completed normally; "length" means MAX_TOKENS
        # truncated it.
        "finish_reasons":  {},
        "errors":          {},  # config_id -> {type, message, transient, attempts}
    }

    lines = [
        "=" * 68,
        f"Q{i}/{total} [{q.category or '?'}/{q.complexity or '?'}]  {question}",
        f"Expected : {expected}  entities: {entity_names}",
        "─" * 68,
    ]

    for config_id, label in CONFIGS:
        capture: dict = {}
        answer, error = run_config_with_retry(
            config_id, question, entity_names, entity_qids, capture, entity_block
        )
        if error is not None:
            lines.append(f"{label:35s} [FAILED after retries] {error['type']}: "
                         f"{error['message'][:80]}")
            row["errors"][config_id] = error
            # No answer is recorded: an infrastructure failure is not a
            # prediction and must not be graded as one.
            continue

        lines.append(f"{label:35s} {answer}")
        row["answers"][config_id] = answer
        # 'length' means MAX_TOKENS cut the answer off. Recorded because a
        # truncated answer is indistinguishable from a short one downstream and
        # would be graded as wrong rather than flagged as unusable. For Config 4
        # this is the ANSWERING call — the condensing call ran earlier and its
        # reason has already been overwritten.
        finish = llm_config.last_finish_reason()
        if finish and finish != "stop":
            row["finish_reasons"][config_id] = finish
        if capture:
            row["contexts"][config_id] = capture
            ctx = capture.get("context")
            if isinstance(ctx, str):
                # Whitespace split: the one unit every context-size figure
                # uses (the CtxW column, tools/compare_context_sizes.py).
                row["context_words"][config_id] = len(ctx.split())

    with _print_lock:
        print("\n".join(lines) + "\n", flush=True)

    _append_partial(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_file", help="Path to questions JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent questions (default {DEFAULT_WORKERS}; "
                             "1 = sequential). Measured to change results no more "
                             "than a sequential rerun does — see src/eval/parallel.py.")
    parser.add_argument("--purpose", default="",
                        help="One line on what this run is for. Goes in report.md. "
                             "A run nobody can state the purpose of is one nobody "
                             "can interpret later.")
    parser.add_argument("--slug", default=None,
                        help="Short name for the run directory (default: sample stem).")
    parser.add_argument("--no-document", action="store_true",
                        help="Skip report generation. Raw answers are still written.")
    parser.add_argument("--test-run", action="store_true",
                        help="Write to data/results/test_runs/ instead of dev_runs/. "
                             "REQUIRED when evaluating the sealed test split, and "
                             "refused for anything else. Enforces the FROZEN "
                             "configuration (see frozen_config_violations), the "
                             "canonical file hash, and one-spend.")
    parser.add_argument("--resume-from", default=None, metavar="PARTIAL_JSONL",
                        help="A .partial.jsonl sidecar left by an interrupted "
                             "run, IN PLACE in its run directory. Its rows are "
                             "kept VERBATIM; only the remaining questions run, "
                             "and the final raw file holds all of them in "
                             "question order. The resume is VERIFIED against "
                             "the interrupted run's own start snapshot (git "
                             "SHA, sample, model, prompts, retrieval, caches, "
                             "workers, rpm, preflights) and refuses any "
                             "mismatch, with no override. Under --test-run the "
                             "one-spend guard excludes the interrupted "
                             "directory itself, so a verified resume needs no "
                             "--allow-override.")
    parser.add_argument("--allow-override", action="store_true",
                        help="Bypass the --test-run launch guards "
                             "(frozen-configuration, file hash, one-spend, "
                             "dirty tree). Requires a non-empty --purpose; "
                             "every guard actually bypassed is recorded in the "
                             "run's purpose line. Does NOT bypass resume "
                             "verification.")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Requests/minute ceiling; 0 disables. Default resolves "
                             "from LLM_RPM_LIMIT, else 33 on NVIDIA endpoints, else "
                             "off. Pacing is FASTER than not pacing on a metered "
                             "endpoint — see llm_config.acquire_slot.")
    # ⭐ THE SWEEP AXES, AS RUN FLAGS (2026-08-16). Both were previously swept by
    # bespoke tools that wrote their own multi-cell file format, which then
    # needed `report.py --cell` to read back. Setting them here instead makes
    # every sweep arm a FIRST-CLASS RUN: its own directory, its own meta.json,
    # scored by the same report as the reference run. `experiment_spec` reads
    # these constants from the live modules at snapshot time, so the value each
    # arm used is recorded automatically and cannot be mislabelled.
    parser.add_argument("--top-k", type=int, default=None,
                        help="Retrieval depth for ALL THREE retrieval configs. "
                             "Omit to use each module's default (30). Held equal "
                             "across configs by construction — an arm where they "
                             "differed would confound depth with configuration.")
    parser.add_argument("--allocation", default=None,
                        choices=list(embedding_retriever.ALLOCATIONS),
                        help="How the top-k budget is split across a question's "
                             "entities: 'global' (primary) or 'floor'. Omit to "
                             "use the shipped default.")
    parser.add_argument("--entity-block", default=None, choices=list(BLOCK_MODES),
                        help="Which entity types get a line in the prompt's "
                             "`Question entities:` block: 'all' (primary — "
                             "linked entities plus normalised literals) or "
                             "'entity' (linked entities only). A declared "
                             "prompt arm; omit to use the shipped default.")
    parser.add_argument("--prompt-variant", default=None,
                        choices=sorted(prompts.PROMPT_VARIANTS),
                        help="A pre-registered prompt arm from prompts."
                             "PROMPT_VARIANTS (see docs/reproduction.md). "
                             "Omit to run the shipped variant F. "
                             "Sets the default slug to prompt-<name> so the "
                             "run directory names its arm.")
    parser.add_argument("--value-desc", default=None,
                        choices=list(wikidata_pool.VALUE_DESC_MODES),
                        help="Value-description enrichment on label collisions "
                             "(the named post-TEST option, see wikidata_pool). "
                             "'off' is the frozen default; 'collision' appends "
                             "the value entity's description when its label "
                             "equals the subject's. A RETRIEVAL change — "
                             "exploratory arms only, refused under --test-run. "
                             "Sets the default slug to value-desc-<mode>.")
    args = parser.parse_args()

    # ⚠️ TEST-RUN HARDENING (2026-08-19): the sealed split runs the frozen
    # configuration or it does not run. Checked FIRST — before the module
    # globals below are mutated — so a refused invocation never half-applies
    # its overrides. --allow-override is the escape hatch for the launch
    # guards (frozen configuration, file hash, one-spend, dirty tree); it
    # requires a stated --purpose, and every guard it actually bypasses is
    # recorded into the run's purpose line. The resume-verification checks
    # have NO override — see _resume_mismatches.
    if args.allow_override and not args.test_run:
        parser.error("--allow-override only means something with --test-run.")
    if args.rpm is not None and args.rpm < 0:
        parser.error(f"--rpm must be >= 0 (got {args.rpm}; 0 disables pacing)")
    if args.test_run and args.allow_override and not args.purpose.strip():
        parser.error("--allow-override requires a non-empty --purpose naming "
                     "the reason — a bypassed guard with no stated reason is "
                     "exactly the provenance hole the guards exist to close.")
    bypassed: list[str] = []
    # Resolved EARLY (env + endpoint default + flag collapse into one number)
    # so the frozen check tests what the run will actually do. Reading module
    # state only — safe before the mutations below.
    rpm = llm_config.default_rpm() if args.rpm is None else args.rpm
    if args.test_run:
        violations = frozen_config_violations(args, rpm)
        if violations:
            if args.allow_override:
                bypassed.append("frozen-configuration")
                print("⚠️  --allow-override: bypassing the frozen-configuration "
                      "guard:\n  " + "\n  ".join(violations))
            else:
                parser.error(
                    "--test-run refuses a non-frozen configuration:\n  "
                    + "\n  ".join(violations)
                    + "\nThe frozen values are the module defaults; the canonical "
                      "command passes no override flags (see "
                      "data/results/test_runs/README.md). --allow-override exists "
                      "for a deliberate, documented exception.")

    # Applied BEFORE the specs snapshot and before any worker starts, so the run
    # is documented with the values it actually used. Mutating a module global
    # mid-run would not be safe (see src/eval/parallel.py); doing it here is.
    if args.top_k is not None:
        rag.TOP_K = graph_rag.TOP_K = graph_rag_rerank.EMBED_TOP_K = args.top_k
    if args.allocation is not None:
        embedding_retriever.ALLOCATION = args.allocation
    if args.entity_block is not None:
        prompts.ENTITY_BLOCK_MODE = args.entity_block
    if args.value_desc is not None:
        wikidata_pool.VALUE_DESC_MODE = args.value_desc
        # Same rule as --prompt-variant: an arm run filed under the bare sample
        # stem would be indistinguishable from a reference run in a listing.
        if args.slug is None:
            args.slug = f"value-desc-{args.value_desc}"
    if args.prompt_variant is not None:
        # Two steps, both required: the constants change in prompts/llm_config,
        # and each pipeline re-derives its template from them — the templates
        # are assembled at import time, so mutation alone changes nothing a
        # pipeline sends. Done here so write_start_snapshot() below records the
        # arm's real templates and framing, not the shipped ones.
        overrides = prompts.apply_variant(args.prompt_variant)
        llm_config.apply_prompt_overrides(persona=overrides.get("persona"),
                                          brevity=overrides.get("brevity"))
        for mod in (base_llm, rag, graph_rag, graph_rag_rerank):
            mod.rebuild_prompts()
        # Every `args.slug or stem` site below then names the arm: an arm run
        # filed under the bare sample stem would be indistinguishable from a
        # reference run in a directory listing, which is how the study's runs
        # are meant to be identified.
        if args.slug is None:
            args.slug = f"prompt-{args.prompt_variant}"

    # Two-way guard, part 1 of 2: the FILENAME heuristic, kept only as a cheap
    # fast-fail before the file is even read. ⚠️ It is not the authority — a
    # sample drawn FROM the test split under a neutral name ("mintaka_
    # sample_100.json" was exactly that) sails straight past a substring test.
    # The authoritative check is by QUESTION ID, after the file is loaded below.
    looks_like_test = "test" in Path(args.questions_file).stem.lower()
    if looks_like_test and not args.test_run:
        parser.error(
            f"{args.questions_file} looks like the TEST split but --test-run was not "
            f"given. The test split is spent once and its results do not belong in "
            f"dev_runs/. Pass --test-run, or use a dev sample.")

    # CLI sanity: argparse's `type=int` happily accepts values Python slicing
    # then reinterprets — `--limit 0` is falsy and runs the WHOLE file,
    # `--limit -5` drops the last 5, `--top-k -5` selects pool-minus-5 items
    # and sends `floor` allocation's per-group reserve negative. All of them
    # would be snapshotted into meta.json as if legitimate.
    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be a positive integer (got {args.limit})")
    if args.top_k is not None and args.top_k <= 0:
        parser.error(f"--top-k must be a positive integer (got {args.top_k})")
    if args.workers is not None and args.workers <= 0:
        parser.error(f"--workers must be a positive integer (got {args.workers})")

    llm_config.set_rpm_limit(rpm)   # resolved above, before the frozen check

    # RAW MINTAKA ONLY — anything else raises NotRawMintakaError rather than
    # being coerced into a shape it does not have.
    questions = load_questions(args.questions_file)
    golds = load_answers(args.questions_file)

    # Two-way guard, part 2 of 2: QUESTION IDS are the authority (2026-08-18).
    # Checked on the FULL file, before --limit slicing and before any API call.
    # Any overlap with the sealed split demands --test-run; --test-run demands
    # that EVERY question belong to it. Ids cannot be renamed away the way a
    # filename can, which closes the sampler bypass above.
    test_ids = _sealed_test_ids()
    run_ids = {q.id for q in questions}
    overlap = run_ids & test_ids
    if overlap and not args.test_run:
        parser.error(
            f"{len(overlap)} of {len(run_ids)} questions in {args.questions_file} "
            f"belong to the SEALED TEST split (checked by question id against "
            f"{_TEST_RAW_FILE.name}). The test split is spent once; pass "
            f"--test-run, or use a dev sample.")
    if args.test_run:
        if not test_ids:
            parser.error(
                f"--test-run given but {_TEST_RAW_FILE} is missing, so the run "
                f"cannot be verified as the test split. Restore it via "
                f"tools/download_mintaka.py before spending the sealed split.")
        outside = run_ids - test_ids
        if outside:
            parser.error(
                f"--test-run was given but {len(outside)} of {len(run_ids)} "
                f"questions in {args.questions_file} are not in the test split. "
                f"Dev runs belong in dev_runs/, where they can be repeated.")
        # Membership proves every question BELONGS to the split; only the hash
        # proves this is the WHOLE split from the canonical file — a
        # test-derived subset passes the id check.
        actual_sha = _sha256(Path(args.questions_file))
        if actual_sha not in _TEST_RAW_SHA256_FORMS:
            if args.allow_override:
                bypassed.append("canonical-file-sha")
                print(f"⚠️  --allow-override: {args.questions_file} is NOT the "
                      f"canonical test file (sha256 {actual_sha[:12]}… != pinned "
                      f"{_TEST_RAW_SHA256[:12]}…). Proceeding because you said so; "
                      f"this run is a partial or non-canonical spend and must be "
                      f"documented as one.")
            else:
                parser.error(
                    f"--test-run requires the canonical test file. "
                    f"{args.questions_file} has sha256 {actual_sha}, pinned is "
                    f"{_TEST_RAW_SHA256} (CRLF checkout) / {_TEST_RAW_SHA256_LF} "
                    f"(LF). If the file is a test-derived subset, "
                    f"that is exactly what this guard refuses; if the canonical "
                    f"download legitimately changed, update _TEST_RAW_SHA256 in "
                    f"a reviewed commit first.")
        # The split is spent ONCE. A second full run would not be independent
        # evidence — the first run's results have been seen.
        prior = prior_test_runs()
        if args.resume_from:
            # A verified resume may relaunch: its own interrupted directory is
            # not a second spend. ONLY that directory is excluded — any other
            # prior run still refuses. The resume itself is verified against
            # that directory's meta.json below, with no override.
            src_dir = Path(args.resume_from).resolve().parent
            prior = [p for p in prior if p.resolve() != src_dir]
        if prior:
            if args.allow_override:
                bypassed.append("one-spend")
                print(f"⚠️  --allow-override: proceeding past {len(prior)} "
                      f"existing test run(s): "
                      f"{', '.join(p.name for p in prior[:3])}.")
            else:
                parser.error(
                    f"test_runs/ already holds {len(prior)} run "
                    f"director{'y' if len(prior) == 1 else 'ies'} "
                    f"({', '.join(p.name for p in prior[:3])}"
                    f"{'…' if len(prior) > 3 else ''}). The sealed split is spent "
                    f"once; a rerun needs --allow-override and a stated reason in "
                    f"--purpose.")
        # ⚠️ The sealed run must be reproducible from its SHA alone. Run
        # outputs under data/results/ are ignored (an interrupted TEST
        # directory must not make its own recovery look dirty); a git failure
        # fails CLOSED — "cannot verify" is not "clean".
        try:
            dirty = code_dirty_paths(strict=True)
        except GitUnavailableError as e:
            parser.error(f"--test-run: cannot verify the working tree ({e}). "
                         f"Refusing to launch blind — fix git, don't bypass.")
        if dirty:
            if args.allow_override:
                bypassed.append("dirty-tree")
                print(f"⚠️  --allow-override: launching with a DIRTY tree — "
                      f"{len(dirty)} path(s), e.g. {', '.join(dirty[:3])}. The "
                      f"recorded SHA will not fully describe this run.")
            else:
                parser.error(
                    f"--test-run requires a clean working tree: {len(dirty)} "
                    f"code/doc path(s) dirty (e.g. {', '.join(dirty[:5])}). "
                    f"Commit or stash them, then launch. Run outputs under "
                    f"data/results/ are already ignored.")
    if args.limit:
        questions = questions[: args.limit]

    # Crash recovery (2026-08-19): rows from an interrupted run's sidecar are
    # kept verbatim; only the remaining questions are run. Structural checks
    # here; the EXPERIMENT-STATE verification (_resume_mismatches, against the
    # interrupted run's own start snapshot) runs after the preflights below,
    # once every field of the signature exists. None of it is bypassable.
    done_rows: dict[str, dict] = {}
    source_meta: dict = {}
    if args.resume_from:
        rpath = Path(args.resume_from)
        if not rpath.exists():
            parser.error(f"--resume-from: {rpath} does not exist")
        # The sidecar must sit INSIDE its run directory, beside the start
        # snapshot written before the interrupted run's first API call — that
        # snapshot is what the resume is verified against. A sidecar copied
        # elsewhere has no provenance to verify.
        meta_path = rpath.parent / "meta.json"
        if not meta_path.exists():
            parser.error(f"--resume-from: no meta.json beside {rpath.name} — "
                         f"the sidecar must stay in its run directory, next to "
                         f"the start snapshot it will be verified against.")
        try:
            source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            parser.error(f"--resume-from: {meta_path} is corrupt ({e}); the "
                         f"resume cannot be verified against it.")
        if source_meta.get("status") != "started":
            parser.error(f"--resume-from: {meta_path.name} has status "
                         f"{source_meta.get('status')!r}, not 'started' — only "
                         f"an interrupted run can be resumed.")
        n_lines = 0
        for line in rpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                parser.error(f"--resume-from: line {n_lines} of {rpath.name} "
                             f"is not valid JSON ({e}). A torn final line means "
                             f"the process died mid-write — delete that line "
                             f"and its question will simply be re-run.")
            if not isinstance(row, dict) or not row.get("id") or "answers" not in row:
                parser.error(f"--resume-from: line {n_lines} of {rpath.name} is "
                             f"not a complete result row (needs 'id' and "
                             f"'answers').")
            done_rows[row["id"]] = row
        if len(done_rows) != n_lines:
            parser.error(f"--resume-from: {rpath.name} holds {n_lines} rows but "
                         f"only {len(done_rows)} distinct ids — duplicates "
                         f"cannot be merged safely.")
        stray = set(done_rows) - {q.id for q in questions}
        if stray:
            parser.error(
                f"--resume-from: {len(stray)} rows in {rpath.name} are not in "
                f"{args.questions_file} (e.g. {sorted(stray)[:3]}). The sidecar "
                f"belongs to a different question set.")
        pending = [q for q in questions if q.id not in done_rows]
        print(f"Resuming: {len(done_rows)} completed rows kept from "
              f"{rpath.name}, {len(pending)} questions remaining. "
              f"⚠️ API/token telemetry will cover the fresh segment only.")
    else:
        pending = questions

    total = len(questions)
    n_pending = len(pending)
    print(f"Running {n_pending} of {total} questions across {len(CONFIGS)} configs "
          f"({args.workers} workers)")
    print(f"model={llm_config.MODEL}  max_tokens={llm_config.MAX_TOKENS}  "
          f"temperature={llm_config.TEMPERATURE}  "
          f"rpm={'off' if rpm <= 0 else rpm}\n")

    # ⚠️ PREFLIGHT BEFORE THE FIRST LLM CALL, not a reminder to run prewarm.
    # A stale or incomplete entity is discovered lazily otherwise -- mid-run,
    # after budget has already been spent on the questions before it, and under
    # strict retrieval it now aborts the question rather than quietly answering
    # from a partial graph. Failing here costs nothing and is fixable in one
    # command; failing at question 140 of 200 costs the run.
    cache_report = preflight_statement_cache(questions, args.questions_file)
    # BOTH arms, before the first LLM call. Checking only the KG arm left C2
    # free to discover a missing article at question 140 and substitute an
    # empty context for it — silently, since that is what the retired
    # retriever returned for every failure mode alike.
    article_report = preflight_article_cache(questions, args.questions_file)

    # ⚠️ RESUME VERIFICATION — after the preflights so every signature field
    # exists, before anything is created or spent. NOT bypassable: merging two
    # experiment states into one raw file is never legitimate.
    if done_rows:
        current_meta = {
            "specs": collect_specs(),
            "sample": {"sha256": _sha256(Path(args.questions_file)),
                       "n": total},
            "start_settings": {"workers": args.workers, "rpm": rpm},
            "preflight": cache_report,
            "article_preflight": article_report,
        }
        problems = _resume_mismatches(source_meta, current_meta)
        if problems:
            sys.exit(
                "--resume-from REFUSED: the interrupted run and this invocation "
                "are not the same experiment:\n  " + "\n  ".join(problems) +
                "\nA resume carries rows VERBATIM; resuming across any of these "
                "differences would blend two experiment states into one raw "
                "file. There is no override — if the difference is real, the "
                "honest outcome is a fresh run.")
        print(f"Resume verified against {meta_path}: git SHA, sample, model, "
              f"prompts, retrieval, caches, workers, rpm and preflights all "
              f"match.")

    # One run, one directory, under version control. Previously answers went to
    # the git-ignored data/results/raw/ and the metrics were printed and lost;
    # a run's prompt, retrieval settings and code SHA were nowhere at all, so
    # interpreting a result months later meant reconstructing them from a
    # filename and a memory. Variant B's data was lost outright this way.
    #
    # The raw file stays a PLAIN LIST so every existing tool (report.py, the sweeps)
    # reads it unchanged; what the run was configured with lives beside it.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.questions_file).stem
    model_slug = re.sub(r"[^a-z0-9]+", "-", llm_config.MODEL.lower()).strip("-")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # ⚠️ COLLISION-SAFE. `run_id` is minute-resolution and this used to be
    # `mkdir(exist_ok=True)`, so two runs started in the same minute with the
    # same slug shared a directory and the second overwrote the first's meta and
    # report. This repo came within one minute of exactly that.
    from tools.document_run import new_run_dir, write_start_snapshot
    run_dir = new_run_dir(TEST_RUNS_DIR if args.test_run else DEV_RUNS_DIR,
                          run_id, model_slug, args.slug or stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Named, not just "raw.json": four tabs called raw.json in an editor are
    # indistinguishable, and a file copied out of its directory loses its
    # identity entirely. The model slug is omitted -- it is already in the
    # directory name, and repeating it makes a 90-character filename whose
    # first 60 characters are the folder.
    out_path = run_dir / f"{run_id}_{args.slug or stem}_raw.json"

    global _partial_path
    _partial_path = out_path.with_suffix(".partial.jsonl")
    # Seed the NEW sidecar with the resumed rows, so a re-interrupted resume
    # still holds everything completed so far in one file. (The source sidecar
    # was fully read into memory above, so this is safe even if the paths were
    # somehow the same.)
    if done_rows:
        with _partial_path.open("w", encoding="utf-8") as fh:
            for row in done_rows.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ⭐ THE FULL SPECS, BEFORE THE FIRST API CALL. This used to be a shallow
    # stub that document_run REPLACED on completion -- which meant the run's
    # provenance was captured from whatever the code looked like at
    # documentation time, and re-documenting later silently restamped an
    # archived run with today's settings. Captured once, here, and preserved.
    purpose = args.purpose or f"{total}-question run on {llm_config.MODEL}."
    if done_rows:
        # Provenance the raw file cannot carry on its own: which rows were
        # produced by THIS process and which were carried from the sidecar.
        purpose += (f" [RESUMED from {Path(args.resume_from).name}: "
                    f"{len(done_rows)} rows carried verbatim, "
                    f"{n_pending} run fresh. API/token telemetry covers the "
                    f"fresh segment only.]")
    if bypassed:
        # Which guards --allow-override actually bypassed — recorded, not just
        # permitted, so the run's meta names the deviation it was launched over.
        purpose += f" [OVERRIDE: bypassed guard(s): {', '.join(bypassed)}.]"
    resume_record = None
    if done_rows:
        resume_record = {
            "source_dir": str(rpath.parent),
            "source_meta_sha256": _sha256(meta_path),
            "source_partial_sha256": _sha256(rpath),
            "carried_ids_count": len(done_rows),
            "carried_ids_sha256": hashlib.sha256(
                "\n".join(sorted(done_rows)).encode("utf-8")).hexdigest(),
            "carried_ids": sorted(done_rows),
            # token_counter starts at zero in this process, so cost/usage in
            # this run's telemetry describe the fresh segment, never the whole
            # logical run — flagged here so retry/answer arithmetic across the
            # combined file is not misread as a rate.
            "api_telemetry": "fresh-segment-only",
        }
    write_start_snapshot(
        run_dir, run_id=run_id, slug=args.slug or stem,
        purpose=purpose,
        questions_file=args.questions_file, n_questions=total,
        workers=args.workers, rpm=rpm, started=timestamp,
        preflight=cache_report,
        article_preflight=article_report,
        resume=resume_record,
    )

    # Build the shared model/client once, before workers contend for them.
    prewarm()
    llm_config.reset_call_stats()

    new_rows = map_questions(
        lambda pair: run_question(pair[0], pair[1], n_pending, golds.get(pair[1].id)),
        list(enumerate(pending, 1)),
        workers=args.workers,
        progress_every=0,   # each question prints its own block
    )

    # Resumed rows and fresh rows, merged back into QUESTION order — the raw
    # file must be indistinguishable in shape from an uninterrupted run's.
    by_id = {**done_rows, **{r["id"]: r for r in new_rows}}
    results = [by_id[q.id] for q in questions]

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {out_path}")
    # The run finished, so the crash-recovery copy is now redundant.
    _partial_path.unlink(missing_ok=True)

    truncated = sum(1 for r in results for v in r.get("finish_reasons", {}).values()
                    if v == "length")

    # Config 4's FIRST call is reported separately. Its truncation is invisible
    # in `finish_reasons` -- that field records the answering call, which
    # overwrites the condensing call's value in llm_config's single thread-local
    # slot. A truncated condensation is worse than a truncated answer: the
    # answering call is handed a context cut off mid-sentence and then graded as
    # if the context were whole.
    cond = [(r.get("contexts", {}).get("rerank") or {}) for r in results]
    cond_trunc = sum(1 for c in cond if isinstance(c, dict)
                     and c.get("condense_finish_reason") == "length")
    if cond_trunc:
        pct = 100 * cond_trunc / max(len(results), 1)
        print(f"\n!! C4 CONDENSE step truncated on {cond_trunc}/{len(results)} "
              f"questions ({pct:.1f}%) at CONDENSE_MAX_TOKENS="
              f"{graph_rag_rerank.CONDENSE_MAX_TOKENS}. Those answering calls "
              f"received an incomplete context. This is NOT counted in the "
              f"finish_reasons figure above.")
    if truncated:
        print(f"\n!! {truncated} answers hit finish_reason='length' "
              f"(MAX_TOKENS={llm_config.MAX_TOKENS}) - these are TRUNCATED, and the "
              f"scorer grades them as wrong rather than unusable. Check "
              f"row['finish_reasons'] before trusting the numbers below.")

    # Gold from the split file, not from the rows just written: one definition,
    # in one module, for this run and for every re-score of it later.
    # `entities_by_id` arms the either-or guard (same shape gold_source builds
    # for every re-score); without it this console summary scored 4 of 200
    # answers leniently on 2026-08-17 and disagreed with report.md by up to
    # 1.5 pts — quotable-looking numbers that were not the record.
    entities_by_id = {
        q.id: {"entity_names": q.entity_mentions, "entity_labels": q.entity_names}
        for q in questions
    }
    scored = score_results(results, entities_by_id, golds=golds)
    print_summary(scored)
    token_counter.print_usage()

    # Documentation is part of the run, not a follow-up chore. The metrics above
    # are PRINTED and would otherwise be lost; more importantly the prompt, the
    # retrieval settings and the code SHA that produced them would be lost too,
    # leaving numbers nobody can interpret in six weeks. Failure here must not
    # cost the answers, which are the expensive, unreproducible part.
    if not args.no_document:
        try:
            from tools.document_run import document
            document(
                raw_path=out_path, run_dir=run_dir, questions_file=args.questions_file,
                purpose=args.purpose or f"{total}-question run on {llm_config.MODEL}.",
                slug=args.slug or stem, run_id=run_id,
                operational={
                    "workers": args.workers,
                    # ⚠️ NOT `sum(len(row["answers"]))`. That counts one per
                    # config per question -- 800 on DEV-200 -- and silently
                    # omits C4's 200 condensing calls, so it undercounted a
                    # documented run by 20%. These come from the choke point
                    # every call goes through; the names are unambiguous
                    # because `calls` was not.
                    **llm_config.call_stats(),
                    "answers_recorded": sum(len(r.get("answers") or {}) for r in results),
                    "truncated": truncated,
                    "started": timestamp,
                },
            )
            print(f"\nDocumented -> {run_dir}\\report.md")
        except Exception as exc:                    # never lose a run over a report
            print(f"\n!! documentation step failed ({type(exc).__name__}: {exc}). "
                  f"Answers are safe in {out_path}. Re-run:\n"
                  f"   venv\\Scripts\\python tools/document_run.py {out_path} "
                  f"--slug {args.slug or stem} --purpose \"...\"")


if __name__ == "__main__":
    main()
