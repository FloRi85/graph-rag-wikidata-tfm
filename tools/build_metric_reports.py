"""
Build the TEST-4,000 metric-report supplement from the sealed run — offline.

WHAT THIS PRODUCES (`data/analysis/test4000_metric_reports/`; see
`docs/reproduction.md`):

    README.md                      populations, axes, units, the faithfulness exception
    manifest.json                  the ONLY volatile file: timestamp, input/code/output hashes
    results.json                   every number, count, delta, interval and cohort hash — unrounded
    explorer.html                  self-contained interactive view of results.json (no network;
                                   every printed number string pre-rendered here in Python)
    full_4000/  attempted/         hallucination_abstention.md · f1_correctness.md · faithfulness.md

DIVISION OF LABOUR, FIXED: Python computes, everything else only displays. The
Markdown reports and the explorer are rendered from the `results` object and
contain no number that is not in it. Regenerating with the same inputs gives
byte-identical results.json, reports and explorer; only manifest.json changes.
Since 2026-09-12 that includes FORMATTING: `display_strings()` pre-renders every
number string the explorer shows with the same formatters the Markdown uses,
embedded beside the untouched results blob, and the page's JavaScript only
selects and colours. An external audit had found the previous JavaScript
formatter disagreeing with `_pct1` in 135 cells by 0.1 point (exact ties such
as 21/400 -> 5.3 instead of 5.2; `Math.round(x*10)/10` drift such as 1238/4000 ->
31.0 instead of 30.9). No JavaScript rounding is left to disagree.

WHAT IT REUSES, DELIBERATELY: `metrics.per_question_scores` (the shared scorer,
all-or-nothing exclusion exactly as `tools/report.py`), `metrics.is_attempted`
(the one attempt contract), `stats.compare_configs` / `mcnemar_exact` /
`unpaired_bootstrap` (seeded, id-sorted), `document_run._pct1` (the divide-first
one-decimal rule) and the stored faithfulness JSON (never re-judged). Nothing
here re-defines an outcome, a threshold or a gold form.

THREE POPULATIONS, NEVER CONFLATED:
  full        all questions; pairs compare the IDENTICAL id set for both configs
  attempted   each config's OWN attempts (CORRECT or HALLUCINATION — Other is
              excluded); pairs use the INTERSECTION of both attempt sets and
              recompute both means there
  scoreable   faithfulness: attempted answers with a stored judge score; C1 has
              no context and is N/A, never zero; unscored attempts are excluded
              from the mean, never zeroed

PRECEDENT CHECKS: where a persisted sanctioned file already analysed the same
population (`report_stats_*.json`, its `_vs-rag` companion,
`faithfulness_comparison_*.json`, `faithfulness_contract_*.json`) the same
numbers must reproduce. A mismatch is fatal — "do not proceed through
unexplained discrepancies".

Usage (from the repo root; ~5 minutes at 10,000 resamples):
    venv/Scripts/python tools/build_metric_reports.py            # regenerate
    venv/Scripts/python tools/build_metric_reports.py --check    # regenerate in memory and compare; write nothing
    venv/Scripts/python tools/build_metric_reports.py --dump-ids PATH   # also write cohort id lists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import (                               # noqa: E402
    ABSTENTION, CONFIGS, CORRECT, HALLUCINATION, OTHER, OUTCOMES, WRONG_THRESHOLD,
    excluded_question_ids, is_attempted, per_question_scores,
)
from src.eval.gold_source import resolve_gold_source          # noqa: E402
from src.eval.parse_answers import ALL_ANSWER_TYPES           # noqa: E402
from src.eval.parse_questions import CATEGORIES, COMPLEXITY_TYPES, raw_type   # noqa: E402
from src.eval.stats import (                                  # noqa: E402
    DEFAULT_RESAMPLES, DEFAULT_SEED, abstention_key, compare_configs, correct_key,
    f1_key, hallucination_key, mcnemar_exact, score_key, unpaired_bootstrap,
)
from tools.document_run import _pct1                          # noqa: E402

# ---------------------------------------------------------------------------
# Constants — the reporting contract
# ---------------------------------------------------------------------------

RUN_DIR = ROOT / "data/results/test_runs/20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw"
DEFAULT_RAW = RUN_DIR / "20260819_2017_mintaka_test_raw_raw.json"
DEFAULT_GOLD = ROOT / "data/questions/mintaka_test_raw.json"
ANALYSIS_DIR = ROOT / "data/analysis"
DEFAULT_FAITH = ANALYSIS_DIR / "faithfulness_20260819_2017_mintaka_test_raw_raw.json"
DEFAULT_OUT = ANALYSIS_DIR / "test4000_metric_reports"
PRECEDENT_FILES = {
    "report_stats": ANALYSIS_DIR / "report_stats_20260819_2017_mintaka_test_raw_raw.json",
    "report_stats_vs_rag": ANALYSIS_DIR / "report_stats_20260819_2017_mintaka_test_raw_raw_vs-rag.json",
    "faithfulness_comparison": ANALYSIS_DIR / "faithfulness_comparison_20260819_2017_mintaka_test_raw_raw.json",
    "faithfulness_contract": ANALYSIS_DIR / "faithfulness_contract_20260819_2017_mintaka_test_raw_raw.json",
}
# Code whose behaviour the numbers depend on; hashed into the manifest.
CODE_FILES = (
    "tools/build_metric_reports.py", "src/eval/metrics.py", "src/eval/stats.py",
    "src/eval/gold_source.py", "src/eval/parse_answers.py", "src/eval/parse_questions.py",
    "tools/document_run.py",
)

CONFIG_INFO = (
    ("base_llm_abstain", "C1", "Base LLM"),
    ("rag", "C2", "RAG"),
    ("graph_rag", "C3", "Graph-RAG"),
    ("rerank", "C4", "Graph-RAG + condensing"),
)
assert tuple(c for c, _, _ in CONFIG_INFO) == tuple(CONFIGS)
CODE = {c: code for c, code, _ in CONFIG_INFO}
NAME = {c: name for c, _, name in CONFIG_INFO}
RETRIEVAL = ("rag", "graph_rag", "rerank")
# A minus B, fixed orientation and order.
PAIRS = (
    ("rag", "base_llm_abstain"), ("graph_rag", "base_llm_abstain"), ("rerank", "base_llm_abstain"),
    ("graph_rag", "rag"), ("rerank", "rag"), ("rerank", "graph_rag"),
)
RETRIEVAL_PAIRS = tuple(p for p in PAIRS if p[1] != "base_llm_abstain")


def pair_key(a: str, b: str) -> str:
    return f"{CODE[a]}-{CODE[b]}"


def pair_label(a: str, b: str) -> str:
    return f"{CODE[a]} − {CODE[b]}"


# The three native dataset attributes, groups in parser-vocabulary order.
AXES = (
    ("answer_type", "Answer type", tuple(ALL_ANSWER_TYPES)),
    ("complexity", "Question complexity", tuple(raw_type(x) for x in COMPLEXITY_TYPES)),
    ("category", "Topic category", tuple(CATEGORIES)),
)
TINY_N = 50                 # cells below this carry a † in every table
GROUNDED_THRESHOLD = 0.5    # = tools/grounding_outcome_matrix.GROUNDED_THRESHOLD (pinned by test)
THRESHOLDS = (0.3, 0.5, 0.7)
SCHEMA = "test4000_metric_reports/v1"


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_ids(ids) -> str:
    """Membership hash of a cohort: sha256 of the sorted ids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def load_inputs(raw: Path, gold: Path, faith: Path) -> dict:
    """Score once, join metadata and the stored judge scores; refuse anything inconsistent."""
    rows = json.loads(raw.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        sys.exit(f"{raw.name}: not a run_eval raw file (expected a list of rows)")
    ids = [r.get("id", "") for r in rows]
    if len(set(ids)) != len(ids) or "" in ids:
        sys.exit(f"{raw.name}: question ids are not unique/non-empty")
    excluded = excluded_question_ids(rows)
    if excluded:
        # A run with infrastructure failures cannot satisfy the identical-id-set
        # contract of the full folder; this tool is for the sealed, clean run.
        sys.exit(f"{raw.name}: {len(excluded)} question(s) carry infrastructure failures — "
                 f"the full-benchmark contract needs a clean run")
    golds, qbi, gold_path = resolve_gold_source(set(ids), explicit=str(gold), verbose=False)
    scores = per_question_scores(rows, qbi, golds=golds)
    if list(scores) != list(CONFIGS):
        sys.exit(f"configs present {list(scores)} != expected {list(CONFIGS)}")
    # Other-threshold scoring for the sensitivity supplement (same rows, same gold).
    scores_at = {t: (scores if t == WRONG_THRESHOLD
                     else per_question_scores(rows, qbi, threshold=t, golds=golds))
                 for t in THRESHOLDS}

    questions = json.loads(gold.read_text(encoding="utf-8"))
    qraw = {q["id"]: q for q in questions}
    missing = [q for q in ids if q not in qraw]
    if missing:
        sys.exit(f"{gold.name} lacks {len(missing)} scored question(s), e.g. {missing[:3]}")
    meta = {q: {"answer_type": qraw[q]["answer"]["answerType"],
                "complexity": qraw[q]["complexityType"],
                "category": qraw[q]["category"]} for q in ids}
    for axis, _, groups in AXES:
        unknown = sorted({m[axis] for m in meta.values()} - set(groups))
        if unknown:
            sys.exit(f"{gold.name}: {axis} value(s) outside the parser vocabulary: {unknown}")
    # The scorer records and the rows carry answer_type/complexity too — all three must agree.
    for c in CONFIGS:
        for q in ids:
            rec = scores[c][q]
            if rec["answer_type"] != meta[q]["answer_type"] or rec["complexity"] != meta[q]["complexity"]:
                sys.exit(f"metadata disagreement on {q}/{c}: scorer {rec['answer_type']}/{rec['complexity']} "
                         f"vs question file {meta[q]}")

    records = json.loads(faith.read_text(encoding="utf-8"))
    seen: set[tuple[str, str]] = set()
    for r in records:
        key = (r.get("config", ""), r.get("id", ""))
        if key in seen:
            sys.exit(f"duplicate (config, id) faithfulness record: {key}")
        seen.add(key)
    fmaps: dict[str, dict[str, dict]] = {c: {} for c in RETRIEVAL}
    for r in records:
        if r["config"] not in fmaps:
            sys.exit(f"faithfulness record for a non-retrieval config: {r['config']!r} "
                     f"(C1 has no context; faithfulness is undefined there)")
        fmaps[r["config"]][r["id"]] = r
    for c in RETRIEVAL:
        attempted = {q for q in ids if is_attempted(scores[c][q])}
        have = set(fmaps[c])
        if have != attempted:
            sys.exit(f"{c}: faithfulness records cover {len(have)} ids but the scorer attempted "
                     f"{len(attempted)} (only-in-judge {len(have - attempted)}, only-in-scorer "
                     f"{len(attempted - have)})")
        bad = [q for q in attempted if fmaps[c][q].get("outcome") != scores[c][q]["outcome"]]
        if bad:
            sys.exit(f"{c}: {len(bad)} faithfulness record(s) carry an outcome the scorer disagrees "
                     f"with, e.g. {bad[:3]}")
    return {
        "ids": sorted(ids), "scores": scores, "scores_at": scores_at, "meta": meta,
        "faith": fmaps, "paths": {"raw": raw, "gold": gold, "faithfulness": faith},
        "hashes": {"raw": sha256_file(raw), "gold": sha256_file(gold), "faithfulness": sha256_file(faith)},
        "gold_path": Path(gold_path) if gold_path else gold,
    }


# ---------------------------------------------------------------------------
# Cohorts and aggregation
# ---------------------------------------------------------------------------

def build_cohorts(inp: dict) -> dict[str, list[str]]:
    ids, scores, faith = inp["ids"], inp["scores"], inp["faith"]
    cohorts: dict[str, list[str]] = {"full": list(ids)}
    att = {c: [q for q in ids if is_attempted(scores[c][q])] for c in CONFIGS}
    for c in CONFIGS:
        cohorts[f"own_attempted:{CODE[c]}"] = att[c]
    for a, b in PAIRS:
        cohorts[f"common_attempted:{pair_key(a, b)}"] = sorted(set(att[a]) & set(att[b]))
    cohorts["common_attempted:all_four"] = sorted(set.intersection(*(set(att[c]) for c in CONFIGS)))
    sc = {c: [q for q in att[c] if faith[c][q]["score"] is not None] for c in RETRIEVAL}
    for c in RETRIEVAL:
        cohorts[f"own_scoreable:{CODE[c]}"] = sc[c]
    for a, b in RETRIEVAL_PAIRS:
        cohorts[f"common_scoreable:{pair_key(a, b)}"] = sorted(set(sc[a]) & set(sc[b]))
    cohorts["common_scoreable:all_three"] = sorted(set.intersection(*(set(sc[c]) for c in RETRIEVAL)))
    return cohorts


def agg_outcomes(smap: dict[str, dict], ids: list[str]) -> dict:
    """Outcome counts, rates, F1 and EM of one config over `ids` (unrounded)."""
    n = len(ids)
    counts = Counter(smap[q]["outcome"] for q in ids)
    att = [q for q in ids if is_attempted(smap[q])]
    f1_all = sum(smap[q]["f1"] for q in ids)
    f1_att = sum(smap[q]["f1"] for q in att)
    return {
        "n": n,
        "counts": {o: counts.get(o, 0) for o in OUTCOMES},
        "rates": {o: (counts.get(o, 0) / n if n else None) for o in OUTCOMES},
        "f1_mean": f1_all / n if n else None,
        "em_mean": (sum(smap[q]["em"] for q in ids) / n) if n else None,
        "n_attempted": len(att),
        "coverage": len(att) / n if n else None,
        "f1_attempted": f1_att / len(att) if att else None,
    }


def agg_faith(fmap: dict[str, dict], smap: dict[str, dict], ids: list[str]) -> dict:
    """Faithfulness of one retrieval config over `ids`: conditional on a scoreable attempt."""
    att = [q for q in ids if is_attempted(smap[q])]
    sc = [q for q in att if fmap[q]["score"] is not None]
    reasons = Counter((fmap[q].get("status") or "unknown") for q in att if fmap[q]["score"] is None)
    return {
        "n_ids": len(ids),
        "n_attempted": len(att),
        "n_scoreable": len(sc),
        "n_unscored_attempts": len(att) - len(sc),
        "unscored_reasons": {k: reasons[k] for k in sorted(reasons)},
        "mean": (sum(fmap[q]["score"] for q in sc) / len(sc)) if sc else None,
        "share_of_ids": (len(sc) / len(ids)) if ids else None,
        "share_of_attempts": (len(sc) / len(att)) if att else None,
    }


def _bootstrap(a: dict, b: dict, ids: list[str], key, resamples: int) -> dict:
    sub_a = {q: a[q] for q in ids}
    sub_b = {q: b[q] for q in ids}
    return compare_configs(sub_a, sub_b, key, n_resamples=resamples)


def _mcnemar(a: dict, b: dict, ids: list[str], key) -> dict:
    return mcnemar_exact({q: a[q] for q in ids}, {q: b[q] for q in ids}, key)


def pair_full(sa: dict, sb: dict, ids: list[str], resamples: int, tests: bool) -> dict:
    """Pair on the identical id set (full folder). `tests` adds bootstrap + McNemar."""
    out = {"n": len(ids), "a": agg_outcomes(sa, ids), "b": agg_outcomes(sb, ids)}
    if not ids:
        out["available"] = False
        return out
    out["available"] = True
    specs = (("hallucination", hallucination_key, HALLUCINATION), ("correct", correct_key, CORRECT),
             ("abstention", abstention_key, ABSTENTION))
    for name, key, outcome in specs:
        entry = {"delta": out["a"]["rates"][outcome] - out["b"]["rates"][outcome]}
        if tests:
            entry["bootstrap"] = _bootstrap(sa, sb, ids, key, resamples)
            entry["mcnemar"] = _mcnemar(sa, sb, ids, key)
        out[name] = entry
    entry = {"delta": out["a"]["f1_mean"] - out["b"]["f1_mean"]}
    if tests:
        entry["bootstrap"] = _bootstrap(sa, sb, ids, f1_key, resamples)
    out["f1"] = entry
    return out


def pair_attempted(sa: dict, sb: dict, ids: list[str], denominator_n: int,
                   resamples: int, tests: bool) -> dict:
    """Pair on the common attempted set; both means recomputed there."""
    out = {"n_common": len(ids), "denominator_n": denominator_n,
           "retained_share": (len(ids) / denominator_n) if denominator_n else None,
           "a": agg_outcomes(sa, ids), "b": agg_outcomes(sb, ids)}
    if not ids:
        out["available"] = False
        return out
    out["available"] = True
    ha, hb = out["a"]["rates"][HALLUCINATION], out["b"]["rates"][HALLUCINATION]
    both = sum(1 for q in ids if sa[q]["outcome"] == HALLUCINATION and sb[q]["outcome"] == HALLUCINATION)
    a_only = sum(1 for q in ids if sa[q]["outcome"] == HALLUCINATION and sb[q]["outcome"] != HALLUCINATION)
    b_only = sum(1 for q in ids if sa[q]["outcome"] != HALLUCINATION and sb[q]["outcome"] == HALLUCINATION)
    h = {"delta": ha - hb,
         "table": {"both_hallucinate": both, "a_only": a_only, "b_only": b_only,
                   "neither": len(ids) - both - a_only - b_only}}
    if tests:
        h["bootstrap"] = _bootstrap(sa, sb, ids, hallucination_key, resamples)
        h["mcnemar"] = _mcnemar(sa, sb, ids, hallucination_key)
    out["hallucination"] = h
    # Within attempts CORRECT is the complement of HALLUCINATION: same test, opposite sign.
    out["correct"] = {"delta": out["a"]["rates"][CORRECT] - out["b"]["rates"][CORRECT],
                      "complement_of": "hallucination"}
    f = {"delta": out["a"]["f1_mean"] - out["b"]["f1_mean"]}
    if tests:
        f["bootstrap"] = _bootstrap(sa, sb, ids, f1_key, resamples)
    out["f1"] = f
    return out


def pair_faith(fa: dict, fb: dict, ids: list[str], resamples: int, tests: bool) -> dict:
    out = {"n_common_scoreable": len(ids)}
    if not ids:
        out["available"] = False
        return out
    out["available"] = True
    out["mean_a"] = sum(fa[q]["score"] for q in ids) / len(ids)
    out["mean_b"] = sum(fb[q]["score"] for q in ids) / len(ids)
    out["delta"] = out["mean_a"] - out["mean_b"]
    if tests:
        out["bootstrap"] = _bootstrap(fa, fb, ids, score_key, resamples)
    return out


def _groups(inp: dict, ids: list[str], axis: str, groups) -> dict[str, list[str]]:
    meta = inp["meta"]
    return {g: [q for q in ids if meta[q][axis] == g] for g in groups}


# ---------------------------------------------------------------------------
# The results object
# ---------------------------------------------------------------------------

def compute(inp: dict, resamples: int = DEFAULT_RESAMPLES) -> dict:
    ids, scores, faith, meta = inp["ids"], inp["scores"], inp["faith"], inp["meta"]
    cohorts = build_cohorts(inp)
    n_full = len(ids)
    att = {c: cohorts[f"own_attempted:{CODE[c]}"] for c in CONFIGS}
    sc = {c: cohorts[f"own_scoreable:{CODE[c]}"] for c in RETRIEVAL}

    res: dict = {
        "schema": SCHEMA,
        "inputs": {k: {"path": str(inp["paths"][k].resolve().relative_to(ROOT)).replace("\\", "/")
                       if inp["paths"][k].resolve().is_relative_to(ROOT) else str(inp["paths"][k]),
                       "sha256": inp["hashes"][k]} for k in ("raw", "gold", "faithfulness")},
        "settings": {
            "threshold": WRONG_THRESHOLD, "seed": DEFAULT_SEED, "n_resamples": resamples,
            "tiny_n": TINY_N, "grounded_threshold": GROUNDED_THRESHOLD,
            "sensitivity_thresholds": list(THRESHOLDS),
            "attempted_definition": "metrics.is_attempted: outcome in {correct, hallucination}; Other excluded",
            "scoreable_definition": "attempted AND stored judge `score` is not null; partial_score never used",
            "pct_rule": "count-based percentages: tools/document_run._pct1 = round(count / n * 100, 1) (divide first); "
                        "F1 printed as mean*100 with one decimal; faithfulness means with three decimals",
            "p_value_kind": "two-sided percentile-bootstrap tail estimate, resolution 2/n_resamples, "
                            "printed as <.001 below resolution; mcnemar = exact binomial on discordant pairs; "
                            "CIs nominal pointwise 95%, unadjusted",
            "delta_orientation": "A minus B in the pair key A-B",
        },
        "configs": [{"key": c, "code": CODE[c], "label": NAME[c]} for c in CONFIGS],
        "pairs": [{"key": pair_key(a, b), "a": a, "b": b, "label": pair_label(a, b)} for a, b in PAIRS],
        "axes": [{"key": k, "label": lbl, "groups": list(groups)} for k, lbl, groups in AXES],
        "outcomes": list(OUTCOMES),
    }

    # -- dataset composition -------------------------------------------------
    res["axis_counts"] = {axis: {g: len(v) for g, v in _groups(inp, ids, axis, groups).items()}
                          for axis, _, groups in AXES}
    comp = {}
    for cx in AXES[1][2]:
        comp[cx] = {at: sum(1 for q in ids if meta[q]["complexity"] == cx and meta[q]["answer_type"] == at)
                    for at in AXES[0][2]}
    res["composition_complexity_x_answer_type"] = comp

    res["cohorts"] = {name: {"n": len(v), "sha256": sha256_ids(v)} for name, v in cohorts.items()}

    # -- full benchmark -------------------------------------------------------
    full: dict = {"overall": {c: agg_outcomes(scores[c], ids) for c in CONFIGS}, "by_axis": {}, "pairs": {}}
    for axis, _, groups in AXES:
        full["by_axis"][axis] = {}
        for g, gids in _groups(inp, ids, axis, groups).items():
            full["by_axis"][axis][g] = {"n": len(gids), "configs": {c: agg_outcomes(scores[c], gids) for c in CONFIGS}}
    for a, b in PAIRS:
        pk = pair_key(a, b)
        entry = {"overall": pair_full(scores[a], scores[b], ids, resamples, tests=True), "by_axis": {}}
        for axis, _, groups in AXES:
            entry["by_axis"][axis] = {g: pair_full(scores[a], scores[b], gids, resamples, tests=False)
                                      for g, gids in _groups(inp, ids, axis, groups).items()}
        full["pairs"][pk] = entry
    # Outcome transitions: rows = B's outcome (the reference config), columns = A's outcome.
    full["transitions"] = {}
    for a, b in PAIRS:
        matrix = {ob: {oa: 0 for oa in OUTCOMES} for ob in OUTCOMES}
        for q in ids:
            matrix[scores[b][q]["outcome"]][scores[a][q]["outcome"]] += 1
        full["transitions"][pair_key(a, b)] = {"rows": b, "columns": a, "matrix": matrix}
    patterns = Counter("+".join(CODE[c] for c in CONFIGS if is_attempted(scores[c][q])) or "none" for q in ids)
    full["attempt_patterns"] = {k: patterns[k] for k in sorted(patterns, key=lambda k: (k == "none", k))}
    full["thresholds"] = {}
    for t in THRESHOLDS:
        smaps = inp["scores_at"][t]
        full["thresholds"][f"{t:.1f}"] = {c: agg_outcomes(smaps[c], ids) for c in CONFIGS}
    res["full"] = full

    # -- attempted ------------------------------------------------------------
    attempted: dict = {"overall": {}, "by_axis": {}, "pairs": {}, "all_four": {}}
    for c in CONFIGS:
        agg = agg_outcomes(scores[c], att[c])
        excl = full["overall"][c]["counts"]
        agg.update({"benchmark_n": n_full, "retained_share": len(att[c]) / n_full,
                    "excluded": {"abstention": excl[ABSTENTION], "other": excl[OTHER]}})
        attempted["overall"][c] = agg
    for axis, _, groups in AXES:
        attempted["by_axis"][axis] = {}
        for g, gids in _groups(inp, ids, axis, groups).items():
            cell = {"n": len(gids), "configs": {}}
            for c in CONFIGS:
                own = [q for q in gids if is_attempted(scores[c][q])]
                agg = agg_outcomes(scores[c], own)
                excl = full["by_axis"][axis][g]["configs"][c]["counts"]
                agg.update({"benchmark_n": len(gids), "retained_share": (len(own) / len(gids)) if gids else None,
                            "excluded": {"abstention": excl[ABSTENTION], "other": excl[OTHER]}})
                cell["configs"][c] = agg
            attempted["by_axis"][axis][g] = cell
    for a, b in PAIRS:
        pk = pair_key(a, b)
        common = cohorts[f"common_attempted:{pk}"]
        entry = {"overall": pair_attempted(scores[a], scores[b], common, n_full, resamples, tests=True),
                 "by_axis": {}}
        for axis, _, groups in AXES:
            entry["by_axis"][axis] = {}
            for g, gids in _groups(inp, ids, axis, groups).items():
                cg = [q for q in gids if is_attempted(scores[a][q]) and is_attempted(scores[b][q])]
                entry["by_axis"][axis][g] = pair_attempted(scores[a], scores[b], cg, len(gids), resamples, tests=False)
        attempted["pairs"][pk] = entry
    four = cohorts["common_attempted:all_four"]
    attempted["all_four"] = {"overall": {"n": len(four), "retained_share": len(four) / n_full,
                                         "configs": {c: agg_outcomes(scores[c], four) for c in CONFIGS}},
                             "by_axis": {}}
    for axis, _, groups in AXES:
        attempted["all_four"]["by_axis"][axis] = {}
        for g, gids in _groups(inp, ids, axis, groups).items():
            fg = [q for q in gids if q in set(four)]
            attempted["all_four"]["by_axis"][axis][g] = {
                "n": len(fg), "benchmark_n": len(gids),
                "retained_share": (len(fg) / len(gids)) if gids else None,
                "configs": {c: agg_outcomes(scores[c], fg) for c in CONFIGS}}
    res["attempted"] = attempted

    # -- faithfulness ---------------------------------------------------------
    fth: dict = {"own": {}, "by_axis": {}, "pairs": {}, "by_outcome": {}, "all_three": {},
                 "coverage_accounting": {}, "not_applicable": {}}
    fth["not_applicable"]["base_llm_abstain"] = ("C1 receives no retrieved context; answer-to-context "
                                                 "faithfulness is undefined, not zero")
    for c in RETRIEVAL:
        fth["own"][c] = agg_faith(faith[c], scores[c], ids)
    for axis, _, groups in AXES:
        fth["by_axis"][axis] = {g: {"n": len(gids), "configs": {c: agg_faith(faith[c], scores[c], gids)
                                                                for c in RETRIEVAL}}
                                for g, gids in _groups(inp, ids, axis, groups).items()}
    for a, b in PAIRS:
        pk = pair_key(a, b)
        if b == "base_llm_abstain" or a == "base_llm_abstain":
            fth["pairs"][pk] = {"available": False,
                                "reason": "involves C1, which has no context: faithfulness N/A"}
            continue
        common = cohorts[f"common_scoreable:{pk}"]
        entry = {"overall": pair_faith(faith[a], faith[b], common, resamples, tests=True), "by_axis": {}}
        cset = set(common)
        for axis, _, groups in AXES:
            entry["by_axis"][axis] = {g: pair_faith(faith[a], faith[b], [q for q in gids if q in cset],
                                                    resamples, tests=False)
                                      for g, gids in _groups(inp, ids, axis, groups).items()}
        fth["pairs"][pk] = entry
    for c in RETRIEVAL:
        corr = [faith[c][q]["score"] for q in sc[c] if scores[c][q]["outcome"] == CORRECT]
        wrong = [faith[c][q]["score"] for q in sc[c] if scores[c][q]["outcome"] == HALLUCINATION]
        grounded = {}
        for oc, vals in (("correct", corr), ("hallucination", wrong)):
            g = sum(1 for v in vals if v >= GROUNDED_THRESHOLD)
            grounded[oc] = {"grounded": g, "ungrounded": len(vals) - g}
        fth["by_outcome"][c] = {
            "correct": {"n": len(corr), "mean": (sum(corr) / len(corr)) if corr else None},
            "hallucination": {"n": len(wrong), "mean": (sum(wrong) / len(wrong)) if wrong else None},
            "unpaired_bootstrap_correct_minus_hallucination": (
                unpaired_bootstrap(corr, wrong, n_resamples=resamples) if corr and wrong else None),
            "grounded": grounded,
        }
    three = cohorts["common_scoreable:all_three"]
    fth["all_three"] = {"overall": {"n": len(three),
                                    "configs": {c: {"n": len(three),
                                                    "mean": (sum(faith[c][q]["score"] for q in three) / len(three))
                                                    if three else None} for c in RETRIEVAL}},
                        "by_axis": {}}
    tset = set(three)
    for axis, _, groups in AXES:
        fth["all_three"]["by_axis"][axis] = {}
        for g, gids in _groups(inp, ids, axis, groups).items():
            tg = [q for q in gids if q in tset]
            fth["all_three"]["by_axis"][axis][g] = {
                "n": len(tg), "configs": {c: {"n": len(tg), "mean": (sum(faith[c][q]["score"] for q in tg) / len(tg))
                                              if tg else None} for c in RETRIEVAL}}
    for c in RETRIEVAL:
        o = full["overall"][c]["counts"]
        f = fth["own"][c]
        acc = {"abstention": o[ABSTENTION], "other": o[OTHER], "unscored_attempt": f["n_unscored_attempts"],
               "scoreable": f["n_scoreable"]}
        acc["total"] = sum(acc.values())
        acc["partition_ok"] = acc["total"] == n_full
        fth["coverage_accounting"][c] = acc
    res["faithfulness"] = fth

    res["precedent_checks"] = precedent_checks(res, inp, resamples)
    return res


# ---------------------------------------------------------------------------
# Precedent checks — the same population must give the same number
# ---------------------------------------------------------------------------

def _close(x, y, tol=1e-9) -> bool:
    if x is None or y is None:
        return x is y
    return abs(float(x) - float(y)) <= tol


def _boot_equal(mine: dict, theirs: dict) -> bool:
    return all(_close(mine.get(k), theirs.get(k)) for k in ("n", "delta", "ci_low", "ci_high", "p_value"))


def precedent_checks(res: dict, inp: dict, resamples: int) -> list[dict]:
    """Compare against the persisted sanctioned files where their input is OUR input."""
    checks: list[dict] = []

    def add(source, name, expected, observed, ok):
        checks.append({"source": source, "check": name, "expected": expected, "observed": observed, "match": bool(ok)})

    boot_ok = resamples == DEFAULT_RESAMPLES

    for tag in ("report_stats", "report_stats_vs_rag"):
        path = PRECEDENT_FILES[tag]
        if not path.exists():
            checks.append({"source": tag, "check": "file present", "match": None, "note": "file absent — skipped"})
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("results_sha256") != inp["hashes"]["raw"]:
            checks.append({"source": tag, "check": "same input", "match": None,
                           "note": "persisted file analysed a different raw file — not applicable"})
            continue
        base = d["baseline"]
        for row in d["sections"]["hallucination_vs_baseline"]:
            c = row["config"]
            a, b = (c, base) if (c, base) in PAIRS else (base, c)
            mine = res["full"]["pairs"][pair_key(a, b)]["overall"]["hallucination"]
            sign = 1.0 if (c, base) in PAIRS else -1.0
            exp_delta = sign * row["bootstrap"]["delta"]
            add(tag, f"full hallucination delta {pair_key(a, b)}", exp_delta, mine["delta"], _close(exp_delta, mine["delta"]))
            if boot_ok and sign == 1.0:
                add(tag, f"full hallucination bootstrap {pair_key(a, b)}", row["bootstrap"], mine["bootstrap"],
                    _boot_equal(mine["bootstrap"], row["bootstrap"]))
            add(tag, f"full hallucination mcnemar p {pair_key(a, b)}", row["mcnemar"]["p_value"],
                mine["mcnemar"]["p_value"], _close(row["mcnemar"]["p_value"], mine["mcnemar"]["p_value"]))
        for row in d["sections"]["within_attempted_subset"]:
            c = row["config"]
            own = res["attempted"]["overall"][c]
            add(tag, f"own attempted n {CODE[c]}", row["n_attempted_shared"], own["n"], row["n_attempted_shared"] == own["n"])
            add(tag, f"own F1@attempted {CODE[c]}", row["config_f1_attempted"], own["f1_mean"],
                _close(row["config_f1_attempted"], own["f1_mean"]))
        for row in d["sections"]["jointly_attempted_pairs"]:
            a, b = row["pair"]
            mine = res["attempted"]["pairs"][pair_key(a, b)]["overall"]
            add(tag, f"common attempted n {pair_key(a, b)}", row["n_joint"], mine["n_common"], row["n_joint"] == mine["n_common"])
            add(tag, f"common attempted F1 A {pair_key(a, b)}", row["f1_a_joint"], mine["a"]["f1_mean"],
                _close(row["f1_a_joint"], mine["a"]["f1_mean"]))
            add(tag, f"common attempted F1 B {pair_key(a, b)}", row["f1_b_joint"], mine["b"]["f1_mean"],
                _close(row["f1_b_joint"], mine["b"]["f1_mean"]))
            if boot_ok:
                add(tag, f"common attempted F1 bootstrap {pair_key(a, b)}", row["bootstrap_f1"], mine["f1"]["bootstrap"],
                    _boot_equal(mine["f1"]["bootstrap"], row["bootstrap_f1"]))
            add(tag, f"common attempted hallucination mcnemar {pair_key(a, b)}", row["mcnemar_hallucination"],
                mine["hallucination"]["mcnemar"],
                _close(row["mcnemar_hallucination"]["p_value"], mine["hallucination"]["mcnemar"]["p_value"])
                and _close(row["mcnemar_hallucination"]["delta"], mine["hallucination"]["mcnemar"]["delta"]))

    tag = "faithfulness_comparison"
    path = PRECEDENT_FILES[tag]
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("input_sha256") != inp["hashes"]["faithfulness"]:
            checks.append({"source": tag, "check": "same input", "match": None,
                           "note": "persisted file analysed a different faithfulness file — not applicable"})
        else:
            for c, row in d["operating_point_means"].items():
                own = res["faithfulness"]["own"][c]
                add(tag, f"own scoreable n {CODE[c]}", row["n"], own["n_scoreable"], row["n"] == own["n_scoreable"])
                add(tag, f"own faithfulness mean {CODE[c]}", row["mean"], own["mean"], _close(row["mean"], own["mean"]))
            for row in d["paired_common_scoreable"]:
                a, b = row["pair"]
                mine = res["faithfulness"]["pairs"][pair_key(a, b)]["overall"]
                add(tag, f"common scoreable n {pair_key(a, b)}", row["bootstrap"]["n"], mine["n_common_scoreable"],
                    row["bootstrap"]["n"] == mine["n_common_scoreable"])
                add(tag, f"common scoreable delta {pair_key(a, b)}", row["bootstrap"]["delta"], mine["delta"],
                    _close(row["bootstrap"]["delta"], mine["delta"]))
                if boot_ok:
                    add(tag, f"common scoreable bootstrap {pair_key(a, b)}", row["bootstrap"], mine["bootstrap"],
                        _boot_equal(mine["bootstrap"], row["bootstrap"]))
    else:
        checks.append({"source": tag, "check": "file present", "match": None, "note": "file absent — skipped"})

    tag = "faithfulness_contract"
    path = PRECEDENT_FILES[tag]
    res["faithfulness"]["contract_flags"] = None
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("input_sha256") == inp["hashes"]["faithfulness"]:
            flagged_sc = Counter(f["config"] for f in d["flagged"] if f.get("score") is not None)
            res["faithfulness"]["contract_flags"] = {
                "source": path.name, "n_flagged": d["n_flagged"], "n_flagged_scoreable": d["n_flagged_scoreable"],
                "per_config_flagged_scoreable": {c: flagged_sc.get(c, 0) for c in RETRIEVAL},
                "policy": "flagged-but-scoreable records are RETAINED (existing availability policy); "
                          "the contract file's exclusion means are a sensitivity check, not the reported number",
                "mean_excl_flagged": {c: d["per_config"][c]["mean_excl_flagged"] for c in RETRIEVAL},
            }
            for c in RETRIEVAL:
                add(tag, f"scoreable n {CODE[c]}", d["per_config"][c]["n_scoreable"],
                    res["faithfulness"]["own"][c]["n_scoreable"],
                    d["per_config"][c]["n_scoreable"] == res["faithfulness"]["own"][c]["n_scoreable"])
        else:
            checks.append({"source": tag, "check": "same input", "match": None,
                           "note": "contract file analysed a different faithfulness file — not applicable"})
    return checks


# ---------------------------------------------------------------------------
# Formatting — the only place numbers become strings
# ---------------------------------------------------------------------------

def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    return "<.001" if p < 0.001 else f"{p:.3f}"


def pct(count: int, n: int) -> str:
    return "—" if not n else f"{_pct1(count, n):.1f}"


def rate_pct(rate: float | None) -> str:
    """A rate that is NOT a count/n (a delta): printed as *100, one decimal, like report.py."""
    return "—" if rate is None else f"{rate * 100:.1f}"


def f1s(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def fs(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def pp(d: float | None) -> str:
    return "—" if d is None else f"{d * 100:+.1f}"


def dfs(d: float | None) -> str:
    return "—" if d is None else f"{d:+.3f}"


def ci_pp(b: dict | None) -> str:
    return "—" if not b else f"[{b['ci_low'] * 100:+.1f}, {b['ci_high'] * 100:+.1f}]"


def ci_f(b: dict | None) -> str:
    return "—" if not b else f"[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]"


def ncell(n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{n}†" if n < TINY_N else str(n)


def share(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def md_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---" if a == "l" else "---:" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def cp(agg: dict, outcome: str) -> str:
    """Count-based percentage of one outcome in an aggregate, through the shared rule."""
    return pct(agg["counts"][outcome], agg["n"])


# ---------------------------------------------------------------------------
# Display strings for the explorer — rendered HERE, by the Markdown formatters
# ---------------------------------------------------------------------------
# The page receives these beside the untouched results blob and never rounds a
# number itself. Each subtree mirrors the shape of the `results` subtree the
# page reads for the value, so the JavaScript selects the string by the path it
# already uses. A cell the page shows as "—" is None here.

def _agg_display(agg: dict | None, outcomes) -> dict | None:
    if not agg or not agg.get("n"):
        return None
    d = {m: cp(agg, m) for m in outcomes}
    d["f1"] = f1s(agg["f1_mean"])
    return d


def _pair_stat_display(ent: dict | None) -> dict | None:
    if not ent:
        return None
    bs, mc = ent.get("bootstrap"), ent.get("mcnemar")
    return {"delta": pp(ent["delta"]), "ci": ci_pp(bs),
            "p": fmt_p(bs["p_value"]) if bs else "—",
            "mcnemar_p": fmt_p(mc["p_value"]) if mc else "—"}


def _outcome_pair_display(o: dict, full: bool, outcomes) -> dict | None:
    if not o.get("available", True):
        return None
    d = {"a": _agg_display(o.get("a"), outcomes), "b": _agg_display(o.get("b"), outcomes),
         "stats": {k: _pair_stat_display(o[k]) for k in ("hallucination", "correct", "abstention", "f1") if o.get(k)}}
    if not full:
        d["retained_share"] = share(o.get("retained_share"))
    return d


def _faith_pair_display(o: dict) -> dict | None:
    if not o.get("available", True):
        return None
    bs = o.get("bootstrap")
    return {"mean_a": fs(o["mean_a"]), "mean_b": fs(o["mean_b"]), "delta": dfs(o["delta"]),
            "ci": ci_f(bs), "p": fmt_p(bs["p_value"]) if bs else "—"}


def _faith_mean_display(v: dict | None) -> dict | None:
    if not v:
        return None
    ns = v.get("n_scoreable", v.get("n"))
    return {"mean": fs(v["mean"])} if ns else None


def _by_outcome_display(o: dict) -> dict:
    ub = o.get("unpaired_bootstrap_correct_minus_hallucination")
    gc, gh = o["grounded"]["correct"], o["grounded"]["hallucination"]
    return {"correct_mean": fs(o["correct"]["mean"]), "hallucination_mean": fs(o["hallucination"]["mean"]),
            "delta": dfs(ub["delta"]) if ub else "—", "ci": ci_f(ub) if ub else "—",
            "p": fmt_p(ub["p_value"]) if ub else "—",
            "ungrounded_correct_pct": pct(gc["ungrounded"], gc["grounded"] + gc["ungrounded"]),
            "ungrounded_hallucination_pct": pct(gh["ungrounded"], gh["grounded"] + gh["ungrounded"])}


def _configs_cell(cell: dict, fn) -> dict:
    return {"configs": {c: fn(v) for c, v in cell["configs"].items()}}


def _by_axis(tree: dict, fn) -> dict:
    return {ax: {g: fn(cell) for g, cell in groups.items()} for ax, groups in tree.items()}


def display_strings(res: dict) -> dict:
    """Every number string the explorer shows, produced by the Markdown formatters (see the header)."""
    outs = res["outcomes"]
    n_full = res["cohorts"]["full"]["n"]
    full, att, fth = res["full"], res["attempted"], res["faithfulness"]
    agg = lambda a: _agg_display(a, outs)                                  # noqa: E731
    return {
        "full": {
            "overall": {c: agg(a) for c, a in full["overall"].items()},
            "by_axis": _by_axis(full["by_axis"], lambda cell: _configs_cell(cell, agg)),
            "pairs": {pk: {"overall": _outcome_pair_display(p["overall"], True, outs),
                           "by_axis": _by_axis(p["by_axis"], lambda o: _outcome_pair_display(o, True, outs))}
                      for pk, p in full["pairs"].items()},
            "transitions": {pk: {"matrix": {ob: {oa: pct(row[oa], sum(row.values())) for oa in outs}
                                            for ob, row in t["matrix"].items()}}
                            for pk, t in full["transitions"].items()},
            "attempt_patterns": {k: pct(v, n_full) for k, v in full["attempt_patterns"].items()},
        },
        "attempted": {
            "overall": {c: agg(a) for c, a in att["overall"].items()},
            "by_axis": _by_axis(att["by_axis"], lambda cell: _configs_cell(cell, agg)),
            "pairs": {pk: {"overall": _outcome_pair_display(p["overall"], False, outs),
                           "by_axis": _by_axis(p["by_axis"], lambda o: _outcome_pair_display(o, False, outs))}
                      for pk, p in att["pairs"].items()},
            "all_four": {"overall": _configs_cell(att["all_four"]["overall"], agg),
                         "by_axis": _by_axis(att["all_four"]["by_axis"], lambda cell: _configs_cell(cell, agg))},
        },
        "faithfulness": {
            "own": {c: _faith_mean_display(v) for c, v in fth["own"].items()},
            "by_axis": _by_axis(fth["by_axis"], lambda cell: _configs_cell(cell, _faith_mean_display)),
            "all_three": {"overall": _configs_cell(fth["all_three"]["overall"], _faith_mean_display),
                          "by_axis": _by_axis(fth["all_three"]["by_axis"],
                                              lambda cell: _configs_cell(cell, _faith_mean_display))},
            "pairs": {pk: (None if not p.get("available", True) else
                           {"overall": _faith_pair_display(p["overall"]),
                            "by_axis": _by_axis(p["by_axis"], _faith_pair_display)})
                      for pk, p in fth["pairs"].items()},
            "by_outcome": {c: _by_outcome_display(o) for c, o in fth["by_outcome"].items()},
        },
    }


def _ordering(values: dict[str, float | None], ascending=True, fmt=None) -> str:
    fmt = fmt or (lambda v: rate_pct(v) + "%")
    items = [(c, v) for c, v in values.items() if v is not None]
    items.sort(key=lambda cv: cv[1], reverse=not ascending)
    joiner = " < " if ascending else " > "
    return joiner.join(f"{CODE[c]} {fmt(v)}" for c, v in items)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

REGEN_CMD = "venv/Scripts/python tools/build_metric_reports.py"
CHECK_CMD = REGEN_CMD + " --check"


def _provenance(res: dict, depth: int) -> str:
    up = "../" * depth
    return "\n".join([
        "## 7. Provenance and reproduction", "",
        f"- Input files and checksums are recorded in [`manifest.json`]({up}manifest.json).",
        f"- Every number here is copied from [`results.json`]({up}results.json); the reading guide is "
        f"[`README.md`]({up}README.md); browse interactively in [`explorer.html`]({up}explorer.html).",
        f"- Regenerate: `{REGEN_CMD}` — verify without overwriting: `{CHECK_CMD}`.",
        f"- Statistics: paired bootstrap ({res['settings']['n_resamples']:,} resamples, seed {res['settings']['seed']}, "
        "nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. "
        "Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.",
        f"- Precedent checks against the persisted sanctioned files: "
        f"{sum(1 for c in res['precedent_checks'] if c.get('match') is True)} matched, "
        f"{sum(1 for c in res['precedent_checks'] if c.get('match') is False)} mismatched "
        f"(list in `results.json` → `precedent_checks`).",
    ])


def _tiny_note() -> str:
    return (f"† marks a cell with fewer than {TINY_N} questions; `n/a` marks an empty group "
            "(unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B "
            "in percentage points (pp) and are computed from the unrounded rates, so they can differ "
            "from the difference of the printed values by 0.1.")


def _limitations_common() -> list[str]:
    return [
        "- The three attribute axes are separate marginal partitions of the same questions; a finding that "
        "repeats across axes is the same questions seen three times, not three findings.",
        "- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup "
        "interaction, and one group being significant while another is not says nothing about their contrast.",
        "- Hallucination here is confident disagreement with the 2021 Mintaka gold (a factuality proxy), scored "
        "by the shared type-aware scorer at threshold 0.5; correctness is that scorer's CORRECT outcome, not "
        "exact match or verified world truth.",
        "- C1 never receives the question-entity block that C2–C4 receive, so a C1-vs-retrieval delta measures "
        "retrieval plus that annotation. C4's prompt differs from C3's in more than the condensing step.",
        "- Run-to-run noise (byte-identical repeats): hallucination ≤ 1.0 pt, correct/F1 2–4 pts, abstention "
        "2–5 pts. A p<.05 on correctness, F1 or abstention is not by itself evidence of an effect.",
    ]


def render_full_hallucination(res: dict) -> str:
    full = res["full"]
    L = ["# Hallucination and abstention — full benchmark (n = 4,000)", "",
         "## 1. Definition and reading guide", "",
         "- **Population:** every question of the sealed TEST split, for every configuration. Pair rows compare the "
         "identical id set; nothing is filtered to common attempts in this folder (see `../attempted/` for that).",
         "- **Hallucination %** = 100 × questions with the scorer's HALLUCINATION outcome / n. "
         "**Abstention %** = 100 × ABSTENTION / n. The four outcomes (correct, hallucination, abstention, other) "
         "partition every group exactly; **coverage** = attempted / n = (correct + hallucination) / n.",
         "- Other = empty, truncated or refusal-shaped-but-non-compliant replies: not evidence about accuracy, "
         "counted but never graded.",
         "- " + _tiny_note(), ""]
    # overview
    L += ["## 2. Overview", ""]
    rows = []
    for c in CONFIGS:
        a = full["overall"][c]
        k = a["counts"]
        rows.append([f"{CODE[c]} {NAME[c]}", str(a["n"]), f"{k[HALLUCINATION]} ({pct(k[HALLUCINATION], a['n'])})",
                     f"{k[ABSTENTION]} ({pct(k[ABSTENTION], a['n'])})", f"{k[CORRECT]} ({pct(k[CORRECT], a['n'])})",
                     f"{k[OTHER]} ({pct(k[OTHER], a['n'])})", pct(a["n_attempted"], a["n"]),
                     pct(k[HALLUCINATION], a["n_attempted"])])
    L += [md_table(["Config", "n", "Hallucination n (%)", "Abstention n (%)", "Correct n (%)", "Other n (%)",
                    "Coverage %", "Hall. % of attempts"], rows), "",
          "The last column is each configuration's OWN attempted set (denominators differ between rows; see "
          "`../attempted/hallucination_abstention.md`).", ""]
    # axes
    L += ["## 3. Dataset stratification", ""]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: hallucination % / abstention % over all questions in the group.", ""]
        rows = []
        for g in groups:
            cell = full["by_axis"][axis][g]
            r = [g, ncell(cell["n"])]
            for c in CONFIGS:
                k = cell["configs"][c]["counts"]
                r.append("—" if cell["n"] == 0 else f"{pct(k[HALLUCINATION], cell['n'])} / {pct(k[ABSTENTION], cell['n'])}")
            rows.append(r)
        L += [md_table(["Group", "n"] + [f"{CODE[c]} H / A" for c in CONFIGS], rows), ""]
        other = [(g, {CODE[c]: full["by_axis"][axis][g]["configs"][c]["counts"][OTHER] for c in CONFIGS
                      if full["by_axis"][axis][g]["configs"][c]["counts"][OTHER]})
                 for g in groups]
        other = [(g, d) for g, d in other if d]
        if other:
            L += ["Other counts (excluded from H and A): " + "; ".join(
                f"{g}: " + ", ".join(f"{k} {v}" for k, v in d.items()) for g, d in other) + ".", ""]
    # pairs
    L += ["## 4. Direct paired comparisons (identical id sets)", "",
          "### 4.1 Overall, six pairs", "",
          "ΔH = hallucination A − B; ΔA = abstention A − B (secondary: abstention has a 2–5 pt noise floor). "
          "McNemar: exact binomial on the discordant pairs (A-only / B-only hallucinations).", ""]
    rows = []
    for a, b in PAIRS:
        p = full["pairs"][pair_key(a, b)]["overall"]
        h, ab = p["hallucination"], p["abstention"]
        rows.append([pair_label(a, b), str(p["n"]), cp(p["a"], HALLUCINATION),
                     cp(p["b"], HALLUCINATION), pp(h["delta"]), ci_pp(h["bootstrap"]),
                     fmt_p(h["bootstrap"]["p_value"]), f"{h['mcnemar']['n10']} / {h['mcnemar']['n01']}",
                     fmt_p(h["mcnemar"]["p_value"]), pp(ab["delta"]), ci_pp(ab["bootstrap"])])
    L += [md_table(["Pair (A − B)", "n", "H% A", "H% B", "ΔH pp", "95% CI", "p", "A-only / B-only", "McNemar p",
                    "ΔA pp", "95% CI (ΔA)"], rows), ""]
    for a, b in PAIRS:
        pk = pair_key(a, b)
        L += [f"### 4.{PAIRS.index((a, b)) + 2} {pair_label(a, b)} by attribute", "",
              "Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.", ""]
        rows = []
        for axis, lbl, groups in AXES:
            rows.append([f"**{lbl}**"] + [""] * 7)
            for g in groups:
                p = full["pairs"][pk]["by_axis"][axis][g]
                if not p["available"]:
                    rows.append([g, "n/a"] + ["—"] * 6)
                    continue
                rows.append([g, ncell(p["n"]), cp(p["a"], HALLUCINATION),
                             cp(p["b"], HALLUCINATION), pp(p["hallucination"]["delta"]),
                             cp(p["a"], ABSTENTION), cp(p["b"], ABSTENTION),
                             pp(p["abstention"]["delta"])])
        L += [md_table(["Group", "n", f"H% {CODE[a]}", f"H% {CODE[b]}", "ΔH pp", f"A% {CODE[a]}", f"A% {CODE[b]}",
                        "ΔA pp"], rows), ""]
    # supplements
    L += ["## 5. Supplements: conditional outcomes", "",
          "### 5.1 Outcome-transition matrices", "",
          "For each pair, rows are the outcome of the reference configuration B and columns the outcome of A on the "
          "same questions; cells are counts with the row share in brackets. Read a row as *\"where B did X, A did …\"*. "
          "These are descriptive comparisons of independent outputs on a selected subset, not causal decompositions.", ""]
    for a, b in PAIRS:
        t = full["transitions"][pair_key(a, b)]
        L += [f"**{pair_label(a, b)}** — rows: {CODE[b]} outcome · columns: {CODE[a]} outcome", ""]
        rows = []
        for ob in OUTCOMES:
            rn = sum(t["matrix"][ob].values())
            rows.append([f"{CODE[b]} {ob}", str(rn)] + [f"{t['matrix'][ob][oa]} ({pct(t['matrix'][ob][oa], rn)})"
                                                         if rn else "—" for oa in OUTCOMES])
        L += [md_table(["", "row n"] + [f"{CODE[a]} {oa}" for oa in OUTCOMES], rows), ""]
    L += ["### 5.2 Exact attempt patterns", "",
          "Which configurations attempted each question. Unlike the common-attempt intersections in `../attempted/`, "
          "these mutually exclusive patterns partition the 4,000 questions. They describe coverage overlap, not accuracy.", ""]
    rows = [[k, str(v), pct(v, 4000)] for k, v in full["attempt_patterns"].items()]
    L += [md_table(["Configurations that attempted", "n", "% of 4,000"], rows), ""]
    # findings
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    hv = {c: full["overall"][c]["rates"][HALLUCINATION] for c in CONFIGS}
    av = {c: full["overall"][c]["rates"][ABSTENTION] for c in CONFIGS}
    L += [f"- Full-benchmark hallucination, lowest to highest: {_ordering(hv)}.",
          f"- Abstention, lowest to highest: {_ordering(av)}."]
    for axis, lbl, groups in AXES:
        parts = []
        for c in CONFIGS:
            vals = {g: full["by_axis"][axis][g]["configs"][c]["rates"][HALLUCINATION] for g in groups
                    if full["by_axis"][axis][g]["n"] >= TINY_N}
            if not vals:
                continue
            hi = max(vals, key=vals.get)
            lo = min(vals, key=vals.get)
            parts.append(f"{CODE[c]} highest on {hi} ({rate_pct(vals[hi])}%), lowest on {lo} ({rate_pct(vals[lo])}%)")
        L.append(f"- By {lbl.lower()} (groups with n ≥ {TINY_N}): " + ("; ".join(parts) if parts else
                                                                     "no group reaches that size") + ".")
    for a, b in PAIRS:
        h = full["pairs"][pair_key(a, b)]["overall"]["hallucination"]
        verdict = "CI excludes zero" if h["bootstrap"]["significant"] else "CI includes zero"
        L.append(f"- {pair_label(a, b)}: ΔH {pp(h['delta'])} pp {ci_pp(h['bootstrap'])}, bootstrap p {fmt_p(h['bootstrap']['p_value'])}, "
                 f"McNemar p {fmt_p(h['mcnemar']['p_value'])} — {verdict}.")
    for a, b in PAIRS:
        t = full["transitions"][pair_key(a, b)]["matrix"]
        for ob in (HALLUCINATION, ABSTENTION):
            rn = sum(t[ob].values())
            if rn:
                L.append(f"- Where {CODE[b]}'s outcome is {ob} ({rn} questions), {CODE[a]}: correct {t[ob][CORRECT]} "
                         f"({pct(t[ob][CORRECT], rn)}%), hallucination {t[ob][HALLUCINATION]} "
                         f"({pct(t[ob][HALLUCINATION], rn)}%), abstention {t[ob][ABSTENTION]} "
                         f"({pct(t[ob][ABSTENTION], rn)}%), other {t[ob][OTHER]}.")
    tiny = [f"{g} (n={res['axis_counts'][axis][g]})" for axis, _, groups in AXES for g in groups
            if res["axis_counts"][axis][g] < TINY_N]
    L.append("- Tiny groups (†): " + (", ".join(tiny) if tiny else "none") + " — no ranking should rest on them.")
    L += ["", "Limitations:", ""] + _limitations_common() + [
        "- Lower hallucination does not imply higher coverage or F1: the configurations with the fewest confident "
        "errors are also the ones that abstain most (see section 2 and `f1_correctness.md`).", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def render_attempted_hallucination(res: dict) -> str:
    att = res["attempted"]
    L = ["# Hallucination — attempted answers (each configuration's own attempts; pairs on common attempts)", "",
         "## 1. Definition and reading guide", "",
         "- **Attempted** = the scorer's `is_attempted`: outcome CORRECT or HALLUCINATION. Abstention AND Other are "
         "excluded — attempted does not mean \"not abstention\".",
         "- **Hallucination % of attempts** = 100 × HALLUCINATION / attempted. Within attempts, correct % is its "
         "complement (they sum to 100), so a test on one IS the test on the other — `f1_correctness.md` does not repeat it.",
         "- Overview and attribute tables use each configuration's OWN attempted ids (denominators differ per config). "
         "Pair rows use the INTERSECTION of the two attempt sets and recompute both rates there; they never subtract "
         "two own-attempted rates.",
         "- These are answer-selected, post-treatment populations: a common id set removes question-set mismatch "
         "within a comparison but not selection bias, and identifies no causal context effect. An inclusive "
         "C1/C2 intersection also contains questions C3/C4 attempted.",
         "- **Retained share** = attempted / benchmark n for the same group.",
         "- " + _tiny_note(), ""]
    L += ["## 2. Overview (own attempts)", ""]
    rows = []
    for c in CONFIGS:
        a = att["overall"][c]
        rows.append([f"{CODE[c]} {NAME[c]}", str(a["benchmark_n"]), str(a["n"]), pct(a["n"], a["benchmark_n"]),
                     str(a["excluded"]["abstention"]), str(a["excluded"]["other"]),
                     f"{a['counts'][HALLUCINATION]} ({pct(a['counts'][HALLUCINATION], a['n'])})",
                     pct(a["counts"][CORRECT], a["n"])])
    L += [md_table(["Config", "Benchmark n", "Attempted n", "Retained %", "Excl. abstention", "Excl. other",
                    "Hallucination n (% of attempts)", "Correct % of attempts"], rows), ""]
    L += ["## 3. Dataset stratification (own attempts)", ""]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: attempted n / hallucination % of those attempts. Group n is the benchmark count.", ""]
        rows = []
        for g in groups:
            cell = att["by_axis"][axis][g]
            r = [g, ncell(cell["n"])]
            for c in CONFIGS:
                a = cell["configs"][c]
                r.append("—" if a["n"] == 0 else f"{ncell(a['n'])} / {pct(a['counts'][HALLUCINATION], a['n'])}")
            rows.append(r)
        L += [md_table(["Group", "n"] + [f"{CODE[c]} att. n / H%" for c in CONFIGS], rows), ""]
    L += ["## 4. Direct paired comparisons (common attempts)", "", "### 4.1 Overall, six pairs", "",
          "Both rates recomputed on the questions BOTH configurations attempted. Discordant cells: A-only = A "
          "hallucinated and B was correct; B-only the reverse.", ""]
    rows = []
    for a, b in PAIRS:
        p = att["pairs"][pair_key(a, b)]["overall"]
        h = p["hallucination"]
        t = h["table"]
        rows.append([pair_label(a, b), str(p["n_common"]), share(p["retained_share"]),
                     cp(p["a"], HALLUCINATION), cp(p["b"], HALLUCINATION), pp(h["delta"]),
                     ci_pp(h["bootstrap"]), fmt_p(h["bootstrap"]["p_value"]),
                     f"{t['both_hallucinate']} / {t['a_only']} / {t['b_only']} / {t['neither']}",
                     fmt_p(h["mcnemar"]["p_value"])])
    L += [md_table(["Pair (A − B)", "Common n", "Retained % of 4,000", "H% A", "H% B", "ΔH pp", "95% CI", "p",
                    "both / A-only / B-only / neither", "McNemar p"], rows), ""]
    for a, b in PAIRS:
        pk = pair_key(a, b)
        L += [f"### 4.{PAIRS.index((a, b)) + 2} {pair_label(a, b)} by attribute", "",
              "Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is "
              "relative to the group's benchmark n.", ""]
        rows = []
        for axis, lbl, groups in AXES:
            rows.append([f"**{lbl}**"] + [""] * 6)
            for g in groups:
                p = att["pairs"][pk]["by_axis"][axis][g]
                if not p["available"]:
                    rows.append([g, ncell(p["denominator_n"]), "n/a", "—", "—", "—", "—"])
                    continue
                rows.append([g, ncell(p["denominator_n"]), ncell(p["n_common"]), share(p["retained_share"]),
                             cp(p["a"], HALLUCINATION), cp(p["b"], HALLUCINATION),
                             pp(p["hallucination"]["delta"])])
        L += [md_table(["Group", "n", "Common n", "Retained %", f"H% {CODE[a]}", f"H% {CODE[b]}", "ΔH pp"], rows), ""]
    four = att["all_four"]
    L += ["## 5. Supplement: questions attempted by all four configurations", "",
          f"One common set for all four (n = {four['overall']['n']}, {share(four['overall']['retained_share'])}% of "
          "the benchmark). The most selected population in this supplement: every configuration chose to answer.", ""]
    rows = [["overall", str(four["overall"]["n"]), "4000", share(four["overall"]["retained_share"])] +
            [pct(four["overall"]["configs"][c]["counts"][HALLUCINATION], four["overall"]["n"]) for c in CONFIGS]]
    for axis, lbl, groups in AXES:
        rows.append([f"**{lbl}**"] + [""] * 7)
        for g in groups:
            cell = four["by_axis"][axis][g]
            rows.append([g, ncell(cell["n"]), str(cell["benchmark_n"]), share(cell["retained_share"])] +
                        [pct(cell["configs"][c]["counts"][HALLUCINATION], cell["n"]) if cell["n"] else "—" for c in CONFIGS])
    L += [md_table(["Group", "Common n", "Benchmark n", "Retained %"] + [f"{CODE[c]} H%" for c in CONFIGS], rows), ""]
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    hv = {c: att["overall"][c]["rates"][HALLUCINATION] for c in CONFIGS}
    cov = {c: att["overall"][c]["retained_share"] for c in CONFIGS}
    L += [f"- Hallucination among own attempts, lowest to highest: {_ordering(hv)}.",
          f"- Retained share (coverage), highest to lowest: {_ordering(cov, ascending=False)}."]
    for a, b in PAIRS:
        p = att["pairs"][pair_key(a, b)]["overall"]
        h = p["hallucination"]
        verdict = "CI excludes zero" if h["bootstrap"]["significant"] else "CI includes zero"
        L.append(f"- {pair_label(a, b)} on {p['n_common']} common attempts: {cp(p['a'], HALLUCINATION)}% vs "
                 f"{cp(p['b'], HALLUCINATION)}%, Δ {pp(h['delta'])} pp {ci_pp(h['bootstrap'])}, p "
                 f"{fmt_p(h['bootstrap']['p_value'])}, McNemar p {fmt_p(h['mcnemar']['p_value'])} — {verdict}.")
    L.append(f"- On the all-four common set ({four['overall']['n']}): " + ", ".join(
        f"{CODE[c]} {pct(four['overall']['configs'][c]['counts'][HALLUCINATION], four['overall']['n'])}%" for c in CONFIGS) + ".")
    L += ["", "Limitations:", ""] + _limitations_common() + [
        "- Own-attempt rates sit on different question sets per configuration and are not testable against each "
        "other; only the common-attempt rows are.",
        "- A common-attempt comparison conditions on both configurations choosing to answer, which is itself a "
        "treatment outcome; it complements the full-benchmark comparison and does not replace it.", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def render_full_f1(res: dict) -> str:
    full = res["full"]
    L = ["# F1 and correctness — full benchmark (n = 4,000)", "",
         "## 1. Definition and reading guide", "",
         "- **Correct %** = 100 × questions with the scorer's CORRECT outcome / n: thresholded operational "
         "correctness (score ≥ 0.5 under the type-aware matcher), not exact match and not verified world truth.",
         "- **F1** = 100 × mean per-question token/set F1 from the shared scorer, on a 0–100 scale. Abstention and "
         "Other carry F1 = 0 under the current contract, so full-benchmark F1 = coverage × F1@attempted exactly.",
         "- **F1@attempted** is each configuration's OWN attempted set (different denominators; see "
         "`../attempted/f1_correctness.md`). The threshold never replaces the outcome classifier.",
         "- " + _tiny_note(), ""]
    L += ["## 2. Overview", ""]
    rows = []
    for c in CONFIGS:
        a = full["overall"][c]
        ident = (a["coverage"] or 0) * (a["f1_attempted"] or 0)
        rows.append([f"{CODE[c]} {NAME[c]}", str(a["n"]), f"{a['counts'][CORRECT]} ({pct(a['counts'][CORRECT], a['n'])})",
                     f1s(a["f1_mean"]), pct(a["n_attempted"], a["n"]), f1s(a["f1_attempted"]), f1s(ident),
                     f1s(a["em_mean"])])
    L += [md_table(["Config", "n", "Correct n (%)", "F1", "Coverage %", "F1@attempted (own)", "coverage × F1@att.",
                    "EM"], rows), "",
          "EM (exact match, 0–100) is shown as a companion only; the thesis reports F1.", ""]
    L += ["## 3. Dataset stratification", ""]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: correct % / F1 over all questions in the group.", ""]
        rows = []
        for g in groups:
            cell = full["by_axis"][axis][g]
            r = [g, ncell(cell["n"])]
            for c in CONFIGS:
                a = cell["configs"][c]
                r.append("—" if cell["n"] == 0 else f"{pct(a['counts'][CORRECT], cell['n'])} / {f1s(a['f1_mean'])}")
            rows.append(r)
        L += [md_table(["Group", "n"] + [f"{CODE[c]} Corr / F1" for c in CONFIGS], rows), ""]
    L += ["## 4. Direct paired comparisons (identical id sets)", "", "### 4.1 Overall, six pairs", "",
          "ΔCorr = correct A − B (pp) with bootstrap CI and exact McNemar; ΔF1 = F1 A − B (points) with bootstrap CI.", ""]
    rows = []
    for a, b in PAIRS:
        p = full["pairs"][pair_key(a, b)]["overall"]
        cr, f = p["correct"], p["f1"]
        rows.append([pair_label(a, b), str(p["n"]), cp(p["a"], CORRECT), cp(p["b"], CORRECT),
                     pp(cr["delta"]), ci_pp(cr["bootstrap"]), fmt_p(cr["bootstrap"]["p_value"]), fmt_p(cr["mcnemar"]["p_value"]),
                     f1s(p["a"]["f1_mean"]), f1s(p["b"]["f1_mean"]), pp(f["delta"]), ci_pp(f["bootstrap"]),
                     fmt_p(f["bootstrap"]["p_value"])])
    L += [md_table(["Pair (A − B)", "n", "Corr% A", "Corr% B", "ΔCorr pp", "95% CI", "p", "McNemar p", "F1 A", "F1 B",
                    "ΔF1", "95% CI", "p"], rows), ""]
    for a, b in PAIRS:
        pk = pair_key(a, b)
        L += [f"### 4.{PAIRS.index((a, b)) + 2} {pair_label(a, b)} by attribute", "", "Descriptive per group (no tests).", ""]
        rows = []
        for axis, lbl, groups in AXES:
            rows.append([f"**{lbl}**"] + [""] * 7)
            for g in groups:
                p = full["pairs"][pk]["by_axis"][axis][g]
                if not p["available"]:
                    rows.append([g, "n/a"] + ["—"] * 6)
                    continue
                rows.append([g, ncell(p["n"]), cp(p["a"], CORRECT), cp(p["b"], CORRECT),
                             pp(p["correct"]["delta"]), f1s(p["a"]["f1_mean"]), f1s(p["b"]["f1_mean"]), pp(p["f1"]["delta"])])
        L += [md_table(["Group", "n", f"Corr% {CODE[a]}", f"Corr% {CODE[b]}", "ΔCorr pp", f"F1 {CODE[a]}", f"F1 {CODE[b]}",
                        "ΔF1"], rows), ""]
    L += ["## 5. Supplement: threshold sensitivity", "",
          "Outcomes re-scored with the wrong-answer cutoff at 0.3 / 0.5 / 0.7 (0.5 is the reported contract). "
          "Cells: correct % / hallucination %. The question is whether the ORDERING depends on the cutoff, not the level.", ""]
    rows = []
    for c in CONFIGS:
        r = [f"{CODE[c]} {NAME[c]}"]
        for t in THRESHOLDS:
            a = full["thresholds"][f"{t:.1f}"][c]
            r.append(f"{pct(a['counts'][CORRECT], a['n'])} / {pct(a['counts'][HALLUCINATION], a['n'])}")
        rows.append(r)
    L += [md_table(["Config"] + [f"t = {t:.1f}: Corr / H" for t in THRESHOLDS], rows), ""]
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    cv = {c: full["overall"][c]["rates"][CORRECT] for c in CONFIGS}
    fv = {c: full["overall"][c]["f1_mean"] for c in CONFIGS}
    fa = {c: full["overall"][c]["f1_attempted"] for c in CONFIGS}
    L += [f"- Full-benchmark correct %, highest to lowest: {_ordering(cv, ascending=False)}.",
          f"- Full-benchmark F1, highest to lowest: {_ordering(fv, ascending=False, fmt=f1s)}.",
          f"- F1 on own attempts, highest to lowest: {_ordering(fa, ascending=False, fmt=f1s)} — different denominators, not testable against each other."]
    for a, b in PAIRS:
        p = full["pairs"][pair_key(a, b)]["overall"]
        L.append(f"- {pair_label(a, b)}: ΔCorr {pp(p['correct']['delta'])} pp {ci_pp(p['correct']['bootstrap'])}, p "
                 f"{fmt_p(p['correct']['bootstrap']['p_value'])}; ΔF1 {pp(p['f1']['delta'])} {ci_pp(p['f1']['bootstrap'])}, p "
                 f"{fmt_p(p['f1']['bootstrap']['p_value'])}.")
    for t in THRESHOLDS:
        hv = {c: full["thresholds"][f"{t:.1f}"][c]["rates"][HALLUCINATION] for c in CONFIGS}
        L.append(f"- Hallucination ordering at t = {t:.1f}: {_ordering(hv)}.")
    L += ["", "Limitations:", ""] + _limitations_common() + [
        "- Full-benchmark F1 and correct % reward answering: a configuration that abstains often scores low here even "
        "when its attempted answers are as accurate — read them beside coverage, never alone.", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def render_attempted_f1(res: dict) -> str:
    att = res["attempted"]
    L = ["# F1 and correctness — attempted answers (own attempts; pairs on common attempts)", "",
         "## 1. Definition and reading guide", "",
         "- **Attempted** = outcome CORRECT or HALLUCINATION (Other excluded). **Correct % of attempts** is the "
         "complement of hallucination % of attempts (they sum to 100): its paired test is the one in "
         "`hallucination_abstention.md` §4.1 and is NOT repeated here as independent evidence.",
         "- **F1@attempted** = 100 × mean per-question F1 over the attempted set (0–100). A partial-credit measure: "
         "it separates from correct % where answers are partially right (set answers, extra tokens).",
         "- Overview and attribute tables use each configuration's OWN attempts; pair rows use the common attempts "
         "of the two configurations and recompute both means there.",
         "- " + _tiny_note(), ""]
    L += ["## 2. Overview (own attempts)", ""]
    rows = []
    for c in CONFIGS:
        a = att["overall"][c]
        rows.append([f"{CODE[c]} {NAME[c]}", str(a["n"]), pct(a["n"], a["benchmark_n"]), pct(a["counts"][CORRECT], a["n"]),
                     f1s(a["f1_mean"]), f1s(a["em_mean"])])
    L += [md_table(["Config", "Attempted n", "Retained %", "Correct % of attempts", "F1@attempted", "EM@attempted"], rows), ""]
    L += ["## 3. Dataset stratification (own attempts)", ""]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: attempted n / correct % / F1 on those attempts.", ""]
        rows = []
        for g in groups:
            cell = att["by_axis"][axis][g]
            r = [g, ncell(cell["n"])]
            for c in CONFIGS:
                a = cell["configs"][c]
                r.append("—" if a["n"] == 0 else f"{ncell(a['n'])} / {pct(a['counts'][CORRECT], a['n'])} / {f1s(a['f1_mean'])}")
            rows.append(r)
        L += [md_table(["Group", "n"] + [f"{CODE[c]} n / Corr / F1" for c in CONFIGS], rows), ""]
    L += ["## 4. Direct paired comparisons (common attempts)", "", "### 4.1 Overall, six pairs", "",
          "ΔF1 with bootstrap CI (the F1 test). ΔCorr is descriptive here — its test is the hallucination test.", ""]
    rows = []
    for a, b in PAIRS:
        p = att["pairs"][pair_key(a, b)]["overall"]
        f = p["f1"]
        rows.append([pair_label(a, b), str(p["n_common"]), f1s(p["a"]["f1_mean"]), f1s(p["b"]["f1_mean"]), pp(f["delta"]),
                     ci_pp(f["bootstrap"]), fmt_p(f["bootstrap"]["p_value"]), cp(p["a"], CORRECT),
                     cp(p["b"], CORRECT), pp(p["correct"]["delta"])])
    L += [md_table(["Pair (A − B)", "Common n", "F1 A", "F1 B", "ΔF1", "95% CI", "p", "Corr% A", "Corr% B",
                    "ΔCorr pp (see H file)"], rows), ""]
    for a, b in PAIRS:
        pk = pair_key(a, b)
        L += [f"### 4.{PAIRS.index((a, b)) + 2} {pair_label(a, b)} by attribute", "", "Descriptive per group (no tests).", ""]
        rows = []
        for axis, lbl, groups in AXES:
            rows.append([f"**{lbl}**"] + [""] * 7)
            for g in groups:
                p = att["pairs"][pk]["by_axis"][axis][g]
                if not p["available"]:
                    rows.append([g, "n/a"] + ["—"] * 6)
                    continue
                rows.append([g, ncell(p["n_common"]), f1s(p["a"]["f1_mean"]), f1s(p["b"]["f1_mean"]), pp(p["f1"]["delta"]),
                             cp(p["a"], CORRECT), cp(p["b"], CORRECT), pp(p["correct"]["delta"])])
        L += [md_table(["Group", "Common n", f"F1 {CODE[a]}", f"F1 {CODE[b]}", "ΔF1", f"Corr% {CODE[a]}", f"Corr% {CODE[b]}",
                        "ΔCorr pp"], rows), ""]
    four = att["all_four"]
    L += ["## 5. Supplement: questions attempted by all four configurations", "",
          f"n = {four['overall']['n']} ({share(four['overall']['retained_share'])}% of the benchmark). Cells: correct % / F1.", ""]
    rows = [["overall", str(four["overall"]["n"])] +
            [f"{pct(four['overall']['configs'][c]['counts'][CORRECT], four['overall']['n'])} / {f1s(four['overall']['configs'][c]['f1_mean'])}"
             for c in CONFIGS]]
    for axis, lbl, groups in AXES:
        rows.append([f"**{lbl}**"] + [""] * 5)
        for g in groups:
            cell = four["by_axis"][axis][g]
            rows.append([g, ncell(cell["n"])] + [
                f"{pct(cell['configs'][c]['counts'][CORRECT], cell['n'])} / {f1s(cell['configs'][c]['f1_mean'])}" if cell["n"] else "—"
                for c in CONFIGS])
    L += [md_table(["Group", "Common n"] + [f"{CODE[c]} Corr / F1" for c in CONFIGS], rows), ""]
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    fv = {c: att["overall"][c]["f1_mean"] for c in CONFIGS}
    cv = {c: att["overall"][c]["rates"][CORRECT] for c in CONFIGS}
    L += [f"- F1 on own attempts, highest to lowest: {_ordering(fv, ascending=False, fmt=f1s)} (different denominators).",
          f"- Correct % of own attempts, highest to lowest: {_ordering(cv, ascending=False)}."]
    for a, b in PAIRS:
        p = att["pairs"][pair_key(a, b)]["overall"]
        f = p["f1"]
        verdict = "CI excludes zero" if f["bootstrap"]["significant"] else "CI includes zero"
        L.append(f"- {pair_label(a, b)} on {p['n_common']} common attempts: F1 {f1s(p['a']['f1_mean'])} vs {f1s(p['b']['f1_mean'])}, "
                 f"Δ {pp(f['delta'])} {ci_pp(f['bootstrap'])}, p {fmt_p(f['bootstrap']['p_value'])} — {verdict}.")
    L.append(f"- All-four common set ({four['overall']['n']}), F1: " + ", ".join(
        f"{CODE[c]} {f1s(four['overall']['configs'][c]['f1_mean'])}" for c in CONFIGS) + ".")
    L += ["", "Limitations:", ""] + _limitations_common() + [
        "- Correct % and F1 on attempts can move by 2–4 pts between byte-identical runs; a common-attempt F1 gap inside "
        "that band is not a finding.", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def _faith_definition(folder: str) -> list[str]:
    return [
        "## 1. Definition and reading guide", "",
        "- **Faithfulness** = the stored answer-level judge score (0–1, RAGAS-style: share of the answer's claims "
        "entailed by the context that configuration was supplied). Means are macro-averages of answer scores, never "
        "total supported claims over total claims.",
        "- **There is no four-configuration, denominator-4,000 faithfulness mean in this experiment.** C1 receives no "
        "context: N/A, not zero, in every table and every pair involving it. C2–C4 means are conditional on a "
        "**scoreable attempt**: the answer was attempted (CORRECT or HALLUCINATION) and the judge returned a usable score.",
        "- Abstentions, Other and judge-unscored attempts are EXCLUDED from every mean, never assigned zero; the "
        "reasons are separated in the coverage accounting. `partial_score` is never used as a substitute.",
        ("- This folder reports scoreable share relative to the FULL benchmark; the means themselves are identical to "
         "`../attempted/faithfulness.md` because the conditional population is the same — the two folders differ only "
         "in what the share is taken over." if folder == "full" else
         "- This folder reports scoreability AMONG ATTEMPTS; the own-scoreable means are identical to "
         "`../full_4000/faithfulness.md` by construction (same conditional population)."),
        "- Pair rows use the COMMON scoreable set (both attempted, both scored, same question). Each score still "
        "evaluates its answer against ITS OWN supplied context, not a shared one.",
        "- Judge contract-audit flags (degenerate verdict lines) are retained under the existing availability policy; "
        "the sensitivity means without them are recorded, not reported.",
        "- " + _tiny_note(), ""]


def render_full_faith(res: dict) -> str:
    F = res["faithfulness"]
    L = ["# Faithfulness — full benchmark view (conditional on a scoreable attempt)", ""] + _faith_definition("full")
    L += ["## 2. Overview", ""]
    c1 = res["full"]["overall"]["base_llm_abstain"]
    rows = [["C1 Base LLM", str(c1["n"]), str(c1["n_attempted"]), "N/A", "N/A", "N/A", "N/A — no context"]]
    for c in RETRIEVAL:
        f = F["own"][c]
        rows.append([f"{CODE[c]} {NAME[c]}", str(f["n_ids"]), str(f["n_attempted"]), str(f["n_scoreable"]),
                     str(f["n_unscored_attempts"]), pct(f["n_scoreable"], f["n_ids"]), fs(f["mean"])])
    L += [md_table(["Config", "Benchmark n", "Attempted n", "Scoreable n", "Unscored attempts", "Scoreable % of benchmark",
                    "Mean faithfulness (conditional)"], rows), ""]
    if F.get("contract_flags"):
        cf = F["contract_flags"]
        per_cfg = ", ".join(f"{CODE[c]} {cf['per_config_flagged_scoreable'][c]}" for c in RETRIEVAL)
        L += [f"Contract-audit flags ({cf['source']}): {cf['n_flagged_scoreable']} scoreable records flagged "
              f"({per_cfg}), retained. "
              "Means excluding them: " + ", ".join(f"{CODE[c]} {fs(cf['mean_excl_flagged'][c])}" for c in RETRIEVAL) + ".", ""]
    L += ["## 3. Dataset stratification", "", ]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: scoreable n (share of the group's benchmark n) / mean. C1: N/A.", ""]
        rows = []
        for g in groups:
            cell = F["by_axis"][axis][g]
            r = [g, ncell(cell["n"]), "N/A"]
            for c in RETRIEVAL:
                f = cell["configs"][c]
                r.append("—" if f["n_scoreable"] == 0 else f"{ncell(f['n_scoreable'])} ({pct(f['n_scoreable'], cell['n'])}%) / {fs(f['mean'])}")
            rows.append(r)
        L += [md_table(["Group", "n", "C1"] + [f"{CODE[c]} scoreable (%) / mean" for c in RETRIEVAL], rows), ""]
    L += ["## 4. Direct paired comparisons (common scoreable attempts)", "", "### 4.1 Overall, six pairs", ""]
    rows = []
    for a, b in PAIRS:
        p = F["pairs"][pair_key(a, b)]
        if not p.get("available", True):
            rows.append([pair_label(a, b), "—", "—", "—", "—", "—", "—", f"N/A: {p['reason']}"])
            continue
        o = p["overall"]
        rows.append([pair_label(a, b), str(o["n_common_scoreable"]), pct(o["n_common_scoreable"], 4000), fs(o["mean_a"]), fs(o["mean_b"]),
                     dfs(o["delta"]), ci_f(o["bootstrap"]), fmt_p(o["bootstrap"]["p_value"])])
    L += [md_table(["Pair (A − B)", "Common scoreable n", "% of 4,000", "Mean A", "Mean B", "Δ", "95% CI", "p"], rows), "",
          "### 4.2 Per-pair attribute breakdowns", "",
          "Rendered once, in `../attempted/faithfulness.md` §4 — identical by construction (same common scoreable sets).", ""]
    L += ["## 5. Supplement: coverage accounting", "",
          "Why a question contributes no faithfulness score. The four reasons partition the benchmark for each configuration.", ""]
    rows = []
    for c in RETRIEVAL:
        a = F["coverage_accounting"][c]
        rows.append([f"{CODE[c]} {NAME[c]}", str(a["abstention"]), str(a["other"]), str(a["unscored_attempt"]),
                     str(a["scoreable"]), str(a["total"]), "yes" if a["partition_ok"] else "NO"])
    L += [md_table(["Config", "Abstention", "Other", "Attempted, judge unscored", "Scoreable", "Total", "= 4,000"], rows), ""]
    reasons = "; ".join(f"{CODE[c]}: " + ", ".join(f"{k} {v}" for k, v in F["own"][c]["unscored_reasons"].items())
                        for c in RETRIEVAL)
    L += [f"Unscored-attempt reasons (judge status): {reasons}.", ""]
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    mv = {c: F["own"][c]["mean"] for c in RETRIEVAL}
    order = sorted(RETRIEVAL, key=lambda c: mv[c], reverse=True)
    L += ["- Conditional mean faithfulness, highest to lowest: " + " > ".join(f"{CODE[c]} {fs(mv[c])}" for c in order) +
          " — on different scoreable sets (" + ", ".join(f"{CODE[c]} n={F['own'][c]['n_scoreable']}" for c in RETRIEVAL) +
          "); not testable against each other.",
          "- Scoreable share of the benchmark: " + ", ".join(
              f"{CODE[c]} {pct(F['own'][c]['n_scoreable'], 4000)}%" for c in RETRIEVAL) +
          "; the rest is abstention, Other or an unscored attempt (section 5)."]
    for a, b in RETRIEVAL_PAIRS:
        o = F["pairs"][pair_key(a, b)]["overall"]
        verdict = "CI excludes zero" if o["bootstrap"]["significant"] else "CI includes zero"
        L.append(f"- {pair_label(a, b)} on {o['n_common_scoreable']} common scoreable attempts: {fs(o['mean_a'])} vs {fs(o['mean_b'])}, "
                 f"Δ {dfs(o['delta'])} {ci_f(o['bootstrap'])}, p {fmt_p(o['bootstrap']['p_value'])} — {verdict}.")
    L += ["", "Limitations:", ""] + _limitations_common()[:2] + [
        "- Faithfulness is undefined for C1; the headline C1-vs-retrieval comparison is a factuality comparison by construction.",
        "- A faithful-but-wrong answer scores 1.0; the metric rewards small contexts and terse answers and does not measure "
        "benchmark correctness. C3's ~30×-smaller context cannot support as many claims as C2's.",
        "- The judge was validated on DEV with moderate answer-level agreement: quote configuration-level comparisons only.", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def render_attempted_faith(res: dict) -> str:
    F = res["faithfulness"]
    L = ["# Faithfulness — attempted answers (scoreable share among attempts; pairs on common scoreable attempts)", ""] + \
        _faith_definition("attempted")
    L += ["## 2. Overview", ""]
    rows = [["C1 Base LLM", str(res["attempted"]["overall"]["base_llm_abstain"]["n"]), "N/A", "N/A", "N/A", "N/A — no context"]]
    for c in RETRIEVAL:
        f = F["own"][c]
        rows.append([f"{CODE[c]} {NAME[c]}", str(f["n_attempted"]), str(f["n_scoreable"]), str(f["n_unscored_attempts"]),
                     pct(f["n_scoreable"], f["n_attempted"]), fs(f["mean"])])
    L += [md_table(["Config", "Attempted n", "Scoreable n", "Unscored attempts", "Scoreable % of attempts",
                    "Mean faithfulness (conditional)"], rows), ""]
    L += ["## 3. Dataset stratification", ""]
    for axis, lbl, groups in AXES:
        L += [f"### 3.{['answer_type', 'complexity', 'category'].index(axis) + 1} By {lbl.lower()}", "",
              "Cells: attempted n / scoreable n / mean. C1: N/A.", ""]
        rows = []
        for g in groups:
            cell = F["by_axis"][axis][g]
            r = [g, ncell(cell["n"]), "N/A"]
            for c in RETRIEVAL:
                f = cell["configs"][c]
                r.append("—" if f["n_scoreable"] == 0 else f"{f['n_attempted']} / {ncell(f['n_scoreable'])} / {fs(f['mean'])}")
            rows.append(r)
        L += [md_table(["Group", "n", "C1"] + [f"{CODE[c]} att. / scor. / mean" for c in RETRIEVAL], rows), ""]
    L += ["## 4. Direct paired comparisons (common scoreable attempts)", "", "### 4.1 Overall, six pairs", ""]
    rows = []
    for a, b in PAIRS:
        p = F["pairs"][pair_key(a, b)]
        if not p.get("available", True):
            rows.append([pair_label(a, b), "—", "—", "—", "—", "—", "—", f"N/A: {p['reason']}"])
            continue
        o = p["overall"]
        na, nb = F["own"][a]["n_scoreable"], F["own"][b]["n_scoreable"]
        rows.append([pair_label(a, b), str(o["n_common_scoreable"]), f"{pct(o['n_common_scoreable'], na)} / {pct(o['n_common_scoreable'], nb)}",
                     fs(o["mean_a"]), fs(o["mean_b"]), dfs(o["delta"]), ci_f(o["bootstrap"]), fmt_p(o["bootstrap"]["p_value"])])
    L += [md_table(["Pair (A − B)", "Common scoreable n", "% of A's / B's scoreable", "Mean A", "Mean B", "Δ", "95% CI", "p"], rows), ""]
    for a, b in PAIRS:
        pk = pair_key(a, b)
        L += [f"### 4.{PAIRS.index((a, b)) + 2} {pair_label(a, b)} by attribute", ""]
        if not F["pairs"][pk].get("available", True):
            L += [f"N/A — {F['pairs'][pk]['reason']}.", ""]
            continue
        L += ["Descriptive per group (no tests).", ""]
        rows = []
        for axis, lbl, groups in AXES:
            rows.append([f"**{lbl}**"] + [""] * 4)
            for g in groups:
                p = F["pairs"][pk]["by_axis"][axis][g]
                if not p["available"]:
                    rows.append([g, "n/a", "—", "—", "—"])
                    continue
                rows.append([g, ncell(p["n_common_scoreable"]), fs(p["mean_a"]), fs(p["mean_b"]), dfs(p["delta"])])
        L += [md_table(["Group", "Common scoreable n", f"Mean {CODE[a]}", f"Mean {CODE[b]}", "Δ"], rows), ""]
    L += ["## 5. Supplements", "", "### 5.1 Faithfulness by outcome within each configuration", "",
          "Correct vs hallucinated answers are DISJOINT groups within one configuration, so the contrast is an unpaired "
          "two-sample bootstrap. Grounded := score ≥ " + f"{GROUNDED_THRESHOLD}" + " (the pre-registered grounded×outcome "
          "cut); shares are within the outcome.", ""]
    rows = []
    for c in RETRIEVAL:
        o = F["by_outcome"][c]
        ub = o["unpaired_bootstrap_correct_minus_hallucination"]
        gc, gh = o["grounded"]["correct"], o["grounded"]["hallucination"]
        rows.append([f"{CODE[c]} {NAME[c]}", str(o["correct"]["n"]), fs(o["correct"]["mean"]), str(o["hallucination"]["n"]),
                     fs(o["hallucination"]["mean"]), dfs(ub["delta"]) if ub else "—", ci_f(ub) if ub else "—",
                     fmt_p(ub["p_value"]) if ub else "—",
                     f"{gc['ungrounded']} ({pct(gc['ungrounded'], gc['grounded'] + gc['ungrounded'])}%)",
                     f"{gh['ungrounded']} ({pct(gh['ungrounded'], gh['grounded'] + gh['ungrounded'])}%)"])
    L += [md_table(["Config", "Correct n", "mean", "Hallucination n", "mean", "Δ corr − hall", "95% CI", "p",
                    "Ungrounded among correct", "Ungrounded among hallucinations"], rows), "",
          "The last column is the confabulation-ANALOGUE rate (ungrounded share of hallucinations): an entailment verdict, "
          "not proof of source — judge error and derivable-but-unstated answers land in the same bucket.", ""]
    three = F["all_three"]
    L += ["### 5.2 Questions scoreable for all three retrieval configurations", "",
          f"n = {three['overall']['n']}. Cells: mean faithfulness on that common set.", ""]
    rows = [["overall", str(three["overall"]["n"])] + [fs(three["overall"]["configs"][c]["mean"]) for c in RETRIEVAL]]
    for axis, lbl, groups in AXES:
        rows.append([f"**{lbl}**"] + [""] * 4)
        for g in groups:
            cell = three["by_axis"][axis][g]
            rows.append([g, ncell(cell["n"])] + [fs(cell["configs"][c]["mean"]) if cell["n"] else "—" for c in RETRIEVAL])
    L += [md_table(["Group", "Common n"] + [f"{CODE[c]} mean" for c in RETRIEVAL], rows), ""]
    L += ["## 6. Findings and limitations", "", "Generated from the tables above; no interpretation beyond them.", ""]
    L += ["- Scoreable share among attempts: " + ", ".join(
        f"{CODE[c]} {pct(F['own'][c]['n_scoreable'], F['own'][c]['n_attempted'])}%" for c in RETRIEVAL) +
        " (unscored: " + ", ".join(str(F["own"][c]["n_unscored_attempts"]) for c in RETRIEVAL) + ")."]
    for a, b in RETRIEVAL_PAIRS:
        o = F["pairs"][pair_key(a, b)]["overall"]
        verdict = "CI excludes zero" if o["bootstrap"]["significant"] else "CI includes zero"
        L.append(f"- {pair_label(a, b)} on {o['n_common_scoreable']} common scoreable attempts: Δ {dfs(o['delta'])} {ci_f(o['bootstrap'])}, "
                 f"p {fmt_p(o['bootstrap']['p_value'])} — {verdict}.")
    for c in RETRIEVAL:
        o = F["by_outcome"][c]
        gh = o["grounded"]["hallucination"]
        L.append(f"- {CODE[c]}: correct answers mean {fs(o['correct']['mean'])} (n={o['correct']['n']}) vs hallucinations "
                 f"{fs(o['hallucination']['mean'])} (n={o['hallucination']['n']}); {gh['ungrounded']} of "
                 f"{gh['grounded'] + gh['ungrounded']} hallucinations ({pct(gh['ungrounded'], gh['grounded'] + gh['ungrounded'])}%) are ungrounded.")
    L.append(f"- All-three common set ({three['overall']['n']}): " + ", ".join(
        f"{CODE[c]} {fs(three['overall']['configs'][c]['mean'])}" for c in RETRIEVAL) + ".")
    L += ["", "Limitations:", ""] + _limitations_common()[:2] + [
        "- Faithfulness is computed only over attempted answers with an available judge score. It is undefined for C1 "
        "and conditional on attempting, which is itself a treatment outcome.",
        "- Higher faithfulness does not imply benchmark correctness. For C4, support is judged against condensed prose; "
        "this does not test whether the condenser preserved the retrieved statements faithfully.", ""]
    L += [_provenance(res, 1), ""]
    return "\n".join(L)


def render_readme(res: dict) -> str:
    n = res["cohorts"]["full"]["n"]
    L = ["# TEST-4,000 metric reports — reading guide", "",
         "A reproducible supplement documenting the sealed TEST run of the frozen configuration across four metrics, "
         "two populations and the dataset's three native attributes. Everything is computed offline from the sealed "
         "answers, the benchmark annotations and the stored judge output by `tools/build_metric_reports.py`; nothing is "
         "re-run, re-judged or re-retrieved. The thesis quotes only what its arguments need.", "",
         "## Files", "",
         "| file | what it is |", "|---|---|",
         "| `results.json` | every number, count, delta, interval and cohort membership hash, unrounded; timestamp-free — the record |",
         "| `manifest.json` | generation time, input/code/output SHA-256s, command line — the only volatile file |",
         "| `explorer.html` | self-contained interactive view of `results.json` (select population × metric × attribute × pair; colour scales); no network; no scoring or statistical inference is performed there, and every printed number string is pre-rendered in Python by the formatters of these reports |",
         "| `full_4000/hallucination_abstention.md` | hallucination and abstention over all questions; six pairs on identical id sets; outcome-transition matrices; attempt patterns |",
         "| `full_4000/f1_correctness.md` | correct % and F1 over all questions; six pairs; threshold sensitivity |",
         "| `full_4000/faithfulness.md` | conditional faithfulness with the scoreable share of the benchmark; coverage accounting |",
         "| `attempted/hallucination_abstention.md` | hallucination among own attempts; six pairs on common attempts; all-four common set |",
         "| `attempted/f1_correctness.md` | F1 and correct % among own attempts; pairs on common attempts; all-four common set |",
         "| `attempted/faithfulness.md` | scoreability among attempts; pairs on common scoreable attempts; faithfulness by outcome |", "",
         "## Three independent choices behind every table", "",
         "1. **Metric.** Benchmark hallucination (confident disagreement with the 2021 Mintaka gold — a factuality proxy), "
         "operational correctness (the scorer's CORRECT outcome at threshold 0.5), token/set F1 (0–100), or answer-to-supplied-context "
         "faithfulness (0–1, judge score). Abstention, Other and coverage are companions, never a fifth report.",
         "2. **Population.** `full_4000`: every question, pairs on the identical id set. `attempted`: each configuration's own "
         "attempts (CORRECT or HALLUCINATION; Other excluded) for overviews, and the INTERSECTION of two attempt sets for pairs, "
         "with both means recomputed there. Faithfulness: scoreable attempts (attempted and judge-scored), pairs on the common "
         "scoreable set; C1 is N/A.",
         "3. **Dataset attribute.** Answer type (`answer.answerType`, 5 groups), question complexity (`complexityType`, 9), topic "
         "category (`category`, 8). Three separate marginal partitions of the same questions — never crossed in the reports, never "
         "pooled as independent observations. Group order is the parser vocabulary order.", "",
         "## Units and rounding", "",
         "- Count-based percentages use the shared divide-first rule (`tools/document_run._pct1`: `round(count / n * 100, 1)`), "
         "so a printed digit here equals the digit in the run's `report.md`. F1 prints as mean × 100, one decimal; faithfulness "
         "means print with three decimals. Deltas are A − B on the unrounded values.",
         f"- `†` marks a cell with fewer than {TINY_N} questions; `n/a` marks an empty group (unavailable, not zero); `—` marks a value "
         "that does not exist (C1 faithfulness, an empty denominator).",
         "- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted) and exact McNemar on "
         "discordant pairs for binary outcomes; unpaired bootstrap only for the disjoint correct-vs-hallucination faithfulness "
         "contrast. Bootstrap p below the Monte Carlo resolution prints as `<.001`. Only OVERALL pair rows carry tests; every "
         "per-group pair delta is descriptive. Where a persisted sanctioned file already analysed the same population, the number "
         "here reproduces it (`results.json` → `precedent_checks`).", "",
         "## The faithfulness exception (both folders)", "",
         "There is no four-configuration, denominator-4,000 faithfulness mean. C1 has no context (N/A, not zero). C2–C4 means are "
         "conditional on a scoreable attempt; abstentions, Other and judge-unscored attempts are excluded, not zeroed, and the "
         "reasons are accounted for separately. The own-scoreable means are identical in the two folders because the conditional "
         "population is the same; the folders differ in what the scoreable SHARE is taken over (benchmark vs attempts). "
         "Pair means need both configurations attempted and scored on the same question; each score still judges its answer "
         "against its own supplied context.", "",
         "## Dataset composition (counts only)", "",
         "Answer type × complexity over the 4,000 test questions — shown so the near-deterministic crossings (`count` → numerical, "
         "`yesno` → boolean) and the one split class (`comparative`: entity vs boolean) are visible. Not a stratification axis.", ""]
    comp = res["composition_complexity_x_answer_type"]
    ats = AXES[0][2]
    rows = [[cx] + [str(comp[cx][at]) for at in ats] + [str(sum(comp[cx].values()))] for cx in AXES[1][2]]
    rows.append(["**total**"] + [str(sum(comp[cx][at] for cx in AXES[1][2])) for at in ats] + [str(n)])
    L += [md_table(["complexity \\ answer type"] + list(ats) + ["n"], rows), "",
          "Topic categories are exactly balanced across complexity classes in Mintaka (every category has the same complexity "
          "proportions), so that crossing carries no information and is not shown.", "",
          "## Cohorts", "",
          "Every population is a named cohort. Membership hashes (SHA-256 of the sorted question IDs) are retained in "
          "`results.json`; `tools/build_metric_reports.py --dump-ids PATH` writes the ID lists.", ""]
    rows = [[k, str(v["n"])] for k, v in res["cohorts"].items()]
    L += [md_table(["cohort", "n"], rows), "",
          "## Regeneration", "",
          f"```", f"{REGEN_CMD}", f"{CHECK_CMD}   # regenerate in memory and compare; exit 1 on any difference", "```", "",
          "Inputs (hashes in `manifest.json`): the sealed answer file (untracked, 215 MB; hash pinned in "
          "`data/results/test_runs/README.md`), `data/questions/mintaka_test_raw.json`, and "
          "`data/analysis/faithfulness_20260819_2017_mintaka_test_raw_raw.json`. Tests: `tests/unit/test_build_metric_reports.py`.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Explorer — a static page that only selects, filters and colours results.json;
# every number string it prints is produced above by display_strings()
# ---------------------------------------------------------------------------

EXPLORER_TEMPLATE = r"""<title>TEST-4000 Metric Explorer</title>
<style>
:root{--bg:#f4f6f9;--fg:#1a1f27;--muted:#5b6473;--line:#d3d9e2;--panel:#ffffff;--accent:#2f5d8a;--tiny:#8a5f00;
--good:#2e8b57;--bad:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#13161b;--fg:#e6eaf0;--muted:#98a2b3;--line:#323946;--panel:#1b2028;--accent:#8fb6e0;--tiny:#e0b354}}
:root[data-theme="dark"]{--bg:#13161b;--fg:#e6eaf0;--muted:#98a2b3;--line:#323946;--panel:#1b2028;--accent:#8fb6e0;--tiny:#e0b354}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;padding:1.25rem 1.5rem 2rem}
.eyebrow{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .35rem;font-variant-numeric:tabular-nums}
h1{font-size:1.35rem;margin:0 0 .3rem;text-wrap:balance;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 1rem;font-size:.9rem;max-width:70ch}
code{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:.9em}
.controls{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.75rem 1rem;margin-bottom:1rem}
.controls label{display:flex;flex-direction:column;gap:.2rem;font-size:.8rem;color:var(--muted)}
select{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.25rem .4rem;min-width:11rem}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.5rem}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:.3rem .55rem;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:.8rem;position:sticky;top:0;background:var(--panel)}
td.block{text-align:left;font-weight:600;color:var(--accent);background:transparent!important}
td.tiny::after{content:"†";color:var(--tiny);margin-left:.15rem}
td.na{color:var(--muted)}
.note{font-size:.85rem;color:var(--muted);margin:.5rem 0 0}
.legend{display:flex;gap:.5rem;align-items:center;font-size:.8rem;color:var(--muted);margin:.5rem 0}
.legend span.bar{display:inline-block;width:8rem;height:.7rem;border-radius:3px;background:linear-gradient(90deg,#2e8b57,#e8d44d,#c0392b)}
</style>
<p class="eyebrow" id="eyebrow"></p>
<h1>TEST-4,000 metric explorer</h1>
<p class="sub">Displays <code>results.json</code> of the sealed run. No scoring or statistical inference is performed here: every value is copied from the record, and every printed number string was pre-rendered by the same Python formatters as the six Markdown reports beside this file, which are the citable record. Colour is per table, on the selected metric, relative to that table's range.</p>
<div class="controls">
 <label>View<select id="view"></select></label>
 <label>Population<select id="pop"></select></label>
 <label>Metric<select id="metric"></select></label>
 <label>Attribute<select id="axis"></select></label>
 <label>Pair (A − B)<select id="pair"></select></label>
 <label>Colour<select id="colour"><option value="on">on</option><option value="off">off</option></select></label>
</div>
<div class="legend"><span>lower</span><span class="bar" id="bar"></span><span>higher</span><span id="legendnote"></span></div>
<div class="wrap" id="out"></div>
<p class="note" id="note"></p>
<script>
const R = __RESULTS__;
const D = __DISPLAY__;   // display strings, pre-rendered in Python; same shape as the subtrees of R the page reads
const CODES = Object.fromEntries(R.configs.map(c=>[c.key,c.code]));
const CONFIGS = R.configs.map(c=>c.key), RETR = CONFIGS.slice(1);
const OUT = R.outcomes;
const TINY = R.settings.tiny_n;
const METRICS = {
  hallucination:{label:"Hallucination %",unit:"%",higherBetter:false,faith:false},
  abstention:{label:"Abstention %",unit:"%",higherBetter:null,faith:false},
  correct:{label:"Correct %",unit:"%",higherBetter:true,faith:false},
  f1:{label:"F1 (0–100)",unit:"",higherBetter:true,faith:false},
  faithfulness:{label:"Faithfulness (0–1)",unit:"",higherBetter:true,faith:true},
};
const VIEWS = {
  configs:{label:"Configurations × attribute",pops:["full","attempted","all_four","faith_own","faith_all_three"]},
  pairs:{label:"Pair A − B × attribute",pops:["full","attempted_common","faith_common"]},
  transitions:{label:"Outcome transitions (pair)",pops:["full"]},
  patterns:{label:"Attempt patterns",pops:["full"]},
  by_outcome:{label:"Faithfulness by outcome",pops:["faith_own"]},
};
const POPS = {
  full:"Full benchmark (n = 4,000)", attempted:"Own attempts", all_four:"Common attempts, all four",
  attempted_common:"Common attempts (pair)", faith_own:"Faithfulness: own scoreable attempts",
  faith_all_three:"Faithfulness: common scoreable, C2–C4", faith_common:"Faithfulness: common scoreable (pair)",
};
const $ = id=>document.getElementById(id);
function opts(sel, items, keep){ const prev=sel.value; sel.innerHTML=""; for(const [v,l] of items){const o=document.createElement("option");o.value=v;o.textContent=l;sel.appendChild(o);} if(keep&&items.some(i=>i[0]===prev)) sel.value=prev; }
// ---- no number formatting here: every printed string comes from D (Python). Values are read from R only for colour. ----
function metricOf(agg, m){ // value on the 0-1 scale from an outcome aggregate (colour scale only)
  if(!agg||!agg.n) return null;
  if(m==="f1") return agg.f1_mean;
  return agg.rates[m];
}
// ---- colour ----
function mix(t){ // 0..1 -> green..yellow..red
  const stops=[[46,139,87],[232,212,77],[192,57,43]]; const s=t<.5? [stops[0],stops[1],t*2] : [stops[1],stops[2],(t-.5)*2];
  const c=s[0].map((v,i)=>Math.round(v+(s[1][i]-v)*s[2])); return `rgba(${c[0]},${c[1]},${c[2]},0.28)`; }
function colourCells(cells, higherBetter){
  if($("colour").value==="off"||higherBetter===null) return;
  const vals=cells.map(c=>c.v).filter(v=>v!=null); if(vals.length<2) return;
  const lo=Math.min(...vals), hi=Math.max(...vals); if(hi===lo) return;
  for(const c of cells){ if(c.v==null) continue; let t=(c.v-lo)/(hi-lo); if(higherBetter) t=1-t; c.td.style.background=mix(t); }
}
function td(text, cls){ const e=document.createElement("td"); e.textContent=text; if(cls) e.className=cls; return e; }
function table(headers, rows){ // rows: arrays of {text, cls, v?}
  const t=document.createElement("table"); const tr=document.createElement("tr");
  for(const h of headers){const th=document.createElement("th");th.textContent=h;tr.appendChild(th);} t.appendChild(tr);
  const cells=[];
  for(const r of rows){ const trr=document.createElement("tr"); for(const c of r){ const e=td(c.text,c.cls); if(c.v!==undefined){cells.push({td:e,v:c.v});} trr.appendChild(e);} t.appendChild(trr); }
  return {t,cells};
}
const ncell=n=> ({text:n===0?"n/a":String(n), cls:n===0?"na":(n<TINY?"tiny":"")});
const block=label=>[{text:label,cls:"block"}];
// ---- views ----
function axisGroups(axis){ return axis==="overall"? null : R.axes.find(a=>a.key===axis).groups; }
function viewConfigs(pop, m, axis){
  const M=METRICS[m]; const rows=[]; const cfgs = M.faith? RETR : CONFIGS;
  const headers=["Group","n",...cfgs.map(c=>CODES[c]+(pop==="attempted"?" (att. n)":""))];
  function rowFor(label, n, aggOf){ // aggOf(c) -> [value node from R, display node from D]
    const r=[{text:label},ncell(n)];
    for(const c of cfgs){ const [a,s]=aggOf(c);
      if(M.faith){ const ns = a ? (a.n_scoreable ?? a.n) : 0; if(!a||!ns||!s){r.push({text:"—",cls:"na",v:null});continue;} r.push({text:s.mean+" (n="+ns+")",v:a.mean, cls:ns<TINY?"tiny":""}); }
      else { if(!a||!a.n||!s){r.push({text:"—",cls:"na",v:null});continue;} let t=s[m]; if(pop==="attempted") t+=" ("+a.n+")"; r.push({text:t,v:metricOf(a,m), cls:(pop!=="full"&&a.n<TINY)?"tiny":""}); } }
    return r; }
  let src, note=""; // src functions take the root (R for values, D for strings) so both are read by one path
  if(pop==="full"){ src={overall:(X,g)=>X.full.overall[g], axis:(X,ax,g)=>X.full.by_axis[ax][g]}; note="All questions in the group; each configuration on the identical set."; }
  else if(pop==="attempted"){ src={overall:(X,g)=>X.attempted.overall[g], axis:(X,ax,g)=>X.attempted.by_axis[ax][g]}; note="Each configuration's OWN attempts (different denominators per column, shown in brackets). Abstention is not defined here."; }
  else if(pop==="all_four"){ src={overall:(X,g)=>X.attempted.all_four.overall.configs[g], axis:(X,ax,g)=>X.attempted.all_four.by_axis[ax][g]}; note="Questions attempted by all four configurations (n = "+R.attempted.all_four.overall.n+")."; }
  else if(pop==="faith_own"){ src={overall:(X,g)=>X.faithfulness.own[g], axis:(X,ax,g)=>X.faithfulness.by_axis[ax][g]}; note="Conditional on a scoreable attempt; C1 is N/A. Means on different sets per column."; }
  else { src={overall:(X,g)=>X.faithfulness.all_three.overall.configs[g], axis:(X,ax,g)=>X.faithfulness.all_three.by_axis[ax][g]}; note="Common scoreable set of C2–C4 (n = "+R.faithfulness.all_three.overall.n+")."; }
  const nOverall = pop==="full"?R.cohorts.full.n : pop==="all_four"?R.attempted.all_four.overall.n : pop==="faith_all_three"?R.faithfulness.all_three.overall.n : R.cohorts.full.n;
  const groups=axisGroups(axis);
  if(!groups){ rows.push(rowFor("overall", nOverall, c=>[src.overall(R,c), src.overall(D,c)])); }
  else { for(const g of groups){ const cell=src.axis(R,axis,g), dcell=src.axis(D,axis,g); rows.push(rowFor(g, cell.n, c=>[cell.configs[c], dcell.configs[c]])); } }
  const {t,cells}=table(headers,rows); colourCells(cells,M.higherBetter); return {t,note};
}
function viewPairs(pop, m, axis, pk){
  const M=METRICS[m]; const P=R.pairs.find(p=>p.key===pk); const A=CODES[P.a],B=CODES[P.b];
  let src, note, headers, rows=[];
  if(pop==="faith_common"){
    const e=R.faithfulness.pairs[pk], de=D.faithfulness.pairs[pk]; if(e.available===false||!de){ return {t:Object.assign(document.createElement("p"),{textContent:"N/A — "+e.reason}), note:""}; }
    headers=["Group","Common scoreable n","Mean "+A,"Mean "+B,"Δ","95% CI","p"];
    const row=(label,o,s)=>{ if(!o.available||!s) return [{text:label},ncell(0),{text:"—"},{text:"—"},{text:"—"},{text:"—"},{text:"—"}];
      return [{text:label},ncell(o.n_common_scoreable),{text:s.mean_a},{text:s.mean_b},{text:s.delta,v:o.delta},{text:s.ci},{text:s.p}]; };
    rows.push(row("overall",e.overall,de.overall)); const groups=axisGroups(axis); if(groups){ for(const g of groups) rows.push(row(g,e.by_axis[axis][g],de.by_axis[axis][g])); }
    note="Both configurations attempted and were judge-scored on the same question; each answer judged against its own context. Tests on the overall row only.";
  } else {
    const isFull = pop==="full"; const e = isFull? R.full.pairs[pk] : R.attempted.pairs[pk]; const de = isFull? D.full.pairs[pk] : D.attempted.pairs[pk];
    if(M.faith){ return {t:Object.assign(document.createElement("p"),{textContent:"Faithfulness pairs live under the population “Faithfulness: common scoreable (pair)”."}), note:""}; }
    if(!isFull && m==="abstention"){ return {t:Object.assign(document.createElement("p"),{textContent:"Abstention is not defined within attempted answers."}), note:""}; }
    const key = m==="correct"&&!isFull ? "correct" : m;
    headers=["Group", isFull?"n":"Common n", ...(isFull?[]:["Retained %"]), M.label+" "+A, M.label+" "+B, "Δ "+(M.unit==="%"?"pp":""), "95% CI","p","McNemar p"];
    const row=(label,o,s)=>{ if(!o.available||!s) return [{text:label},ncell(0),...(isFull?[]:[{text:"—"}]),{text:"—"},{text:"—"},{text:"—"},{text:"—"},{text:"—"},{text:"—"}];
      const ent=o[key]; const st=s.stats[key]||{};
      return [{text:label}, ncell(isFull?o.n:o.n_common), ...(isFull?[]:[{text:s.retained_share}]),
        {text:s.a?s.a[m]:"—"},{text:s.b?s.b[m]:"—"},{text:st.delta??"—",v:ent?ent.delta:null},
        {text:st.ci??"—"},{text:st.p??"—"},{text:st.mcnemar_p??"—"}]; };
    rows.push(row("overall",e.overall,de.overall)); const groups=axisGroups(axis); if(groups){ for(const g of groups) rows.push(row(g,e.by_axis[axis][g],de.by_axis[axis][g])); }
    note = isFull? "Identical id sets for both configurations. Tests on the overall row only; per-group deltas are descriptive."
                 : "Intersection of the two attempt sets; both means recomputed there. Within attempts correct % is the complement of hallucination % (same test). Tests on the overall row only.";
  }
  const {t,cells}=table(headers,rows); colourCells(cells, M.higherBetter===null?null:M.higherBetter); return {t,note};
}
function viewTransitions(pk){
  const P=R.pairs.find(p=>p.key===pk); const T=R.full.transitions[pk], DT=D.full.transitions[pk]; const A=CODES[P.a],B=CODES[P.b];
  const headers=[B+" outcome ↓ / "+A+" outcome →","row n",...OUT.map(o=>A+" "+o)]; const rows=[];
  for(const ob of OUT){ const rn=OUT.reduce((s,oa)=>s+T.matrix[ob][oa],0); const r=[{text:B+" "+ob},{text:String(rn)}];
    for(const oa of OUT){ r.push({text:`${T.matrix[ob][oa]} (${DT.matrix[ob][oa]}%)`, v: rn? T.matrix[ob][oa]/rn : null}); } rows.push(r); }
  const {t,cells}=table(headers,rows); colourCells(cells,true);
  return {t, note:"Rows: outcome of the reference configuration B; columns: what A did on the same questions; row share in brackets. Descriptive, not causal."};
}
function viewPatterns(){
  const rows=Object.entries(R.full.attempt_patterns).map(([k,v])=>[{text:k},{text:String(v)},{text:D.full.attempt_patterns[k],v:v}]);
  const {t,cells}=table(["Configurations that attempted","n","% of 4,000"],rows); colourCells(cells,true);
  return {t,note:"Mutually exclusive patterns that partition the benchmark — coverage overlap, not accuracy."};
}
function viewByOutcome(){
  const rows=[]; for(const c of RETR){ const o=R.faithfulness.by_outcome[c], s=D.faithfulness.by_outcome[c]; const gc=o.grounded.correct, gh=o.grounded.hallucination;
    rows.push([{text:CODES[c]},{text:String(o.correct.n)},{text:s.correct_mean,v:o.correct.mean},{text:String(o.hallucination.n)},{text:s.hallucination_mean,v:o.hallucination.mean},
      {text:s.delta},{text:s.ci},{text:s.p},
      {text:`${gc.ungrounded} (${s.ungrounded_correct_pct}%)`},{text:`${gh.ungrounded} (${s.ungrounded_hallucination_pct}%)`}]); }
  const {t,cells}=table(["Config","Correct n","mean","Hallucination n","mean","Δ corr − hall","95% CI","p","Ungrounded among correct","Ungrounded among hallucinations"],rows);
  colourCells(cells,true); return {t,note:"Disjoint groups within one configuration (unpaired bootstrap). Grounded := score ≥ "+R.settings.grounded_threshold+"."};
}
function render(){
  const view=$("view").value; const V=VIEWS[view];
  opts($("pop"), V.pops.map(p=>[p,POPS[p]]), true);
  const pop=$("pop").value;
  const faithPop = pop.startsWith("faith");
  const mItems = faithPop? [["faithfulness",METRICS.faithfulness.label]] : Object.entries(METRICS).filter(([k])=>k!=="faithfulness" && !(pop!=="full"&&k==="abstention")).map(([k,v])=>[k,v.label]);
  opts($("metric"), mItems, true);
  opts($("axis"), [["overall","overall only"],...R.axes.map(a=>[a.key,a.label])], true);
  const pairItems = pop==="faith_common"? R.pairs.filter(p=>p.a!=="base_llm_abstain"&&p.b!=="base_llm_abstain") : R.pairs;
  opts($("pair"), pairItems.map(p=>[p.key,p.label]), true);
  const needPair = view==="pairs"||view==="transitions"; $("pair").disabled=!needPair; $("axis").disabled=!(view==="configs"||view==="pairs"); $("metric").disabled=!(view==="configs"||view==="pairs");
  const m=$("metric").value, axis=$("axis").value, pk=$("pair").value;
  let out;
  if(view==="configs") out=viewConfigs(pop,m,axis); else if(view==="pairs") out=viewPairs(pop,m,axis,pk); else if(view==="transitions") out=viewTransitions(pk); else if(view==="patterns") out=viewPatterns(); else out=viewByOutcome();
  const M=METRICS[m]; $("legendnote").textContent = (view==="configs"||view==="pairs")&&M ? (M.higherBetter===null? " (no colour: abstention has no direction)" : M.higherBetter? " (green = higher "+M.label+")" : " (green = lower "+M.label+")") : "";
  $("out").innerHTML=""; $("out").appendChild(out.t); $("note").textContent=out.note+"  † = fewer than "+TINY+" questions.";
}
$("eyebrow").textContent = "Sealed run " + R.inputs.raw.path.split("/").pop().replace("_raw.json","") + " · n = " + R.cohorts.full.n.toLocaleString("en") + " · answers sha256 " + R.inputs.raw.sha256.slice(0,12) + "… · " + R.settings.n_resamples.toLocaleString("en") + " bootstrap resamples, seed " + R.settings.seed;
opts($("view"), Object.entries(VIEWS).map(([k,v])=>[k,v.label]));
for(const id of ["view","pop","metric","axis","pair","colour"]) $(id).addEventListener("change", render);
render();
</script>
"""


def render_explorer(res: dict) -> str:
    """The results blob verbatim (it IS results.json) plus the Python-rendered display strings."""
    def blob(obj):
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return EXPLORER_TEMPLATE.replace("__RESULTS__", blob(res)).replace("__DISPLAY__", blob(display_strings(res)))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_all(res: dict) -> dict[str, str]:
    """Relative path -> text, for every deterministic file."""
    return {
        "README.md": render_readme(res),
        "full_4000/hallucination_abstention.md": render_full_hallucination(res),
        "full_4000/f1_correctness.md": render_full_f1(res),
        "full_4000/faithfulness.md": render_full_faith(res),
        "attempted/hallucination_abstention.md": render_attempted_hallucination(res),
        "attempted/f1_correctness.md": render_attempted_f1(res),
        "attempted/faithfulness.md": render_attempted_faith(res),
        "explorer.html": render_explorer(res),
    }


def results_text(res: dict) -> str:
    return json.dumps(res, indent=1, ensure_ascii=False) + "\n"


def write_outputs(out: Path, res: dict, files: dict[str, str], argv: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_4000").mkdir(exist_ok=True)
    (out / "attempted").mkdir(exist_ok=True)
    written: dict[str, str] = {}
    rt = results_text(res)
    (out / "results.json").write_text(rt, encoding="utf-8", newline="\n")
    written["results.json"] = hashlib.sha256(rt.encode("utf-8")).hexdigest()
    for rel, text in files.items():
        (out / rel).write_text(text, encoding="utf-8", newline="\n")
        written[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(argv),
        "python": platform.python_version(),
        "schema": SCHEMA,
        "inputs": res["inputs"],
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()},
        "settings": res["settings"],
        "outputs_sha256": written,
        "precedent_checks": {"matched": sum(1 for c in res["precedent_checks"] if c.get("match") is True),
                             "mismatched": sum(1 for c in res["precedent_checks"] if c.get("match") is False),
                             "not_applicable": sum(1 for c in res["precedent_checks"] if c.get("match") is None)},
        "note": "The only volatile file. results.json, the reports and explorer.html are byte-identical across "
                "regenerations with the same inputs and code.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    return manifest


def check_against(out: Path, res: dict, files: dict[str, str]) -> list[str]:
    """Names of deterministic files that differ from what `out` holds (missing counts as differing)."""
    diffs = []
    expected = {"results.json": results_text(res), **files}
    for rel, text in expected.items():
        p = out / rel
        if not p.exists() or p.read_text(encoding="utf-8") != text:
            diffs.append(rel)
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--faithfulness", default=str(DEFAULT_FAITH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                    help="bootstrap resamples (tests only; the tracked output is produced at the default)")
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and compare with --out; exit 1 on any difference, write nothing")
    ap.add_argument("--dump-ids", default=None, metavar="PATH",
                    help="also write every cohort's sorted question ids as JSON to PATH (for auditing)")
    ap.add_argument("--allow-precedent-mismatch", action="store_true",
                    help="record precedent mismatches instead of aborting (never for the tracked output)")
    args = ap.parse_args(argv)
    if args.resamples <= 0:
        ap.error("--resamples must be positive")

    raw, gold, faith, out = Path(args.raw), Path(args.gold), Path(args.faithfulness), Path(args.out)
    for p in (raw, gold, faith):
        if not p.exists():
            sys.exit(f"not found: {p}")
    print(f"loading + scoring {raw.name} …", flush=True)
    inp = load_inputs(raw, gold, faith)
    print(f"computing ({args.resamples:,} resamples) …", flush=True)
    res = compute(inp, resamples=args.resamples)
    bad = [c for c in res["precedent_checks"] if c.get("match") is False]
    for c in bad:
        print(f"PRECEDENT MISMATCH [{c['source']}] {c['check']}: expected {c['expected']!r}, got {c['observed']!r}")
    if bad and not args.allow_precedent_mismatch:
        sys.exit(f"{len(bad)} precedent check(s) failed — refusing to write. Explain the discrepancy first.")
    files = render_all(res)

    if args.dump_ids:
        cohorts = build_cohorts(inp)
        Path(args.dump_ids).write_text(json.dumps({k: {"n": len(v), "sha256": sha256_ids(v), "ids": v}
                                                   for k, v in cohorts.items()}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote cohort ids to {args.dump_ids}")

    if args.check:
        diffs = check_against(out, res, files)
        if diffs:
            print("DIFFERS from tracked output: " + ", ".join(diffs))
            return 1
        print(f"OK — {len(files) + 1} deterministic files identical to {out}")
        return 0

    manifest = write_outputs(out, res, files, sys.argv if argv is None else ["build_metric_reports.py", *argv])
    print(f"wrote {len(files) + 2} files to {out}")
    print(f"precedent checks: {manifest['precedent_checks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
