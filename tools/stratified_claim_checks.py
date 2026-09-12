"""
Exploratory stratified hallucination comparisons against C1.

Three cuts, all paired against C1 on the same question subsets via the seeded
paired bootstrap:

  1. By Mintaka complexity and category — does "retrieval cuts hallucinations
     vs C1" hold in every stratum?
  2. Questions with a gold answer form detected or not detected in each
     retrieval configuration's supplied context.
  3. Within C3's not-detected group, by complexity.

Result on the TEST-4000 run of record (2026-08-21): the vs-C1 hallucination
rate is lower in all 51 comparisons (3 configurations × 9 complexity groups
and 3 × 8 topic categories). Reductions are larger in the not-detected
groups (C2 −24.8 / C3 −22.3 / C4 −17.8 percentage points) than in detected
groups (−14.0 / −9.3 / −10.1 points).

Gold-form presence is a surface-form proxy, not evidence sufficiency. The
groups are configuration-specific and defined after retrieval or condensing;
their differences do not isolate calibration or a causal effect of context.
Tests are exploratory and unadjusted for multiple comparisons. Headline
numbers stay with tools/report.py; see docs/reproduction.md.

Usage (from repo root, after tools/context_gold_coverage.py has produced the
joined table for the run):
    venv\\Scripts\\python tools/stratified_claim_checks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import (  # noqa: E402
    ABSTENTION, CORRECT, HALLUCINATION, per_question_scores,
)
from src.eval.stats import compare_configs, hallucination_key  # noqa: E402
from src.eval.gold_source import resolve_gold_source  # noqa: E402

DEFAULT_RAW = (ROOT / "data/results/test_runs"
               / "20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw"
               / "20260819_2017_mintaka_test_raw_raw.json")

C1 = "base_llm_abstain"
CONFIGS = {"rag": "C2", "graph_rag": "C3", "rerank": "C4"}


def rate(sub: dict, outcome: str) -> float:
    n = len(sub)
    return 100 * sum(1 for s in sub.values() if s["outcome"] == outcome) / n if n else 0.0


def delta_line(scores, cfg, qids, label):
    a = {q: scores[cfg][q] for q in qids}
    b = {q: scores[C1][q] for q in qids}
    r = compare_configs(a, b, hallucination_key)
    star = "***" if r["p_value"] < .001 else ("*" if r["significant"] else "n.s.")
    return (f"    {label:<26} n={len(qids):>5}  "
            f"halluc {CONFIGS[cfg]} {rate(a, HALLUCINATION):5.1f} vs C1 {rate(b, HALLUCINATION):5.1f}  "
            f"Δ {100*r['delta']:+6.1f} [{100*r['ci_low']:+6.1f},{100*r['ci_high']:+6.1f}] "
            f"p={r['p_value']:.3f} {star}   "
            f"(abst {CONFIGS[cfg]} {rate(a, ABSTENTION):4.1f} / C1 {rate(b, ABSTENTION):4.1f}; "
            f"corr {CONFIGS[cfg]} {rate(a, CORRECT):4.1f} / C1 {rate(b, CORRECT):4.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default=str(DEFAULT_RAW),
                    help="A run's raw results file (default: the TEST-4000 run of record)")
    ap.add_argument("--joined", default=None,
                    help="Joined table from tools/context_gold_coverage.py "
                         "(default: data/analysis/context_gold_coverage_<raw stem>.json)")
    args = ap.parse_args()
    raw_path = Path(args.raw)
    joined_path = Path(args.joined) if args.joined else (
        ROOT / "data" / "analysis" / f"context_gold_coverage_{raw_path.stem}.json")
    if not joined_path.exists():
        sys.exit(f"{joined_path} not found — run tools/context_gold_coverage.py "
                 f"{raw_path} first.")

    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    golds, qbi, _ = resolve_gold_source({r["id"] for r in rows}, verbose=False)
    scores = per_question_scores(rows, qbi, golds=golds)
    joined = json.loads(joined_path.read_text(encoding="utf-8"))
    meta = {r["id"]: (r.get("complexity", ""), r.get("type", "")) for r in rows}
    all_qids = sorted(scores[C1].keys())

    # ---- 1. vs-C1 hallucination delta per complexity / category ----------
    for field_ix, field_name in ((0, "complexity"), (1, "category")):
        print("=" * 110)
        print(f"1{'ab'[field_ix]}. C1-PAIRED HALLUCINATION DELTA BY {field_name.upper()}"
              f"  (Δ = config − C1; negative = lower hallucination rate)")
        print("=" * 110)
        keys = sorted({meta[q][field_ix] for q in all_qids})
        for cfg in CONFIGS:
            print(f"  {CONFIGS[cfg]}:")
            for k in keys:
                qids = [q for q in all_qids if meta[q][field_ix] == k]
                print(delta_line(scores, cfg, qids, k))
            print()

    # ---- 2. gold-form not-detected vs detected groups --------------------
    print("=" * 110)
    print("2. GOLD-FORM PRESENCE GROUPS  (defined separately for each configuration)")
    print("   Undefined checks (Boolean answers/no usable gold form) are excluded.")
    print("=" * 110)
    for cfg in CONFIGS:
        flags = joined[cfg]
        scoreable = [q for q in all_qids if flags.get(q, {}).get("in_ctx") is not None]
        miss = [q for q in scoreable if flags[q]["in_ctx"] is False]
        hit = [q for q in scoreable if flags[q]["in_ctx"] is True]
        print(f"  {CONFIGS[cfg]}:")
        print(delta_line(scores, cfg, miss, "form not detected"))
        print(delta_line(scores, cfg, hit, "form detected"))
        print(delta_line(scores, cfg, scoreable, "all eligible"))
        print()

    # ---- 3. within C3's not-detected group, by complexity -----------------
    print("=" * 110)
    print("3. WITHIN THE NOT-DETECTED GROUP, BY COMPLEXITY (C3; groups with n>=50)")
    print("=" * 110)
    flags = joined["graph_rag"]
    miss = [q for q in all_qids if flags.get(q, {}).get("in_ctx") is False]
    for k, _ in Counter(meta[q][0] for q in miss).most_common():
        qids = [q for q in miss if meta[q][0] == k]
        if len(qids) >= 50:
            print(delta_line(scores, "graph_rag", qids, k))


if __name__ == "__main__":
    main()
