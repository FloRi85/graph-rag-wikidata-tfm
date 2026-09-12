"""
Faithfulness statistics over a STORED faithfulness JSON — the sanctioned source
for every faithfulness comparison the thesis quotes.

Three tables, three different constructs — do not mix them in prose:

1. OPERATING-POINT MEANS — each config's mean over its OWN scoreable set.
   These are the headline 0.723 / 0.616 / 0.859 numbers. They are descriptive
   marginals on different question sets and CANNOT be tested against each
   other directly.
2. PAIRED CONTRASTS — config pairs on the COMMON scoreable set (both configs
   attempted the question and the judge scored both). Seeded paired bootstrap
   CI + p. These subsets are conditional and post-treatment-selected (chosen
   by the configs' own attempt behaviour), so they support "on the questions
   both answered" claims only.
3. CORRECT-vs-WRONG within one config — unpaired by construction (a question
   is in exactly one group), so the interval comes from a two-sample
   bootstrap. This is the CI behind any "wrong answers are less grounded"
   sentence.

Offline by construction: reads the stored JSON, makes zero API calls.
Persists a deterministic JSON (input sha256, seed, resamples, results) to
data/analysis/faithfulness_comparison_{stem}.json — the thesis copies from
that file, not from stdout scrollback.

Usage (from repo root):
    venv/Scripts/python tools/compare_faithfulness.py            # TEST default
    venv/Scripts/python tools/compare_faithfulness.py data/analysis/faithfulness_<run>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.faithfulness import RETRIEVAL_CONFIGS
from src.eval.metrics import CORRECT, HALLUCINATION, _CONFIG_LABELS
from src.eval.stats import (
    DEFAULT_RESAMPLES, DEFAULT_SEED, compare_configs, unpaired_bootstrap,
)

ANALYSIS_DIR = ROOT / "data" / "analysis"
DEFAULT_INPUT = ANALYSIS_DIR / "faithfulness_20260819_2017_mintaka_test_raw_raw.json"

# Fixed pair order so the persisted JSON is stable across invocations.
_PAIRS = (("graph_rag", "rag"), ("rerank", "rag"), ("rerank", "graph_rag"))


def fmt_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}"


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    # Duplicate (config, id) pairs would silently double-weight a question in
    # every mean below. That is a corrupt input, not a judgement call — fatal.
    seen: set[tuple[str, str]] = set()
    for r in records:
        key = (r.get("config", ""), r.get("id", ""))
        if key in seen:
            sys.exit(f"duplicate (config, id) record: {key} — refusing to score "
                     f"a file that double-counts a question.")
        seen.add(key)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("faithfulness_file", nargs="?", default=str(DEFAULT_INPUT),
                    help=f"Stored faithfulness JSON (default: {DEFAULT_INPUT.name})")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = ap.parse_args()

    path = Path(args.faithfulness_file)
    if not path.exists():
        sys.exit(f"not found: {path}")
    records = load_records(path)
    src_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    scored = {c: {r["id"]: r for r in records
                  if r["config"] == c and r["score"] is not None}
              for c in RETRIEVAL_CONFIGS}

    # ---- 1. operating-point means ------------------------------------------
    print("=" * 78)
    print("1. OPERATING-POINT MEANS  — each config over its OWN scoreable set")
    print("   (descriptive marginals on DIFFERENT question sets; not testable")
    print("    against each other — that is what section 2 exists for)")
    print("=" * 78)
    marginals: dict[str, dict] = {}
    print(f"{'Config':<20}{'n scoreable':>12}{'mean':>9}")
    for c in RETRIEVAL_CONFIGS:
        vals = [r["score"] for r in scored[c].values()]
        mean = sum(vals) / len(vals) if vals else None
        marginals[c] = {"n": len(vals), "mean": mean}
        print(f"{_CONFIG_LABELS.get(c, c):<20}{len(vals):>12}"
              + (f"{mean:>9.3f}" if mean is not None else f"{'—':>9}"))

    # ---- 2. paired contrasts on common scoreable sets ----------------------
    print()
    print("=" * 78)
    print("2. PAIRED CONTRASTS  — common scoreable set per pair (both configs")
    print("   attempted AND the judge scored both; conditional,")
    print("   post-treatment-selected subsets)")
    print("=" * 78)
    pairs: list[dict] = []
    print(f"{'Pair (A − B)':<36}{'n common':>9}{'delta':>9}{'95% CI':>20}{'p':>8}")
    for a, b in _PAIRS:
        r = compare_configs(scored[a], scored[b], lambda s: s["score"],
                            n_resamples=args.resamples)
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        pair_label = (f"{_CONFIG_LABELS.get(a, a)} − {_CONFIG_LABELS.get(b, b)}")
        print(f"{pair_label:<36}{r['n']:>9}{r['delta']:>+9.3f}{ci:>20}"
              f"{fmt_p(r['p_value']):>8}")
        pairs.append({"pair": [a, b], "bootstrap": r})

    # ---- 3. correct vs wrong within each config ----------------------------
    print()
    print("=" * 78)
    print("3. CORRECT vs WRONG within each config  — UNPAIRED two-sample")
    print("   bootstrap (disjoint groups by construction)")
    print("=" * 78)
    by_outcome: list[dict] = []
    print(f"{'Config':<20}{'n corr':>7}{'mean':>8}{'n wrong':>8}{'mean':>8}"
          f"{'delta':>8}{'95% CI':>20}{'p':>8}")
    for c in RETRIEVAL_CONFIGS:
        corr = [r["score"] for r in scored[c].values() if r["outcome"] == CORRECT]
        wrong = [r["score"] for r in scored[c].values()
                 if r["outcome"] == HALLUCINATION]
        r = unpaired_bootstrap(corr, wrong, n_resamples=args.resamples)
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        mc = sum(corr) / len(corr) if corr else None
        mw = sum(wrong) / len(wrong) if wrong else None
        print(f"{_CONFIG_LABELS.get(c, c):<20}{len(corr):>7}"
              + (f"{mc:>8.3f}" if mc is not None else f"{'—':>8}")
              + f"{len(wrong):>8}"
              + (f"{mw:>8.3f}" if mw is not None else f"{'—':>8}")
              + f"{r['delta']:>+8.3f}{ci:>20}{fmt_p(r['p_value']):>8}")
        by_outcome.append({"config": c, "n_correct": len(corr),
                           "mean_correct": mc, "n_wrong": len(wrong),
                           "mean_wrong": mw, "unpaired_bootstrap": r})

    print("\n  Intervals are nominal pointwise 95% CIs, unadjusted for multiple")
    print("  comparisons; p-values are two-sided bootstrap tail estimates with")
    print(f"  finite Monte Carlo resolution (smallest positive value "
          f"2/{args.resamples:,}).")

    # ---- persist ------------------------------------------------------------
    payload = {
        "input_file": path.name,
        "input_sha256": src_sha,
        "seed": DEFAULT_SEED,
        "n_resamples": args.resamples,
        "p_value_kind": ("two-sided percentile-bootstrap tail estimate; "
                         "resolution 2/n_resamples; CIs nominal pointwise 95%, "
                         "unadjusted"),
        "operating_point_means": marginals,
        "paired_common_scoreable": pairs,
        "correct_vs_wrong_unpaired": by_outcome,
    }
    stem = path.stem
    if stem.startswith("faithfulness_"):
        stem = stem[len("faithfulness_"):]   # avoid faithfulness_comparison_faithfulness_…
    out_path = ANALYSIS_DIR / f"faithfulness_comparison_{stem}.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:          # ANALYSIS_DIR redirected outside the repo (tests)
        shown = out_path
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
