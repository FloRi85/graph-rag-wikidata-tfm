"""
Compare the SAME config across two runs, paired by question id.

`tools/report.py` compares configs WITHIN one run against a baseline. That is
the wrong shape for every arm comparison this project has left: the k-sweep asks
whether C3 at k=30 beats C3 at k=10, the allocation arm asks whether `floor`
moves C2/C3/C4 against `global`, and the entity-block variant asks the same of
its two prompt modes. All three are one config, two runs, same 200 questions.

⚠️ IT SCORES BOTH RUNS ITSELF, through `gold_source` and the shared scorer,
rather than reading numbers out of two `report.md` files. Two reports generated
at different times can sit under different scorer versions; scoring here means
a difference between the arms cannot be an artifact of when each was documented.

⚠️ PAIRING IS BY QUESTION ID AND NON-OVERLAP IS REPORTED, never silently
dropped. `n_dropped > 0` means the two runs did not see the same questions, and
a paired test over a shifting cohort is not a paired test.

Usage (from repo root):
    venv/Scripts/python tools/compare_runs.py RUN_A.json RUN_B.json
    venv/Scripts/python tools/compare_runs.py A.json B.json --metric hallucination
    venv/Scripts/python tools/compare_runs.py A.json B.json --label-a k30 --label-b k10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import (_CONFIG_LABELS, failed_configs,          # noqa: E402
                              per_question_scores)
from src.eval.gold_source import resolve_gold_source                  # noqa: E402
from src.eval import stats                                            # noqa: E402

METRICS = {
    "f1": (stats.f1_key, "F1", 100.0),
    "correct": (stats.correct_key, "correct %", 100.0),
    "hallucination": (stats.hallucination_key, "hallucination %", 100.0),
    "abstention": (stats.abstention_key, "abstention %", 100.0),
}


def load_scored(path: Path, questions: str | None) -> tuple[dict, dict]:
    """Scored per-question records plus per-config infrastructure-failure counts.

    ⚠️ Scored with PER-CONFIG exclusion (2026-08-18): this tool pairs ONE
    config across two runs, so a question is dropped only where the compared
    config itself failed. The default all-or-nothing rule — right for a
    within-run table, where four columns need one denominator — silently
    removed a healthy C3 row because an unrelated C4 call had 500'd.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path.name}: not a run_eval raw file (expected a list of rows)")
    golds, qbi, _ = resolve_gold_source({r.get("id", "") for r in rows},
                                        explicit=questions, verbose=False)
    failures: dict[str, int] = {}
    for row in rows:
        for c in failed_configs(row):
            failures[c] = failures.get(c, 0) + 1
    scored = per_question_scores(rows, qbi, golds=golds, per_config_exclusion=True)
    return scored, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--metric", default="f1", choices=sorted(METRICS))
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--questions", default=None,
                    help="Raw Mintaka file supplying the gold. Auto-resolved and "
                         "coverage-checked when omitted.")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()

    if args.resamples <= 0:
        ap.error(f"--resamples must be a positive integer (got {args.resamples})")

    a_path, b_path = Path(args.run_a), Path(args.run_b)
    label_a = args.label_a or a_path.stem
    label_b = args.label_b or b_path.stem

    a, failures_a = load_scored(a_path, args.questions)
    b, failures_b = load_scored(b_path, args.questions)
    key, metric_label, scale = METRICS[args.metric]

    print("=" * 78)
    print(f"{metric_label}:  {label_a}  MINUS  {label_b}   "
          f"(paired bootstrap, {args.resamples:,} resamples)")
    print("=" * 78)
    print(f"{'Config':<20}{'delta':>9}{'95% CI':>20}{'p':>9}   verdict")

    shared_configs = [c for c in a if c in b]
    if not shared_configs:
        raise SystemExit("the two runs share no config keys")

    for config in shared_configs:
        r = stats.compare_configs(a[config], b[config], key,
                                  n_resamples=args.resamples)
        ci = f"[{r['ci_low'] * scale:+.1f}, {r['ci_high'] * scale:+.1f}]"
        verdict = "significant" if r["significant"] else "n.s."
        # p in [.02, .10] is flagged SEED-SENSITIVE, the same convention
        # report.py uses: it is what caught the C3 selection-effect claim being
        # borderline rather than significant.
        if 0.02 <= r["p_value"] <= 0.10:
            verdict += "  ⚠️ seed-sensitive"
        label = _CONFIG_LABELS.get(config, config)
        print(f"{label:<20}{r['delta'] * scale:>+9.1f}{ci:>20}"
              f"{r['p_value']:>9.3f}   {verdict}")
        if r["n_dropped"]:
            print(f"{'':<20}⚠️ {r['n_dropped']} question(s) not in both runs — "
                  f"the pairing is incomplete")
        # Per-config exclusions are otherwise invisible: a question this config
        # failed on in BOTH runs is absent from both sides and n_dropped stays
        # 0, so silence here would read as "nothing was excluded".
        fa, fb = failures_a.get(config, 0), failures_b.get(config, 0)
        if fa or fb:
            print(f"{'':<20}ℹ️ infrastructure failures for this config: "
                  f"{label_a} {fa}, {label_b} {fb} (those questions are "
                  f"excluded for this config only)")

    print(f"\nPositive = {label_a} scores HIGHER than {label_b} on {metric_label}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
