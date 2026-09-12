"""
The thesis results report: hallucination rate, stratification, and paired CIs.

Turns a saved eval run into the numbers §5 is written from. Every figure here
was previously computed by hand in a scratchpad, which meant the strongest
result in the project was not reproducible from the repository — the thing a
submitted thesis most needs its code to be.

Six sections
------------
1. OUTCOMES        Abst / Correct / Hallucination per config. The headline.
                   F1 cannot express it: F1 scores an abstention and a confident
                   error identically (both 0), and the whole claim is that
                   Graph-RAG converts the second into the first.
2. HALLUCINATION   Paired bootstrap of each config's hallucination rate against
   VS BASELINE     a baseline (default C1), plus an exact McNemar p on the
                   binary outcome as a sensitivity check on the bootstrap.
3. THRESHOLD       Section 1 recomputed at several wrong-answer thresholds.
   SENSITIVITY     Publishing one threshold invites the charge that it was
                   chosen to suit the result; showing the ordering is stable
                   across 0.3/0.5/0.7 retires the objection.
4. STRATIFIED      F1 and abstention per answer_type and per complexity. Failure
                   modes separate by answer TYPE (C3 collapses on numerical,
                   nearly holds on boolean) in a way aggregate F1 hides.
5. WITHIN-         For each retrieval config, its F1 on the questions it CHOSE
   ATTEMPTED       to answer vs the baseline's F1 on those same questions — AND
   SUBSET          the baseline's F1 on the not-attempted complement, because
                   the attempted subsets are substantially easier and "no
                   selection" must not be read into this section. A selection
                   diagnostic, not proof of its absence.
6. JOINTLY         Retrieval-config pairs compared on the questions BOTH
   ATTEMPTED       attempted — the only subset on which their attempted-set
                   accuracy can be tested against each other; the marginal
                   F1@attempted values live on different subsets.

Sections 2, 5 and 6 are also persisted as deterministic JSON under
data/analysis/report_stats_<run>.json (input sha, seed, resamples, results) —
the sanctioned source the thesis copies statistical figures from.

Usage (from repo root):
    venv\\Scripts\\python tools/report.py data/results/dev_runs/<run>/raw.json
    venv\\Scripts\\python tools/report.py <run>.json --baseline rag
    venv\\Scripts\\python tools/report.py <run>.json --thresholds 0.3 0.5 0.7
    venv\\Scripts\\python tools/report.py <sweep>.json --cell property_type_k30
    venv\\Scripts\\python tools/report.py <run>.json --latex     # §5 tables
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

from src.eval.metrics import (
    WRONG_THRESHOLD, _CONFIG_LABELS, is_attempted, per_question_scores,
    score_results,
)
from src.eval.stats import (
    DEFAULT_SEED, compare_configs, f1_key, hallucination_key, mcnemar_exact,
)
from src.eval.gold_source import resolve_gold_source

DEFAULT_BASELINE = "base_llm_abstain"   # Config 1; see run_eval.CONFIGS
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7]
ANALYSIS_DIR = ROOT / "data" / "analysis"

# The three retrieval configs in a fixed pair order, so section 6 and its
# persisted JSON are stable across runs of the tool.
_PAIRS = (("graph_rag", "rag"), ("rerank", "rag"), ("rerank", "graph_rag"))


def fmt_p(p: float) -> str:
    """p for display. The bootstrap p is a tail count with finite Monte Carlo
    resolution (smallest positive value 2/resamples, literal 0.0 possible), so
    anything below .001 prints as `<.001` rather than as a false exact zero."""
    return "<.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Loading — plain eval runs and sweep files
# ---------------------------------------------------------------------------

def load_rows(path: Path, cell: str | None) -> tuple[list[dict], str, str | None]:
    """
    Return (rows, description, sample_name). Accepts either a run_eval.py output
    (a list of result rows) or a sweep file (cells, each holding its own rows).
    """
    blob = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(blob, list):
        return blob, path.name, None

    cells = blob.get("cells") or {}
    if not cells:
        sys.exit(f"{path.name}: not an eval run and has no 'cells' — cannot report on it.")
    if cell is None:
        sys.exit(f"{path.name} is a sweep file. Pick a cell with --cell:\n"
                 + "\n".join(f"  {k}" for k in cells))
    if cell not in cells:
        sys.exit(f"No cell {cell!r}. Available:\n" + "\n".join(f"  {k}" for k in cells))
    return cells[cell]["rows"], f"{path.name} [{cell}]", blob.get("sample")


def label(config_id: str) -> str:
    return _CONFIG_LABELS.get(config_id, config_id)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_outcomes(scored: dict, threshold: float) -> None:
    pc = scored["per_config"]
    print("=" * 84)
    print(f"1. OUTCOMES  (hallucination = answered AND scored < {threshold:.2f})")
    print("   'Hallucination' = a clean attempted answer that disagrees with the Mintaka")
    print("   benchmark gold — a factuality proxy against the 2021 annotation, not")
    print("   independently verified world-factuality.")
    print("=" * 84)
    # ⚠️ The @att pair is over the ATTEMPTED subset only. It answers "when it
    # answers, how often is it right / confidently wrong" -- which the
    # all-questions rates cannot, since abstaining lowers both at once. The
    # all-questions columns stay the headline: see metrics._aggregate.
    print(f"{'Config':<18}{'N':>5}{'Abst%':>8}{'Correct%':>10}{'HALLUC%':>9}"
          f"{'Other%':>8}{'Corr@att':>10}{'Hall@att':>10}{'F1':>7}{'F1@att':>8}")
    for c, d in pc.items():
        ca = d.get("correct_rate_attempted")
        ha = d.get("hallucination_rate_attempted")
        print(f"{label(c):<18}{d['n']:>5}{d['abstention_rate']*100:>8.1f}"
              f"{d['correct_rate']*100:>10.1f}{d['hallucination_rate']*100:>9.1f}"
              f"{d.get('other_rate', 0.0)*100:>8.1f}"
              + (f"{ca*100:>10.1f}" if ca is not None else f"{'—':>10}")
              + (f"{ha*100:>10.1f}" if ha is not None else f"{'—':>10}")
              + f"{d['f1']*100:>7.1f}{d['f1_attempted']*100:>8.1f}")

    if any(d.get("other_rate") for d in pc.values()):
        print("\n  Other% = neither a clean answer nor the required refusal: empty,")
        print("  truncated, or a refusal in the model's own words instead of the")
        print("  instructed sentinel. Reported rather than forced into one of the")
        print("  other three, because guessing moves the headline in both directions —")
        print("  terse refusals used to inflate hallucination, and a hedge inside a")
        print("  real answer once hid a wrong one. A large Other% is a finding about")
        print("  instruction-following, not a rounding detail.")

    err = scored.get("errors") or {}
    if err.get("n_questions_excluded"):
        print(f"\n  {err['n_questions_excluded']} question(s) excluded for infrastructure "
              f"failure (dropped for ALL configs to keep the sets matched).")


def section_vs_baseline(scores: dict, baseline: str, resamples: int) -> list[dict]:
    print()
    print("=" * 84)
    print(f"2. HALLUCINATION RATE vs {label(baseline)}  (paired bootstrap, "
          f"{resamples:,} resamples; exact McNemar as sensitivity check)")
    print("=" * 84)
    if baseline not in scores:
        print(f"  baseline {baseline!r} not in this run — skipped.")
        return []
    rows: list[dict] = []
    print(f"{'Config':<18}{'delta pts':>11}{'95% CI':>20}{'p':>9}{'McN p':>9}   verdict")
    for c in scores:
        if c == baseline:
            continue
        r = compare_configs(scores[c], scores[baseline], hallucination_key,
                            n_resamples=resamples)
        mc = mcnemar_exact(scores[c], scores[baseline], hallucination_key)
        ci = f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]"
        verdict = "significant" if r["significant"] else "not significant"
        print(f"{label(c):<18}{r['delta']*100:>+11.1f}{ci:>20}{fmt_p(r['p_value']):>9}"
              f"{fmt_p(mc['p_value']):>9}   {verdict}")
        rows.append({"config": c, "baseline": baseline,
                     "metric": "hallucination_rate",
                     "bootstrap": r, "mcnemar": mc})
    print("\n  Negative = FEWER confident errors than the baseline. This is the direction\n"
          "  the thesis predicts for the KG configs.\n"
          "  p = two-sided percentile-bootstrap tail estimate (finite Monte Carlo\n"
          f"  resolution: smallest positive value 2/{resamples:,}; <.001 printed as such).\n"
          "  McN p = exact McNemar binomial on the discordant pairs of the binary\n"
          "  hallucination outcome — a sensitivity check beside the instrument, not a\n"
          "  replacement. CIs are nominal pointwise 95%, unadjusted.")
    return rows


def section_thresholds(rows: list[dict], thresholds: list[float],
                       questions_by_id: dict[str, dict] | None = None,
                       golds: dict | None = None) -> None:
    print()
    print("=" * 84)
    print("3. THRESHOLD SENSITIVITY  (hallucination % at each wrong-answer cutoff)")
    print("=" * 84)
    per_thr = {t: score_results(rows, questions_by_id, threshold=t,
                                golds=golds)["per_config"]
               for t in thresholds}
    configs = list(next(iter(per_thr.values())))
    print(f"{'Config':<18}" + "".join(f"{'t=' + format(t, '.2f'):>12}" for t in thresholds))
    for c in configs:
        print(f"{label(c):<18}"
              + "".join(f"{per_thr[t][c]['hallucination_rate']*100:>12.1f}" for t in thresholds))

    # The claim worth making is about ORDERING, not the absolute rate: if the
    # ranking holds at every cutoff, the threshold stops being a lever anyone can
    # accuse the result of resting on.
    #
    # But a bare "ordering changed" verdict is useless, because configs sitting
    # within a point of each other trade places on noise. So: name the pairs that
    # actually swap, and report whether the swaps are confined to configs the
    # headline does not distinguish anyway.
    ranks = {t: sorted(per_thr[t], key=lambda c: per_thr[t][c]["hallucination_rate"])
             for t in thresholds}
    for t in thresholds:
        print(f"\n  t={t:.2f} ranking (fewest confident errors first): "
              + " < ".join(label(c) for c in ranks[t]))

    swaps: set[tuple[str, str]] = set()
    for i, t1 in enumerate(thresholds):
        for t2 in thresholds[i + 1:]:
            pos1, pos2 = {c: i for i, c in enumerate(ranks[t1])}, {c: i for i, c in enumerate(ranks[t2])}
            for a in ranks[t1]:
                for b in ranks[t1]:
                    if a < b and (pos1[a] < pos1[b]) != (pos2[a] < pos2[b]):
                        swaps.add((a, b))

    if not swaps:
        print("\n  Ordering is IDENTICAL at every threshold — the headline does not depend\n"
              "  on where the cutoff is drawn.")
        return

    print("\n  Pairs that swap across thresholds:")
    for a, b in sorted(swaps):
        gaps = [abs(per_thr[t][a]["hallucination_rate"] - per_thr[t][b]["hallucination_rate"]) * 100
                for t in thresholds]
        print(f"    {label(a)} vs {label(b)}   (gap {min(gaps):.1f}-{max(gaps):.1f} pts)")
    print("  Check whether these are pairs the headline actually separates. Two configs\n"
          "  a fraction of a point apart trade places on noise, and that is not the same\n"
          "  as the result being threshold-dependent.")


def section_stratified(scored: dict) -> None:
    for field, title in (("per_answer_type", "answer_type"), ("per_complexity", "complexity")):
        groups = scored.get(field) or {}
        if not groups:
            continue
        configs = list(scored["per_config"])
        keys = sorted(groups)
        print()
        print("=" * 84)
        print(f"4. STRATIFIED BY {title.upper()}  — F1 (abstention %)")
        print("=" * 84)
        print(f"{'Config':<18}" + "".join(f"{k[:12]:>17}" for k in keys))
        n_row = f"{'  (n)':<18}" + "".join(
            f"{groups[k][configs[0]]['n']:>17}" for k in keys)
        print(n_row)
        for c in configs:
            row = f"{label(c):<18}"
            for k in keys:
                d = groups[k][c]
                row += f"{d['f1']*100:>11.1f} ({d['abstention_rate']*100:>3.0f})"
            print(row)


def section_attempted_subset(scores: dict, baseline: str, resamples: int) -> list[dict]:
    print()
    print("=" * 84)
    print("5. WITHIN-ATTEMPTED-SUBSET COMPARISON  — each config vs the baseline on the")
    print("   questions the config CHOSE to answer (selection diagnostic)")
    print("=" * 84)
    if baseline not in scores:
        print(f"  baseline {baseline!r} not in this run — skipped.")
        return []
    base = scores[baseline]
    rows: list[dict] = []
    print(f"{'Config':<18}{'n att':>7}{'its F1':>9}{'base F1':>9}{'base n-att':>11}"
          f"{'delta':>9}{'95% CI':>20}{'p':>8}   verdict")
    for c in scores:
        if c == baseline:
            continue
        # is_attempted, NOT `not abstained`: an OTHER reply (empty, truncated,
        # or a refusal in the model's own words) is not a question the config
        # "CHOSE to answer". Testing `abstained` put C4's ten OTHER replies in
        # this subset and reported n=120 / delta -2.9 where the truth is
        # n=110 / +2.7 -- wrong sample, wrong sign, wrong CI.
        attempted = {q: r for q, r in scores[c].items() if is_attempted(r)}
        if not attempted:
            continue
        shared = {q: base[q] for q in attempted if q in base}
        if not shared:
            continue
        # The NOT-ATTEMPTED complement — abstentions AND Other rows, i.e.
        # everything the config did not cleanly answer. "Declined" would be
        # wrong: an Other row is not an abstention.
        complement = [base[q]["f1"] for q in scores[c]
                      if q not in attempted and q in base]
        comp_f1 = sum(complement) / len(complement) if complement else None
        r = compare_configs(attempted, shared, f1_key, n_resamples=resamples)
        its = sum(v["f1"] for q, v in attempted.items() if q in shared) / len(shared)
        bas = sum(v["f1"] for v in shared.values()) / len(shared)
        ci = f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]"
        verdict = ("beats baseline" if r["significant"] and r["delta"] > 0 else
                   "WORSE than baseline" if r["significant"] else "no difference")
        # A p within a whisker of .05 flips on the resampling seed. Flagging it
        # is the difference between an honest borderline and a claim that
        # quietly depends on which seed happened to be used.
        if 0.02 <= r["p_value"] <= 0.10:
            verdict += "  [BORDERLINE — seed-sensitive]"
        print(f"{label(c):<18}{len(shared):>7}{its*100:>9.1f}{bas*100:>9.1f}"
              + (f"{comp_f1*100:>11.1f}" if comp_f1 is not None else f"{'—':>11}")
              + f"{r['delta']*100:>+9.1f}{ci:>20}{fmt_p(r['p_value']):>8}   {verdict}")
        rows.append({"config": c, "baseline": baseline,
                     "n_attempted_shared": len(shared),
                     "config_f1_attempted": its,
                     "baseline_f1_attempted": bas,
                     "baseline_f1_not_attempted": comp_f1,
                     "bootstrap": r})
    print("\n  Two facts, both needed for an honest reading: (1) the attempted subsets are\n"
          "  substantially EASIER for the baseline too — compare 'base F1' with\n"
          "  'base n-att', its F1 on the not-attempted complement — so retrieval DOES\n"
          "  select easier questions and this section cannot show otherwise; (2) within\n"
          "  those easier subsets each config still beats (or fails to beat) the baseline\n"
          "  on the SAME questions. These are conditional, post-treatment-selected\n"
          "  comparisons — the subset is chosen by the config's own behaviour.")
    return rows


def section_pairwise_attempted(scores: dict, resamples: int) -> list[dict]:
    pairs = [(a, b) for a, b in _PAIRS if a in scores and b in scores]
    if not pairs:
        return []
    print()
    print("=" * 84)
    print("6. CONFIG PAIRS ON JOINTLY ATTEMPTED QUESTIONS  — F1 where BOTH attempted")
    print("=" * 84)
    rows: list[dict] = []
    print(f"{'Pair (A − B)':<36}{'n joint':>8}{'F1 A':>8}{'F1 B':>8}{'delta':>8}"
          f"{'95% CI':>20}{'p':>8}{'McN p':>9}")
    for a, b in pairs:
        att_a = {q: r for q, r in scores[a].items() if is_attempted(r)}
        att_b = {q: r for q, r in scores[b].items() if is_attempted(r)}
        joint = sorted(set(att_a) & set(att_b))
        if not joint:
            continue
        ja = {q: att_a[q] for q in joint}
        jb = {q: att_b[q] for q in joint}
        r = compare_configs(ja, jb, f1_key, n_resamples=resamples)
        mc = mcnemar_exact(ja, jb, hallucination_key)
        fa = sum(v["f1"] for v in ja.values()) / len(joint)
        fb = sum(v["f1"] for v in jb.values()) / len(joint)
        ci = f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]"
        print(f"{label(a) + ' − ' + label(b):<36}{len(joint):>8}{fa*100:>8.1f}"
              f"{fb*100:>8.1f}{r['delta']*100:>+8.1f}{ci:>20}"
              f"{fmt_p(r['p_value']):>8}{fmt_p(mc['p_value']):>9}")
        rows.append({"pair": [a, b], "n_joint": len(joint),
                     "f1_a_joint": fa, "f1_b_joint": fb,
                     "bootstrap_f1": r, "mcnemar_hallucination": mc})
    print("\n  This is 'F1 on jointly attempted questions', NOT the marginal F1@attempted\n"
          "  pair: each marginal is computed over a DIFFERENT attempted set, so the\n"
          "  marginals cannot be tested against each other. The joint subset is\n"
          "  conditional and post-treatment-selected (both configs chose to answer).\n"
          "  McN p = exact McNemar on the binary hallucination outcome, same subset.")
    return rows


def section_latex(scored: dict, threshold: float) -> None:
    pc = scored["per_config"]
    print()
    print("=" * 84)
    print("LATEX  — main results table for §5")
    print("=" * 84)
    print(r"\begin{table}[h]")
    print(r"  \caption{Answer outcomes per configuration. Hallucination denotes a clean "
          rf"attempted answer that disagrees with the Mintaka benchmark gold (scored below "
          rf"{threshold:.1f}) --- a factuality proxy against the 2021 annotation, not "
          rf"independently verified world-factuality.}}")
    print(r"  \label{tab:results}")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrrr}")
    print(r"    \toprule")
    print(r"    \textbf{Configuration} & \textbf{Abst.\%} & \textbf{Correct\%} & "
          r"\textbf{Halluc.\%} & \textbf{F1} & \textbf{F1@att} \\")
    print(r"    \midrule")
    for c, d in pc.items():
        print(f"    {label(c)} & {d['abstention_rate']*100:.1f} & "
              f"{d['correct_rate']*100:.1f} & {d['hallucination_rate']*100:.1f} & "
              f"{d['f1']*100:.1f} & {d['f1_attempted']*100:.1f} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_file", help="Eval run JSON, or a sweep file plus --cell")
    ap.add_argument("--cell", default=None, help="Cell key, for sweep files")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help=f"Config to compare against (default {DEFAULT_BASELINE})")
    ap.add_argument("--threshold", type=float, default=WRONG_THRESHOLD,
                    help=f"Wrong-answer cutoff for the headline (default {WRONG_THRESHOLD})")
    ap.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS,
                    help="Cutoffs for the sensitivity section")
    ap.add_argument("--resamples", type=int, default=10_000,
                    help="Bootstrap resamples (default 10000)")
    ap.add_argument("--latex", action="store_true", help="Also emit a LaTeX table")
    ap.add_argument("--questions", default=None,
                    help="Raw Mintaka file supplying the gold answers. "
                         "Auto-resolved from a sweep file's `sample` field, "
                         "then the dev and test splits; a candidate is used "
                         "only if it covers every question id being scored.")
    args = ap.parse_args()

    if args.resamples <= 0:
        ap.error(f"--resamples must be a positive integer (got {args.resamples})")

    results_path = Path(args.results_file)
    rows, described, sample = load_rows(results_path, args.cell)
    golds, qbi, gold_path = resolve_gold_source({r.get("id", "") for r in rows},
                                                explicit=args.questions,
                                                recorded=sample)
    scored = score_results(rows, qbi, threshold=args.threshold, golds=golds)
    scores = per_question_scores(rows, qbi, threshold=args.threshold, golds=golds)

    print(f"\nRESULTS REPORT — {described}   n={len(rows)} questions\n")
    section_outcomes(scored, args.threshold)
    vs_rows = section_vs_baseline(scores, args.baseline, args.resamples)
    section_thresholds(rows, args.thresholds, qbi, golds)
    section_stratified(scored)
    sel_rows = section_attempted_subset(scores, args.baseline, args.resamples)
    pair_rows = section_pairwise_attempted(scores, args.resamples)
    if args.latex:
        section_latex(scored, args.threshold)

    # Persist the statistical sections. The thesis copies these figures from
    # THIS file, never from a scratchpad or an external recomputation — the
    # sanctioned-source rule. Deterministic given the same inputs: the
    # bootstraps are seeded and ids are processed in sorted order.
    payload = {
        "results_file": results_path.name,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "cell": args.cell,
        "gold_file": Path(gold_path).name if gold_path else None,
        "threshold": args.threshold,
        "baseline": args.baseline,
        "seed": DEFAULT_SEED,
        "n_resamples": args.resamples,
        "p_value_kind": ("two-sided percentile-bootstrap tail estimate; "
                         "resolution 2/n_resamples; mcnemar = exact binomial "
                         "on discordant pairs (sensitivity check)"),
        "sections": {
            "hallucination_vs_baseline": vs_rows,
            "within_attempted_subset": sel_rows,
            "jointly_attempted_pairs": pair_rows,
        },
    }
    suffix = f"_{args.cell}" if args.cell else ""
    if args.baseline != DEFAULT_BASELINE:
        suffix += f"_vs-{args.baseline}"   # a --baseline run must not clobber the default one
    out_path = ANALYSIS_DIR / f"report_stats_{results_path.stem}{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
