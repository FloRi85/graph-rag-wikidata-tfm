"""
Gold-form presence check against each configuration's supplied context.

Joins each retrieval configuration's stored context (no live retrieval) to
its scored response outcome, then checks whether any accepted gold answer
form is detected. Outcomes are reported separately for detected and
not-detected forms; these groups do not identify the cause of a response.

The TEST-4000 analysis covers 3,426 eligible questions per configuration:
a form is detected in 71.5% of C2 contexts, 38.1% of C3 contexts and 37.7%
of C4 contexts. C4 retains a detected form on 79.8% of questions with one
detected in the ranked facts. A form is newly detected after condensing
on 251 questions; this does not determine where that form came from.

Two before/after views are printed separately: C4's ranked fact lines alone,
and C3's full rendered graph context (including its question-entity header),
each compared with C4's condensed prose. The latter reproduces the thesis's
reported full-context comparison. Header-only detections are not evidence
that a retrieved fact states the answer; see docs/reproduction.md.

This is a surface-form proxy, not a test of whether the context entails the
answer: presence does not establish sufficient evidence, and absence does
not rule out a paraphrase. C4 is measured after condensing. Boolean answers
and records without usable gold forms are excluded (flag undefined).
Comparative questions often name answer candidates, so form presence is
especially weak evidence of derivability there. This analysis is exploratory;
headline outcome metrics remain in tools/report.py. See docs/reproduction.md.

Usage (from repo root):
    venv\\Scripts\\python tools/context_gold_coverage.py            # TEST-4000
    venv\\Scripts\\python tools/context_gold_coverage.py <raw.json>

Writes the per-question joined table (outcome, in_ctx, stratum fields) to
data/analysis/, which tools/stratified_claim_checks.py reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import (  # noqa: E402
    ABSTENTION, CORRECT, HALLUCINATION, OTHER,
    _extract_years, normalize, parsed_gold_forms, per_question_scores,
)
from src.eval.gold_source import resolve_gold_source  # noqa: E402

DEFAULT_RAW = (ROOT / "data/results/test_runs"
               / "20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw"
               / "20260819_2017_mintaka_test_raw_raw.json")
OUT_DIR = ROOT / "data" / "analysis"

CONFIGS = {"rag": "C2 RAG", "graph_rag": "C3 Graph-RAG", "rerank": "C4 condense"}
OUTCOME_ORDER = [CORRECT, HALLUCINATION, ABSTENTION, OTHER]


def gold_in_context(ctx: str, forms: list[str], answer_type: str) -> bool | None:
    """None = undefined (boolean / empty gold). Word-boundary normalized match,
    plus a year match for date golds (a date's surface form rarely matches
    verbatim, but its year does)."""
    if answer_type == "boolean":
        return None
    usable = [f for f in forms if f and normalize(f)]
    if not usable:
        return None
    nctx = " " + normalize(ctx) + " "
    for g in usable:
        if f" {normalize(g)} " in nctx:
            return True
    if answer_type == "date":
        gyears = set()
        for g in usable:
            gyears |= _extract_years(g)
        if gyears and gyears & _extract_years(ctx):
            return True
    return False


def stored_context(row: dict, cfg: str) -> str:
    ctx = (row.get("contexts") or {}).get(cfg) or ""
    if isinstance(ctx, dict):
        ctx = ctx.get("context") or ""
    return ctx


def pct(a: int, b: int) -> str:
    return f"{100*a/b:5.1f}%" if b else "    —"


def build_joined(rows: list[dict], golds: dict, scores: dict) -> dict[str, dict[str, dict]]:
    """Per config: qid -> {outcome, in_ctx, empty_ctx, ctx_words, complexity,
    category, answer_type}. The table stratified_claim_checks.py consumes."""
    rows_by_id = {r["id"]: r for r in rows}
    joined: dict[str, dict[str, dict]] = {}
    for cfg in CONFIGS:
        j = {}
        for qid, s in scores[cfg].items():
            row = rows_by_id[qid]
            ctx = stored_context(row, cfg)
            g = golds[qid]
            forms, _ = parsed_gold_forms(g)
            j[qid] = {
                "outcome": s["outcome"],
                "in_ctx": gold_in_context(ctx, forms, g.answer_type),
                "empty_ctx": not ctx.strip(),
                "ctx_words": s.get("context_words") or 0,
                "complexity": s["complexity"],
                "category": row.get("type", ""),
                "answer_type": g.answer_type,
            }
        joined[cfg] = j
    return joined


def print_rendered_context_transitions(joined: dict) -> Counter:
    """Print the thesis's full C3-context versus C4-prose comparison.

    Uses the existing per-configuration presence flags without altering them.
    C3's rendered context includes the question-entity names in its header;
    the separate fact-lines-only check deliberately excludes that header.
    """
    transitions: Counter = Counter()
    outcomes: dict = {}
    for qid in sorted(set(joined["graph_rag"]) & set(joined["rerank"])):
        before = joined["graph_rag"][qid]["in_ctx"]
        after = joined["rerank"][qid]["in_ctx"]
        if before is None or after is None:
            continue
        cell = (bool(before), bool(after))
        transitions[cell] += 1
        outcomes.setdefault(cell, Counter())[joined["rerank"][qid]["outcome"]] += 1

    print("\n" + "=" * 78)
    print("5b. FULL RENDERED C3 GRAPH CONTEXT -> C4 CONDENSED PROSE")
    print("    Thesis comparison: includes question-entity names in the C3 header.")
    print("    Header-only form presence does not establish retrieved answer evidence.")
    print("=" * 78)
    n = sum(transitions.values())
    for (before, after), count in sorted(transitions.items(), reverse=True):
        cells = outcomes[(before, after)]
        print(f"  in-full-C3={str(before):<5} in-C4-prose={str(after):<5} "
              f"n={count:>5} ({pct(count, n).strip()})   outcomes: "
              + "  ".join(f"{label} {pct(cells[label], count).strip()}"
                          for label in OUTCOME_ORDER if cells[label]))
    kept = transitions[(True, True)]
    lost = transitions[(True, False)]
    absent = transitions[(False, True)] + transitions[(False, False)]
    print(f"  eligible n={n}; detected in full C3 context={kept + lost}; "
          f"no longer detected in C4 prose={lost}")
    if kept + lost:
        print(f"  retention: {kept}/{kept + lost} = {pct(kept, kept + lost).strip()}")
    print(f"  newly detected in C4 prose: {transitions[(False, True)]}/{absent} "
          f"= {pct(transitions[(False, True)], absent).strip()}")
    return transitions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default=str(DEFAULT_RAW),
                    help="A run's raw results file (default: the TEST-4000 run of record)")
    args = ap.parse_args()
    raw_path = Path(args.raw)

    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    golds, qbi, gold_path = resolve_gold_source({r["id"] for r in rows}, verbose=False)
    print(f"n={len(rows)}   gold source: {Path(gold_path).name}")
    scores = per_question_scores(rows, qbi, golds=golds)
    joined = build_joined(rows, golds, scores)

    # ---- 1. gold-form presence x outcome ---------------------------------
    print("\n" + "=" * 78)
    print("1. GOLD-FORM PRESENCE x OUTCOME  (non-boolean questions with a usable gold form)")
    print("=" * 78)
    for cfg, name in CONFIGS.items():
        j = [d for d in joined[cfg].values() if d["in_ctx"] is not None]
        n = len(j)
        n_in = sum(d["in_ctx"] for d in j)
        print(f"\n{name}:  eligible n={n}   form detected: {n_in} ({100*n_in/n:.1f}%)")
        print(f"  {'outcome':<14}{'n':>6}{'detected':>13}{'not detected':>15}{'detected %':>12}")
        for oc in OUTCOME_ORDER:
            sub = [d for d in j if d["outcome"] == oc]
            i = sum(d["in_ctx"] for d in sub)
            print(f"  {oc:<14}{len(sub):>6}{i:>13}{len(sub)-i:>15}{pct(i, len(sub)):>12}")

    # ---- 2. outcome conditioned on gold-form presence --------------------
    print("\n" + "=" * 78)
    print("2. OUTCOME GIVEN GOLD-FORM PRESENCE  (same eligible subset)")
    print("=" * 78)
    for cfg, name in CONFIGS.items():
        j = [d for d in joined[cfg].values() if d["in_ctx"] is not None]
        for label, cond in [("form detected    ", True), ("form not detected", False)]:
            sub = [d for d in j if d["in_ctx"] is cond]
            c = Counter(d["outcome"] for d in sub)
            n = len(sub)
            if not n:
                continue
            print(f"  {name:<16} {label} n={n:>5}  "
                  f"correct {pct(c[CORRECT], n)}  halluc {pct(c[HALLUCINATION], n)}  "
                  f"abstain {pct(c[ABSTENTION], n)}  other {pct(c[OTHER], n)}")
        print()

    # ---- 3. gold-form presence within abstentions and hallucinations ------
    print("=" * 78)
    print("3. GOLD-FORM PRESENCE WITHIN ABSTENTIONS AND HALLUCINATIONS")
    print("=" * 78)
    for cfg, name in CONFIGS.items():
        j = [d for d in joined[cfg].values() if d["in_ctx"] is not None]
        ab = [d for d in j if d["outcome"] == ABSTENTION]
        ha = [d for d in j if d["outcome"] == HALLUCINATION]
        print(f"\n{name}:")
        print(f"  abstentions   : {len(ab):>5}  form not detected "
              f"{pct(sum(not d['in_ctx'] for d in ab), len(ab))}"
              f"   form detected {pct(sum(d['in_ctx'] for d in ab), len(ab))}")
        print(f"  hallucinations: {len(ha):>5}  form not detected "
              f"{pct(sum(not d['in_ctx'] for d in ha), len(ha))}"
              f"   form detected {pct(sum(d['in_ctx'] for d in ha), len(ha))}")

    # ---- 4. gold-in-context by complexity / category / answer type -------
    print("\n" + "=" * 78)
    print("4. GOLD-FORM DETECTION RATE BY COMPLEXITY / CATEGORY / ANSWER TYPE")
    print("   (Comparative questions often name answer candidates; presence != derivability.)")
    print("=" * 78)
    for field in ("complexity", "category", "answer_type"):
        keys = sorted({d[field] for j in joined.values() for d in j.values()
                       if d["in_ctx"] is not None})
        print(f"\n  by {field}:{'':<{max(0, 14-len(field))}}"
              + "".join(f"{CONFIGS[c].split()[0]:>10}" for c in CONFIGS))
        for k in keys:
            line = f"  {k:<20}"
            for cfg in CONFIGS:
                sub = [d for d in joined[cfg].values()
                       if d[field] == k and d["in_ctx"] is not None]
                line += pct(sum(d["in_ctx"] for d in sub), len(sub)).rjust(10)
            print(line)

    # ---- 5. C4 condense transformation -----------------------------------
    print("\n" + "=" * 78)
    print("5a. C4 RANKED FACT LINES ONLY -> C4 CONDENSED PROSE")
    print("    Fact-lines check: excludes the rendered graph-context header.")
    print("=" * 78)
    rows_by_id = {r["id"]: r for r in rows}
    trans: Counter = Counter()
    out_by_cell: dict = {}
    for qid, d4 in joined["rerank"].items():
        if d4["in_ctx"] is None:
            continue
        c4 = rows_by_id[qid]["contexts"].get("rerank") or {}
        g = golds[qid]
        forms, _ = parsed_gold_forms(g)
        top = " \n ".join(c4.get("top_facts") or []) if isinstance(c4, dict) else ""
        in_top = gold_in_context(top, forms, g.answer_type)
        trans[(bool(in_top), bool(d4["in_ctx"]))] += 1
        out_by_cell.setdefault((bool(in_top), bool(d4["in_ctx"])), Counter())[d4["outcome"]] += 1
    n = sum(trans.values())
    for (t, c), cnt in sorted(trans.items(), reverse=True):
        oc = out_by_cell[(t, c)]
        print(f"  in-top30={str(t):<5} in-condensed={str(c):<5} n={cnt:>5} ({100*cnt/n:4.1f}%)"
              f"   outcomes: " + "  ".join(f"{k} {100*v/cnt:.0f}%" for k, v in oc.most_common()))
    kept, lost = trans[(True, True)], trans[(True, False)]
    if kept + lost:
        print(f"\n  retention when a gold form was detected in the ranked facts: "
              f"{kept}/{kept+lost} = {100*kept/(kept+lost):.1f}%")
    inj = trans[(False, True)]
    print(f"  gold form newly detected after condensing: {inj} questions")
    print("  This check cannot distinguish paraphrasing, inference, information from")
    print("  the question or entity annotations, and parametric knowledge.")

    print_rendered_context_transitions(joined)

    # ---- 6. median context words by outcome ------------------------------
    print("\n" + "=" * 78)
    print("6. MEDIAN CONTEXT WORDS BY OUTCOME")
    print("=" * 78)
    print(f"  {'config':<16}" + "".join(f"{oc:>16}" for oc in OUTCOME_ORDER))
    for cfg, name in CONFIGS.items():
        line = f"  {name:<16}"
        for oc in OUTCOME_ORDER:
            ws = [d["ctx_words"] for d in joined[cfg].values() if d["outcome"] == oc]
            line += f"{statistics.median(ws):>16.0f}" if ws else f"{'—':>16}"
        print(line)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"context_gold_coverage_{raw_path.stem}.json"
    out.write_text(json.dumps(joined, indent=0), encoding="utf-8")
    print(f"\njoined table saved: {out}")


if __name__ == "__main__":
    main()
