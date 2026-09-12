"""
Score a stored run for faithfulness — is each answer supported by its context?

Reads the answers and contexts already in a run file and adds the measurement
the hallucination rate cannot make (see `src/eval/faithfulness.py`). It scores
**attempted** answers only, for the three retrieval configurations only:
abstentions and `other` are not claims about the world, and Configuration 1 has
no context to be faithful to.

Faithfulness is reported by response outcome to distinguish benchmark
agreement from judged support in the supplied context. A correct answer can
have low context faithfulness, and a hallucinated answer can have high
faithfulness. Neither combination identifies the source of the answer or the
cause of an error. See docs/reproduction.md for the analysis and validation.

⚠️ This calls the LLM: roughly two calls per scored answer. On the reference run
that is ~650 calls, ~30 min at the paced rate. Use --limit first.

Usage (from repo root):
    venv\\Scripts\\python tools/score_faithfulness.py <run>_raw.json --limit 6
    venv\\Scripts\\python tools/score_faithfulness.py <run>_raw.json --workers 3
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

from src import llm_config, token_counter
from src.eval import faithfulness
from src.eval.metrics import (
    CORRECT, HALLUCINATION, WRONG_THRESHOLD, _CONFIG_LABELS, is_attempted,
    per_question_scores,
)
from src.eval.parallel import map_questions
from src.eval.gold_source import resolve_gold_source

OUT_DIR = ROOT / "data" / "analysis"


def context_for(row: dict, config: str) -> str:
    """`contexts[config]` is the pipeline's capture dict, not a bare string."""
    capture = (row.get("contexts") or {}).get(config) or {}
    if isinstance(capture, str):
        return capture
    ctx = capture.get("context")
    return ctx if isinstance(ctx, str) else ""


def build_jobs(rows: list[dict], qbyid: dict, golds: dict, threshold: float,
               limit: int | None) -> list[dict]:
    scored = per_question_scores(rows, qbyid, threshold, golds)
    by_id = {r.get("id", ""): r for r in rows}

    jobs: list[dict] = []
    for config in faithfulness.RETRIEVAL_CONFIGS:
        for qid, rec in (scored.get(config) or {}).items():
            # Abstentions and OTHER replies are not claims about the world, so
            # there is nothing to be faithful (or unfaithful) to. Same contract
            # as metrics._aggregate and report.py's attempted-subset diagnostic.
            if not is_attempted(rec):
                continue
            row = by_id.get(qid, {})
            jobs.append({
                "id": qid,
                "config": config,
                "outcome": rec["outcome"],
                "question": row.get("question", ""),
                "answer": (row.get("answers") or {}).get(config, ""),
                "context": context_for(row, config),
            })
    jobs.sort(key=lambda j: (faithfulness.RETRIEVAL_CONFIGS.index(j["config"]), j["id"]))
    if limit:
        # Take a slice per config, not the first N overall, so a smoke run
        # exercises all three rather than only Configuration 2.
        per = max(1, limit // len(faithfulness.RETRIEVAL_CONFIGS))
        kept: list[dict] = []
        for config in faithfulness.RETRIEVAL_CONFIGS:
            kept += [j for j in jobs if j["config"] == config][:per]
        jobs = kept
    return jobs


def run_job(job: dict) -> dict:
    """
    Score one answer, surviving a failure that outlives the retries.

    An exception here used to abort the pool and discard every completed score —
    a full run was lost that way to one 429 after several hundred good calls.
    The failure is recorded as an unscored row instead, exactly as run_eval
    records an infrastructure error rather than grading it: a call that never
    returned is not evidence that an answer was unfaithful.
    """
    try:
        result = faithfulness.score(job["question"], job["answer"], job["context"])
    except Exception as e:                                        # noqa: BLE001
        result = {"score": None, "n_claims": 0, "n_supported": 0,
                  "claims": [], "verdicts": [], "status": "error",
                  "instrument_version": faithfulness.INSTRUMENT_VERSION,
                  "n_out_of_range_initial": None,
                  "n_out_of_range_repair": None,
                  "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {**job, **result, "context": None}   # context is already in the run file


def summarise(records: list[dict]) -> None:
    print()
    print("=" * 78)
    print("FAITHFULNESS  (share of the answer's claims supported by its own context)")
    print("=" * 78)
    print(f"{'Config':<20}{'n':>5}{'mean':>9}{'correct':>10}{'wrong':>9}"
          f"{'unscored':>10}")

    for config in faithfulness.RETRIEVAL_CONFIGS:
        subset = [r for r in records if r["config"] == config]
        if not subset:
            continue
        ok = [r for r in subset if r["score"] is not None]
        unscored = len(subset) - len(ok)

        def mean(rs: list[dict]) -> str:
            vals = [r["score"] for r in rs if r["score"] is not None]
            return f"{sum(vals) / len(vals):.3f}" if vals else "  —  "

        print(f"{_CONFIG_LABELS.get(config, config):<20}{len(ok):>5}"
              f"{mean(ok):>9}"
              f"{mean([r for r in ok if r['outcome'] == CORRECT]):>10}"
              f"{mean([r for r in ok if r['outcome'] == HALLUCINATION]):>9}"
              f"{unscored:>10}")

    print()
    print("  Faithfulness measures judged context support, not benchmark correctness.")
    print("  A hallucinated answer may have supported claims; a correct answer may")
    print("  have unsupported claims. These scores do not identify the source of")
    print("  the answer or the cause of an error.")

    # ⚠️ MISSINGNESS IS PER CONFIG-ANSWER, NOT PER QUESTION. A judge that could
    # not be parsed for C3 says nothing about C2's answer to the same question,
    # so the record is dropped from THIS config's marginal and the question is
    # dropped only from comparisons that need both. That is the opposite of the
    # run_eval rule, where an infrastructure failure excludes a question from
    # EVERY config -- there the pipeline failed, here only the post-hoc judge did.
    unscorable = {c: sum(1 for r in records
                         if r["config"] == c and r["score"] is None)
                  for c in faithfulness.RETRIEVAL_CONFIGS}
    if any(unscorable.values()):
        print()
        print("  ⚠️ Unscored answers are EXCLUDED from the means above, not counted")
        print("     as 0.0 — missing scores differ from no claims judged supported.")
        print("     Per config: "
              + ", ".join(f"{c} {n}" for c, n in unscorable.items() if n))
        print("     For a PAIRED comparison, use only ids scoreable in BOTH configs")
        print("     and report how many were dropped.")

    repaired = sum(1 for r in records if r.get("repair_used"))
    if repaired:
        print(f"  {repaired} answers needed a format-repair call on the verdict list.")

    statuses: dict[str, int] = {}
    for r in records:
        if r["score"] is None:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    if statuses:
        print()
        print("  unscored: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--questions")
    ap.add_argument("--threshold", type=float, default=WRONG_THRESHOLD)
    ap.add_argument("--workers", type=int, default=3,
                    help="3 is the validated concurrency for NVIDIA Build")
    ap.add_argument("--limit", type=int, help="score only this many, spread across configs")
    ap.add_argument("--rpm", type=int, help="override the request-per-minute cap")
    args = ap.parse_args()

    if args.limit is not None and args.limit <= 0:
        ap.error(f"--limit must be a positive integer (got {args.limit})")
    if args.workers is not None and args.workers <= 0:
        ap.error(f"--workers must be a positive integer (got {args.workers})")

    if args.rpm is not None:
        llm_config.set_rpm_limit(args.rpm)

    rows = json.loads(args.run.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        sys.exit(f"{args.run.name}: expected a run_eval result list.")

    golds, qbyid, _ = resolve_gold_source({r.get("id", "") for r in rows},
                                          explicit=args.questions)
    jobs = build_jobs(rows, qbyid, golds, args.threshold, args.limit)
    print(f"{args.run.name} — scoring {len(jobs)} attempted answers "
          f"(~{2 * len(jobs)} calls) at {args.workers} workers")

    records = map_questions(run_job, jobs, workers=args.workers,
                            progress_every=25, label="faithfulness")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"faithfulness_{args.run.stem}.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    summarise(records)

    print()
    token_counter.print_usage()
    print(f"\n  written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
