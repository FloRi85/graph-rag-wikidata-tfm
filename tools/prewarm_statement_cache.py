"""
Fetch every gold entity of a question sample into the statement cache.

WHY THIS IS A SEPARATE STEP. `run_eval` would fetch these lazily, but then the
eval competes with SPARQL latency: the public endpoint is erratic under shared
load (the same query has measured 3 s, 7 s and a 30 s timeout on consecutive
calls), and a fetch failure mid-run costs API budget for the LLM calls already
spent on that question. Warming first separates "can we retrieve" from "how does
the system score".

Idempotent: an entity already cached at the current version is skipped, so an
interrupted run resumes by being run again.

⚠️ It does NOT cache an entity whose label query failed — `fetch_statements`
refuses, because a QID-shaped label renders every incoming statement unreadable.
Those show up in the summary as failures and are retried on the next run.

Run from repo root:
    venv/Scripts/python tools/prewarm_statement_cache.py
    venv/Scripts/python tools/prewarm_statement_cache.py --questions <file> --workers 3
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.parse_questions import load_questions
from src.retrieval.wikidata import (
    STATEMENT_CACHE_VERSION, cache_entry_status, fetch_statements,
)

DEFAULT_QUESTIONS = ROOT / "data" / "questions" / "mintaka_sample_dev_200.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    # 3, matching the eval's validated concurrency. Higher rates draw 429s and
    # 504s from the public endpoint, which cost more time than they save.
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    # ⚠️ The same expression `run_eval.preflight_statement_cache` uses, and it
    # has to stay that way: prewarm and preflight disagreeing about which
    # entities a run needs is how "nothing to do" preceded a run that then
    # fetched hundreds of entities live.
    questions = load_questions(args.questions)
    qids = sorted({qid for question in questions for qid in question.qids})

    # 🔴 VALIDATE, DO NOT TEST EXISTENCE. `Path.exists()` was the whole check
    # until 2026-08-13, so after a cache-version bump every stale entry still
    # existed and this tool reported "nothing to do" -- then the eval run it
    # exists to de-risk absorbed hundreds of live fetches, exactly the failure
    # mode described at the top of this file. One shared validator now.
    statuses = {q: cache_entry_status(q)["state"] for q in qids}
    todo = [q for q, s in statuses.items() if s != "ok"]
    by_state: dict[str, int] = {}
    for s in statuses.values():
        by_state[s] = by_state.get(s, 0) + 1

    print(f"sample      : {Path(args.questions).name}")
    print(f"gold QIDs   : {len(qids)}")
    print(f"cached (v{STATEMENT_CACHE_VERSION})  : {by_state.get('ok', 0)}")
    print(f"to fetch    : {len(todo)}"
          + (f"  ({', '.join(f'{k} {v}' for k, v in sorted(by_state.items()) if k != 'ok')})"
             if len(todo) else "") + "\n")
    if not todo:
        print("nothing to do.")
        return

    started = time.time()
    done = failed = 0
    failures: list[tuple[str, str]] = []

    def one(qid: str) -> tuple[str, int | None, str | None]:
        try:
            n = len(fetch_statements(qid))
        except Exception as e:            # noqa: BLE001 — reported, not swallowed
            return qid, None, f"{type(e).__name__}: {e}"
        # ⚠️ RETURNING STATEMENTS IS NOT THE SAME AS CACHING THEM.
        # `fetch_statements` deliberately skips the cache write when the entity's
        # own label does not resolve or a direction failed, since either leaves a
        # permanently damaged entry. Without this check the tool reports a clean
        # run while leaving the entity to be refetched on every future call —
        # exactly the silent-success pattern this project keeps finding.
        #
        # Re-validated through the SAME validator used to build `todo`, so
        # "warmed" here and "usable" at preflight cannot mean different things.
        status = cache_entry_status(qid)
        if status["state"] != "ok":
            return qid, n, f"not cached ({status['state']}: {status.get('reason', '')})"
        return qid, n, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, q) for q in todo]
        for fut in as_completed(futures):
            qid, n, err = fut.result()
            if err:
                failed += 1
                failures.append((qid, err))
            else:
                done += 1
            elapsed = time.time() - started
            rate = (done + failed) / elapsed if elapsed else 0
            left = (len(todo) - done - failed) / rate if rate else 0
            status = f"{n} statements" if err is None else f"FAILED {err[:60]}"
            print(f"  [{done + failed:3d}/{len(todo)}] {qid}: {status}"
                  f"   ~{left / 60:.0f} min left")

    print(f"\nfetched {done}, failed {failed}, in {(time.time() - started) / 60:.1f} min")
    if failures:
        print("\n⚠️ FAILURES — rerun to retry (an entity is only cached when its "
              "label resolves, so these are safe to repeat):")
        for qid, err in failures:
            print(f"  {qid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
