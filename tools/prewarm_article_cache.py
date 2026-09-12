"""
Fetch every gold entity's Wikipedia article into the article cache.

THE TEXT ARM'S COUNTERPART to `prewarm_statement_cache.py`, and it exists for
the same reason: warming first separates "can we retrieve" from "how does the
system score". A fetch that fails mid-run costs the LLM budget already spent on
that question, and until 2026-08-16 the text arm had no retry at all, so a
single transient timeout produced an empty context that was then graded as a
retrieval result.

⚠️ AN ENTITY WITH NO ENGLISH ARTICLE IS A SUCCESS, NOT A FAILURE, and it is
cached as one so it is not re-fetched every run. It is reported separately in
the summary because it is a real property of the corpus that a run should be
able to state: those entities contribute nothing to C2's pool.

Idempotent: an entity already cached at the current version and chunk geometry
is skipped, so an interrupted run resumes by being run again.

Run from repo root:
    venv/Scripts/python tools/prewarm_article_cache.py
    venv/Scripts/python tools/prewarm_article_cache.py --questions <file> --workers 3
    venv/Scripts/python tools/prewarm_article_cache.py --refetch      # ignore the cache
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
from src.retrieval.wikipedia import (
    ARTICLE_CACHE_VERSION, CACHE_DIR, cache_entry_status, fetch_chunks,
)

DEFAULT_QUESTIONS = ROOT / "data" / "questions" / "mintaka_sample_dev_200.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    # 3, matching the eval's validated concurrency. The MediaWiki APIs are more
    # forgiving than the SPARQL endpoint, but there is nothing to gain from
    # racing them and a 429 costs more time than it saves.
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--refetch", action="store_true",
                    help="Re-fetch entities that are already cached and usable. "
                         "Wikipedia is LIVE, so this is how a run gets a corpus "
                         "dated on one day rather than accumulated over weeks.")
    args = ap.parse_args()

    # ⚠️ The same expression `run_eval.preflight_article_cache` uses. Prewarm
    # and preflight disagreeing about which entities a run needs is how
    # "nothing to do" precedes a run that then fetches hundreds of entities.
    questions = load_questions(args.questions)
    qids = sorted({qid for question in questions for qid in question.qids})

    # 🔴 VALIDATE, DO NOT TEST EXISTENCE. On the KG side `Path.exists()` was the
    # whole check until 2026-08-13, so after a version bump every stale entry
    # still existed and the tool reported "nothing to do" while every entry was
    # unusable. One shared validator, here too.
    statuses = {q: cache_entry_status(q)["state"] for q in qids}
    todo = qids if args.refetch else [q for q, s in statuses.items() if s != "ok"]

    by_state: dict[str, int] = {}
    for s in statuses.values():
        by_state[s] = by_state.get(s, 0) + 1

    print(f"sample      : {Path(args.questions).name}")
    print(f"gold QIDs   : {len(qids)}")
    print(f"cache dir   : {CACHE_DIR}")
    print(f"cached (v{ARTICLE_CACHE_VERSION})  : {by_state.get('ok', 0)}")
    print(f"to fetch    : {len(todo)}"
          + (f"  ({', '.join(f'{k} {v}' for k, v in sorted(by_state.items()) if k != 'ok')})"
             if not args.refetch and todo else "")
          + ("   [--refetch: ignoring the cache]" if args.refetch else "") + "\n")
    if not todo:
        print("nothing to do.")
        return

    started = time.time()
    done = failed = 0
    no_article: list[str] = []
    failures: list[tuple[str, str]] = []

    def one(qid: str) -> tuple[str, int | None, str | None]:
        try:
            # `strict=False`: this tool REPORTS failures rather than aborting on
            # the first one, so a single unreachable article does not stop the
            # other 237 from being warmed. The eval stays strict.
            chunks = fetch_chunks(qid, strict=False)
        except Exception as e:                # noqa: BLE001 — reported, not swallowed
            return qid, None, f"{type(e).__name__}: {e}"
        # ⚠️ RETURNING CHUNKS IS NOT THE SAME AS CACHING THEM. Re-validated
        # through the SAME validator that built `todo`, so "warmed" here and
        # "usable" at preflight cannot mean different things.
        status = cache_entry_status(qid)
        if status["state"] != "ok":
            return qid, None, f"not cached ({status['state']}: {status.get('reason', '')})"
        return qid, len(chunks), None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, q) for q in todo]
        for fut in as_completed(futures):
            qid, n, err = fut.result()
            if err:
                failed += 1
                failures.append((qid, err))
            else:
                done += 1
                if n == 0:
                    no_article.append(qid)
            elapsed = time.time() - started
            rate = (done + failed) / elapsed if elapsed else 0
            left = (len(todo) - done - failed) / rate if rate else 0
            label = f"{n} chunks" if err is None else f"FAILED {err[:60]}"
            if err is None and n == 0:
                label = "no English article"
            print(f"  [{done + failed:3d}/{len(todo)}] {qid}: {label}"
                  f"   ~{left / 60:.0f} min left")

    print(f"\nfetched {done}, failed {failed}, in {(time.time() - started) / 60:.1f} min")
    if no_article:
        # Stated, not buried: these entities contribute nothing to C2's pool,
        # and a reader of the run should know how many there were.
        print(f"{len(no_article)} entities have NO English Wikipedia article "
              f"(a real property of the corpus, cached as such): "
              f"{', '.join(no_article[:8])}"
              + (" ..." if len(no_article) > 8 else ""))
    if failures:
        print("\nfailures (re-run to retry):")
        for qid, err in failures[:20]:
            print(f"  {qid}: {err[:100]}")


if __name__ == "__main__":
    main()
