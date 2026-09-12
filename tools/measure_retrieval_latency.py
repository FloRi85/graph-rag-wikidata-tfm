"""
Local retrieval + ranking cost per configuration — LLM excluded, offline.

Answers "what does the retrieval stage itself cost?" without the one term that
is not a property of this system: the model call. Every LLM round trip in this
project goes to a shared free-tier endpoint whose latency is dominated by
queueing at the provider, so an end-to-end wall clock measures NVIDIA on the
day, not Graph-RAG. What IS attributable to the architecture is everything
before the prompt is sent — cache read, rendering, embedding, ranking — and
that is what this tool times.

WHAT IS TIMED, and why it matches the pipelines
    The call sequences are copied from the pipelines rather than reimplemented:

      C2   wikipedia_pool.build_pool(qids)          -> chunks, groups, records
           embedding_retriever.retrieve_context(...)

      C3   wikidata_pool.build_pool(qids, names)    -> facts, groups, _
      C4   embedding_retriever.retrieve_context(...)
           context_format.format_ranked(...)        (C3's bullet rendering)

    ⚠️ C3 AND C4 SHARE THIS STAGE EXACTLY. graph_rag.py:139-141 and
    graph_rag_rerank.py:241-243 make the same two calls with the same
    arguments; C4's only extra work is its condensing LLM call, which is
    excluded here by design. They are therefore reported as ONE row. Timing
    them as two would not measure a difference, it would measure C3 having
    warmed the page cache for C4 — the artifact demo_server.execute_run's
    docstring already warns about.

    C1 performs no retrieval at all. Its row is structurally zero, not fast.

WHY NO API KEY AND NO NETWORK
    Only questions whose entities are already cache-resident are measured, and
    residency is decided by the SAME validators the preflight uses
    (`wikidata.statement_cache_is_usable`, `wikipedia.article_cache_is_usable`)
    — never `Path.exists()`, which once reported a fully stale cache as ready.
    Questions failing that check are skipped and counted, so a fetch cannot
    silently inflate a timing.

    The two arms are filtered independently: a QID can be usable in the
    statement cache and missing from the article cache.

COLD VS WARM
    Every invocation prints the PREVIOUS run's figures beside this one. First
    touch of a cache file pays real disk I/O; later touches are served from the
    OS page cache, so a first run against a cold corpus reads slower than the
    repeat. `--repeats` shows the same effect within a single invocation.

    ⚠️ The embedder is warmed before timing starts. Without that, the first
    ranking call would absorb the SentenceTransformer model load (seconds) and
    swamp the measurement.

Usage
    venv\\Scripts\\python tools/measure_retrieval_latency.py
    venv\\Scripts\\python tools/measure_retrieval_latency.py --limit 50 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.parse_questions import load_questions          # noqa: E402
from src.pipelines import graph_rag, rag                      # noqa: E402
from src.retrieval import wikidata, wikipedia                 # noqa: E402
from src.retrieval.context_format import format_ranked        # noqa: E402
from src.retrieval.embedding_retriever import retrieve_context  # noqa: E402
from src.retrieval.wikidata_pool import build_pool as kg_pool  # noqa: E402
from src.retrieval.wikipedia_pool import build_pool as text_pool  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "data" / "questions" / "mintaka_sample_dev_200.json"
STORE = ROOT / "data" / "analysis" / "retrieval_latency.json"
HISTORY_CAP = 20

# Reported as one row: the two configurations execute the same two calls.
KG_ARM = "C3/C4 Graph-RAG"
TEXT_ARM = "C2 RAG"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def cache_ready(q, arm: str) -> bool:
    """Is every entity of this question usable from the arm's cache?

    Uses the arm's own validator, so a version bump or a fingerprint change
    counts as absent rather than as a fast read of stale bytes.
    """
    if not q.qids:
        return False
    check = (wikidata.statement_cache_is_usable if arm == "kg"
             else wikipedia.article_cache_is_usable)
    return all(check(qid) for qid in q.qids)


def time_kg(q) -> dict[str, float]:
    """Mirrors graph_rag.answer up to (not including) the LLM call."""
    qid_names = dict(zip(q.qids, q.entity_names))

    t0 = time.perf_counter()
    facts, groups, _ = kg_pool(q.qids, qid_names, verbose=False)
    t1 = time.perf_counter()
    ranked = retrieve_context(q.text, facts, top_k=graph_rag.TOP_K, groups=groups)
    t2 = time.perf_counter()
    label = " / ".join(q.entity_names) if q.entity_names else "entities"
    format_ranked(label, ranked)
    t3 = time.perf_counter()

    return {"pool": (t1 - t0) * 1000, "rank": (t2 - t1) * 1000,
            "format": (t3 - t2) * 1000, "total": (t3 - t0) * 1000,
            "pool_size": float(len(facts))}


def time_text(q) -> dict[str, float]:
    """Mirrors rag.answer up to (not including) the LLM call."""
    t0 = time.perf_counter()
    chunks, groups, _ = text_pool(q.qids, verbose=False)
    t1 = time.perf_counter()
    if chunks:
        retrieve_context(q.text, chunks, top_k=rag.TOP_K, groups=groups)
    t2 = time.perf_counter()

    return {"pool": (t1 - t0) * 1000, "rank": (t2 - t1) * 1000,
            "format": 0.0, "total": (t2 - t0) * 1000,
            "pool_size": float(len(chunks))}


def measure(questions, repeats: int) -> dict:
    """Per-arm medians over every (question, repeat) pair."""
    arms = {
        KG_ARM: {"fn": time_kg, "key": "kg", "samples": [], "skipped": 0},
        TEXT_ARM: {"fn": time_text, "key": "text", "samples": [], "skipped": 0},
    }

    for name, arm in arms.items():
        eligible = [q for q in questions if cache_ready(q, arm["key"])]
        arm["skipped"] = len(questions) - len(eligible)
        print(f"  {name}: {len(eligible)} cache-resident, "
              f"{arm['skipped']} skipped (would need a fetch)")
        for r in range(repeats):
            for q in eligible:
                arm["samples"].append(arm["fn"](q))

    out = {}
    for name, arm in arms.items():
        s = arm["samples"]
        if not s:
            out[name] = None
            continue
        out[name] = {
            "n": len(s),
            "skipped": arm["skipped"],
            "pool_ms": statistics.median(x["pool"] for x in s),
            "rank_ms": statistics.median(x["rank"] for x in s),
            "format_ms": statistics.median(x["format"] for x in s),
            "total_ms": statistics.median(x["total"] for x in s),
            "total_mean_ms": statistics.fmean(x["total"] for x in s),
            "pool_size_median": statistics.median(x["pool_size"] for x in s),
        }
    return out


# ---------------------------------------------------------------------------
# Persistence + reporting
# ---------------------------------------------------------------------------

def load_store() -> dict:
    if not STORE.exists():
        return {"latest": None, "history": []}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt store must not lose the measurement about to be taken.
        print(f"  note: {STORE.name} unreadable; treating as no previous run")
        return {"latest": None, "history": []}


def save_store(store: dict, record: dict) -> None:
    history = ([store["latest"]] if store.get("latest") else []) + \
        list(store.get("history", []))
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(
        {"latest": record, "history": history[:HISTORY_CAP]},
        indent=2), encoding="utf-8")


def _fmt(ms) -> str:
    return "—" if ms is None else f"{ms:8.1f} ms"


def _delta(now, before) -> str:
    if now is None or before is None:
        return "—"
    d = now - before
    sign = "+" if d >= 0 else "−"
    pct = f"{sign}{abs(d) / before * 100:3.0f}%" if before else "   — "
    return f"{sign}{abs(d):7.1f} ms  ({pct})"


def report(this: dict, prev: dict | None) -> None:
    prev_arms = (prev or {}).get("arms") or {}
    when = (prev or {}).get("timestamp", "—")

    print(f"\n  last run: {when}")
    print(f"  this run: {this['timestamp']}\n")

    head = f"  {'':<22} {'last run':>12}  {'this run':>12}  {'change':<20}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    print(f"  {'C1 Base LLM':<22} {'no retrieval':>12}  {'no retrieval':>12}"
          f"  {'—':>18}")

    for name in (TEXT_ARM, KG_ARM):
        cur = this["arms"].get(name)
        old = prev_arms.get(name)
        if cur is None:
            print(f"  {name:<22} {'—':>12}  {'no eligible questions':>12}")
            continue
        print(f"\n  {name}   (n={cur['n']} timings, "
              f"{cur['skipped']} questions skipped, "
              f"median pool {cur['pool_size_median']:.0f} candidates)")
        for label, key in (("pool build", "pool_ms"),
                           ("embed + rank", "rank_ms"),
                           ("format", "format_ms"),
                           ("TOTAL (median)", "total_ms"),
                           ("total (mean)", "total_mean_ms")):
            o = old.get(key) if old else None
            print(f"    {label:<20} {_fmt(o):>12}  {_fmt(cur[key]):>12}"
                  f"  {_delta(cur[key], o):<20}")

    print("\n  LLM calls are excluded. C4 additionally issues a condensing call "
          "and\n  C1–C3 one answering call; those are provider-bound and not "
          "measured here.")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N questions")
    ap.add_argument("--repeats", type=int, default=1,
                    help="time every question N times (shows the warm-up effect "
                         "within one invocation)")
    ap.add_argument("--no-save", action="store_true",
                    help="report without updating the stored previous run")
    args = ap.parse_args(argv)

    if not args.questions.exists():
        print(f"questions file not found: {args.questions}")
        return 1

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    print(f"\nLocal retrieval + ranking cost — LLM excluded")
    print(f"  questions: {len(questions)} from {args.questions.name}"
          f"   repeats: {args.repeats}")

    # Before any timer starts: the first ranking call would otherwise absorb
    # the SentenceTransformer load and swamp every figure below.
    t0 = time.perf_counter()
    retrieve_context("warm up the encoder", ["a warm up line"], top_k=1)
    print(f"  embedder warmed in {time.perf_counter() - t0:.1f}s")

    arms = measure(questions, args.repeats)
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "questions_file": args.questions.name,
        "n_questions": len(questions),
        "repeats": args.repeats,
        "top_k": {"C2": rag.TOP_K, "C3/C4": graph_rag.TOP_K},
        "arms": arms,
    }

    store = load_store()
    report(record, store.get("latest"))

    if args.no_save:
        print(f"\n  --no-save: {STORE.name} left unchanged")
    else:
        save_store(store, record)
        print(f"\n  saved: {STORE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
