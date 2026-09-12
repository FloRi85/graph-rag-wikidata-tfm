"""
Ordered parallel map over questions, for eval runs.

Eval is a loop of independent, network-bound LLM calls: ~1s each, and a canonical
run is thousands of them, so a sequential run is dominated by waiting. Threads
(not processes) are the right tool — the work is I/O, and the pipelines share an
in-process embedding model and HTTP client that would have to be rebuilt per
process.

Guarantees
----------
- **Order preserved.** Results come back in input order regardless of completion
  order, so a parallel run's rows line up with a sequential run's.
- **No effect on results beyond the provider's own nondeterminism.** Measured on
  8 questions x 5 configs, 3 sequential and 3 parallel runs: sequential-vs-
  sequential reruns disagreed on 6.0 of 40 answers on average, parallel-vs-
  parallel on 6.0, and sequential-vs-parallel on 5.0 — i.e. across-mode variation
  is no larger than within-mode. temperature=0 is NOT a determinism guarantee
  from the API, and that variation is present with or without threads.
  The disagreements are almost all phrasing: across those 6 runs, C1/C3/C4
  aggregate F1 was identical to the decimal and only C2 moved (1.1 F1, one
  question). `workers=1` runs inline with no pool at all — the debugging path.
- **Exceptions belong to their item.** A failure is returned in place rather than
  cancelling the batch, leaving retry/record policy to the caller (see
  run_eval.run_config_with_retry).

Shared state this relies on being thread-safe (all fixed alongside this module):
token_counter's accumulator (locked), the lazily-built OpenAI client and
sentence-transformer (double-checked locks), and the QID-keyed disk caches
(atomic writes — two questions can share an entity). Callers must not mutate
pipeline module globals (e.g. graph_rag_rerank.EMBED_TOP_K,
wikidata.NOISE_FILTER_MODE) from inside a worker; set them before the batch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Concurrency default. Chosen well under gpt-4o-mini's rate limits: 429s are
# retried as transient, but a burst that trips them wastes wall-clock on backoff
# and gains nothing. Raise via --workers once a run's headroom is known.
DEFAULT_WORKERS = 8


def map_questions(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int = DEFAULT_WORKERS,
    progress_every: int = 50,
    label: str = "",
) -> list[R]:
    """
    Apply `fn` to every item, up to `workers` at a time, returning results in
    input order. `workers <= 1` runs sequentially in the calling thread.

    Progress counts COMPLETIONS, not positions — with a pool those differ, and
    reporting a position would imply an ordering the run does not have.
    """
    n = len(items)
    if n == 0:
        return []

    tag = f"[{label}] " if label else ""

    if workers <= 1:
        out: list[R] = []
        for i, item in enumerate(items, 1):
            out.append(fn(item))
            if progress_every and i % progress_every == 0:
                print(f"    {tag}...{i}/{n}", flush=True)
        return out

    done = 0

    def wrapped(item: T) -> R:
        nonlocal done
        try:
            return fn(item)
        finally:
            # GIL makes this increment safe; it is only used for progress.
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"    {tag}...{done}/{n} done", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map yields in input order and re-raises per item on access.
        return list(pool.map(wrapped, items))


def prewarm(embedding: bool = True, client: bool = True) -> None:
    """
    Build the shared singletons before a pool starts.

    Both are double-checked-locked, so this is belt-and-braces rather than
    required — but it keeps the first worker from paying a multi-second model
    load while the rest queue behind the lock, and it surfaces a missing API key
    immediately instead of as N identical worker failures.
    """
    if embedding:
        from src.retrieval.embedding_retriever import _get_model
        _get_model()
    if client:
        from src import llm_config
        llm_config.get_client()
