"""
Module-level token usage accumulator.

Pipelines call record(response.usage) after every OpenAI API call.
run_eval.py calls get() to print totals and reset() between runs.
"""

from __future__ import annotations

import threading

from src import llm_config

_usage: dict[str, int] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

# `+=` on a dict value is read-modify-write, so concurrent pipeline calls would
# silently lose increments and under-report cost. Eval runs are parallelized
# across questions (see src/eval/parallel.py), so the accumulator needs a lock.
_lock = threading.Lock()


def record(usage) -> None:
    """Accumulate usage from an OpenAI CompletionUsage object (None-safe)."""
    if usage is None:
        return
    with _lock:
        _usage["calls"] += 1
        _usage["prompt_tokens"]      += usage.prompt_tokens      or 0
        _usage["completion_tokens"]  += usage.completion_tokens  or 0


def get() -> dict:
    with _lock:
        return dict(_usage)


def reset() -> None:
    with _lock:
        _usage.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


def print_usage() -> None:
    u = get()
    price_in, price_out = llm_config.price_per_million()
    cost = (
        u["prompt_tokens"]       / 1_000_000 * price_in
        + u["completion_tokens"] / 1_000_000 * price_out
    )
    # ⚠️ `usage_records`, NOT "calls". This counter increments only when a
    # response arrives carrying a usage field, so it counts neither failed
    # attempts nor providers that omit usage -- displaying it as "calls" invited
    # exactly the confusion that let a documented run report 800 calls for a
    # 1,002-call run. The call count lives in `llm_config.call_stats()`.
    print(
        f"\nToken usage  usage_records={u['calls']}"
        f"  input={u['prompt_tokens']:,}"
        f"  output={u['completion_tokens']:,}"
        f"  total={u['prompt_tokens'] + u['completion_tokens']:,}"
        f"  est. cost=${cost:.4f} ({llm_config.MODEL})"
    )
    stats = llm_config.call_stats()
    print(f"Calls        attempts={stats['completion_attempts']}"
          f"  responses={stats['completion_responses']}"
          f"  exceptions={stats['completion_exceptions']}"
          f"  with_usage={stats['responses_with_usage']}"
          f"  (excludes SDK-internal HTTP retries)")
