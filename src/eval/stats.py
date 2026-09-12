"""
Paired bootstrap for config-vs-config comparisons.

Why paired, and why bootstrap
-----------------------------
Every config answers the SAME questions, so the comparison is paired: the
question-to-question variation (some are simply harder) is shared and must be
differenced out, not treated as independent noise. Resampling *questions* — and
carrying each question's per-config difference with it — does exactly that.

Bootstrap rather than a t-test because the per-question quantities are not
normal and often not even continuous: token-F1 is bounded in [0, 1] and piles up
at the endpoints, and the hallucination indicator is binary. The bootstrap
makes no parametric distributional assumption; it asks how much the mean difference moves
when the question sample is redrawn.

What a CI here does and does not license
----------------------------------------
The interval covers sampling variation over QUESTIONS from one fixed run. It
does NOT cover LLM nondeterminism (temperature=0 is not a determinism guarantee
from the API — see src/eval/parallel.py), nor the fact that a knob was chosen by
looking at a previous sample. A difference whose confidence interval includes
zero is not established by that interval.

Usage:
    from src.eval.stats import paired_bootstrap, compare_configs
    r = compare_configs(scores["rerank"], scores["base_llm_abstain"], lambda s: s["f1"])
    print(r["delta"], r["ci_low"], r["ci_high"], r["p_value"], r["significant"])
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


def paired_bootstrap(
    deltas: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap the mean of per-question paired differences.

    `deltas` holds one A-minus-B value per question. Returns the observed mean,
    a percentile CI, a two-sided p-value, and whether the CI excludes zero.

    The p-value is a two-sided percentile-bootstrap TAIL estimate with finite
    Monte Carlo resolution: it counts resampled means on the far side of zero,
    so it CAN be literally 0.0 (no resample crossed), and its smallest positive
    value is 2/n_resamples — there is no 1/n floor. Report values below the
    resolution as "<.001", never as an exact zero. Intervals are nominal
    pointwise 95% CIs, unadjusted for multiple comparisons.

    The seed is fixed by default so a reported interval is reproducible: an
    unseeded CI that shifts on every run is not a number anyone can check.
    """
    n = len(deltas)
    if n == 0:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0, "significant": False}

    observed = sum(deltas) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]

    # Two-sided p: how often the resampled mean lands on the other side of zero
    # from the observed effect. Doubling the smaller tail is the standard
    # percentile-bootstrap convention; clamped because it can exceed 1 when the
    # effect sits essentially on zero.
    n_le = sum(1 for m in means if m <= 0)
    n_ge = sum(1 for m in means if m >= 0)
    p_value = min(1.0, 2 * min(n_le, n_ge) / n_resamples)

    return {
        "n": n,
        "delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "significant": lo > 0 or hi < 0,
    }


def compare_configs(
    a: dict[str, dict],
    b: dict[str, dict],
    key: Callable[[dict], float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Paired bootstrap of `key` between two per-question score maps (qid -> record,
    as produced by metrics.per_question_scores).

    Only questions present in BOTH maps are compared, and they are taken in
    sorted id order so the result does not depend on dict insertion order — a
    bootstrap that silently changes with input ordering is not reproducible.
    Non-overlapping questions are reported as `n_dropped` rather than dropped
    silently; a large value means the two runs are not really comparable.
    """
    shared = sorted(set(a) & set(b))
    deltas = [key(a[q]) - key(b[q]) for q in shared]
    out = paired_bootstrap(deltas, n_resamples=n_resamples, seed=seed)
    out["n_dropped"] = len(set(a) | set(b)) - len(shared)
    return out


def mcnemar_exact(
    a: dict[str, dict],
    b: dict[str, dict],
    key: Callable[[dict], float],
) -> dict:
    """
    Exact paired McNemar test on a BINARY per-question outcome.

    Sensitivity check beside the bootstrap, not a replacement: for a binary
    indicator (e.g. hallucination yes/no) the discordant pairs carry all the
    paired information, and the exact binomial needs no resampling at all.

    `key` must return a binary value (0/1, 0.0/1.0 or bool) for every shared
    record — anything else raises, because a McNemar on a non-binary quantity
    is a category error the caller should hear about.

    Orientation: `n10` counts questions where A has the event and B does not;
    `n01` the reverse. `delta` is the A-minus-B event-rate difference over the
    shared pairs, so its sign matches compare_configs on the same key.

    p is the exact two-sided binomial tail: with n = n01 + n10 discordant
    pairs, p = min(1, 2 * sum_{k<=min(n01,n10)} C(n,k) / 2^n). The tail sum
    stays an exact integer and is divided by `1 << n` at the end — a float
    `comb(n,k) * 0.5**n` overflows/underflows for large discordant counts.
    """
    shared = sorted(set(a) & set(b))
    n01 = n10 = 0
    for q in shared:
        xa, xb = key(a[q]), key(b[q])
        va, vb = float(xa), float(xb)
        if va not in (0.0, 1.0) or vb not in (0.0, 1.0):
            raise ValueError(
                f"mcnemar_exact needs a binary key; got {xa!r}/{xb!r} for id {q!r}")
        if va == 1.0 and vb == 0.0:
            n10 += 1
        elif va == 0.0 and vb == 1.0:
            n01 += 1

    n = n01 + n10
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(n01, n10) + 1))
        p_value = min(1.0, (2 * tail) / (1 << n))

    return {
        "n_pairs": len(shared),
        "n01": n01,
        "n10": n10,
        "delta": (n10 - n01) / len(shared) if shared else 0.0,
        "p_value": p_value,
    }


def unpaired_bootstrap(
    xs: Sequence[float],
    ys: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sample bootstrap CI on a difference of means over DISJOINT groups.

    For comparisons that are unpaired by construction — e.g. mean faithfulness
    of correct vs wrong answers within one config: no question appears in both
    groups, so there is no per-question delta to pair on. Each group is
    resampled independently, sizes preserved.

    Same p-value convention and caveats as paired_bootstrap: a two-sided
    percentile tail estimate, resolution 2/n_resamples, nominal pointwise
    95% interval, unadjusted. Empty input on either side returns the null
    result rather than raising — "no members in a group" is a reportable
    finding, not a crash.
    """
    xs, ys = list(xs), list(ys)
    if not xs or not ys:
        return {"n_a": len(xs), "n_b": len(ys), "delta": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                "significant": False}

    observed = sum(xs) / len(xs) - sum(ys) / len(ys)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        tx = sum(xs[rng.randrange(len(xs))] for _ in range(len(xs))) / len(xs)
        ty = sum(ys[rng.randrange(len(ys))] for _ in range(len(ys))) / len(ys)
        diffs.append(tx - ty)
    diffs.sort()

    lo = diffs[int(alpha / 2 * n_resamples)]
    hi = diffs[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    n_le = sum(1 for d in diffs if d <= 0)
    n_ge = sum(1 for d in diffs if d >= 0)
    p_value = min(1.0, 2 * min(n_le, n_ge) / n_resamples)

    return {
        "n_a": len(xs),
        "n_b": len(ys),
        "delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "significant": lo > 0 or hi < 0,
    }


# Common comparison keys.
f1_key = lambda s: s["f1"]                                  # noqa: E731
score_key = lambda s: s["score"]                            # noqa: E731
hallucination_key = lambda s: 1.0 if s["outcome"] == "hallucination" else 0.0      # noqa: E731
abstention_key = lambda s: 1.0 if s["outcome"] == "abstention" else 0.0     # noqa: E731
correct_key = lambda s: 1.0 if s["outcome"] == "correct" else 0.0          # noqa: E731
