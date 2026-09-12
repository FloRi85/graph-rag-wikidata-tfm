"""
Sentence-transformer embedding + in-memory cosine similarity retriever.
Uses NumPy for exact in-memory cosine ranking.
Public entry point: retrieve_context(question, chunks, top_k) -> str
"""

from __future__ import annotations

import threading

import numpy as np

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# ---------------------------------------------------------------------------
# How the top-k budget is allocated across a question's entities
# ---------------------------------------------------------------------------
#
# ⭐ A DECLARED SENSITIVITY ARM (2026-08-16), not a setting, and it lives HERE
# rather than in the three pipelines so all of C2, C3 and C4 read one value.
# Putting it in each pipeline would be three places to change and three chances
# for the arms to differ in a way indistinguishable from a retrieval effect.
#
#   "global"  pool everything, z-normalise within each source entity, then take
#             ONE global top-k. The status quo, and the PRIMARY arm — declared
#             before any post-rebuild number existed.
#   "floor"   reserve floor(k/n) slots per question entity, fill the remainder
#             globally. The challenger.
#
# ⚠️ IT IS A NO-OP ON SINGLE-ENTITY QUESTIONS — with one group, floor(k/1) = k.
# On DEV-200, 114 of 200 questions have exactly one distinct QID, so this can
# move at most 86 of them. Both arms are run and both are reported.
#
# The retained reference-v8 and allocation-floor DEV runs provide the
# statement-model sensitivity comparison.
ALLOCATION_GLOBAL = "global"
ALLOCATION_FLOOR = "floor"
ALLOCATIONS = (ALLOCATION_GLOBAL, ALLOCATION_FLOOR)

ALLOCATION = ALLOCATION_GLOBAL

_model = None
# Guards construction only. Without it, parallel eval workers racing on the first
# call each load their own copy of the transformer. Once built, the model is
# shared and used read-only (inference, no grad), which is safe to call
# concurrently — so encode() itself is deliberately NOT serialized.
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # re-check: another thread may have won the race
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


def _normalize_within_groups(scores: np.ndarray, groups: list) -> np.ndarray:
    """Z-score each group's scores independently.

    Centres and scales each source entity's score distribution before global
    selection. This controls differences in score location and scale, but
    does not guarantee equal representation or improve relevance in every case.

    For a single group this is a monotonic transform of the raw scores, so any
    selection made on it is identical to one made on the raw scores — a provable
    no-op on single-entity questions.
    """
    norm = np.empty_like(scores)
    for g in set(groups):
        mask = np.fromiter((gg == g for gg in groups), dtype=bool, count=len(groups))
        s = scores[mask]
        std = s.std()
        norm[mask] = (s - s.mean()) / std if std > 1e-9 else 0.0
    return norm


def _topk_indices(scores: np.ndarray, groups: list | None, top_k: int,
                  allocation: str | None = None) -> list[int]:
    """
    Indices of the top_k highest-scoring items, best first.

    Plain top-k when `groups` is None. Otherwise scores are z-normalised within
    each group (see `_normalize_within_groups`) and `allocation` decides how the
    budget is split:

      "global"  one global cut over the normalised scores. The primary arm.
      "floor"   floor(k/n) slots reserved per group, remainder filled globally.

    `allocation` defaults to the module's ALLOCATION, resolved at CALL time —
    the same convention `top_k` follows in the pipelines, so a sweep tool that
    assigns `embedding_retriever.ALLOCATION` between batches takes effect
    instead of being frozen at import.
    """
    if allocation is None:
        allocation = ALLOCATION
    if allocation not in ALLOCATIONS:
        raise ValueError(f"unknown allocation {allocation!r}; expected one of {ALLOCATIONS}")

    if groups is None:
        return list(np.argsort(scores)[::-1][:top_k])

    norm = _normalize_within_groups(scores, groups)
    order = np.argsort(norm)[::-1]          # every index, best first

    if allocation == ALLOCATION_GLOBAL:
        return list(order[:top_k])

    # --- floor(k/n) per group, then fill globally ---------------------------
    #
    # ⚠️ A GROUP WITH FEWER CANDIDATES THAN ITS RESERVATION SIMPLY CONTRIBUTES
    # FEWER, and the unused slots return to the global fill rather than being
    # lost. On the text arm this is the common case, not an edge case: the
    # median article yields 28 chunks and C2 already fails to fill k=30 on 64 of
    # 200 questions, so a reservation that could not be given back would shrink
    # an already under-filled pool.
    n_groups = len(set(groups))
    reserve = top_k // n_groups

    chosen: list[int] = []
    seen: set[int] = set()
    if reserve:
        per_group: dict = {g: 0 for g in set(groups)}
        for i in order:
            g = groups[i]
            if per_group[g] < reserve:
                per_group[g] += 1
                chosen.append(int(i))
                seen.add(int(i))

    for i in order:                          # global fill, still best-first
        if len(chosen) >= top_k:
            break
        if int(i) not in seen:
            chosen.append(int(i))
            seen.add(int(i))

    # Re-sorted so the returned order is by score, not by which pass claimed it.
    # The context is read top-down by the answering model, and a reserved-but-
    # weaker line appearing above a stronger one would make the ranking a lie.
    chosen.sort(key=lambda i: -norm[i])
    return chosen[:top_k]


def retrieve_context(
    question: str,
    chunks: list[str],
    top_k: int = 3,
    groups: list | None = None,
    allocation: str | None = None,
) -> str:
    """
    Embed the question and all chunks; return the top_k most similar chunks
    joined as a single context string. Returns "" if chunks is empty.

    Pass `groups` (parallel to `chunks`, one entity label per chunk) to rank
    with per-group score normalization — see `_topk_indices`. Omit it (the
    default) for plain global cosine ranking.

    `allocation` defaults to the module's ALLOCATION, resolved at call time, so
    C2, C3 and C4 share one value and a sweep can flip it without touching the
    pipelines.
    """
    if not chunks:
        return ""

    model = _get_model()
    embeddings = model.encode(
        [question] + chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores = _cosine_similarity(embeddings[0], embeddings[1:])
    top_indices = _topk_indices(scores, groups, top_k, allocation)
    return "\n\n".join(chunks[i] for i in top_indices)
