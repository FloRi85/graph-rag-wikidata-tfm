"""
Unit tests for the shared ranker — the one stage C2, C3 and C4 have in common.

No model is loaded: every test drives `_topk_indices` directly with synthetic
scores, which is the whole selection rule.

⚠️ WHAT IS PINNED HERE IS THE COMPARISON'S SYMMETRY. All three retrieval configs
route through this function, so a change in its behaviour is a change to every
arm at once — and a change that affected them unevenly would be
indistinguishable from a retrieval effect in the results.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_embedding_retriever.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval import embedding_retriever as er
from src.retrieval.embedding_retriever import (
    ALLOCATION_FLOOR, ALLOCATION_GLOBAL, _topk_indices,
)


def scores(*values):
    return np.array(values, dtype=float)


class TestGlobalAllocation:
    """The primary arm, and the status quo every recorded number was produced under."""

    def test_plain_top_k_without_groups(self):
        got = _topk_indices(scores(0.1, 0.9, 0.5, 0.7), None, 2)
        assert got == [1, 3]

    def test_best_first(self):
        got = _topk_indices(scores(0.1, 0.9, 0.5), None, 3)
        assert got == [1, 2, 0]

    def test_a_single_group_selects_identically_to_no_groups(self):
        """The provable no-op: with one group the z-transform is monotonic, so
        it cannot reorder anything. 114 of 200 DEV questions are single-entity."""
        s = scores(0.1, 0.9, 0.5, 0.7, 0.3)
        assert (_topk_indices(s, ["Q1"] * 5, 3, ALLOCATION_GLOBAL)
                == _topk_indices(s, None, 3))

    def test_normalisation_rescues_a_quiet_group(self):
        """The bias this exists to fix: one entity scoring absolutely higher
        across the board would otherwise take every slot."""
        s = scores(0.90, 0.89, 0.88, 0.20, 0.10, 0.05)
        groups = ["loud"] * 3 + ["quiet"] * 3
        got = _topk_indices(s, groups, 2, ALLOCATION_GLOBAL)
        assert {groups[i] for i in got} == {"loud", "quiet"}

    def test_top_k_larger_than_the_pool_returns_everything(self):
        got = _topk_indices(scores(0.1, 0.2), ["Q1", "Q2"], 30, ALLOCATION_GLOBAL)
        assert sorted(got) == [0, 1]


class TestFloorAllocation:
    """The declared sensitivity arm: floor(k/n) reserved per entity."""

    def test_it_is_a_no_op_on_a_single_entity_question(self):
        """floor(k/1) == k, so the reservation is the whole budget. This is why
        the arm can move at most 86 of 200 DEV questions."""
        s = scores(0.1, 0.9, 0.5, 0.7, 0.3)
        groups = ["Q1"] * 5
        assert (_topk_indices(s, groups, 3, ALLOCATION_FLOOR)
                == _topk_indices(s, groups, 3, ALLOCATION_GLOBAL))

    def test_each_entity_is_guaranteed_its_share(self):
        """⭐ THE MECHANISM IS GROUP SIZE, and it is worth stating because it is
        not obvious. Z-normalisation equalises each group's MEAN and SPREAD, but
        a bigger group still reaches a higher maximum z — with 10 items the top
        one sits ~1.57 SDs above its mean, with 2 items only 1.0. So the global
        rule systematically favours the entity with more material, which
        normalisation alone does not fix. The reservation does.

        ⚠️ Two groups of equal size and spread select IDENTICALLY under both
        arms: z-normalisation has already balanced them. That is why this arm
        can only move multi-entity questions with UNEVEN pools.
        """
        s = scores(*[float(i) for i in range(10)], 0.0, 1.0)
        groups = ["big"] * 10 + ["small"] * 2

        glob = _topk_indices(s, groups, 4, ALLOCATION_GLOBAL)
        floor = _topk_indices(s, groups, 4, ALLOCATION_FLOOR)

        def share(sel):
            return {g: sum(1 for i in sel if groups[i] == g) for g in ("big", "small")}

        assert share(glob) == {"big": 3, "small": 1}
        assert share(floor) == {"big": 2, "small": 2}

    def test_a_small_group_gives_its_unused_slots_back(self):
        """⚠️ The common case on the TEXT arm, not an edge case: the median
        article yields 28 chunks and C2 already fails to fill k=30 on 64 of 200
        questions. A reservation that could not be returned would shrink an
        already under-filled pool."""
        s = scores(0.9, 0.8, 0.7, 0.6, 0.5, 0.1)
        groups = ["big"] * 5 + ["tiny"]
        got = _topk_indices(s, groups, 6, ALLOCATION_FLOOR)
        assert len(got) == 6, "unused reservations must return to the global fill"

    def test_the_result_is_ordered_by_score_not_by_reservation(self):
        """The answering model reads the context top-down; a reserved-but-weaker
        line above a stronger one would make the ranking a lie."""
        s = scores(5.0, 4.0, 3.0, 0.5, 0.4, 0.3)
        groups = ["a"] * 3 + ["b"] * 3
        got = _topk_indices(s, groups, 4, ALLOCATION_FLOOR)
        norm = er._normalize_within_groups(s, groups)
        assert list(got) == sorted(got, key=lambda i: -norm[i])

    def test_it_never_returns_more_than_k(self):
        s = scores(*[float(i) for i in range(12)])
        groups = ["a", "b", "c"] * 4
        assert len(_topk_indices(s, groups, 5, ALLOCATION_FLOOR)) == 5

    def test_it_never_returns_a_duplicate(self):
        s = scores(*[float(i) for i in range(9)])
        groups = ["a", "b", "c"] * 3
        got = _topk_indices(s, groups, 7, ALLOCATION_FLOOR)
        assert len(got) == len(set(got))

    def test_k_smaller_than_the_group_count_reserves_nothing(self):
        """floor(2/3) == 0 — the arm degenerates to the global rule rather than
        raising or reserving a fractional slot."""
        s = scores(0.9, 0.1, 0.5, 0.4, 0.3, 0.2)
        groups = ["a", "a", "b", "b", "c", "c"]
        assert (_topk_indices(s, groups, 2, ALLOCATION_FLOOR)
                == _topk_indices(s, groups, 2, ALLOCATION_GLOBAL))


class TestAllocationIsResolvedAtCallTime:

    def test_the_module_global_is_the_default(self, monkeypatch):
        s = scores(*[float(i) for i in range(10)], 0.0, 1.0)
        groups = ["big"] * 10 + ["small"] * 2
        monkeypatch.setattr(er, "ALLOCATION", ALLOCATION_FLOOR)
        assert _topk_indices(s, groups, 4) == _topk_indices(s, groups, 4, ALLOCATION_FLOOR)
        assert _topk_indices(s, groups, 4) != _topk_indices(s, groups, 4, ALLOCATION_GLOBAL)

    def test_an_explicit_argument_overrides_the_global(self, monkeypatch):
        # Uneven pools, so the two arms genuinely differ — see
        # test_each_entity_is_guaranteed_its_share.
        s = scores(*[float(i) for i in range(10)], 0.0, 1.0)
        groups = ["big"] * 10 + ["small"] * 2
        monkeypatch.setattr(er, "ALLOCATION", ALLOCATION_FLOOR)
        assert _topk_indices(s, groups, 4, ALLOCATION_GLOBAL) != \
               _topk_indices(s, groups, 4, ALLOCATION_FLOOR)

    def test_the_shipped_default_is_the_primary_arm(self):
        """Declared before any post-rebuild number existed. If this ever flips
        silently, the reference run stops being the arm it claims to be."""
        assert er.ALLOCATION == ALLOCATION_GLOBAL

    def test_an_unknown_allocation_raises(self):
        with pytest.raises(ValueError, match="unknown allocation"):
            _topk_indices(scores(0.1), ["a"], 1, "per_entity")
