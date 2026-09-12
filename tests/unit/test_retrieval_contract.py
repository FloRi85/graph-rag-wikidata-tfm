"""
The two retrieval arms expose the same surface.

WHY THIS FILE EXISTS. `src/retrieval/wikidata*.py` and `src/retrieval/wikipedia*.py`
are deliberately PARALLEL rather than sharing a base class: statements and prose
windows are genuinely different things, and a common abstraction would have to
hide that difference to work — which is the difference the thesis measures.

The cost of that choice is drift. `run_eval`'s preflight, the prewarm tools and
`experiment_spec` all treat the two arms as interchangeable, so a function that
loses an argument on one side, or a cache validator that stops reporting
`state`, breaks a caller that looks identical. These tests make drift fail here
instead of mid-run.

⚠️ THEY ASSERT SHAPE, NOT BEHAVIOUR. Nothing here says the two arms should
retrieve alike — they must not. It says a caller can ask both the same
questions.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_retrieval_contract.py -v
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval import wikidata, wikidata_pool, wikipedia, wikipedia_pool
from src.retrieval.chunk import Chunk
from src.retrieval.statement import Statement

ARMS = (("wikidata", wikidata, wikidata_pool, "fetch_statements"),
        ("wikipedia", wikipedia, wikipedia_pool, "fetch_chunks"))


class TestCacheValidatorContract:
    """`run_eval`'s preflight and the prewarm tools route through these."""

    @pytest.mark.parametrize("name,module", [(a[0], a[1]) for a in ARMS])
    def test_each_arm_has_the_three_cache_functions(self, name, module):
        for fn in ("cache_entry_status", "cache_fingerprint"):
            assert callable(getattr(module, fn, None)), f"{name} lacks {fn}"

    @pytest.mark.parametrize("name,module", [(a[0], a[1]) for a in ARMS])
    def test_status_takes_a_qid_and_reports_a_state(self, name, module):
        status = module.cache_entry_status("Q-does-not-exist-999999")
        assert status["state"] == "missing", f"{name} mis-reports an absent entry"
        assert status["qid"] == "Q-does-not-exist-999999"

    @pytest.mark.parametrize("name,module", [(a[0], a[1]) for a in ARMS])
    def test_fingerprint_is_a_plain_json_able_dict(self, name, module):
        fp = module.cache_fingerprint()
        assert isinstance(fp, dict) and fp, f"{name} has an empty fingerprint"
        for k, v in fp.items():
            assert isinstance(k, str)
            assert isinstance(v, (str, int, float, bool, list, tuple)), \
                f"{name}.{k} is not serialisable into meta.json"

    @pytest.mark.parametrize("name,module", [(a[0], a[1]) for a in ARMS])
    def test_each_arm_has_a_usability_predicate(self, name, module):
        fn = getattr(module, "statement_cache_is_usable", None) or \
             getattr(module, "article_cache_is_usable", None)
        assert callable(fn), f"{name} has no cache_is_usable predicate"
        assert fn("Q-does-not-exist-999999") is False


class TestFetchContract:

    @pytest.mark.parametrize("name,module,_pool,fetch", ARMS)
    def test_fetch_takes_a_qid_and_a_strict_flag(self, name, module, _pool, fetch):
        """`strict` must be resolvable at CALL time, so the demo path can lower
        it without changing the eval's behaviour — the same convention `top_k`
        follows in the pipelines."""
        sig = inspect.signature(getattr(module, fetch))
        assert list(sig.parameters)[0] == "qid", f"{name}: first arg is not qid"
        assert "strict" in sig.parameters, f"{name}: no strict flag"
        assert sig.parameters["strict"].default is None, \
            f"{name}: strict must default to None and read the module global"

    @pytest.mark.parametrize("name,module,_pool,_fetch", ARMS)
    def test_each_arm_has_a_strict_switch(self, name, module, _pool, _fetch):
        assert module.STRICT_RETRIEVAL is True, \
            f"{name} is not strict by default; a canonical run would answer " \
            f"from a partial corpus"

    def test_both_arms_raise_the_same_exception_type(self):
        """One class, imported not redefined, so `run_eval._is_transient`'s
        single non-transient registration covers both arms. Two classes with
        the same name would leave one arm's failures retried 4x3 times."""
        assert wikipedia.IncompleteRetrievalError is wikidata.IncompleteRetrievalError


class TestPoolContract:

    @pytest.mark.parametrize("name,_m,pool,_f", ARMS)
    def test_build_pool_has_the_same_signature(self, name, _m, pool, _f):
        sig = inspect.signature(pool.build_pool)
        params = list(sig.parameters)
        assert params[:2] == ["qids", "qid_names"], \
            f"{name}.build_pool: positional args diverged"
        assert sig.parameters["verbose"].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("name,_m,pool,_f", ARMS)
    def test_build_pool_returns_three_parallel_lists(self, name, _m, pool, _f):
        lines, groups, records = pool.build_pool([], {})
        assert lines == [] and groups == [] and records == []

    def test_the_record_types_are_deliberately_different(self):
        """The contract is on the SURROUNDING machinery, not the record. If
        these ever unify, the two arms have stopped retrieving different things
        and the experiment has lost its independent variable."""
        assert Statement is not Chunk
        assert hasattr(Statement, "render") and hasattr(Chunk, "render")


class TestRecordedOutcomeContract:
    """Each arm records its own permitted-but-notable outcome per entity.

    Truncation (KG) and an absent article (text) are different events, but both
    are legal, both change what a config saw, and both used to reach the user
    only as console output that scrolls away.
    """

    def test_kg_arm_reports_truncated_entities(self):
        assert wikidata.truncated_entities([]) == {}

    def test_text_arm_reports_empty_articles(self):
        assert wikipedia.empty_articles([]) == {}
