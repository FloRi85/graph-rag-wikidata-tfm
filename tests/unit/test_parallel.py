"""
Unit tests for src/eval/parallel.py and the thread-safety fixes it depends on.

No network and no API key: the mapped function is a local stub. What matters here
is that parallelizing an eval run cannot change its RESULTS — only its wall clock.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_parallel.py -v
"""

import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.eval.parallel import map_questions, DEFAULT_WORKERS
from src.retrieval.cache_io import atomic_write_text, read_text_or_none
from src import token_counter


# ---------------------------------------------------------------------------
# map_questions — ordering, equivalence, error propagation
# ---------------------------------------------------------------------------

class TestMapQuestions:
    @pytest.mark.parametrize("workers", [1, 2, 8])
    def test_preserves_input_order(self, workers):
        # Reverse-staggered sleeps: later items finish FIRST, so a naive
        # as-completed collection would scramble the output file's row order.
        import time

        items = list(range(20))

        def slow(i):
            time.sleep((20 - i) * 0.001)
            return i * 10

        assert map_questions(slow, items, workers=workers, progress_every=0) == \
            [i * 10 for i in items]

    def test_parallel_matches_sequential(self):
        items = [{"q": i} for i in range(50)]
        fn = lambda d: d["q"] ** 2
        seq = map_questions(fn, items, workers=1, progress_every=0)
        par = map_questions(fn, items, workers=8, progress_every=0)
        assert seq == par

    def test_empty_input(self):
        assert map_questions(lambda x: x, [], workers=4) == []

    def test_workers_one_runs_inline(self):
        # workers<=1 must not spawn a pool — the debugging path stays on the
        # calling thread so a pdb break or traceback is usable.
        main = threading.current_thread().ident
        seen = map_questions(lambda x: threading.current_thread().ident,
                             [1, 2, 3], workers=1, progress_every=0)
        assert seen == [main, main, main]

    def test_actually_concurrent(self):
        # Guard against a regression to a sequential implementation: with 8
        # workers, at least 2 must be in flight at once.
        import time

        peak = 0
        current = 0
        lock = threading.Lock()

        def track(_):
            nonlocal peak, current
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.05)
            with lock:
                current -= 1
            return None

        map_questions(track, list(range(16)), workers=8, progress_every=0)
        assert peak >= 2

    @pytest.mark.parametrize("workers", [1, 4])
    def test_exception_propagates(self, workers):
        def boom(i):
            if i == 3:
                raise ValueError("kaboom")
            return i

        with pytest.raises(ValueError, match="kaboom"):
            map_questions(boom, list(range(10)), workers=workers, progress_every=0)


# ---------------------------------------------------------------------------
# token_counter — the accumulator is read-modify-write under threads
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class TestTokenCounterThreadSafety:
    def test_concurrent_record_loses_nothing(self):
        token_counter.reset()
        try:
            def hammer():
                for _ in range(500):
                    token_counter.record(_Usage(2, 3))

            threads = [threading.Thread(target=hammer) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            u = token_counter.get()
            assert u["calls"] == 8 * 500
            assert u["prompt_tokens"] == 8 * 500 * 2
            assert u["completion_tokens"] == 8 * 500 * 3
        finally:
            token_counter.reset()


# ---------------------------------------------------------------------------
# atomic_write_text — a reader must never observe a torn cache file
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "q42.json"
        atomic_write_text(p, json.dumps({"v": 2, "triples": [["Q1", "p", "o"]]}))
        assert json.loads(p.read_text(encoding="utf-8"))["v"] == 2

    def test_overwrite_replaces_cleanly(self, tmp_path):
        p = tmp_path / "q.json"
        atomic_write_text(p, "old")
        atomic_write_text(p, "new")
        assert p.read_text(encoding="utf-8") == "new"

    def test_creates_missing_parent(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "q.json"
        atomic_write_text(p, "x")
        assert p.read_text(encoding="utf-8") == "x"

    def test_concurrent_writers_never_yield_partial_content(self, tmp_path):
        # Two entities' fetches landing on the same key must leave one whole
        # payload, never a spliced one. Readers run throughout.
        p = tmp_path / "shared.json"
        a, b = "A" * 20000, "B" * 20000
        atomic_write_text(p, a)
        bad: list[str] = []
        write_errors: list[BaseException] = []
        stop = threading.Event()

        def writer():
            for i in range(40):
                try:
                    atomic_write_text(p, a if i % 2 else b)
                except BaseException as e:      # noqa: BLE001
                    # On Windows os.replace fails while a reader holds the
                    # destination open; the retry loop must absorb that.
                    write_errors.append(e)

        def reader():
            # Uses the same locked reader the cache call sites use — a bare
            # Path.read_text does NOT interlock with the swap.
            while not stop.is_set():
                txt = read_text_or_none(p)
                if txt is not None and txt not in (a, b):
                    bad.append(txt[:40])

        w = [threading.Thread(target=writer) for _ in range(4)]
        r = [threading.Thread(target=reader) for _ in range(2)]
        for t in r + w:
            t.start()
        for t in w:
            t.join()
        stop.set()
        for t in r:
            t.join()

        assert not bad, f"observed {len(bad)} torn reads, e.g. {bad[:1]}"
        assert not write_errors, (
            f"{len(write_errors)} writes failed under contention, "
            f"e.g. {write_errors[0]!r} — the replace retry is not absorbing "
            "Windows sharing violations"
        )

    def test_no_temp_files_left_behind(self, tmp_path):
        for _ in range(5):
            atomic_write_text(tmp_path / "q.json", "data")
        assert [p.name for p in tmp_path.iterdir()] == ["q.json"]


# ---------------------------------------------------------------------------
# Lazy singletons — double-checked locks must hand every caller the SAME object
# ---------------------------------------------------------------------------

class TestSingletonRace:
    def test_get_client_returns_one_instance(self, monkeypatch):
        import src.llm_config as llm_config

        built = []

        class FakeOpenAI:
            def __init__(self, **kw):
                built.append(1)

        monkeypatch.setattr(llm_config, "OpenAI", FakeOpenAI)
        monkeypatch.setattr(llm_config, "_client", None)

        out: list = []
        barrier = threading.Barrier(8)

        def grab():
            barrier.wait()          # maximize the chance of a real race
            out.append(llm_config.get_client())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1, f"client constructed {len(built)} times"
        assert all(c is out[0] for c in out)


def test_default_workers_is_sane():
    assert 1 < DEFAULT_WORKERS <= 32
