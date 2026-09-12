"""
Unit tests for src/retrieval/wikipedia.py — the text arm's fetch/cache layer.

No network: `_get_json` is patched throughout.

⚠️ THE LOAD-BEARING DISTINCTION IS "ABSENT" vs "UNKNOWN". The retriever this
replaces returned [] for a missing sitelink, an empty extract AND a network
error alike, and all three reached the pipeline as "No Wikipedia articles
found" with nothing recorded. Most of what follows pins them apart.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_wikipedia.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval import wikipedia
from src.retrieval.wikidata import IncompleteRetrievalError

SITELINK_OK = {"entities": {"Q42": {"sitelinks": {"enwiki": {"title": "Douglas Adams"}}}}}
SITELINK_NONE = {"entities": {"Q42": {"sitelinks": {}}}}


def article(text: str, revid: int = 7788):
    return {"query": {"pages": {"1": {"extract": text, "revisions": [{"revid": revid}]}}}}


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Every test gets an empty cache dir; none can touch the real one."""
    monkeypatch.setattr(wikipedia, "CACHE_DIR", tmp_path / "wikipedia")
    yield


def responses(*payloads):
    """A `_get_json` stub returning each payload in turn."""
    seq = list(payloads)
    return lambda url, **kw: seq.pop(0) if seq else None


class TestAbsentIsNotUnknown:

    def test_no_english_article_is_a_SUCCESSFUL_fetch(self, monkeypatch):
        """Real data, not a failure: many Wikidata items have no enwiki page.

        The parallel is the KG arm, where six DEV entities legitimately have
        zero incoming statements and are still `ok`.
        """
        monkeypatch.setattr(wikipedia, "_get_json", responses(SITELINK_NONE))
        assert wikipedia.fetch_chunks("Q42") == []
        status = wikipedia.cache_entry_status("Q42")
        assert status["state"] == "ok"
        assert status["entry"]["reason"] == "no_sitelink"
        assert status["entry"]["chunks"] == []

    def test_an_empty_extract_is_recorded_distinctly(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("")))
        assert wikipedia.fetch_chunks("Q42") == []
        assert wikipedia.cache_entry_status("Q42")["entry"]["reason"] == "empty_extract"

    def test_a_failed_sitelink_lookup_raises_under_strict(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json", responses(None))
        with pytest.raises(IncompleteRetrievalError, match="sitelink"):
            wikipedia.fetch_chunks("Q42")

    def test_a_failed_article_fetch_raises_under_strict(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json", responses(SITELINK_OK, None))
        with pytest.raises(IncompleteRetrievalError, match="article fetch failed"):
            wikipedia.fetch_chunks("Q42")

    def test_a_failure_is_NOT_cached(self, monkeypatch):
        """So a later run can recover instead of inheriting the damage — the
        same rule the KG arm applies to a failed own-label query, where caching
        the failure poisoned every incoming statement of that entity."""
        monkeypatch.setattr(wikipedia, "_get_json", responses(SITELINK_OK, None))
        with pytest.raises(IncompleteRetrievalError):
            wikipedia.fetch_chunks("Q42")
        assert wikipedia.cache_entry_status("Q42")["state"] == "missing"

    def test_non_strict_returns_empty_instead_of_raising(self, monkeypatch):
        """The demo path, where a partial answer beats no answer."""
        monkeypatch.setattr(wikipedia, "_get_json", responses(None))
        assert wikipedia.fetch_chunks("Q42", strict=False) == []


class TestChunksCarryProvenance:

    def test_a_chunk_knows_where_it_came_from(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("word " * 700, revid=999)))
        chunks = wikipedia.fetch_chunks("Q42")
        assert len(chunks) == 3            # 700 words, 300-word windows, 50 overlap
        first = chunks[0]
        assert first.source_entity_id == "Q42"
        assert first.title == "Douglas Adams"
        assert first.revision_id == 999
        assert (first.index, first.n_chunks) == (0, 3)

    def test_render_returns_the_bare_prose(self, monkeypatch):
        """Prefixing the title would change what the embedding scores — a
        retrieval change wearing the clothes of a formatting one."""
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("Douglas Adams was a writer.")))
        [chunk] = wikipedia.fetch_chunks("Q42")
        assert chunk.render() == "Douglas Adams was a writer."

    def test_windows_match_the_retired_implementation(self):
        """Byte-identical chunking: this rebuild changed the machinery around
        the fetch, not the arm's retrieval behaviour."""
        words = [f"w{i}" for i in range(700)]
        got = wikipedia._windows(" ".join(words))
        assert [len(w.split()) for w in got] == [300, 300, 200]
        assert got[1].split()[0] == "w250"      # 300 - 50 overlap

    def test_a_cached_entity_is_served_without_a_fetch(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("a b c")))
        wikipedia.fetch_chunks("Q42")

        def explode(*a, **kw):
            raise AssertionError("re-fetched an entity that was cached")

        monkeypatch.setattr(wikipedia, "_get_json", explode)
        assert [c.text for c in wikipedia.fetch_chunks("Q42")] == ["a b c"]


class TestCacheValidation:

    def _write(self, qid, payload):
        wikipedia.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (wikipedia.CACHE_DIR / f"{qid}.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def test_a_pre_v1_bare_list_is_stale_not_usable(self):
        """The 345 files this replaces were `json.dumps(chunks)` — a list of
        strings that cannot say which article, which revision, or what chunk
        geometry produced it."""
        self._write("Q42", ["some chunk", "another chunk"])
        status = wikipedia.cache_entry_status("Q42")
        assert status["state"] == "stale"
        assert "pre-v1" in status["reason"]

    def test_a_version_bump_invalidates_every_entry(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("a b c")))
        wikipedia.fetch_chunks("Q42")
        assert wikipedia.cache_entry_status("Q42")["state"] == "ok"
        monkeypatch.setattr(wikipedia, "ARTICLE_CACHE_VERSION",
                            wikipedia.ARTICLE_CACHE_VERSION + 1)
        assert wikipedia.cache_entry_status("Q42")["state"] == "stale"

    def test_changed_chunk_geometry_invalidates_the_cache(self, monkeypatch):
        """Otherwise a pool silently mixes 300-word and 500-word windows."""
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("a b c")))
        wikipedia.fetch_chunks("Q42")
        monkeypatch.setattr(wikipedia, "CHUNK_SIZE", 500)
        status = wikipedia.cache_entry_status("Q42")
        assert status["state"] == "stale" and "geometry" in status["reason"]

    def test_malformed_json_is_reported_not_raised(self):
        wikipedia.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (wikipedia.CACHE_DIR / "Q42.json").write_text("{not json", encoding="utf-8")
        assert wikipedia.cache_entry_status("Q42")["state"] == "malformed"

    def test_empty_articles_lists_entities_that_yielded_nothing(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json", responses(SITELINK_NONE))
        wikipedia.fetch_chunks("Q42")
        # A DIFFERENT qid needs its own sitelink payload — wbgetentities keys
        # the response by the id asked for.
        sitelink_q1 = {"entities": {"Q1": {"sitelinks": {"enwiki": {"title": "Universe"}}}}}
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(sitelink_q1, article("a b c")))
        wikipedia.fetch_chunks("Q1")
        assert wikipedia.empty_articles(["Q42", "Q1"]) == {"Q42": "no_sitelink"}


class TestRetry:

    @pytest.mark.parametrize("article_stage", [False, True])
    def test_api_error_with_http_success_is_not_cached(self, monkeypatch, article_stage):
        """HTTP 200 with an API error is unknown data, not an absent article."""
        calls = []

        class Resp:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return json.dumps(self.payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def api_error(request, timeout=None):
            calls.append(request.full_url)
            if article_stage and len(calls) == 1:
                return Resp(SITELINK_OK)
            return Resp({"error": {"code": "maxlag", "info": "Please retry"}})

        monkeypatch.setattr(wikipedia.urllib.request, "urlopen", api_error)
        monkeypatch.setattr(wikipedia.time, "sleep", lambda *_: None)
        with pytest.raises(IncompleteRetrievalError):
            wikipedia.fetch_chunks("Q42")
        assert len(calls) == wikipedia._FETCH_ATTEMPTS + int(article_stage)
        assert wikipedia.cache_entry_status("Q42")["state"] == "missing"

    def test_api_error_can_recover_without_changing_success_payload(self, monkeypatch):
        payloads = [{"error": {"code": "maxlag"}}, SITELINK_OK]

        class Resp:
            def read(self):
                return json.dumps(payloads.pop(0)).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(wikipedia.urllib.request, "urlopen", lambda *a, **kw: Resp())
        monkeypatch.setattr(wikipedia.time, "sleep", lambda *_: None)
        assert wikipedia._get_json("http://example.invalid", attempts=2) == SITELINK_OK
        assert payloads == []

    def test_a_transient_failure_is_retried(self, monkeypatch):
        """The retired retriever had NO retry, so one timeout produced an empty
        context that was then graded as a retrieval result."""
        calls = {"n": 0}

        def always_times_out(url, timeout=None):
            calls["n"] += 1
            raise TimeoutError("slow")

        monkeypatch.setattr(wikipedia.urllib.request, "urlopen", always_times_out)
        monkeypatch.setattr(wikipedia.time, "sleep", lambda *_: None)
        assert wikipedia._get_json("http://x", attempts=3) is None
        assert calls["n"] == 3, "exactly `attempts` tries, no more and no fewer"

    def test_a_recovered_failure_returns_the_payload(self, monkeypatch):
        calls = {"n": 0}

        class Resp:
            def read(self):
                return b'{"ok": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def flaky(url, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("slow")
            return Resp()

        monkeypatch.setattr(wikipedia.urllib.request, "urlopen", flaky)
        monkeypatch.setattr(wikipedia.time, "sleep", lambda *_: None)
        assert wikipedia._get_json("http://x", attempts=3) == {"ok": True}


class TestArticleMeta:

    def test_meta_reports_the_article_and_revision(self, monkeypatch):
        monkeypatch.setattr(wikipedia, "_get_json",
                            responses(SITELINK_OK, article("a b c", revid=4242)))
        wikipedia.fetch_chunks("Q42")
        meta = wikipedia.fetch_article_meta("Q42")
        assert meta == {"title": "Douglas Adams", "revision_id": 4242,
                        "reason": "", "fetch_succeeded": True}

    def test_meta_on_an_uncached_entity_says_so(self):
        meta = wikipedia.fetch_article_meta("Q42")
        assert meta["fetch_succeeded"] is False and meta["title"] is None
