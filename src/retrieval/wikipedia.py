r"""
Wikipedia article text for a Wikidata entity: QID -> sitelink -> prose -> chunks.

THE MIRROR OF `wikidata.py`, and it exists because the two arms of the
comparison were not built to the same standard. `text_retriever.py`, which this
replaces, was 113 lines unchanged since May: no retry, no cache version, and
`return []` for three different failures -- no English sitelink, an empty
extract, and a network error -- all of which reached the pipeline as
"No Wikipedia articles found" with nothing recorded anywhere.

Meanwhile C3/C4 gained strict retrieval, a versioned cache, a per-entity status
and a preflight. So the knowledge-graph arm aborted loudly on incomplete
retrieval while the text arm degraded silently, and those two arms are exactly
what the research question compares.

THE FAILURE CONTRACT, MIRRORED FROM THE KG SIDE:

    state "ok" means THE FETCH SUCCEEDED -- including for an entity that
    genuinely has no English Wikipedia article. That is real data, in the same
    way six DEV entities legitimately have zero incoming statements. It is
    recorded (`reason: "no_sitelink"`, `n_chunks: 0`) rather than silently
    producing an empty context.

    state "failed" means we could not find out. Under STRICT_RETRIEVAL that
    raises `IncompleteRetrievalError` -- the SAME exception the KG side raises,
    imported rather than redefined, so `run_eval._is_transient` keeps one
    non-transient registration and neither arm can drift from it.

⚠️ ABSENCE IS LEGAL; NOT KNOWING IS NOT. That distinction is the whole design,
and it is why `no_sitelink` is not an error: 83 Mintaka questions have no linked
entity at all, and many linked entities are minor enough to have no article.
Treating those as failures would abort a fifth of the split.

Public entry points:
    fetch_chunks(qid, strict=None) -> list[Chunk]
    fetch_article_meta(qid)        -> {title, revision_id, redirect, state}
    cache_entry_status(qid)        -> {state, ...}
    cache_fingerprint()            -> dict
    article_cache_is_usable(qid)   -> bool
    empty_articles(qids)           -> {qid: reason}
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.retrieval.cache_io import atomic_write_text, read_text_or_none
from src.retrieval.chunk import Chunk
# ⚠️ IMPORTED, NOT REDEFINED. One exception type across both retrieval arms, so
# `run_eval._NON_TRANSIENT_TYPES` names one class and cannot cover the KG arm
# while missing the text arm.
from src.retrieval.wikidata import IncompleteRetrievalError

MEDIAWIKI_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TFM-GraphRAG/0.1 (florispam@outlook.com)"

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "wikipedia"

# ⭐ VERSIONED FROM 2026-08-16. The 345 files this replaces carried no version
# and no fetch date, so a chunking change left stale windows in place and
# "which text did that run read?" had no answer. A bump invalidates every entry.
ARTICLE_CACHE_VERSION = 1

# Word-window chunking. UNCHANGED VALUES, deliberately: they produced every
# recorded C2 number, and this rebuild is about the machinery around the fetch,
# not about retuning the arm. They are in the fingerprint, so changing them
# later invalidates the cache instead of silently mixing window sizes.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Same posture as `wikidata.STRICT_RETRIEVAL`, for the same reason: a canonical
# run must not answer from a corpus it could not fully read. The delivered
# web demo also pins strict retrieval; explicit callers may override it.
STRICT_RETRIEVAL = True

# Matches the KG side's `_STATEMENT_QUERY_ATTEMPTS`. The text arm had NO retry
# at all, so a single transient timeout produced an empty context that was then
# graded as a retrieval result.
_FETCH_ATTEMPTS = 3
_BACKOFF_BASE = 2.0
_TIMEOUT = 20


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get_json(url: str, *, attempts: int = _FETCH_ATTEMPTS) -> dict | None:
    """GET and parse JSON, retrying transient failures. None when all fail.

    Returns None rather than raising so the CALLER decides whether a failure is
    fatal -- `fetch_chunks` needs to distinguish "the API said this entity has
    no article" from "we never reached the API", and only the second is a
    failure. Raising here would collapse them again.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode())
            # MediaWiki can report an API error with HTTP 200. Treat it as a
            # failed request, never as evidence that an article is absent.
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"]
                code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
                raise urllib.error.URLError(f"MediaWiki API error: {code}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            if attempt == attempts:
                print(f"  [wikipedia] giving up after {attempts} attempts: "
                      f"{type(exc).__name__}: {str(exc)[:90]}")
                return None
            time.sleep(_BACKOFF_BASE ** attempt)
    return None


def _sitelink_title(qid: str) -> tuple[str | None, bool]:
    """(enwiki title, fetch_succeeded).

    (None, True)  -> the API answered and this entity has no English article.
    (None, False) -> we could not ask. A different finding, and the reason this
                     returns a pair instead of an optional string.
    """
    params = urllib.parse.urlencode({
        "action": "wbgetentities", "ids": qid, "props": "sitelinks",
        "sitefilter": "enwiki", "format": "json",
    })
    data = _get_json(f"{MEDIAWIKI_API}?{params}")
    if data is None:
        return None, False
    entity = (data.get("entities") or {}).get(qid) or {}
    entry = (entity.get("sitelinks") or {}).get("enwiki")
    return (entry["title"] if entry else None), True


def _article(title: str) -> tuple[str | None, int | None, bool]:
    """(plain text, revision id, fetch_succeeded) for an article title.

    ⭐ ASKS FOR THE REVISION ID ALONGSIDE THE TEXT, in one call. Wikipedia is
    live and the thesis compares against a 2021 answer key, so a run that cannot
    name the revision it read cannot be reproduced or dated -- and the drift
    penalises the retrieval configs specifically.
    """
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts|revisions", "titles": title,
        "format": "json", "explaintext": True, "exlimit": 1, "rvprop": "ids",
        "redirects": 1,
    })
    data = _get_json(f"{WIKIPEDIA_API}?{params}")
    if data is None:
        return None, None, False
    pages = (data.get("query") or {}).get("pages") or {}
    if not pages:
        return None, None, True
    page = next(iter(pages.values()))
    revisions = page.get("revisions") or [{}]
    return page.get("extract", ""), revisions[0].get("revid"), True


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _windows(text: str, chunk_size: int = CHUNK_SIZE,
             overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Overlapping word windows. Byte-identical to the retired implementation."""
    words = text.split()
    step = chunk_size - overlap
    out: list[str] = []
    for i in range(0, len(words), step):
        window = " ".join(words[i: i + chunk_size])
        if window:
            out.append(window)
        if i + chunk_size >= len(words):
            break
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_fingerprint() -> dict:
    """The settings that determine WHAT was cached.

    Chunk geometry belongs here because it is applied at FETCH time and baked
    into the stored windows -- unlike the KG side, where `_clean_statements`
    filters at return time and its settings are therefore deliberately absent
    from the fingerprint. Changing either value must invalidate the cache
    rather than silently mixing 300-word and 500-word windows in one pool.
    """
    return {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "source": "enwiki"}


def cache_entry_status(qid: str) -> dict:
    """
    Whether `qid`'s cache entry may be used, and why not when it may not.

    THE ONE VALIDATOR, mirroring `wikidata.cache_entry_status` -- fetch, the
    preflight and the prewarm tool all route through it. On the KG side those
    three once disagreed, and prewarm's `Path.exists()` reported "nothing to do"
    while every entry was stale.

    state:
      missing     no file
      malformed   unreadable, or not the expected shape
      stale       older cache version, or different chunk geometry
      failed      the fetch did not complete; nothing can be concluded
      ok          usable -- INCLUDING an entity with no English article, which
                  is recorded with `n_chunks: 0` and a reason
    """
    raw = read_text_or_none(CACHE_DIR / f"{qid}.json")
    if raw is None:
        return {"state": "missing", "qid": qid}
    try:
        cached = json.loads(raw)
    except json.JSONDecodeError:
        return {"state": "malformed", "qid": qid, "reason": "not valid JSON"}
    # ⚠️ A BARE LIST IS THE PRE-v1 SHAPE. The retired retriever wrote
    # `json.dumps(chunks)` -- a list of strings with no version, no fetch date
    # and no article title. Reported as stale rather than tolerated: those files
    # cannot say what they contain.
    if not isinstance(cached, dict) or "chunks" not in cached:
        return {"state": "stale", "qid": qid, "reason": "pre-v1 cache entry"}
    if cached.get("v") != ARTICLE_CACHE_VERSION:
        return {"state": "stale", "qid": qid,
                "reason": f"cache v{cached.get('v')} != v{ARTICLE_CACHE_VERSION}"}
    if cached.get("fingerprint") != cache_fingerprint():
        return {"state": "stale", "qid": qid, "reason": "chunk geometry changed"}
    if not cached.get("fetch_succeeded"):
        return {"state": "failed", "qid": qid,
                "reason": cached.get("reason") or "fetch did not complete"}
    return {"state": "ok", "qid": qid, "entry": cached,
            "reason": cached.get("reason", "")}


def article_cache_is_usable(qid: str) -> bool:
    """True when `qid` can be served from cache without a live fetch."""
    return cache_entry_status(qid)["state"] == "ok"


def empty_articles(qids: list[str]) -> dict[str, str]:
    """{qid: reason} for cached entities that yielded NO chunks.

    The text arm's counterpart to `wikidata.truncated_entities`: a legitimate,
    permitted outcome that must nonetheless be recorded per run rather than
    left in console output. An entity with no English article contributes
    nothing to C2's pool, and a run should be able to say how many did.
    """
    out: dict[str, str] = {}
    for qid in qids:
        status = cache_entry_status(qid)
        if status["state"] != "ok":
            continue
        if not status["entry"].get("chunks"):
            out[qid] = status["entry"].get("reason") or "empty_extract"
    return out


def _write_cache(qid: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(CACHE_DIR / f"{qid}.json",
                          json.dumps(payload, ensure_ascii=False))
    except OSError:
        # Caching is an optimisation; the chunks are already in hand. Losing the
        # write costs one refetch, whereas raising would fail the question.
        pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def fetch_article_meta(qid: str) -> dict:
    """{title, revision_id, reason, fetch_succeeded} for `qid`, from cache if usable.

    The counterpart of `wikidata.fetch_entity_meta`. Kept separate from
    `fetch_chunks` so a caller that only needs to know WHICH article was read
    -- a report, a provenance line -- does not pull the prose along with it.
    """
    status = cache_entry_status(qid)
    if status["state"] == "ok":
        e = status["entry"]
        return {"title": e.get("title"), "revision_id": e.get("revision_id"),
                "reason": e.get("reason", ""), "fetch_succeeded": True}
    return {"title": None, "revision_id": None,
            "reason": status.get("reason", status["state"]),
            "fetch_succeeded": False}


def fetch_chunks(qid: str, strict: bool | None = None,
                 verbose: bool = False) -> list[Chunk]:
    """
    Wikipedia chunks for `qid`, from cache when usable.

    Returns [] both for an entity with no English article and for one whose
    article is empty -- and RECORDS which, so the two are distinguishable
    afterwards even though both yield nothing to rank.

    `strict` defaults to STRICT_RETRIEVAL, resolved at CALL time. Under strict, a
    fetch that fails after its own retries raises IncompleteRetrievalError
    rather than returning an empty list that reads like an absent article.
    """
    if strict is None:
        strict = STRICT_RETRIEVAL

    status = cache_entry_status(qid)
    if status["state"] == "ok":
        return _chunks_from(status["entry"])

    title, ok = _sitelink_title(qid)
    if not ok:
        if strict:
            raise IncompleteRetrievalError(
                f"{qid}: could not read the Wikidata sitelink after "
                f"{_FETCH_ATTEMPTS} attempts, so whether it has an English "
                f"Wikipedia article is UNKNOWN. Not cached — a later run can "
                f"recover.")
        return []

    if title is None:
        # A real, recorded, permitted outcome: this entity has no English
        # article. Cached as `ok` so it is not re-fetched every run.
        _write_cache(qid, _entry(qid, title=None, revision_id=None, chunks=[],
                                 reason="no_sitelink"))
        if verbose:
            print(f"  [wikipedia] {qid} has no English Wikipedia article")
        return []

    text, revision_id, ok = _article(title)
    if not ok:
        if strict:
            raise IncompleteRetrievalError(
                f"{qid} ('{title}'): the article fetch failed after "
                f"{_FETCH_ATTEMPTS} attempts. Not cached.")
        return []

    windows = _windows(text or "")
    _write_cache(qid, _entry(qid, title=title, revision_id=revision_id,
                             chunks=windows,
                             reason="" if windows else "empty_extract"))
    if verbose:
        print(f"  [wikipedia] {qid} '{title}' rev {revision_id}: "
              f"{len(windows)} chunks")
    return [Chunk(text=w, source_entity_id=qid, title=title, index=i,
                  n_chunks=len(windows), revision_id=revision_id)
            for i, w in enumerate(windows)]


def _entry(qid: str, *, title: str | None, revision_id: int | None,
           chunks: list[str], reason: str) -> dict:
    return {
        "v": ARTICLE_CACHE_VERSION,
        "fingerprint": cache_fingerprint(),
        "qid": qid,
        "title": title,
        "revision_id": revision_id,
        # ⚠️ STRUCTURED AND DIRECTIONAL, like the KG side's `query_succeeded`.
        # True means "we found out", not "there was an article".
        "fetch_succeeded": True,
        "reason": reason,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks": chunks,
    }


def _chunks_from(entry: dict) -> list[Chunk]:
    windows = entry.get("chunks") or []
    return [Chunk(text=w, source_entity_id=entry.get("qid", ""),
                  title=entry.get("title") or "", index=i,
                  n_chunks=len(windows), revision_id=entry.get("revision_id"))
            for i, w in enumerate(windows)]
