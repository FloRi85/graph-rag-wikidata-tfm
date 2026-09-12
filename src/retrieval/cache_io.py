"""
Atomic disk-cache writes, shared by the Wikidata and Wikipedia retrievers.

Path.write_text truncates the target and then writes, so a concurrent reader can
observe a half-written file and a concurrent writer can interleave. Both caches
are keyed by QID, and eval runs are parallelized across questions (see
src/eval/parallel.py) — two questions sharing an entity will fetch and write the
same key at the same time. Writing to a temp file in the same directory and then
renaming makes the swap atomic, so a reader sees either the old contents or the
new ones, never a truncated middle.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

# One lock per cache path, shared by readers and the writer.
#
# All eval concurrency is threads in ONE process (see src/eval/parallel.py), so
# an in-process lock is sufficient and — unlike retrying alone — deterministic.
# It matters most on Windows, where os.replace fails outright while any other
# handle holds the destination open: excluding readers for the microseconds of
# the swap removes that conflict instead of retrying around it.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.abspath(str(path))
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def read_text_or_none(path: Path, encoding: str = "utf-8") -> str | None:
    """
    Read a cache file under its lock, or return None if it is absent.

    Cache readers must use this rather than Path.read_text so they interlock
    with atomic_write_text; a bare read can hold the file open across a
    concurrent swap.
    """
    with _lock_for(path):
        try:
            return path.read_text(encoding=encoding)
        except FileNotFoundError:
            return None

# Windows refuses to replace a file another handle currently has open, raising
# PermissionError (WinError 5/32) even though the operation is otherwise valid.
# With parallel workers sharing an entity, a reader can hold the destination for
# the microseconds the swap needs, so a few quick retries clear it. POSIX never
# hits this path.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF = 0.02


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Write `text` to `path` atomically via a same-directory temp file + rename.

    Raises OSError if the swap cannot be completed. Callers writing a CACHE
    should treat that as non-fatal — the payload is already in hand, and a
    missing cache entry only costs a refetch (see get_chunks / fetch_statements).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
        # os.replace is atomic on POSIX and on Windows (unlike os.rename, which
        # raises there if the destination exists). The lock excludes in-process
        # readers; the retry covers an external holder (an editor, a virus
        # scanner, another process) that the lock cannot reach.
        with _lock_for(path):
            for attempt in range(1, _REPLACE_ATTEMPTS + 1):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == _REPLACE_ATTEMPTS:
                        raise
                    time.sleep(_REPLACE_BACKOFF * attempt)
    except BaseException:
        # Never leave a stray temp file behind on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
