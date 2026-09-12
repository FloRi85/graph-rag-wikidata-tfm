"""
The ranking pool for Configuration 2.

WHY THIS MODULE EXISTS. C3 and C4 build their pool through
`wikidata_pool.build_pool`, which exists because the loop was duplicated in two
pipelines and a divergence between the copies would silently invalidate the
C3-vs-C4 comparison. C2's equivalent loop lived INLINE in `rag.py` -- so the
text arm had no named pool stage at all, and `rag.py`'s capture dict could not
report a pool size while `graph_rag.py`'s could. That is the one field the §3
experiment-flow figure could not fill.

Extracting it makes the two arms structurally comparable: both now answer
"what was available to rank, and which entity did each candidate come from" the
same way, through a function with the same name and signature.

⚠️ IT IS A PARALLEL, NOT A SHARED ABSTRACTION. `wikidata_pool` renders
statements; this renders prose windows. A common base class would have to hide
that difference to work, and the difference IS the experiment. What is shared
is the ranker (`embedding_retriever`), which both pools feed identically.
"""

from __future__ import annotations

from src.retrieval.chunk import Chunk
from src.retrieval.wikipedia import fetch_chunks


def build_pool(
    qids: list[str],
    qid_names: dict[str, str] | None = None,
    *,
    verbose: bool = False,
) -> tuple[list[str], list[str], list[Chunk]]:
    """
    Fetch Wikipedia chunks for every question entity and render them for ranking.

    Returns `(lines, groups, chunks)` — three parallel lists, the same contract
    `wikidata_pool.build_pool` returns:

    `lines`
        The rendered text the embedding ranker scores and the answering model
        reads. For this arm that is the passage itself — see `Chunk.render`.

    `groups`
        The source entity QID per line. The ranker z-normalises within each
        group before the global cut, matching C3/C4. This normalises score
        location and scale; it does not guarantee equal numbers from each
        article, and pool size can still affect selection.

    `chunks`
        The `Chunk` behind each line, kept parallel so downstream code can
        reach the article title, the revision id and the position in the
        article without re-parsing the string.

    ⚠️ `qid_names` IS ACCEPTED AND DELIBERATELY UNUSED. The signature matches
    `wikidata_pool.build_pool` so the two arms are interchangeable to a caller
    and to the contract test — but a Wikipedia passage is already prose and
    prepending the question's mention to it would change what the embedding
    scores. On the KG side that prefix is unavoidable (a statement has no
    natural surface form); here it would be a retrieval change disguised as
    formatting, on the arm that serves as the comparison's baseline.

    ⚠️ DEDUPLICATION IS ON THE RENDERED LINE, matching the KG side. Two entities
    in one question can share an article — Mintaka links the same QID twice on
    22 questions, and distinct QIDs occasionally resolve to one sitelink — and a
    duplicate passage would waste a top-k slot.
    """
    lines: list[str] = []
    groups: list[str] = []
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for qid in qids:
        for chunk in fetch_chunks(qid, verbose=verbose):
            line = chunk.render()
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            groups.append(qid)
            chunks.append(chunk)

    if verbose:
        print(f"  Pooled {len(lines)} Wikipedia chunks, ranking by cosine similarity...")

    return lines, groups, chunks
