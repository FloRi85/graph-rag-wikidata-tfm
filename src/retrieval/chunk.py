"""
One passage of Wikipedia prose, with the provenance that makes it reproducible.

THE MIRROR OF `statement.py`, and it exists for the same reason. Config 2 used
to pass bare strings around: `get_chunks(qid) -> list[str]`. A string cannot say
which article it came from, which revision of it, or where in the article it
sat, so nothing downstream could tell a chunk of the Tom Hanks article from a
chunk of the Philadelphia article, and no run could state which text it had
read. C3/C4 have carried that structure since the statement-model rebuild; the
text arm carrying only strings is why "did the corpus change between runs?" was
unanswerable for C2.

⚠️ IT IS NOT A `Statement`, AND THE DIFFERENCE IS THE POINT OF THE EXPERIMENT.
A statement is a claim with a subject, a property and a value; a chunk is 300
words of prose that may contain many claims, one claim, or none. The two arms
of the comparison retrieve genuinely different things, and giving them one
dataclass would hide exactly the difference the thesis is measuring. The
symmetry that matters is in the SURROUNDING machinery -- cache versioning, a
failure contract, a pool builder -- not in the record itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A word-window of one Wikipedia article.

    `source_entity_id`
        The Wikidata QID whose sitelink led here. Parallel to
        `Statement.source_entity_id`, and what `wikipedia_pool.build_pool` uses
        as the group label for per-entity score normalisation.

    `title`
        The English Wikipedia article title the sitelink resolved to. Kept
        because it is NOT derivable from the QID after the fact: sitelinks
        change, and an article can be renamed between runs.

    `index` / `n_chunks`
        Position in the article and how many windows it was split into.
        Together they say WHERE in the article a passage sat, which is what
        makes "the answer was in the lead" and "the answer was in a trivia
        section at the end" distinguishable in an error analysis.

    `revision_id`
        The MediaWiki revision this text came from, or None when the API did
        not supply one. ⭐ THIS IS THE FIELD THAT MAKES A C2 RUN REPRODUCIBLE.
        Wikipedia is live; without it, "the corpus changed under us" is a
        suspicion rather than a measurement -- the same drift already in
        §Limitations for Wikidata, which penalises the retrieval configs
        against a 2021 answer key.
    """

    text: str
    source_entity_id: str
    title: str
    index: int
    n_chunks: int
    revision_id: int | None = None

    def render(self) -> str:
        """The text as the ranker and the answering model see it.

        ⚠️ RETURNS THE BARE PROSE, deliberately. `Statement.render()` composes
        `[subject] property: value` because a statement has no natural surface
        form; a Wikipedia passage already is one. Prefixing it with the article
        title would add ~2 words to every chunk and change what the embedding
        scores -- a retrieval change wearing the clothes of a formatting one,
        on the arm that is the comparison's baseline.

        The method exists anyway so the two pools have the same shape: both
        build their ranking input by calling `.render()` on their records.
        """
        return self.text
