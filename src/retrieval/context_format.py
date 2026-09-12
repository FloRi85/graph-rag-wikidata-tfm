"""
How ranked facts become the `{context}` block the answering LLM receives.

Shared by C3 and C4's fallback. Word counts must include headers and bullet
markers, not just the bare facts, to measure the context actually supplied.

Deliberately import-light -- no LLM client, no retrieval, no embedding model --
so a diagnostic can measure exactly what a pipeline builds without dragging in
the pipeline.
"""

from __future__ import annotations

BULLET = "  • "


def format_facts(entity_label: str, facts: list[str]) -> str:
    """The context block for a list of already-ranked fact strings.

    THE one renderer for Config 3 and for Config 4's fallback. Anything that
    needs to know how big a context is must call this rather than measuring the
    facts alone.
    """
    if not facts:
        return f"No relevant facts found for {entity_label}."
    bullets = "\n".join(f"{BULLET}{line}" for line in facts)
    return f"Wikidata facts about {entity_label}:\n{bullets}"


def format_ranked(entity_label: str, ranked: str) -> str:
    """Same, from `embed_retrieve`'s double-newline-joined output."""
    if not ranked:
        return f"No relevant facts found for {entity_label}."
    return format_facts(entity_label, [l for l in ranked.split("\n\n") if l.strip()])


def context_words(text: str) -> int:
    """Word count of a context block, the way every stored measurement counts it.

    `len(text.split())`, matching `metrics.context_words` and the `context_words`
    field recorded in every run file -- so a live measurement and an archived one
    are the same statistic.
    """
    return len(text.split())
