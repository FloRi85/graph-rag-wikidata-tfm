"""
The context formatter is shared, and its OVERHEAD is part of the measurement.

WHY THIS FILE EXISTS. The renderer was duplicated in graph_rag and in
graph_rag_rerank's fallback, and a third caller (the context-size diagnostic)
measured the bare fact strings instead of the rendered block. The published
inflation figure was computed from that third instrument and compared against a
baseline measured on the full stored contexts, so "context did NOT inflate,
235 -> 244 (1.04x)" was an artifact of the mismatch. The same-instrument answer
is 235 -> 280 (1.19x).

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_context_format.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines import graph_rag, graph_rag_rerank
from src.retrieval.context_format import (
    context_words, format_facts, format_ranked,
)


class TestFormat:
    def test_header_and_bullets_are_part_of_the_context(self):
        out = format_facts("Tom Hanks", ["a: b", "c: d"])
        assert out == "Wikidata facts about Tom Hanks:\n  • a: b\n  • c: d"

    def test_empty_facts_say_so_rather_than_rendering_a_bare_header(self):
        assert format_facts("X", []) == "No relevant facts found for X."
        assert format_ranked("X", "") == "No relevant facts found for X."

    def test_ranked_input_is_split_on_blank_lines(self):
        assert format_ranked("X", "a: b\n\nc: d") == format_facts("X", ["a: b", "c: d"])

    def test_overhead_is_material_and_must_not_be_dropped(self):
        """
        One bullet marker per fact, plus the header. At top_k=30 that is ~35
        words — about 15% of a C3 context, and exactly the amount by which
        measuring the bare facts under-reported the median. It is what made a
        1.19x inflation look like 1.04x.
        """
        facts = [f"property {i}: value" for i in range(30)]
        bare = sum(len(f.split()) for f in facts)
        header_words = len("Wikidata facts about Tom Hanks:".split())
        rendered = context_words(format_facts("Tom Hanks", facts))
        assert rendered - bare == header_words + len(facts)
        assert rendered - bare == 35
        assert (rendered - bare) / rendered > 0.10

    def test_context_words_matches_the_stored_measurement(self):
        """Same statistic as metrics.context_words and the run files' field."""
        text = format_facts("X", ["a: b"])
        assert context_words(text) == len(text.split())


class TestPipelinesShareOneRenderer:
    def test_config_3_delegates_to_the_shared_formatter(self):
        assert (graph_rag._format_ranked_context("X", "a: b\n\nc: d")
                == format_facts("X", ["a: b", "c: d"]))

    def test_config_4_fallback_renders_identically_to_config_3(self):
        """
        C4's condensing-failure path had its own inline copy, so a change to
        C3's format would silently not apply to it.
        """
        facts = ["a: b", "c: d"]
        assert format_facts("X", facts) == graph_rag._format_ranked_context(
            "X", "\n\n".join(facts))
        assert graph_rag_rerank.format_facts is format_facts
