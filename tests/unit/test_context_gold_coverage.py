"""Distinguish fact-line presence from full rendered-context presence."""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import CORRECT
from src.eval.parse_answers import build_gold_answer
from tools.context_gold_coverage import (
    CONFIGS, build_joined, gold_in_context, print_rendered_context_transitions,
)


def test_header_only_gold_form_is_not_a_fact_line_detection(capsys):
    facts = ["[John McCain] candidacy in election: 2008 presidential election"]
    full_context = "Wikidata facts about John McCain / Mitt Romney:\n  • " + facts[0]
    prose = "The available facts do not provide the requested vote comparison."
    gold = build_gold_answer({
        "answerType": "entity",
        "answer": [{"name": "Q4496", "label": {"en": "Mitt Romney"}}],
        "mention": "Mitt Romney",
    })
    rows = [{
        "id": "header-only", "type": "politics",
        "contexts": {
            "rag": {"context": ""},
            "graph_rag": {"context": full_context},
            "rerank": {"context": prose, "top_facts": facts},
        },
    }]
    scores = {c: {"header-only": {"outcome": CORRECT, "complexity": "comparative"}}
              for c in CONFIGS}
    joined = build_joined(rows, {"header-only": gold}, scores)
    saved = deepcopy(joined)

    assert gold_in_context("\n".join(facts), ["Mitt Romney"], "entity") is False
    assert joined["graph_rag"]["header-only"]["in_ctx"] is True
    assert joined["rerank"]["header-only"]["in_ctx"] is False
    transitions = print_rendered_context_transitions(joined)

    assert transitions[(True, False)] == 1
    assert transitions[(False, False)] == 0
    assert joined == saved
    output = capsys.readouterr().out
    assert "FULL RENDERED C3 GRAPH CONTEXT" in output
    assert "includes question-entity names" in output
    assert "no longer detected in C4 prose=1" in output


def test_full_context_transition_excludes_undefined_checks(capsys):
    joined = {
        "graph_rag": {"q1": {"in_ctx": None}, "q2": {"in_ctx": True}},
        "rerank": {
            "q1": {"in_ctx": None, "outcome": CORRECT},
            "q2": {"in_ctx": None, "outcome": CORRECT},
        },
    }
    transitions = print_rendered_context_transitions(joined)
    assert sum(transitions.values()) == 0
    assert "eligible n=0" in capsys.readouterr().out
