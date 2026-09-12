"""
Tests for the --prompt-variant mechanism (docs/reproduction.md).

WHY THESE EXIST. The prompt-abstention study varies the most result-moving
strings in the repo (brevity line +-54 pts, refusal clause +-33 pts), through a
mechanism that mutates module constants and re-derives import-time templates.
Two silent failure modes are pinned here:

1. The default drifting. Every recorded result was produced by the exact
   variant-F strings; extracting the templates into builder functions must be a
   pure refactor. The default templates and styles are pinned as LITERALS, so
   a change to any fragment or builder surfaces as a diff against the recorded
   prompt, not as a mysteriously moved number.

2. A variant applying partially. The templates are assembled at import time,
   so apply_variant() without rebuild_prompts() changes NOTHING a pipeline
   sends -- an arm would run the shipped prompt under the arm's label, which is
   the exact mislabelling meta.json exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import llm_config, prompts
import src.pipelines.base_llm as base_llm
import src.pipelines.rag as rag
import src.pipelines.graph_rag as graph_rag
import src.pipelines.graph_rag_rerank as graph_rag_rerank

_PIPELINES = (base_llm, rag, graph_rag, graph_rag_rerank)

# The shipped strings, captured at import of THIS module (before any test
# mutates them) so the restore fixture cannot depend on what it restores.
_SHIPPED_PERSONA = llm_config._PERSONA
_SHIPPED_BREVITY = llm_config._BREVITY

_NEMOTRON = "nvidia/llama-3.3-nemotron-super-49b-v1"


def _rebuild_all() -> None:
    for mod in _PIPELINES:
        mod.rebuild_prompts()


@pytest.fixture()
def variant_sandbox():
    """Register a synthetic variant; restore the shipped state afterwards.

    Restoration goes through the same public path a run uses (apply_variant("f")
    + apply_prompt_overrides + rebuild), so the fixture also exercises the claim
    that "f" restores the byte-identical shipped prompt.
    """
    yield
    prompts.PROMPT_VARIANTS.pop("_test", None)
    prompts.apply_variant("f")
    llm_config.apply_prompt_overrides(persona=_SHIPPED_PERSONA,
                                      brevity=_SHIPPED_BREVITY)
    _rebuild_all()


class TestDefaultsAreByteIdentical:
    """The refactor must not have changed one byte of the shipped prompt."""

    C1_TEMPLATE = (
        "Answer the following question as concisely as possible.\n"
        'Refuse if you don\'t find the answer and reply exactly: '
        '"The answer is not in the context."\n\n'
        "Question: {question}\n"
        "Answer:"
    )
    RETRIEVAL_TEMPLATE = (
        "Use the following information to answer the question as concisely as possible.\n"
        'Refuse if you don\'t find the answer and reply exactly: '
        '"The answer is not in the context."\n\n'
        "{entities}"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    def test_c1_template(self):
        assert base_llm.PROMPT_TEMPLATE == self.C1_TEMPLATE

    def test_retrieval_templates(self):
        assert rag.PROMPT_TEMPLATE == self.RETRIEVAL_TEMPLATE
        assert graph_rag.PROMPT_TEMPLATE == self.RETRIEVAL_TEMPLATE
        assert graph_rag_rerank.ANSWER_PROMPT == self.RETRIEVAL_TEMPLATE

    def test_c1_clause_is_the_shared_clause(self):
        # The parity argument in base_llm's docstring, as a property.
        assert prompts.ABSTAIN_CLAUSE_C1 == prompts.ABSTAIN_CLAUSE

    def test_nemotron_style(self):
        answer = llm_config.prompt_style(_NEMOTRON, role="answer")
        condense = llm_config.prompt_style(_NEMOTRON, role="condense")
        assert answer == {
            "system": "detailed thinking off",
            "preamble": _SHIPPED_PERSONA + " " + _SHIPPED_BREVITY + "\n\n",
        }
        assert condense == {
            "system": "detailed thinking off",
            "preamble": _SHIPPED_PERSONA + "\n\n",
        }

    def test_default_style(self):
        assert llm_config.prompt_style("gpt-4o-mini", role="answer") == {
            "system": _SHIPPED_PERSONA, "preamble": ""}
        assert llm_config.prompt_style("gpt-4o-mini", role="condense") == {
            "system": None, "preamble": ""}

    def test_default_variant_is_f_and_registered(self):
        assert prompts.PROMPT_VARIANT == "f"
        assert prompts.PROMPT_VARIANTS["f"] == {}


class TestApplyVariant:
    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown prompt variant"):
            prompts.apply_variant("nope")

    def test_unknown_override_key_raises(self, variant_sandbox):
        prompts.PROMPT_VARIANTS["_test"] = {"sentinel": "I don't know."}
        with pytest.raises(KeyError, match="unknown override keys"):
            prompts.apply_variant("_test")

    def test_shared_clause_reaches_all_four_templates(self, variant_sandbox):
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause": 'Do not guess. Reply exactly: "The answer is not in the context."'}
        prompts.apply_variant("_test")
        _rebuild_all()
        for tpl in (base_llm.PROMPT_TEMPLATE, rag.PROMPT_TEMPLATE,
                    graph_rag.PROMPT_TEMPLATE, graph_rag_rerank.ANSWER_PROMPT):
            assert "Do not guess." in tpl
            assert "Refuse if you don't find" not in tpl
        assert prompts.PROMPT_VARIANT == "_test"

    def test_c1_only_clause_leaves_c2_c4_alone(self, variant_sandbox):
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause_c1": 'If you do not know the answer, reply exactly: '
                                 '"The answer is not in the context."'}
        prompts.apply_variant("_test")
        _rebuild_all()
        assert "If you do not know" in base_llm.PROMPT_TEMPLATE
        for tpl in (rag.PROMPT_TEMPLATE, graph_rag.PROMPT_TEMPLATE,
                    graph_rag_rerank.ANSWER_PROMPT):
            assert "If you do not know" not in tpl
            assert "Refuse if you don't find" in tpl

    def test_empty_c1_clause_drops_line_cleanly(self, variant_sandbox):
        prompts.PROMPT_VARIANTS["_test"] = {"abstain_clause_c1": ""}
        prompts.apply_variant("_test")
        _rebuild_all()
        assert base_llm.PROMPT_TEMPLATE == (
            "Answer the following question as concisely as possible.\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        # No stray blank line: exactly one \n\n between instruction and question.
        assert "\n\n\n" not in base_llm.PROMPT_TEMPLATE

    def test_condense_prompt_is_never_rebuilt(self, variant_sandbox):
        before = graph_rag_rerank.CONDENSE_PROMPT
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause": 'X reply exactly: "The answer is not in the context."',
            "answer_instruction": "Y"}
        prompts.apply_variant("_test")
        _rebuild_all()
        assert graph_rag_rerank.CONDENSE_PROMPT == before

    def test_variants_do_not_compose(self, variant_sandbox):
        # Each variant derives from the SHIPPED baseline, not from the previous
        # variant -- two arms applied in one process must not stack.
        prompts.PROMPT_VARIANTS["_test"] = {"answer_instruction": "Y instruction."}
        prompts.apply_variant("_test")
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause": 'Z reply exactly: "The answer is not in the context."'}
        prompts.apply_variant("_test")
        assert prompts.ANSWER_INSTRUCTION.startswith("Use the following")
        assert prompts.ABSTAIN_CLAUSE.startswith("Z ")

    def test_f_restores_byte_identical_defaults(self, variant_sandbox):
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause": 'W reply exactly: "The answer is not in the context."'}
        prompts.apply_variant("_test")
        _rebuild_all()
        prompts.apply_variant("f")
        _rebuild_all()
        assert base_llm.PROMPT_TEMPLATE == TestDefaultsAreByteIdentical.C1_TEMPLATE
        assert rag.PROMPT_TEMPLATE == TestDefaultsAreByteIdentical.RETRIEVAL_TEMPLATE


class TestStyleOverrides:
    def test_persona_and_brevity_override(self, variant_sandbox):
        llm_config.apply_prompt_overrides(persona="P.", brevity="B.")
        style = llm_config.prompt_style(_NEMOTRON, role="answer")
        assert style["preamble"] == "P. B.\n\n"
        assert llm_config.prompt_style(_NEMOTRON, role="condense")["preamble"] == "P.\n\n"
        assert llm_config.prompt_style("gpt-4o-mini", role="answer")["system"] == "P."

    def test_empty_brevity_leaves_no_stray_space(self, variant_sandbox):
        llm_config.apply_prompt_overrides(brevity="")
        style = llm_config.prompt_style(_NEMOTRON, role="answer")
        assert style["preamble"] == _SHIPPED_PERSONA + "\n\n"

    def test_empty_persona_and_brevity_mean_no_preamble(self, variant_sandbox):
        llm_config.apply_prompt_overrides(persona="", brevity="")
        assert llm_config.prompt_style(_NEMOTRON, role="answer")["preamble"] == ""
        assert llm_config.prompt_style(_NEMOTRON, role="condense")["preamble"] == ""
        # An empty persona means NO system message on the OpenAI style, not "".
        assert llm_config.prompt_style("gpt-4o-mini", role="answer")["system"] is None

    def test_none_means_keep(self, variant_sandbox):
        llm_config.apply_prompt_overrides(persona=None, brevity=None)
        assert llm_config.prompt_style(_NEMOTRON, role="answer")["preamble"] == (
            _SHIPPED_PERSONA + " " + _SHIPPED_BREVITY + "\n\n")

    def test_mode_switch_survives_every_override(self, variant_sandbox):
        # The system slot selects Nemotron's reasoning mode; no wording arm may
        # touch it.
        llm_config.apply_prompt_overrides(persona="", brevity="")
        for role in ("answer", "condense"):
            assert llm_config.prompt_style(_NEMOTRON, role=role)["system"] == \
                "detailed thinking off"


class TestSpecRecordsVariant:
    def test_prompt_variant_in_specs(self, variant_sandbox):
        from src.experiment_spec import collect_specs
        prompts.PROMPT_VARIANTS["_test"] = {
            "abstain_clause": 'V reply exactly: "The answer is not in the context."'}
        prompts.apply_variant("_test")
        _rebuild_all()
        specs = collect_specs()
        frags = specs["prompt_fragments"]
        assert frags["PROMPT_VARIANT"] == "_test"
        assert frags["ABSTAIN_CLAUSE"].startswith("V ")
        assert frags["ABSTAIN_CLAUSE_C1"].startswith("V ")
        # The templates section carries the REBUILT template, so the arm's
        # meta.json shows what was actually sent.
        assert specs["prompt_templates"]["C2_rag"].startswith(
            "Use the following")
        assert "V reply exactly" in specs["prompt_templates"]["C2_rag"]
