"""
Base LLM pipeline — Config 1.
No retrieval. The LLM answers purely from parametric knowledge.
Entry point: answer(question) -> str

Variant F uses the same refusal clause and sentinel as C2-C4. This controls
the wording of abstention permission, not every prompt difference: C1 has no
retrieved context or question-entity block and uses a different opening line.
The scorer recognises the exact sentinel and handles other refusal phrasing
separately. Declared C1-only prompt variants test the refusal clause.
"""

from __future__ import annotations

from src import llm_config, prompts, token_counter

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# The persona/system framing lives in llm_config.build_messages, which places it
# where the active model expects it (system slot on OpenAI-style models, user
# prompt on Nemotron). Only the config-specific body belongs here.


def _build_prompt_template() -> str:
    """The C1 template from the LIVE prompts constants.

    The abstention clause sits on its own line directly after the instruction,
    matching the C2-C4 template layout. C1 reads its OWN copy of the clause
    (prompts.ABSTAIN_CLAUSE_C1) -- byte-identical to the shared one except in
    the Group-5 arms of the prompt-abstention study, where an empty string
    drops the line without leaving a stray blank.
    """
    clause = f"{prompts.ABSTAIN_CLAUSE_C1}\n" if prompts.ABSTAIN_CLAUSE_C1 else ""
    return (
        f"{prompts.ANSWER_INSTRUCTION_NO_CONTEXT}\n"
        f"{clause}\n"
        "Question: {question}\n"
        "Answer:"
    )


PROMPT_TEMPLATE = _build_prompt_template()


def rebuild_prompts() -> None:
    """Re-derive PROMPT_TEMPLATE after prompts.apply_variant().

    The template is assembled at import time, so mutating the prompts constants
    alone changes nothing this pipeline sends -- the same reason --top-k
    reassigns the module globals rather than a default argument. Called by
    run_eval before the specs snapshot and before any worker starts; NOT safe
    mid-run (see src/eval/parallel.py).
    """
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = _build_prompt_template()

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer(question: str) -> str:
    """Answer a question using only the LLM's parametric knowledge."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    response = llm_config.complete(
        model=llm_config.MODEL,
        temperature=llm_config.TEMPERATURE,
        max_tokens=llm_config.MAX_TOKENS,
        messages=llm_config.build_messages(prompt),
    )
    token_counter.record(response.usage)
    return llm_config.extract_text(response)
