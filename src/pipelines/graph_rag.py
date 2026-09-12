"""
Graph-RAG pipeline — Configuration 3.

Gold QIDs -> Wikidata STATEMENTS -> embedding ranking -> top-k bullets -> answer.

Statements preserve qualifiers, rank and typed values beyond a bare
subject-property-value triple. The pipeline renders and ranks these records;
it does not execute symbolic aggregation or multi-hop reasoning.

The ranker is shared with Configurations 2 and 4 (`embedding_retriever`), and
the pool builder with Configuration 4 (`wikidata_pool`), so the only thing that
distinguishes C4 is its condensing call.

Entry point: `answer(question, entity_names, qids, ...) -> str`
"""

from __future__ import annotations

from src import llm_config, prompts, token_counter
from src.prompts import entity_section
from src.retrieval.wikidata_pool import build_pool
from src.retrieval.embedding_retriever import retrieve_context as embed_retrieve
from src.retrieval.context_format import format_ranked

# ---------------------------------------------------------------------------
# Parameters (model/generation params shared via llm_config)
# ---------------------------------------------------------------------------

# Ranked statements handed to the answering LLM. Selected on DEV-200 from
# k in {10, 30, 50}; run_eval --top-k sets the same depth for C2-C4. Selection
# cannot recover evidence absent from the one-hop candidate pool.
TOP_K = 30

# Persona/system framing is placed per-model by llm_config.build_messages.


def _build_prompt_template() -> str:
    """The C3 template from the LIVE prompts constants.

    {entities} is the question-entity block, or nothing at all -- see
    prompts.entity_section. An empty clause (a declared study arm) drops its
    line without leaving a stray blank.
    """
    clause = f"{prompts.ABSTAIN_CLAUSE}\n" if prompts.ABSTAIN_CLAUSE else ""
    return (
        f"{prompts.ANSWER_INSTRUCTION}\n"
        f"{clause}\n"
        "{entities}"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )


PROMPT_TEMPLATE = _build_prompt_template()


def rebuild_prompts() -> None:
    """Re-derive PROMPT_TEMPLATE after prompts.apply_variant().

    Assembled at import time, so mutating the prompts constants alone changes
    nothing this pipeline sends. Called by run_eval before the specs snapshot
    and before any worker starts; NOT safe mid-run (src/eval/parallel.py).
    """
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = _build_prompt_template()

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_ranked_context(entity_label: str, ranked: str) -> str:
    """Reformat embed_retrieve output (double-newline-joined) as a bullet list.

    Delegates to the renderer shared with Config 4's fallback so context
    measurements include the same header and bullet markers as the prompt.
    """
    return format_ranked(entity_label, ranked)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer(
    question: str,
    entity_names: list[str],
    qids: list[str],
    top_k: int | None = None,
    verbose: bool = False,
    capture: dict | None = None,
    entity_block: str = "",
) -> str:
    """
    Answer a question using embedding-ranked Wikidata statements for all question entities.

    Pass `qids` during controlled eval (Mintaka) to bypass entity linking.
    Statements from all entities are pooled before embedding ranking.
    Pass a dict as `capture` to receive the retrieved context (error analysis).

    `top_k` defaults to the module's TOP_K, resolved at CALL time rather than
    bound as a default argument, so sweep tools that set the module global
    between batches take effect (a `top_k: int = TOP_K` default would freeze the
    import-time value and silently ignore them).
    """
    if top_k is None:
        top_k = TOP_K
    # `qids` and `entity_names` are parallel lists from
    # `parse_questions.Question.qids` / `.entity_mentions`. Mintaka supplies the
    # QIDs, so entity linking is bypassed and cannot confound retrieval quality
    # — a stated methodological choice, and since 2026-08-16 a property of this
    # signature rather than a branch nothing could reach. A demo resolves names
    # first, via `src/retrieval/entity_linking.py`.
    qid_names = dict(zip(qids, entity_names))

    # Build the ranking pool. Shared with Config 4 (src/retrieval/wikidata_pool.py) so the
    # two configs cannot drift apart in their retrieval stage — the comparison
    # depends on them differing in exactly one step, the condensing call.
    fact_strings, groups, _ = build_pool(qids, qid_names, verbose=verbose)

    ranked = embed_retrieve(question, fact_strings, top_k=top_k, groups=groups)

    if verbose:
        print(f"  Selected top-{top_k} statements after embedding ranking")

    label = " / ".join(entity_names) if entity_names else "entities"
    context = _format_ranked_context(label, ranked)

    if capture is not None:
        capture["pool_size"] = len(fact_strings)
        capture["context"] = context

    prompt = PROMPT_TEMPLATE.format(entities=entity_section(entity_block),
                                    context=context, question=question)
    response = llm_config.complete(
        model=llm_config.MODEL,
        temperature=llm_config.TEMPERATURE,
        max_tokens=llm_config.MAX_TOKENS,
        messages=llm_config.build_messages(prompt),
    )
    token_counter.record(response.usage)
    return llm_config.extract_text(response)
