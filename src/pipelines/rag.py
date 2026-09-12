"""
Classic RAG pipeline — Config 2.
Retrieves Wikipedia text via Wikidata sitelinks, embeds it, retrieves the
most relevant chunks by cosine similarity, then answers with the LLM.
Entry point: answer(question, entity_names, qids, ...) -> str
"""

from __future__ import annotations

from src import llm_config, prompts, token_counter
from src.prompts import entity_section
from src.retrieval.wikipedia_pool import build_pool
from src.retrieval.embedding_retriever import retrieve_context as embed_retrieve

# ---------------------------------------------------------------------------
# Parameters (model/generation params shared via llm_config)
# ---------------------------------------------------------------------------

# Ranked Wikipedia chunks handed to the answering LLM. The shared k=30 was
# selected on DEV-200; run_eval --top-k changes all three retrieval configs.
# Equal item counts do not equalise context length: chunks contain up to 300
# words, whereas each graph item is a rendered statement.
TOP_K = 30

# Persona/system framing is placed per-model by llm_config.build_messages.


def _build_prompt_template() -> str:
    """The C2 template from the LIVE prompts constants.

    The question-entity block ({entities}) is placed BEFORE Context: as a
    legend for it: those lines lead with the canonical label while the context
    lines are prefixed with the question's mention, so the mapping is supplied
    before the facts that rely on it. An empty clause (a declared study arm)
    drops its line without leaving a stray blank.
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
    Answer a question using Wikipedia text for all question entities.

    `qids` is REQUIRED — during controlled eval (Mintaka) the gold QIDs bypass
    entity linking, and since 2026-08-16 there is no in-pipeline fallback: a
    demo caller must resolve names via `src/retrieval/entity_linking.py` first.
    Chunks from all entities are pooled before embedding ranking.
    Pass a dict as `capture` to receive the retrieved context (error analysis).

    `top_k` defaults to the module's TOP_K, resolved at call time — matching
    Configs 3 and 4, so all three retrieval configs share one signature.
    """
    if top_k is None:
        top_k = TOP_K
    # Keep pool construction separate from the shared ranking stage.
    all_chunks, groups, records = build_pool(qids, verbose=verbose)

    if not all_chunks:
        context = f"No Wikipedia articles found for: {', '.join(entity_names)}."
    else:
        # C2-C4 share per-entity z-normalisation and the same allocation rule.
        context = embed_retrieve(question, all_chunks, top_k=top_k, groups=groups)

    if capture is not None:
        # `pool_size` matches what graph_rag.py records, so "how many candidates
        # were available to rank" is answerable for all three retrieval configs.
        capture["pool_size"] = len(all_chunks)
        capture["context"] = context
        # Preserve article revisions in the run record independently of later
        # cache refreshes. One entry per source entity and article title.
        articles: dict[tuple, dict] = {}
        for c in records:
            key = (c.source_entity_id, c.title)
            if key not in articles:
                articles[key] = {"qid": c.source_entity_id, "title": c.title,
                                 "revision_id": c.revision_id,
                                 "n_chunks": c.n_chunks}
        capture["articles"] = list(articles.values())

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
