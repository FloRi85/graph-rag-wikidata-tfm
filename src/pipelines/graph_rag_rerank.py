"""
Graph-RAG + LLM condensing pipeline — Configuration 4.

Gold QIDs -> Wikidata STATEMENTS -> embedding ranking -> LLM condensing -> answer.

Both C3 and C4 call the same statement pool builder and ranker. C4 adds an LLM
call that rewrites the ranked statements as prose before answering. This tests
the complete condensing step, including its additional prompt and generation.

Entry point: `answer(question, entity_names, qids, ...) -> str`
"""

from __future__ import annotations

from src import llm_config, token_counter
from src import prompts
from src.prompts import entity_section
from src.retrieval.wikidata_pool import build_pool
from src.retrieval.embedding_retriever import retrieve_context as embed_retrieve
from src.retrieval.context_format import format_facts

# ---------------------------------------------------------------------------
# Parameters (model/generation params shared via llm_config)
# ---------------------------------------------------------------------------

# Ranked facts passed to the condensing LLM; the same depth as C2 and C3.
EMBED_TOP_K         = 30
# Condensing has a separate budget from short-answer generation. Its finish
# reason is captured before the answering call overwrites the thread-local slot.
CONDENSE_MAX_TOKENS = 384
# Answer generation remains matched to the other configurations.
ANSWER_MAX_TOKENS   = llm_config.MAX_TOKENS  # Mintaka answers are short — matches other configs

# Persona/system framing is placed per-model by llm_config.build_messages. This
# config makes TWO calls per question and they take different roles: "condense"
# must not inherit the answering style's brevity instruction, or the step would
# be told to emit a bare fact instead of the prose Config 4 exists to test.

# Intermediate condensing prompt — runs AFTER embedding ranking, BEFORE answering LLM.
# Output is prose that becomes {context} in the answering prompt.
# Fixed before reference-v8: the explicit length and facts-only instructions
# address excessive commentary observed during DEV checks. Removing overruns
# did not establish a corresponding improvement in abstention or correctness.
CONDENSE_PROMPT = (
    "Given the following Wikidata facts and the question, select only the relevant facts "
    "and summarise them in at most two sentences.\n"
    "Write only the facts themselves. Do not explain your reasoning, do not comment "
    "on what the facts fail to cover, and do not use knowledge beyond them.\n\n"
    # Both C4 calls receive the same question-entity block.
    "{entities}"
    "Facts:\n{top_k_facts}\n\n"
    "Question: {question}\n"
    "Summary:"
)

def _build_answer_prompt() -> str:
    """The C4 ANSWERING template from the LIVE prompts constants.

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


ANSWER_PROMPT = _build_answer_prompt()


def rebuild_prompts() -> None:
    """Re-derive ANSWER_PROMPT after prompts.apply_variant().

    ⚠️ CONDENSE_PROMPT is deliberately NOT rebuilt: it shares no constant with
    the answering prompts, and the prompt-abstention study holds it fixed in
    every arm -- the condensing step is what C4 measures, and varying it would
    confound two experiments. Called by run_eval before the specs snapshot and
    before any worker starts; NOT safe mid-run (src/eval/parallel.py).
    """
    global ANSWER_PROMPT
    ANSWER_PROMPT = _build_answer_prompt()

# ---------------------------------------------------------------------------
# Condensing step (step 2 of Config 4)
# ---------------------------------------------------------------------------

def _condense(question: str, facts: list[str], capture: dict | None = None,
              entity_block: str = "") -> str:
    """
    Ask the LLM to select the relevant facts from the embedding-ranked pool
    and rewrite them as natural prose. Returns the prose summary.

    `capture` receives this call's own finish reason before the answering call
    replaces the thread-local value. Empty input skips the condensing call.
    """
    if not facts:
        return ""
    facts_block = "\n".join(f"- {fact}" for fact in facts)
    prompt = CONDENSE_PROMPT.format(entities=entity_section(entity_block),
                                    top_k_facts=facts_block, question=question)
    response = llm_config.complete(
        model=llm_config.MODEL,
        temperature=llm_config.TEMPERATURE,
        max_tokens=CONDENSE_MAX_TOKENS,
        messages=llm_config.build_messages(prompt, role="condense"),
    )
    token_counter.record(response.usage)
    text = llm_config.extract_text(response)
    if capture is not None:
        capture["condense_finish_reason"] = llm_config.last_finish_reason()
        capture["condense_words"] = len(text.split())
    return text


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
    Answer a question using embedding-ranked, then LLM-condensed Wikidata statements
    for all question entities.

    Pass `qids` during controlled eval (Mintaka) to bypass entity linking.
    Statements from all entities are pooled before embedding ranking.
    Step 1 (embedding ranking) is identical to Config 3. Step 2 (LLM condensing)
    rewrites the top-k statements as prose — the only difference from Config 3.

    `top_k` matches Config 3's signature and defaults to EMBED_TOP_K, resolved at
    CALL time. Until 2026-07-27 this config had no depth parameter at all: it read
    the module global only, so run_eval.py could not vary C4's retrieval depth and
    the sweep tools had to mutate `graph_rag_rerank.EMBED_TOP_K` from outside.
    That still works — the global remains the default — but is no longer required.
    """
    if top_k is None:
        top_k = EMBED_TOP_K
    # `qids` and `entity_names` are parallel lists from
    # `parse_questions.Question.qids` / `.entity_mentions`. Mintaka supplies the
    # QIDs, so entity linking is bypassed and cannot confound retrieval quality
    # — a stated methodological choice, and since 2026-08-16 a property of this
    # signature rather than a branch nothing could reach. A demo resolves names
    # first, via `src/retrieval/entity_linking.py`.
    qid_names = dict(zip(qids, entity_names))

    # Step 1: embedding ranking — the SAME pool builder Config 3 calls
    # (src/retrieval/wikidata_pool.py), so the shared retrieval stage is enforced by the
    # code rather than by two copies staying in sync. The only difference from
    # Config 3 is step 2 below.
    fact_strings, groups, _ = build_pool(qids, qid_names, verbose=verbose)

    ranked = embed_retrieve(question, fact_strings, top_k=top_k, groups=groups)
    top_facts = [line for line in ranked.split("\n\n") if line.strip()]

    if verbose:
        print(f"  Kept top-{top_k} via embedding, condensing to prose...")

    # Step 2: LLM condenses the embedding-ranked facts into natural prose
    context = _condense(question, top_facts, capture=capture,
                        entity_block=entity_block)
    # Written True AND False (2026-08-19): an explicit False marks the row as
    # INSTRUMENTED, so a reader can tell "no fallback happened" from "a file
    # written before the flag existed" — absence means not measured.
    if capture is not None:
        capture["condense_fallback"] = False
    if not context:
        # The SAME renderer Config 3 uses (src/retrieval/context_format.py).
        # It was an inline copy here, so a change to C3's format would silently
        # not apply to C4's fallback and the two would drift apart.
        label = " / ".join(entity_names) if entity_names else "entities"
        context = format_facts(label, top_facts)
        # Record fallback explicitly. With facts it supplies the C3-style
        # bullet context; with an empty pool it supplies only the no-facts
        # message, not retrieved evidence.
        if capture is not None:
            capture["condense_fallback"] = True

    if verbose:
        print(f"  Condensed to prose ({len(context)} chars)")

    if capture is not None:
        capture["pool_size"] = len(fact_strings)
        capture["top_facts"] = top_facts
        capture["context"] = context

    prompt = ANSWER_PROMPT.format(entities=entity_section(entity_block),
                                  context=context, question=question)
    response = llm_config.complete(
        model=llm_config.MODEL,
        temperature=llm_config.TEMPERATURE,
        max_tokens=ANSWER_MAX_TOKENS,
        messages=llm_config.build_messages(prompt),
    )
    token_counter.record(response.usage)
    return llm_config.extract_text(response)
