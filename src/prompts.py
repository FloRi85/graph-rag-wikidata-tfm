"""
Prompt fragments shared across all four configurations.

Why this module exists at all: the abstention clause has to be IDENTICAL in
every config, and the scorer has to recognise exactly the string the clause
asks for. Three copy-pasted literals in the pipelines plus a fourth in the
scorer is four chances to drift. Defining them once makes identity structural
rather than something a reviewer has to verify by eye.

Deliberately import-light -- no openai, no torch -- so `metrics.py` can import
the sentinel without pulling a client library into a pure scoring path.
"""

from __future__ import annotations

# The exact string every config is instructed to emit when it declines.
# metrics.is_abstention() matches this first, before its paraphrase regex.
ABSTAIN_SENTINEL = "The answer is not in the context."

# Variant F gives all four configurations identical refusal wording and names
# an exact sentinel so compliance can be recognised without guessing paraphrases.
# C1 still differs from C2-C4 in its opening line and lack of context/entity block.
ABSTAIN_CLAUSE = (
    f'Refuse if you don\'t find the answer and reply exactly: "{ABSTAIN_SENTINEL}"'
)

# "Don't find" avoids explicitly conditioning C1's refusal on a missing
# Context block. The sensitivity study tests alternative clauses; wording alone
# does not guarantee how a model interprets the instruction or uses context.

# C1's copy of the clause. Defaults to the SHARED clause -- byte-identical, so
# the parity argument in base_llm.py's docstring stays true by construction.
# It exists as a separate constant because the prompt-abstention study
# (docs/reproduction.md, C1-only variants) runs arms where ONLY C1's
# clause changes while C2-C4 keep variant F. Outside those arms it must never
# be set independently: two constants that can silently diverge is exactly the
# drift this module exists to prevent, which is why apply_variant() below is
# the only sanctioned writer.
ABSTAIN_CLAUSE_C1 = ABSTAIN_CLAUSE

# ---------------------------------------------------------------------------
# Opening instruction line
# ---------------------------------------------------------------------------
#
# The first line of every answering prompt. Split in two because C1 has no
# Context: block -- telling a no-retrieval config to "use the following
# information" would name a section that is not there.
#
# These opening lines are held fixed outside the declared instruction variant.
# The question-entity block is another difference: it is absent from C1.
ANSWER_INSTRUCTION = (
    "Use the following information to answer the question as concisely as possible."
)

ANSWER_INSTRUCTION_NO_CONTEXT = (
    "Answer the following question as concisely as possible."
)

# ---------------------------------------------------------------------------
# The question-entity block
# ---------------------------------------------------------------------------
#
# Mintaka annotates what each question is ABOUT: the Wikidata QID and canonical
# label of every entity mentioned, plus normalised literals for the dates and
# numbers. Until 2026-08-16 the pipelines read only the QIDs, so the labels and
# literals reached nothing. This puts them in the prompt as a short block above
# `Context:`.
#
# ⚠️ C2, C3 AND BOTH OF C4'S CALLS RECEIVE IT; C1 DOES NOT (decisions of
# 2026-08-16). The consequence has to be stated in §5 rather than discovered by
# a reader: the C1-vs-retrieval delta now measures retrieval PLUS an entity
# annotation, so C1 is no longer a clean control.
#
# ⚠️ Placed BEFORE `Context:` because it is a legend for it. The context lines
# are prefixed with the question's MENTION (`[Academy Award] winner: ...`) while
# the block leads with the canonical LABEL, so reading the block first supplies
# the mapping the context relies on.
#
# ⚠️ IT IS A DECLARED PROMPT ARM, and both values are run. This project has
# measured prompt wording moving its headline by tens of points three separate
# times (the brevity line, the refusal clause, and the C3-vs-C2 separation
# itself), so a new prompt section is an experimental variable, not formatting.
#   "all"     linked entities + normalised literals   <- the reference run
#   "entity"  linked entities only                    <- the variant
# `experiment_spec` records the value, so a result file always says which
# prompt produced it.
ENTITY_BLOCK_MODE = "all"


# ---------------------------------------------------------------------------
# Prompt variants (the abstention sensitivity study, 2026-08-18)
# ---------------------------------------------------------------------------
#
# One registry entry per declared prompt arm (see docs/reproduction.md).
# Keys an entry may carry (absent key = the shipped variant-F value):
#
#   abstain_clause                shared clause, C1-C4. Also sets the C1 copy
#                                 unless abstain_clause_c1 is given explicitly.
#   abstain_clause_c1             C1's clause alone (Group 5). "" = no clause.
#   answer_instruction            C2-C4 opening line
#   answer_instruction_no_context C1 opening line
#   persona                       llm_config persona ("" = removed)
#   brevity                       llm_config Nemotron brevity line ("" = removed)
#
# ⚠️ THE SENTINEL IS NOT A KEY, DELIBERATELY. metrics.is_abstention() matches
# ABSTAIN_SENTINEL verbatim, and a stored run is re-scored months later against
# the LIVE sentinel -- an arm with a different sentinel would re-score with all
# its abstentions invisible. Only the condition around the sentinel varies.
#
# "f" is the shipped prompt: an empty override, present so the default is a
# named member of the same family rather than a special case.
PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "f": {},
    # --- Group 1: refusal clause (shared, all four configs) ---------------
    # Earlier clause, re-tested on the statement-model DEV reference setup.
    # It explicitly names context, although C1 has no Context block.
    "c-prime": {
        "abstain_clause": (
            f'If the answer is not in the context, reply exactly: "{ABSTAIN_SENTINEL}"'
        ),
    },
    # Verification framing: refusal conditioned on checkability, not presence.
    "verify": {
        "abstain_clause": (
            f'If you cannot verify the answer from the context, reply exactly: "{ABSTAIN_SENTINEL}"'
        ),
    },
    # Prohibition-of-guessing frame around the unchanged F condition.
    "no-guess": {
        "abstain_clause": (
            f'Do not guess. If you don\'t find the answer, reply exactly: "{ABSTAIN_SENTINEL}"'
        ),
    },
    # --- Group 2: brevity line (Nemotron answering preamble) --------------
    # Soft length pressure without the fact-production demand.
    "keep-short": {
        "brevity": "Keep your answer short.",
    },
    # The shipped line plus an explicit counterweight sentence.
    "refusal-ok": {
        "brevity": ("Answer with the fact only - no explanation, no restatement "
                    "of the question. Refusing is an acceptable answer."),
    },
    # --- Group 3: persona ---------------------------------------------------
    # Humility norm as a standing disposition rather than a per-question clause.
    "decline-over-guess": {
        "persona": ("You are a careful assistant. When you are not sure, "
                    "it is better to decline than to guess."),
    },
    # Lower anchor: does the persona do anything at all?
    "no-persona": {
        "persona": "",
    },
    # --- Group 4: instruction line ------------------------------------------
    # Exclusivity instruction, the standard RAG grounding guardrail. C1's
    # opening line is unchanged (it has no "below" to point at).
    "only-below": {
        "answer_instruction": "Answer using ONLY the information below.",
    },
    # --- Group 5: C1-only clause (C2-C4 keep F verbatim) --------------------
    # The on-record recommendation, finally measured at n=200.
    "c1-dont-know": {
        "abstain_clause_c1": (
            f'If you do not know the answer, reply exactly: "{ABSTAIN_SENTINEL}"'
        ),
    },
    # No abstention permission for C1 at all (the historical "C1a", run
    # properly as a declared arm).
    "c1-no-clause": {
        "abstain_clause_c1": "",
    },
}

# Which variant the module constants currently reflect. Recorded into every
# meta.json via experiment_spec, so two runs differing only in prompt wording
# are distinguishable -- the same argument as ENTITY_BLOCK_MODE.
PROMPT_VARIANT = "f"

# The variant-F values, captured once at import so apply_variant() can always
# derive any variant from the SHIPPED baseline rather than from whatever the
# previous call left behind. Without this, applying two variants in one process
# would compose them.
_DEFAULTS = {
    "abstain_clause": ABSTAIN_CLAUSE,
    "abstain_clause_c1": ABSTAIN_CLAUSE_C1,
    "answer_instruction": ANSWER_INSTRUCTION,
    "answer_instruction_no_context": ANSWER_INSTRUCTION_NO_CONTEXT,
}


def apply_variant(name: str) -> dict[str, str]:
    """Set this module's constants to the named variant; return its overrides.

    Returns the raw override dict so the caller can forward the two keys this
    module does not own (persona, brevity) to llm_config.apply_prompt_overrides
    -- prompts must stay import-light, so it cannot import llm_config itself.

    ⚠️ The caller must then rebuild the pipeline templates (each pipeline's
    rebuild_prompts()): they are assembled from these constants at import time,
    so mutating the constants alone changes nothing a pipeline sends. run_eval
    is the one sanctioned call site and does both steps together.

    Unknown names raise rather than falling back to the default -- a mistyped
    arm name must not silently run the shipped prompt under an arm's label.
    """
    global ABSTAIN_CLAUSE, ABSTAIN_CLAUSE_C1
    global ANSWER_INSTRUCTION, ANSWER_INSTRUCTION_NO_CONTEXT, PROMPT_VARIANT
    if name not in PROMPT_VARIANTS:
        raise KeyError(
            f"Unknown prompt variant {name!r}. Registered: "
            f"{sorted(PROMPT_VARIANTS)}. Arms are registered in "
            f"prompts.PROMPT_VARIANTS; see docs/reproduction.md.")
    ov = PROMPT_VARIANTS[name]
    unknown = set(ov) - (set(_DEFAULTS) | {"persona", "brevity"})
    if unknown:
        raise KeyError(f"Variant {name!r} carries unknown override keys: "
                       f"{sorted(unknown)}")
    ABSTAIN_CLAUSE = ov.get("abstain_clause", _DEFAULTS["abstain_clause"])
    # The shared clause reaches C1 too unless the arm names a C1-specific one --
    # a Group-1 arm changes ALL FOUR configs, which is the house invariant.
    ABSTAIN_CLAUSE_C1 = ov.get("abstain_clause_c1",
                               ov.get("abstain_clause",
                                      _DEFAULTS["abstain_clause_c1"]))
    ANSWER_INSTRUCTION = ov.get("answer_instruction",
                                _DEFAULTS["answer_instruction"])
    ANSWER_INSTRUCTION_NO_CONTEXT = ov.get(
        "answer_instruction_no_context",
        _DEFAULTS["answer_instruction_no_context"])
    PROMPT_VARIANT = name
    return dict(ov)


def entity_section(block: str) -> str:
    """A rendered block plus its trailing blank line, or "" when there is none.

    The templates interpolate this rather than the block itself so an absent
    block leaves NO trace -- no header, no stray blank line. 83 questions have
    no linked entity at all, and a prompt with an empty section above `Context:`
    would differ structurally from one without, on exactly the questions where
    retrieval already has nothing.
    """
    return f"{block}\n\n" if block else ""
