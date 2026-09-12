"""
The one description of what a run was configured with.

WHY THIS IS IN `src/` AND NOT IN `tools/`. It used to live inside
`tools/document_run.py`, which runs AFTER a run completes. That meant the full
specs could only be written at documentation time, so `run_eval` wrote a
shallow stub before the first API call and hoped the run survived to be
documented. It also meant `run_eval` would have had to import from `tools/` to
capture provenance up front. Moving it here fixes both: the snapshot is taken
before anything is spent, and `tools/document_run.py` re-exports it unchanged.

WHY NOT A NEW SOURCE OF TRUTH. Nothing here DEFINES a setting. Every value is
read from the module that owns it, at call time, so a renamed or retuned
constant shows up in the next snapshot automatically and cannot drift out of
sync with a hand-maintained list. Reading in this direction (spec -> modules)
also keeps the import graph acyclic: `wikidata.py` must never learn about C4's
condensing cap, which is what putting the registry in the retrieval layer would
have required.

READ DIRECTLY, NEVER THROUGH `getattr(..., default)`. These are all first-party
constants in this repo, so there is no version skew to tolerate -- and a default
turns a renamed constant into a plausible value instead of an error.
`getattr(wikidata, "FILTER_MODE", None)` did exactly that: the live name is
NOISE_FILTER_MODE, so every run documented before 2026-08-10 recorded
`"noise_filter_mode": null` while the runs actually used `property_type`. An
AttributeError here is the cheapest possible outcome; a null in an archived spec
is the dearest.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Bumped when the SHAPE of a spec changes, so a reader can tell a v1 snapshot
# from a v2 one instead of inferring it from which keys happen to be present.
# ⚠️ Runs documented before this existed carry no `schema_version`. They are
# LEGACY and must be read as such -- never backfilled, because inventing
# historical provenance is worse than admitting it was not captured.
SPEC_SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    """A git command's output, stripped. For values that are single tokens."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "(unavailable)"


class GitUnavailableError(RuntimeError):
    """git itself failed — cleanliness is UNKNOWN, not clean."""


def _git_status_porcelain(strict: bool = False) -> list[str]:
    """Dirty paths, parsed from NUL-delimited porcelain records.

    ⚠️ A SEPARATE READER FROM `_git`, AND THAT IS THE WHOLE POINT. Porcelain
    lines are `XY <path>` where X or Y may be a SPACE (" M file"). `_git`
    strips, which silently removes the leading space of the FIRST line only --
    so `line[3:]` then cut one character too many and exactly one recorded path
    came out mangled: `ata/analysis/gold_qid_existence.json`. It read as a typo
    rather than a bug, which is why it survived. `_git` still strips, because
    SHAs and branch names need it.

    `-z` also removes the quoting/escaping `--porcelain` applies to paths with
    spaces or non-ASCII characters.

    `strict=False` (the default) swallows a git failure and returns [] — right
    for SNAPSHOT recording, where a missing git means "unknown", and wrong for
    a GUARD, where it would read as "clean". A guard passes `strict=True` and
    fails closed on `GitUnavailableError`.
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain", "-z"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:
        if strict:
            raise GitUnavailableError(f"git status failed: {e}") from e
        return []
    records = [r for r in out.split("\0") if r]
    paths: list[str] = []
    skip_next = False
    for rec in records:
        if skip_next:                 # rename/copy: the ORIGIN path is its own
            skip_next = False         # record and is not a dirty path itself
            continue
        if len(rec) < 4:
            continue
        status, path = rec[:2], rec[3:]
        if status[0] in ("R", "C"):
            skip_next = True
        paths.append(path)
    return paths


# Paths a RUN ITSELF writes. Their presence in porcelain says "uncommitted run
# outputs exist", not "the code differs from the SHA" -- and the dirty flag
# exists to answer the second question. Without this filter every run flagged
# its own scorer as unverifiable simply by having written its own untracked
# output directory before scoring it (observed 2026-08-17: all six post-rebuild
# runs carried scorer_tree_dirty=true, the dirt being the run's own directory,
# so the flag carried no information).
_RUN_OUTPUT_PREFIXES = ("data/results/",)


def code_dirty_paths(strict: bool = False) -> list[str]:
    """Dirty paths that can change what a run DOES: porcelain minus run outputs.

    `strict=True` raises GitUnavailableError instead of returning [] when git
    itself fails — for guards that must fail closed.
    """
    return [p for p in _git_status_porcelain(strict=strict)
            if not p.replace("\\", "/").startswith(_RUN_OUTPUT_PREFIXES)]


def collect_specs() -> dict:
    """Everything needed to interpret or repeat a run, read from the live code."""
    from src import llm_config, prompts
    from src.pipelines import base_llm, rag, graph_rag, graph_rag_rerank as rr
    from src.retrieval import wikidata, wikidata_pool, wikipedia, embedding_retriever

    dirty = code_dirty_paths()
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the SHA does NOT fully describe what ran.
            # Recorded rather than hidden -- an unrecorded dirty tree is how a
            # run becomes uninterpretable six weeks later.
            "dirty": bool(dirty),
            "dirty_files": dirty[:40],
        },
        "model": {
            "name": llm_config.MODEL,
            # Read from the environment, not a module constant: llm_config
            # resolves LLM_BASE_URL inside get_client() and never stores it.
            # Recording "(provider default)" for an NVIDIA run would misdescribe it.
            "base_url": os.getenv("LLM_BASE_URL") or "(OpenAI default)",
            "temperature": llm_config.TEMPERATURE,
            "max_tokens": llm_config.MAX_TOKENS,
            # Provider flags complete() adds for this model (e.g. nemotron-3's
            # enable_thinking=False). {} for the reference model and gpt-4o-mini
            # -- recorded so a run on a reasoning model says which mode it ran in.
            "request_extras": llm_config.request_extras(),
        },
        # prompt_style's signature is (model, role) -- passing the role
        # POSITIONALLY silently records the wrong framing rather than raising:
        # "answer" is looked up as a model name, misses _PROMPT_STYLES, and
        # falls back to the default OpenAI style for both roles. Every Nemotron
        # run documented before 2026-08-10 recorded the factual-assistant
        # persona in the system slot when the real calls sent "detailed thinking
        # off". Keyword-only here, and pinned by TestPromptFramingIsRecorded.
        "prompt_framing": {
            role: dict(llm_config.prompt_style(role=role))
            for role in ("answer", "condense")
        } if hasattr(llm_config, "prompt_style") else {},
        "prompt_fragments": {
            "ABSTAIN_SENTINEL": prompts.ABSTAIN_SENTINEL,
            "ABSTAIN_CLAUSE": prompts.ABSTAIN_CLAUSE,
            # C1's copy of the clause — byte-identical to the shared one except
            # in the Group-5 arms of the prompt-abstention study, which is
            # exactly when a reader needs to see the difference.
            "ABSTAIN_CLAUSE_C1": prompts.ABSTAIN_CLAUSE_C1,
            # ⭐ THE THIRD DECLARED ARM AXIS (after ENTITY_BLOCK_MODE and
            # topk_allocation): which pre-registered prompt variant the run's
            # wording reflects — "f" is the shipped prompt. The exact strings
            # above and the templates below are the ground truth; this is the
            # arm's NAME, so a study run identifies itself without a diff.
            "PROMPT_VARIANT": prompts.PROMPT_VARIANT,
            "ANSWER_INSTRUCTION": prompts.ANSWER_INSTRUCTION,
            "ANSWER_INSTRUCTION_NO_CONTEXT": prompts.ANSWER_INSTRUCTION_NO_CONTEXT,
            # ⭐ A DECLARED PROMPT ARM, and the reason this row exists rather
            # than the value being assumed: both "all" and "entity" are run and
            # both are reported. Without it in the snapshot, two runs differing
            # only in this would be indistinguishable in their meta.json --
            # which is exactly how a prompt variant becomes unattributable.
            # Read directly, never via getattr with a default: a renamed
            # constant must surface as an error, not as a plausible value.
            "ENTITY_BLOCK_MODE": prompts.ENTITY_BLOCK_MODE,
        },
        "prompt_templates": {
            "C1_base_llm": base_llm.PROMPT_TEMPLATE,
            "C2_rag": rag.PROMPT_TEMPLATE,
            "C3_graph_rag": graph_rag.PROMPT_TEMPLATE,
            "C4_answer": rr.ANSWER_PROMPT,
            "C4_condense": rr.CONDENSE_PROMPT,
        },
        # SPLIT INTO ACTIVE / DORMANT on purpose. A flat list gave a reader no
        # way to tell a setting that shaped this run from one that is wired but
        # switched off — which is what made `HOPS = 1` read like a live 1-hop
        # decision rather than a parked axis.
        #
        # A third section, `legacy_truthy`, recorded `allow_truthy_path` so an
        # archived spec proved the retired `wdt:` path was off. It went when that
        # path was deleted outright (2026-08-16): there is no flag left to prove
        # anything about. Specs written before then still carry it.
        "retrieval": {
            "active": {
                "top_k": {"C2": rag.TOP_K, "C3": graph_rag.TOP_K,
                          "C4": rr.EMBED_TOP_K},
                "embedding_model": embedding_retriever.MODEL_NAME,
                # ⭐ THE SECOND DECLARED ARM (the first is ENTITY_BLOCK_MODE).
                # "global" is primary and was fixed before any post-rebuild
                # number existed; "floor" reserves floor(k/n) slots per question
                # entity. Recorded because two runs differing only in this would
                # otherwise be indistinguishable in their meta.json — and it is
                # a property of the RANKER, which all three retrieval configs
                # share, so it moves all of them together or none.
                "topk_allocation": embedding_retriever.ALLOCATION,
                # ⭐ THE FOURTH DECLARED ARM AXIS (2026-08-21): value-description
                # enrichment on label collisions. "off" is frozen and shipped;
                # "collision" is an exploratory post-TEST arm ONLY — it changes
                # what the ranker embeds, so a run carrying it is never
                # comparable to the record without saying so.
                "value_desc_mode": wikidata_pool.VALUE_DESC_MODE,
                "noise_filter_mode": wikidata.NOISE_FILTER_MODE,
                # ⚠️ `reverse_limit` was recorded here until 2026-08-16 and was
                # NEVER ACTIVE: it capped the retired truthy reverse query and
                # the statement path never read it. Removed with that path.
                "reverse_per_prop_cap": wikidata.REVERSE_PER_PROP_CAP,
                "statement_cache_version": wikidata.STATEMENT_CACHE_VERSION,
                "statement_row_limit": wikidata.STATEMENT_ROW_LIMIT,
                "rank_policy": list(wikidata.RANK_POLICY),
                "label_langs": wikidata.LABEL_LANGS,
                "statement_query_attempts": wikidata._STATEMENT_QUERY_ATTEMPTS,
                "strict_retrieval": wikidata.STRICT_RETRIEVAL,
                # Incoming statements are discovered exactly (FILTER EXISTS over
                # every entity-valued property) and fetched per property. Both
                # numbers shape WHAT is retrieved, so both belong in provenance:
                # the per-property cap is now a query bound rather than a
                # post-hoc filter over an arbitrary global slice.
                "reverse_discovery_chunk": wikidata.REVERSE_DISCOVERY_CHUNK,
                "reverse_fetch_batch": wikidata.REVERSE_FETCH_BATCH,
                # Both are behaviours a result depends on and neither is
                # inferable from the code SHA alone once this file is archived.
                "follows_redirects": True,
                "cache_fingerprint": wikidata.cache_fingerprint(),

                # --- the TEXT arm (Configuration 2) ---------------------------
                # ⚠️ `article_cache_version` STARTS AT 1 WHILE THE STATEMENT
                # CACHE IS AT 7, and that is not the text arm lagging six
                # generations behind. The two counters are independent: each
                # counts schema changes to ITS OWN cache. The statement cache
                # reached 7 through seven real changes; the article cache had no
                # version at all until 2026-08-16, so this is genuinely its
                # first. `report.md` labels each line with what it versions.
                "article_cache_version": wikipedia.ARTICLE_CACHE_VERSION,
                "article_cache_fingerprint": wikipedia.cache_fingerprint(),
                "article_strict_retrieval": wikipedia.STRICT_RETRIEVAL,
                "article_fetch_attempts": wikipedia._FETCH_ATTEMPTS,
            },
            "dormant": {
                # Wired but not varied. Hop depth is a SEPARATE axis with its own
                # evidence; moving two axes at once makes any outcome
                # unattributable, which this project has already paid for once.
                "hops": wikidata.HOPS,
            },
        },
        # How the ranked material becomes the {context} the answering LLM sees.
        # Distinct from retrieval: the same pool rendered differently is a
        # different experiment, and C2's selection rule lived undocumented for
        # months precisely because there was nowhere to write it down.
        "context_construction": {
            # ⭐ Closed 2026-08-12: all three retrieval configs z-normalise
            # similarity within each source entity before the global cut. Until
            # then only C3/C4 did, so "one identical ranking stage" was false in
            # the one respect the research question turns on.
            "per_entity_z_normalisation": {"C2": True, "C3": True, "C4": True},
            "c3_c4_format": "bullet list under a 'Wikidata facts about X:' header",
            "c2_format": "concatenated Wikipedia prose chunks",
        },
        "generation": {
            "answer_max_tokens": llm_config.MAX_TOKENS,
            # C4's condensing call is asked for prose, so it needs a larger cap
            # than the 128-token answer budget -- at 128 its contexts ended
            # mid-sentence.
            "c4_condense_max_tokens": rr.CONDENSE_MAX_TOKENS,
            "c4_answer_max_tokens": rr.ANSWER_MAX_TOKENS,
            "temperature": llm_config.TEMPERATURE,
        },
    }
