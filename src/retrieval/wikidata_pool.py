"""
The ranking pool shared by Configurations 3 and 4.

WHY THIS MODULE EXISTS. `graph_rag.py` and `graph_rag_rerank.py` carried this
loop as VERBATIM DUPLICATES (`graph_rag.py:108-117`, `graph_rag_rerank.py:153-163`).
The experiment's central claim is that C3 and C4 differ in exactly one step —
the condensing call — so a divergence between those two copies would not be a
tidiness problem, it would silently invalidate the comparison. Extracting it
makes "C3 and C4 share one retrieval stage" a property of the code rather than
a claim a reviewer has to verify by reading two files side by side.

⚠️ Config 2 does NOT use this. It pools Wikipedia prose chunks, not statements.
What C2 shares with C3/C4 is the ranker (`embedding_retriever`), and — from the
2026-08-12 rebuild onward — the selection rule; see `rag.py`.
"""

from __future__ import annotations

import os
from dataclasses import replace

from src.retrieval.statement import SnakValue, Statement
from src.retrieval.wikidata import fetch_entity_meta, fetch_statements

# Entity-level lines: the aliases and the one-line description. Rendered as
# synthetic Statements so they are RANKED like everything else rather than
# force-fed into the context — they compete for a top-k slot on their merits,
# and stay index-parallel with the `statements` list callers rely on.
#
# ⭐ ALIASES EXIST TO FIX A MEASURED FAILURE. The groundedness audit found
# confidently-wrong answers whose gold entity was present under a different
# surface form (Caligula as "Gaius Julius Caesar Germanicus Major"). An alias
# line lets the ranker match the question's wording to the entity.
#
# ⚠️ A description is PROSE, not a fact, and it is labelled `description:` so
# neither the ranker nor the reader mistakes it for a Wikidata claim.
ALIAS_PROPERTY = "also known as"
DESCRIPTION_PROPERTY = "description"
# One line, not one per alias: N alias lines would crowd the top-k with near
# duplicates of each other.
MAX_ALIASES = 8

# ⏸ THE NAMED POST-TEST OPTION (declared 2026-08-19, implemented 2026-08-21):
# conditional value-description enrichment. When a statement's value label is
# identical to its subject label ("[Gone With The Wind] based on: Gone with the
# Wind"), neither the answering model nor the uncased encoder can tell which
# side is which — 135/5,918 reference-v8 top-k lines, measured by
# tools/measure_context_noise.py. In `collision` mode the value entity's
# one-line description is appended ("… based on: Gone with the Wind (1936 novel
# by Margaret Mitchell)"), via the same cached fetch_entity_meta the synthetic
# description line uses.
#
# ⚠️ "off" IS THE FROZEN CONFIGURATION and the module default. `render()` feeds
# the ranker, so this is a RETRIEVAL change: every result of record
# (reference-v8, TEST-4000) ran without it, and run_eval's frozen-config guard
# refuses it on the sealed split. It exists as a declared EXPLORATORY arm —
# quantifying the §Future Work sentence, not amending the record.
#
# The predicate is the same case-insensitive label equality the measurement
# used. A QID comparison is what separates real facts between distinct
# same-named entities from self-description; appending the description handles
# BOTH honestly — a real fact gains a disambiguator, self-description becomes
# visibly self-description — which is why this was named over a drop filter
# (ruled UNSAFE 2026-08-19).
VALUE_DESC_MODES = ("off", "collision")
VALUE_DESC_MODE = os.getenv("VALUE_DESC_MODE", "off")
if VALUE_DESC_MODE not in VALUE_DESC_MODES:
    raise ValueError(f"VALUE_DESC_MODE={VALUE_DESC_MODE!r} — must be one of {VALUE_DESC_MODES}")

# Descriptions for VALUE entities, memoized per process: collision values recur
# across a run (franchise entities especially), and unlike question entities
# they are usually absent from the statement cache, so each distinct QID may
# cost one small live query. Cleared by tests that mock fetch_entity_meta.
_VALUE_DESC_MEMO: dict[str, str | None] = {}


def _value_description(qid: str) -> str | None:
    if qid not in _VALUE_DESC_MEMO:
        _VALUE_DESC_MEMO[qid] = fetch_entity_meta(qid).get("description") or None
    return _VALUE_DESC_MEMO[qid]


def _enrich_on_collision(st: Statement) -> Statement:
    """Append the value entity's description when its label collides with the subject's.

    Applied AFTER the dataset-name overrides in `build_pool`, so the comparison
    sees exactly the labels that would render. No-ops unless the value is
    entity-valued (a collision on a string value has no QID to describe) and a
    description actually exists.
    """
    if VALUE_DESC_MODE != "collision":
        return st
    v = st.value
    if not v.id or not v.label:
        return st
    if v.label.strip().lower() != st.subject_label.strip().lower():
        return st
    desc = _value_description(v.id)
    if not desc:
        return st
    return replace(st, value=replace(v, label=f"{v.label} ({desc})"))


def _meta_statements(qid: str, name: str) -> list[Statement]:
    meta = fetch_entity_meta(qid)
    out: list[Statement] = []

    def synth(prop: str, text: str) -> Statement:
        return Statement(
            subject_id=qid, subject_label=name,
            property_id="", property_label=prop,
            value=SnakValue("value", "string", label=text),
            rank="normal", direction="outgoing", source_entity_id=qid,
            # No statement GUID: Wikidata does not model these as statements, and
            # inventing one would corrupt the drift signal `statement_id` carries.
            statement_id="",
        )

    aliases = [a for a in meta.get("aliases", []) if a and a != name][:MAX_ALIASES]
    if aliases:
        out.append(synth(ALIAS_PROPERTY, "; ".join(aliases)))
    if meta.get("description"):
        out.append(synth(DESCRIPTION_PROPERTY, meta["description"]))
    return out


def build_pool(
    qids: list[str],
    qid_names: dict[str, str],
    *,
    verbose: bool = False,
) -> tuple[list[str], list[str], list[Statement]]:
    """
    Fetch statements for every question entity and render them for ranking.

    Returns `(lines, groups, statements)` — three parallel lists:

    `lines`
        The rendered text the embedding ranker scores and the answering model
        reads, e.g. `[Tom Hanks] award received: Academy Award for Best Actor
        (for work: Philadelphia; point in time: 1994-03-21)`.

    `groups`
        The source entity QID per line. The ranker z-normalises within each
        group before the global cut. This normalises score location and scale
        without guaranteeing equal representation of entities.

    `statements`
        The `Statement` behind each line, kept parallel so downstream code can
        reach structure — `value.id` for QID matching against Mintaka's unused
        `answer_qids`, `rank`, `statement_id` — without re-parsing the string.

    ⚠️ DEDUPLICATION IS ON THE RENDERED LINE, not on `statement_id`, and that is
    deliberate. The same fact reached from two question entities arrives as two
    Statements differing in `direction` and `source_entity_id` but rendering
    identically; the pool must carry it once, because a duplicate line wastes a
    top-k slot. Conversely two statements sharing a property and a value but
    differing in qualifiers render DIFFERENTLY and are both kept — which is the
    entire point of the statement model, and is why the old truthy pool showed
    Tom Hanks's two Best Actor Oscars as one line.
    """
    lines: list[str] = []
    groups: list[str] = []
    statements: list[Statement] = []
    seen: set[str] = set()

    for qid in qids:
        name = qid_names.get(qid)
        entity_lines = _meta_statements(qid, name) if name else _meta_statements(
            qid, fetch_entity_meta(qid).get("label") or qid)
        for st in list(entity_lines) + list(fetch_statements(qid)):
            # ⚠️ Prefer the name the DATASET used for this entity, matching the
            # behaviour of the truthy pool this replaces (`qid_names.get(subj)`).
            # Wikidata's own label is the fallback, not the default: the question
            # says "Tom Hanks", and prepending a differently-worded label would
            # weaken the very match the prepend exists to create.
            #
            # Only OUTGOING statements are overridden. An incoming statement's
            # subject is some other entity entirely ("[Ronnie Wood] member of:
            # The Rolling Stones") and its label must stay as fetched.
            if st.direction == "outgoing":
                name = qid_names.get(st.subject_id)
                if name and name != st.subject_label:
                    st = replace(st, subject_label=name)
            else:
                # An INCOMING statement's value is this entity, so the same name
                # belongs there — "[Fred Weasley] present in work: Harry Potter",
                # not "... : Q8337".
                #
                # ⚠️ Also a guard, not only a preference. `fetch_statements` falls
                # back to the QID when the label query fails, which silently
                # degrades EVERY incoming statement of that entity at once; using
                # the dataset's name keeps the line readable even then. The fetch
                # side now refuses to cache such an entity, so this is the second
                # of two independent defences against one observed failure.
                name = qid_names.get(st.value.id or "")
                if name and name != st.value.label:
                    st = replace(st, value=replace(st.value, label=name))

            st = _enrich_on_collision(st)
            line = st.render()
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            groups.append(qid)
            statements.append(st)

    if verbose:
        print(f"  Pooled {len(lines)} statements, ranking by cosine similarity...")

    return lines, groups, statements
