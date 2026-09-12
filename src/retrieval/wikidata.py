"""
Wikidata SPARQL retriever.

Reads statement nodes, qualifiers, ranks and typed values instead of the
flattened `wdt:` truthy view. Not every statement has a distinct truthy edge:
rank selection excludes some statements, and repeated values may collapse.
References are not fetched. The stored records preserve the information used
by the deterministic renderer; the pipeline performs no symbolic aggregation.

🔴 QID IN, STATEMENTS OUT. That is the whole surface. Entity lookup by NAME
lives in `src/retrieval/entity_linking.py` (demo only, moved 2026-08-16),
because the controlled evaluation takes gold QIDs from Mintaka and never
resolves a name.

What it does:
  1. Entity metadata — label, description, aliases, and `owl:sameAs`, because a
     MERGED gold QID otherwise returns nothing at all
  2. `fetch_statements(qid)` — statements in BOTH directions (outgoing, plus a
     capped set of incoming, since many relations live on the other entity),
     with property datatypes, units and time precision resolved in the same
     query. Cleaned at RETURN time, so filter changes need no refetch

The submission implements the statement-model pipeline used for the final
evaluation. The retired truthy retriever and its private development history
are not part of this snapshot.

Public entry points: `fetch_statements`, `fetch_entity_meta`.
Configurations 3 and 4 do not call this directly — they go through
`src/retrieval/wikidata_pool.py`, which is shared so they cannot drift apart.
"""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

from dataclasses import replace

from src.retrieval.cache_io import atomic_write_text, read_text_or_none
from src.retrieval.statement import (
    Qualifier, SnakValue, Statement, sort_qualifiers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
MEDIAWIKI_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "TFM-GraphRAG/0.1 (florispam@outlook.com)"

# Cap per incoming property. A high-fan-in property (every album carries
# "performer: <band>") must not monopolize the ranking pool and crowd out
# low-fan-in relations like "member of". The cap limits pool imbalance but can
# omit relevant incoming facts; it does not guarantee complete answer sets.
#
# Since 2026-08-14 it bounds the reverse QUERY as well as the return-time
# filter, so changing it changes what was retrieved — hence its presence in
# `cache_fingerprint`.
REVERSE_PER_PROP_CAP = 30

# ---------------------------------------------------------------------------
# Step 1 — Run a SPARQL query via HTTP POST and return JSON rows
# ---------------------------------------------------------------------------


def _sparql_query(query: str, *, timeout: int = 30, attempts: int = 1) -> list[dict]:
    """
    Send a SPARQL query to Wikidata via POST and return the result bindings.

    ⚠️ EVERY RETRIEVAL CALLER PASSES `attempts=_STATEMENT_QUERY_ATTEMPTS`. The
    default of 1 serves the one-off analysis tools in `tools/`, which call this
    bare and would rather fail than silently take three times as long.

    Retry is not optional on the retrieval path: the public endpoint's latency is
    genuinely erratic under shared load — the same Q2263 query measured 6.7 s,
    then a 30 s timeout, then 3.0 s on three consecutive calls, and query SHAPE
    was ruled out (a variant with no datatype join and an ontology-joined variant
    showed the same spread). Without retry a 350-entity refetch fails on load
    rather than on anything we control.
    """
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        SPARQL_ENDPOINT,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
            return result["results"]["bindings"]
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    return []  # unreachable; keeps the return type honest


# ---------------------------------------------------------------------------
# Step 2 — Entity disambiguation: MOVED OUT (2026-08-16)
# ---------------------------------------------------------------------------
#
# Candidate search lives in `src/retrieval/entity_linking.py`; the demo user
# selects the entity. Controlled evaluation takes gold QIDs from Mintaka.


# ---------------------------------------------------------------------------
# Step 3 — The noise filter (which properties are facts, which are plumbing)
# ---------------------------------------------------------------------------

# A Wikidata property's *datatype* (wikibase:propertyType) is the principled
# signal for "fact vs. plumbing": ExternalId / CommonsMedia / Url / … are
# identifiers, media and links excluded by this study's content policy. They
# can rank well lexically. An inventory of every property used by the Mintaka gold
# QIDs — measured 2026-06, the tool since retired — found ExternalId alone is 53% of all
# triples on those entities — the single biggest source of top-k noise, and one
# a label heuristic misses for "image", "official website", "ISNI" or
# "X (Twitter) username".
#
# ⭐ THE DATATYPE ARRIVES WITH THE DATA, so filtering is STRUCTURAL. A June
# attempt to resolve `wikibase:propertyType` in-query timed out and forced a
# shipped static label→datatype map (`property_datatypes.json`, 6,631 labels);
# that attempt was GLOBAL, a join across all properties. The statement fetch's
# join is ENTITY-SCOPED and costs ~1-2 s (Q2263 414 rows 1.1s -> 2.5s with
# qualifier datatypes resolved too; Q8337 162 rows 1.8s -> 0.7s), so the map was
# deleted 2026-08-12. A property created or retyped after that crawl was
# invisible to the frozen snapshot; it is not to this.
#
# The filter runs at RETURN time on raw cached statements, so changing it never
# requires a refetch.

# Datatypes whose statements are identifiers / media / links / technical values —
# excluded from the candidate pool. WikibaseItem, Time, Quantity, Monolingualtext and
# GlobeCoordinate carry the facts and are deliberately absent. (String is also
# absent — a mixed bag; its plumbing subset is caught by the label rules below.)
#
# ⚠️ A datatype in NEITHER list is KEPT. Measured over the gold inventory that is
# 5 properties: 4 `WikibaseProperty` schema links ("main Wikidata property",
# "properties for this type") and 1 whose type did not resolve. Low volume, but
# the default is "fact", so this list is a blocklist and not a partition.
_NOISE_DATATYPES = {
    "ExternalId",
    "CommonsMedia",
    "Url",
    "GeoShape",
    "TabularData",
    "Math",
    "MusicalNotation",
    "WikibaseForm",
    "WikibaseSense",
    "WikibaseLexeme",
    "WikibaseEntitySchema",
}

# Fact-datatype (WikibaseItem / String) properties that are still pure plumbing —
# Wikimedia maintenance, Commons/category/template wiring, "same as"/"different
# from" identity links. The datatype signal can't catch these, so they stay on a
# small label blocklist (e.g. "topic's main category" ranks for any topical Q).
_NOISE_PROP_EXACT = {
    "described by source",
    "different from",
    "said to be the same as",
    "permanent duplicated item",
    "facet of",
}


# Which noise filter to apply — an ABLATION ARM, not a fix and its bug:
#
#   "property_type"    keyed on each property's Wikidata propertyType (below).
#   "label_heuristic"  the 2026-07-19 rules, matching on the property LABEL only.
#
# Measured on DEV-200 (2026-07-27, paired bootstrap) the two are STATISTICALLY
# INDISTINGUISHABLE at both top_k=10 and top_k=30 — all four deltas < 2.4 F1,
# every CI crosses zero. `property_type` is the default because it is the
# principled, documentable signal at no measured cost, NOT because it won. top_k
# is the knob that moves results (+4.9 C3 / +9.2 C4 for k30 over k10).
# ⚠️ That measurement was taken on truthy triples; it is inherited, not
# re-validated under the statement model.
#
# Override per run without editing this file:  NOISE_FILTER_MODE=label_heuristic
#
# NAMING: "property type" is a property's OWN datatype (ExternalId, WikibaseItem,
# Quantity, Time…), unrelated to Mintaka's `answer_type`, which describes a
# QUESTION's gold answer and selects a scorer in metrics.py. The modes were
# called "datatype"/"string" until 2026-07-27; both old names still resolve, so
# an older command or result file does not silently change meaning.
_NOISE_FILTER_MODES = ("property_type", "label_heuristic")
_LEGACY_MODE_ALIASES = {"datatype": "property_type", "string": "label_heuristic"}


def _resolve_mode(mode: str) -> str:
    """Map a mode name (incl. the pre-2026-07-27 aliases) to a canonical mode."""
    return _LEGACY_MODE_ALIASES.get(mode, mode)


NOISE_FILTER_MODE = _resolve_mode(os.environ.get("NOISE_FILTER_MODE", "property_type"))
if NOISE_FILTER_MODE not in _NOISE_FILTER_MODES:
    # Fail loudly: a silently-ignored typo here would mislabel an entire eval run.
    raise ValueError(
        f"NOISE_FILTER_MODE must be one of {_NOISE_FILTER_MODES} "
        f"(or legacy {tuple(_LEGACY_MODE_ALIASES)}), got {NOISE_FILTER_MODE!r}"
    )


def _is_noise_prop_label_heuristic(pl: str) -> bool:
    """
    The 2026-07-19 pre-property_type heuristic. `pl` is lowercased.

    ⚠️ A RECONSTRUCTION from the documented spec, not the original code: that
    never reached a commit (d7f4fa0 introduced reverse triples, dedup and the
    property_type filter together, and its parent has no filter at all).

    Its blind spots relative to the property_type mode's label tiers are the
    point of the comparison and are deliberate: no "commons"/"topic has" rules,
    and the identifier catch is ungated by property type.
    """
    if "wikiproject" in pl or "wikimedia" in pl:
        return True
    if pl.startswith(("category", "template", "topic's main")):
        return True
    if pl in _NOISE_PROP_EXACT:
        return True
    if pl.endswith(" id") or "identifier" in pl:
        return True
    return False


def _is_noise_prop(prop_label: str, datatype: str | None = None) -> bool:
    """
    True for properties whose statements are metadata/identifiers, not facts.

    Primary signal is the property's Wikidata PROPERTY TYPE. The label rules then
    catch fact-typed plumbing the property type can't.

    ⚠️ ALWAYS SUPPLY `datatype`. The statement fetch resolves
    `wikibase:propertyType` in the same query, for statement properties AND
    qualifier properties, so every caller in `src/` has it. Omitting it simply
    skips tier 1 and materially weakens the filter — `ISNI`, `image` and
    `official website` are caught by datatype alone and match no label rule. It
    is accepted only so the free-form demo path degrades rather than crashes.

    Set NOISE_FILTER_MODE = "label_heuristic" for the 2026-07-19 rules instead.
    """
    pl = prop_label.lower()
    if _resolve_mode(NOISE_FILTER_MODE) == "label_heuristic":
        return _is_noise_prop_label_heuristic(pl)

    # 1) Property-type-driven (primary): identifiers, media, URLs, technical values.
    if datatype in _NOISE_DATATYPES:
        return True

    # 2) Fact-typed PLUMBING the datatype signal can't catch (WikibaseItem/String
    #    maintenance + Commons/category/template wiring).
    if "wikiproject" in pl or "wikimedia" in pl or "commons" in pl:
        return True
    if pl.startswith(("category", "template", "topic's main", "topic has")):
        return True
    if pl in _NOISE_PROP_EXACT:
        return True
    # 3) Label-based identifier catch for the genuinely ambiguous case: String-
    #    typed properties are a mixed bag whose identifier members ("NCI Thesaurus
    #    ID") the datatype set cannot distinguish from real String facts ("postal
    #    code"). Genuine facts don't end in " id" or contain "identifier".
    #    `None` is still accepted so a caller without a datatype degrades to
    #    label-only rather than crashing.
    if datatype in (None, "String") and (pl.endswith(" id") or "identifier" in pl):
        return True
    return False


# ---------------------------------------------------------------------------
# Step 4 — Fetch STATEMENTS (qualifiers + ranks). See docs/reproduction.md
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Until 2026-08-12 this module read Wikidata's TRUTHY view
# (`wdt:`), one flat edge per (property, value) pair. That view cannot express
# CARDINALITY, TIME or RANK: Tom Hanks's two Best Actor Oscars share a property
# and a value, so they are one edge. Measured over the first 60 DEV gold QIDs,
# 25.3% of statements carry no distinct truthy edge and 31.1% carry qualifiers
# that were never read.
#
# The cache directory is separate from the retired `data/cache/wikidata/` for
# that reason: the two representations are not interchangeable, and a v2 triple
# cache must never be read as if it were statements.
STATEMENT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "wikidata_statements"
)
# Bumped whenever a fetch starts capturing something the previous version did
# not, because the dict shape stays valid and the gap is therefore INVISIBLE —
# old entries deserialise cleanly and simply mean less. A mixed cache would
# render and filter two different ways with nothing to show it.
#   3: qualifier propertyTypes (filtering)
#   4: value-node unit + time precision, entity meta (aliases, description)
#   5: (statement model v1)
#   6: structured completeness -- `query_succeeded` + `truncated` + `fingerprint`.
#      ⚠️ v5 ENTRIES MAY BE SILENTLY FORWARD-ONLY. Until 2026-08-13 a failed
#      reverse query was swallowed and the forward-only result was cached anyway,
#      so a v5 file cannot be distinguished from a complete one. The bump is what
#      forces those to be refetched rather than trusted.
#   7: per-property reverse fetch. Incoming statements are now discovered exactly
#      and fetched per property, so hub entities have incoming data AT ALL (six
#      were stored with zero) and every other entity gets a per-property slice
#      instead of an arbitrary global 4,000-row cut. Both the coverage and the
#      truncation semantics changed, so v6 entries are not comparable.
#   8: somevalue snaks typed as such. v7 stamped every snak "value", so a skolem
#      /.well-known/genid/ IRI was stored as an ordinary value and RENDERED as a
#      literal URL (53 entities). A v7 entry deserialises cleanly and keeps
#      doing that — exactly the invisible-gap case this constant exists for.
STATEMENT_CACHE_VERSION = 8

# Ranks to keep. `deprecated` means Wikidata itself judges the statement wrong or
# superseded -- on Q2263 the two deprecated statements carry the qualifier
# "reason for deprecated rank: source known to be unreliable". Serving those as
# current would be strictly worse than the truthy view, which excluded them by
# accident. Keeping `normal` alongside `preferred` is the deliberate change:
# truthy drops EVERY normal-rank statement once any preferred one exists for a
# property, which is why superseded values (population in 1990, former
# office-holders) are absent rather than merely unranked.
#
# A PARAMETER, not a constant, so the rank axis can be isolated later without a
# rebuild -- this change moves qualifiers and ranks together, and an ablation
# should cost a flag flip.
RANK_POLICY: tuple[str, ...] = ("preferred", "normal")

# Dormant. Hop depth is a SEPARATE axis with its own evidence (KAPING reports
# 40% of Mintaka answerable from 1-hop against 62% from 2-hop; EFSum runs 2-hop
# on Mintaka). Deliberately not wired: moving two axes at once makes any outcome
# unattributable, which this project has already paid for once.
HOPS = 1

# The qualifier join multiplies rows -- 341 truthy edges become 414 rows on
# Q2263, and far more on qualifier-heavy entities. The old LIMIT 500 was already
# a silent truncation risk with no ORDER BY, so what survived was arbitrary.
# Raised, and binding is RECORDED rather than assumed away (see _statement_rows).
STATEMENT_ROW_LIMIT = 4000


class IncompleteRetrievalError(RuntimeError):
    """A required part of an entity's statements could not be retrieved.

    ⚠️ NOT TRANSIENT BY THE TIME IT REACHES A CALLER. `_statement_rows` already
    makes three endpoint attempts of its own, so this is raised only after those
    are exhausted. `run_eval` must therefore NOT retry it -- with MAX_ATTEMPTS=4
    outer retries that would be 12 endpoint hits per entity, and the four
    attempts differ in nothing. It is registered as non-transient in
    `run_eval._is_transient` so the question is excluded once, cleanly.
    """


# Strict by default: a canonical run must not silently answer from a partial
# graph. A failed reverse fetch drops every INCOMING statement, which is where
# "films directed by X" and "people born in Y" live -- degrading to forward-only
# produces a plausible context and an unmarked, unrecoverable measurement error.
#
# The demo/interactive path sets this False deliberately (see `retrieve_context`):
# there, an answer from a partial graph beats no answer at all.
STRICT_RETRIEVAL = True

_p = "http://www.wikidata.org/prop/"

# 🔴 "en,mul" — NOT "en", AND THIS IS NOT A NICETY.
#
# Wikidata introduced the `mul` ("multiple languages") label code in 2024 for
# names that are IDENTICAL across languages, and editors have been migrating
# labels into it and REMOVING the per-language duplicates. Proper nouns are
# exactly the case it targets, so a large share of the entities this thesis
# retrieves — people, films, bands, book series — now carry their name only
# under `mul`.
#
# Asking for "en" alone gets the QID back as the label, which this pipeline then
# treats as an unresolved value and DROPS. Verified live:
#
#     language "en"      -> ['Q134773', 'Q8337',        'Tom Hanks']
#     language "en,mul"  -> ['Forrest Gump', 'Harry Potter', 'Tom Hanks']
#
# ⚠️ This retracts an earlier conclusion recorded in this file and in the
# `feat(retrieval): fetch statements` commit message, that Q134773 "genuinely has
# no English label, only Arabic and Persian". It has one; it lives under `mul`.
# The Arabic and Persian rows were simply the first ones returned.
#
# Order matters: "en" first, so a genuine English label still wins over the
# multilingual default where both exist (Q2263 has both).
LABEL_LANGS = "en,mul"

# ⚠️ PLAIN strings, not f-strings, and SPARQL's own braces stay DOUBLED. These
# templates are filled by str.format() in _statement_rows, which runs AFTER this
# module is imported -- so a literal `{` left in the body is read by format() as
# the start of a replacement field and raises KeyError on the SPARQL source text.
# Every substitution point (`{qid}`, `{limit}`, `{p}`) is therefore single-braced
# and every SPARQL block brace is doubled. Do not "simplify" this back into an
# f-string: an f-string collapses `{{` to `{` at import time and hands format() a
# body it cannot parse.
_FORWARD_STATEMENTS = """
SELECT ?st ?prop ?propLabel ?dt ?value ?valueLabel ?rank ?unitLabel ?prec
       ?qualProp ?qualPropLabel ?qualDt ?qualValue ?qualValueLabel ?qUnitLabel ?qPrec WHERE {{
  wd:{qid} ?p ?st .
  FILTER(STRSTARTS(STR(?p), "{p}P"))
  ?st ?psP ?value .
  FILTER(STRSTARTS(STR(?psP), "{p}statement/P"))
  BIND(IRI(REPLACE(STR(?p), "prop/", "entity/")) AS ?prop)
  ?st wikibase:rank ?rank .
  OPTIONAL {{ ?prop wikibase:propertyType ?dt . }}
  OPTIONAL {{
    ?prop wikibase:statementValue ?psvP .
    ?st ?psvP ?vn .
    OPTIONAL {{ ?vn wikibase:quantityUnit ?unit . }}
    OPTIONAL {{ ?vn wikibase:timePrecision ?prec . }}
  }}
  OPTIONAL {{
    ?st ?pqP ?qualValue .
    FILTER(STRSTARTS(STR(?pqP), "{p}qualifier/P"))
    BIND(IRI(REPLACE(STR(?pqP), "prop/qualifier/", "entity/")) AS ?qualProp)
    OPTIONAL {{ ?qualProp wikibase:propertyType ?qualDt . }}
    OPTIONAL {{
      ?qualProp wikibase:qualifierValue ?pqvP .
      ?st ?pqvP ?qvn .
      OPTIONAL {{ ?qvn wikibase:quantityUnit ?qUnit . }}
      OPTIONAL {{ ?qvn wikibase:timePrecision ?qPrec . }}
    }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}}
LIMIT {limit}
"""

# ---------------------------------------------------------------------------
# INCOMING STATEMENTS — discover the properties, then fetch per property
#
# Incoming means someone else's statement whose VALUE is our entity. Essential
# for questions whose answer points AT the question entity (band members, works
# by an author); an outgoing-only fetch can never answer those.
#
# 🔴 A ONE-QUERY REVERSE FETCH CANNOT SERVE HUB ENTITIES, AND FAILING ON THEM
# WAS SILENTLY COSTING REAL DATA. That shape (used until 2026-08-14) asked for
# every statement whose value is this entity, in one request. For Tom Hanks that
# is a few hundred. For Q30 (United States) it is millions: MEASURED 2,246,467
# subjects with `country = Q30` and 686,693 with `citizenship = Q30`, on ONE
# property each. The request cannot be served at any timeout, and on DEV-200 it
# failed for exactly the seven highest-fan-in gold entities -- USA, France, UK,
# Japan, Brazil, "film", "actor" -- while every other entity succeeded.
#
# ⚠️ DATATYPE FILTERING DOES NOT HELP HERE, and the reason is structural, not
# incidental. An incoming statement's value IS this entity, so its property must
# be entity-valued: measured over all incoming statements, Q2263 is 236/236
# WikibaseItem and Q76 is 1199/1199. Nothing points at an entity through an
# ExternalId, so excluding identifier properties from the REVERSE query filters
# exactly zero rows. Identifier noise is an outgoing-direction problem, and the
# outgoing query has never timed out.
#
# THE FIX IS THE QUERY SHAPE, IN THREE PARTS, each measured:
#
# 1. `hint:Query hint:optimizer "None"`. Without it Blazegraph plans the join
#    from the wrong side and scans millions of statement nodes to return thirty.
#    With it, the object-bound triple runs first and hits an index.
#    MEASURED on Q30/P17 (2.2M subjects): timeout -> 0.5 s. This single hint is
#    the difference between impossible and instant.
#
# 2. Bound the LIMIT inside a subquery, so the label service decorates thirty
#    rows instead of participating in the scan. Q30/P17 with the label service
#    outside a subquery took 30 s; inside, 0.5 s.
#
# 3. Batch properties as a UNION of individually-limited subqueries. One request
#    per property would be 152 round trips for Q30; at 20 per request it is 8.
#    MEASURED: 20 properties, 308 rows, 2.5 s.
#
# ⭐ AND DISCOVERY IS COMPLETE, NOT SAMPLED. The obvious way to learn which
# properties point at an entity -- read a bounded sample of its incoming
# statements -- does not converge for hubs: Q30 yielded 22 properties from a
# 10,000-row sample and 35 from 50,000. Probing every entity-valued property
# with FILTER EXISTS instead is exact, because a bound predicate is an index
# probe. MEASURED: Q30 has **152** incoming properties, so sampling was missing
# ~80% of them. Cost: Q2263 1.8 s, Q11424 2.6 s, Q33999 2.9 s, Q30 29.4 s.
# ---------------------------------------------------------------------------

# Every property whose value can BE an entity (~1,776). Entity-independent, so
# it is fetched once per process and cached on disk -- it is the candidate set
# for discovery, not per-entity data.
_ITEM_PROPERTIES = """
SELECT ?p WHERE {{ ?p wikibase:propertyType wikibase:WikibaseItem . }}
"""

# Exact discovery. `VALUES` binds the predicate, so each FILTER EXISTS is an
# index probe rather than a scan.
_REVERSE_PROPERTY_PROBE = """
SELECT ?psP WHERE {{
  VALUES ?psP {{ {values} }}
  FILTER EXISTS {{ ?st ?psP wd:{qid} . }}
}}
"""

# Fallback probe for ONE property, used only when a chunk probe fails.
# Blazegraph MATERIALIZES a FILTER EXISTS sub-plan, so a property with
# millions of incoming statements blows the endpoint's 60 s limit even though
# the answer is trivially "yes" — Q5/P31 (~12M `instance of: human` rows) is
# the measured case (2026-08-19: EXISTS 60.2 s timeout, this form 0.3 s; both
# forms agree on P6/P19/P91 negative and P21/P31 positive). LIMIT 1 asks the
# same boolean — "does at least one incoming statement exist".
_REVERSE_PROBE_SINGLE = """
SELECT ?st WHERE {{ ?st ps:{pid} wd:{qid} . }} LIMIT 1
"""

# One request, several properties, each independently capped. The per-property
# cap is `REVERSE_PER_PROP_CAP`, which `_clean_statements` already enforced
# downstream -- so this stops fetching what was always going to be discarded,
# and the cap now bounds the QUERY rather than trimming an arbitrary global
# slice after the fact.
_REVERSE_BY_PROPERTIES = """
PREFIX hint: <http://www.bigdata.com/queryHints#>
SELECT ?st ?subj ?subjLabel ?prop ?propLabel ?dt ?rank ?unitLabel ?prec
       ?qualProp ?qualPropLabel ?qualDt ?qualValue ?qualValueLabel ?qUnitLabel ?qPrec WHERE {{
{blocks}
  ?st wikibase:rank ?rank .
  OPTIONAL {{ ?prop wikibase:propertyType ?dt . }}
  OPTIONAL {{
    ?prop wikibase:statementValue ?psvP .
    ?st ?psvP ?vn .
    OPTIONAL {{ ?vn wikibase:quantityUnit ?unit . }}
    OPTIONAL {{ ?vn wikibase:timePrecision ?prec . }}
  }}
  OPTIONAL {{
    ?st ?pqP ?qualValue .
    FILTER(STRSTARTS(STR(?pqP), "{p}qualifier/P"))
    BIND(IRI(REPLACE(STR(?pqP), "prop/qualifier/", "entity/")) AS ?qualProp)
    OPTIONAL {{ ?qualProp wikibase:propertyType ?qualDt . }}
    OPTIONAL {{
      ?qualProp wikibase:qualifierValue ?pqvP .
      ?st ?pqvP ?qvn .
      OPTIONAL {{ ?qvn wikibase:quantityUnit ?qUnit . }}
      OPTIONAL {{ ?qvn wikibase:timePrecision ?qPrec . }}
    }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}}
"""

_REVERSE_BLOCK = """  {{ SELECT ?st ?subj ?prop WHERE {{
      hint:Query hint:optimizer "None" .
      ?st ps:{pid} wd:{qid} .
      ?subj p:{pid} ?st .
      BIND(wd:{pid} AS ?prop)
    }} LIMIT {cap} }}"""


# Entity-level data that is NOT a statement and therefore has no place in the
# statement query: the label, the one-line description, and the aliases.
#
# ⭐ ALIASES ADDRESS A MEASURED FAILURE, not a hunch. The groundedness audit found
# confidently-wrong answers where the gold entity WAS in the context under a
# different surface form — Caligula stored as "Gaius Julius Caesar Germanicus
# Major". 115 en/mul aliases across 29 DEV gold entities (~4 each).
#
# Descriptions are one line per entity and every DEV gold entity has one. They
# are prose rather than facts, which is why they are rendered as an explicit
# `description:` line rather than blended in.
# 🔴 `owl:sameAs` IS A REDIRECT, AND IGNORING IT IS TOTAL DATA LOSS FOR THAT
# ENTITY. Mintaka was annotated against a 2021 Wikidata snapshot; Wikidata is
# live, and items get MERGED. The stale QID stays syntactically valid, resolves
# to nothing, and carries no statements — so the question silently gets an empty
# context while the gold QID looks perfectly well-formed.
#
# Verified: DEV-200 gold `Q4439148` redirects to `Q315625` ("intern"). The stale
# QID returns 0 statements; the target returns 92.
_ENTITY_META = """
SELECT ?l ?d ?a ?redirect WHERE {{
  OPTIONAL {{ wd:{qid} rdfs:label ?l . FILTER(LANG(?l) IN ("en", "mul")) }}
  OPTIONAL {{ wd:{qid} schema:description ?d . FILTER(LANG(?d) = "en") }}
  OPTIONAL {{ wd:{qid} skos:altLabel ?a . FILTER(LANG(?a) IN ("en", "mul")) }}
  OPTIONAL {{ wd:{qid} owl:sameAs ?redirect . }}
}}
"""


def _val(row: dict, key: str) -> str | None:
    v = (row.get(key) or {}).get("value")
    return v if v not in ("", None) else None


def _local(uri: str | None) -> str:
    """Last path/fragment segment of an IRI: .../entity/P166 -> P166."""
    if not uri:
        return ""
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _prec(text: str | None) -> int | None:
    """Wikidata time precision as an int, or None when the value node had none."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _snak(id_uri: str | None, label: str | None, datatype: str,
          unit: str | None = None, precision: int | None = None) -> SnakValue:
    """
    Build a value, distinguishing "no English label" from "not an entity".

    A label equal to the QID means the label service resolved nothing in any of
    LABEL_LANGS. We keep the id so the statement is still identifiable, and leave
    `label` None so the renderer and the filter can act on it.

    ⚠️ The test is an EXACT match on QID shape, never `startswith("Q")` — that
    prefix test (the truthy path's) also discards "Quebec", "Queen Victoria" and
    "Quentin Tarantino".

    ⚠️ An unresolved label is a genuinely rare case, and it was not before. Until
    the `mul` fallback was added, the commonest cause was not missing data at all
    — it was asking only for "en" while the name lived under Wikidata's
    multilingual `mul` code. Q134773 (*Forrest Gump*) was written up here as an
    entity with "only Arabic and Persian labels"; it has an English name, under
    `mul`. See LABEL_LANGS. Do not read a QID-shaped label as evidence that
    Wikidata lacks the name.
    """
    # ⚠️ A SOMEVALUE snak reaches us as a skolem IRI ("/.well-known/genid/…"):
    # Wikidata's RDF for "the value exists but is unknown". Until 2026-08-18
    # this function stamped every row `snaktype="value"`, so the renderer's
    # raw fallback printed the literal genid URL into the context (53 cached
    # entities; one context per k30 DEV run). SnakValue.render() has
    # handled `somevalue` since the dataclass was written — the type just was
    # never produced. `raw` keeps the IRI for provenance; render ignores it.
    #
    # NOVALUE never arrives here at all: it has no ps: triple, so the query
    # cannot return it. That is a documented, bounded exclusion (23 of 45,945
    # statements in the completeness sweep, 0.05%) — not supported behaviour.
    if id_uri and "/.well-known/genid/" in id_uri:
        return SnakValue(
            snaktype="somevalue",
            datatype=datatype or "",
            raw=id_uri,
            precision=precision,
        )

    qid = _local(id_uri) if id_uri and "/entity/" in id_uri else None
    unresolved = label is not None and re.fullmatch(r"Q\d+", label) is not None
    # Q199 is Wikidata's "unitless" marker (counts, ordinals, ratios). Rendering
    # it would append the literal word to every plain number.
    if unit in (None, "", "1", "Q199"):
        unit = None
    return SnakValue(
        snaktype="value",
        datatype=datatype or "",
        id=qid,
        label=None if unresolved else label,
        raw=id_uri if (id_uri and not qid) else None,
        unit=unit,
        precision=precision,
    )


def _rows_to_statements(rows: list[dict], *, qid: str, own_label: str,
                        direction: str) -> list[Statement]:
    """
    Group SPARQL rows into statements. One row per (statement x qualifier), so a
    statement with three qualifiers arrives three times and must be folded.
    """
    by_st: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        sid = _local(_val(row, "st"))
        if not sid:
            continue
        if sid not in by_st:
            order.append(sid)
            datatype = _local(_val(row, "dt"))
            unit = _val(row, "unitLabel")
            precision = _prec(_val(row, "prec"))
            if direction == "outgoing":
                subject_id, subject_label = qid, own_label
                value = _snak(_val(row, "value"), _val(row, "valueLabel"), datatype,
                              unit, precision)
            else:
                subject_id = _local(_val(row, "subj"))
                subject_label = _val(row, "subjLabel")
                # The reverse statement's value IS our entity; naming it keeps the
                # line readable ("[Ronnie Wood] member of: The Rolling Stones").
                value = SnakValue("value", datatype, id=qid, label=own_label)
                if subject_label and re.fullmatch(r"Q\d+", subject_label):
                    subject_label = None
            by_st[sid] = {
                "property_id": _local(_val(row, "prop")),
                "property_label": _val(row, "propLabel"),
                "datatype": datatype,
                "value": value,
                "subject_id": subject_id,
                "subject_label": subject_label,
                "rank": _local(_val(row, "rank")).replace("Rank", "").lower(),
                "quals": [],
            }
        qprop = _val(row, "qualProp")
        if qprop:
            qlabel = _val(row, "qualPropLabel")
            # The qualifier property's OWN datatype, resolved in-query alongside
            # the statement property's — without it, qualifier filtering would
            # have to fall back to matching on the label alone.
            qval = _snak(_val(row, "qualValue"), _val(row, "qualValueLabel"),
                         _local(_val(row, "qualDt")),
                         _val(row, "qUnitLabel"), _prec(_val(row, "qPrec")))
            if qlabel and not re.fullmatch(r"P\d+", qlabel):
                by_st[sid]["quals"].append(Qualifier(_local(qprop), qlabel, qval))

    out: list[Statement] = []
    for sid in order:
        d = by_st[sid]
        # A statement whose property or value has no English label cannot be read
        # by the answering model, so it is dropped -- but by an EXACT test, not by
        # a string prefix (see _snak).
        if not d["property_label"] or not d["subject_label"]:
            continue
        if d["value"].label is None and d["value"].snaktype == "value":
            continue
        out.append(Statement(
            subject_id=d["subject_id"], subject_label=d["subject_label"],
            property_id=d["property_id"], property_label=d["property_label"],
            value=d["value"], rank=d["rank"] or "normal",
            qualifiers=sort_qualifiers(
                [q for q in d["quals"] if q.value.label is not None]
            ),
            direction=direction, source_entity_id=qid, statement_id=sid,
        ))
    return out


# Endpoint attempts made INSIDE one statement query. This is where retry lives;
# `run_eval` must not layer its own on top (see IncompleteRetrievalError).
_STATEMENT_QUERY_ATTEMPTS = 3

# Candidate properties per discovery probe. 500 keeps the VALUES clause well
# inside the endpoint's query-size limit while holding the probe count to four.
REVERSE_DISCOVERY_CHUNK = 500

# Properties per reverse fetch request. Measured at 20: 308 rows in 2.5 s on the
# largest DEV entity. Higher batches grow the query text without helping, since
# the per-property subqueries are already individually capped.
REVERSE_FETCH_BATCH = 20

_item_properties_memo: list[str] | None = None
_item_properties_lock = threading.Lock()


def _statement_rows(query: str, qid: str) -> tuple[list[dict], bool]:
    """Run one statement query; report whether the row limit bound."""
    rows = _sparql_query(
        query.format(qid=qid, limit=STATEMENT_ROW_LIMIT, p=_p, lang=LABEL_LANGS),
        timeout=60, attempts=_STATEMENT_QUERY_ATTEMPTS,
    )
    return rows, len(rows) >= STATEMENT_ROW_LIMIT


def item_properties() -> list[str]:
    """Every entity-valued property id, e.g. ["P6", "P17", ...] (~1,776).

    Entity-INDEPENDENT, so it is fetched once and reused: it is the candidate
    set that makes discovery exact rather than sampled. Cached on disk beside
    the statement cache under a name no QID can collide with.
    """
    global _item_properties_memo
    with _item_properties_lock:
        if _item_properties_memo is not None:
            return _item_properties_memo

    STATEMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = STATEMENT_CACHE_DIR / "_item_properties.json"
    raw = read_text_or_none(cache_file)
    pids: list[str] | None = None
    if raw is not None:
        try:
            cached = json.loads(raw)
            if isinstance(cached, dict) and cached.get("v") == STATEMENT_CACHE_VERSION:
                pids = cached["properties"]
        except (json.JSONDecodeError, KeyError):
            pids = None

    if pids is None:
        rows = _sparql_query(_ITEM_PROPERTIES.format(), timeout=60,
                             attempts=_STATEMENT_QUERY_ATTEMPTS)
        pids = sorted({r["p"]["value"].rsplit("/", 1)[1] for r in rows},
                      key=lambda x: int(x[1:]))
        try:
            atomic_write_text(cache_file, json.dumps(
                {"v": STATEMENT_CACHE_VERSION, "properties": pids},
                ensure_ascii=False))
        except OSError:
            pass

    with _item_properties_lock:
        _item_properties_memo = pids
    return pids


def incoming_properties(qid: str) -> list[str]:
    """Exactly the properties through which something points at `qid`.

    ⚠️ EXACT, NOT SAMPLED, and the difference is large. Reading a bounded sample
    of incoming statements gave 22 properties for Q30 at 10,000 rows and 35 at
    50,000 -- it does not converge, so the answer would have depended on a
    number we picked. Probing every candidate property with FILTER EXISTS finds
    **152**. Sampling was missing about 80% of them.
    """
    pids = item_properties()
    found: list[str] = []
    for i in range(0, len(pids), REVERSE_DISCOVERY_CHUNK):
        chunk = pids[i:i + REVERSE_DISCOVERY_CHUNK]
        try:
            rows = _sparql_query(
                _REVERSE_PROPERTY_PROBE.format(
                    qid=qid, values=" ".join(f"ps:{p}" for p in chunk)),
                timeout=90, attempts=_STATEMENT_QUERY_ATTEMPTS,
            )
            found += [r["psP"]["value"].rsplit("/", 1)[1] for r in rows]
        except Exception:
            # The chunk's EXISTS materialization can time out on one huge
            # property alone (see _REVERSE_PROBE_SINGLE). Re-probe the chunk
            # one property at a time with the LIMIT-1 form: slower, same
            # exact answer. A property whose own probe fails still raises —
            # silently skipping it would silently lose every incoming
            # statement it carries.
            for p in chunk:
                if _sparql_query(
                        _REVERSE_PROBE_SINGLE.format(pid=p, qid=qid),
                        timeout=30, attempts=_STATEMENT_QUERY_ATTEMPTS):
                    found.append(p)
    return sorted(set(found), key=lambda x: int(x[1:]))


def _reverse_rows(qid: str) -> tuple[list[dict], bool]:
    """Incoming-statement rows for `qid`, fetched per property in batches.

    Returns `(rows, truncated)`. `truncated` is True when any property hit
    `REVERSE_PER_PROP_CAP` -- that cap is now what bounds the QUERY, so the
    slice it takes is per property rather than an arbitrary global cut, but it
    is still a slice and still recorded.
    """
    props = incoming_properties(qid)
    if not props:
        return [], False

    rows: list[dict] = []
    per_prop: dict[str, set[str]] = {}
    for i in range(0, len(props), REVERSE_FETCH_BATCH):
        batch = props[i:i + REVERSE_FETCH_BATCH]
        blocks = "\n  UNION\n".join(
            _REVERSE_BLOCK.format(pid=p, qid=qid, cap=REVERSE_PER_PROP_CAP)
            for p in batch)
        rows += _sparql_query(
            _REVERSE_BY_PROPERTIES.format(blocks=blocks, p=_p, lang=LABEL_LANGS),
            timeout=90, attempts=_STATEMENT_QUERY_ATTEMPTS,
        )

    # A property that came back at exactly the cap was cut. Counted on distinct
    # STATEMENTS, not rows: the qualifier join multiplies rows per statement, so
    # a row count would report truncation that did not happen.
    for r in rows:
        pid = r.get("prop", {}).get("value", "").rsplit("/", 1)[-1]
        st = r.get("st", {}).get("value", "")
        if pid and st:
            per_prop.setdefault(pid, set()).add(st)
    truncated = any(len(v) >= REVERSE_PER_PROP_CAP for v in per_prop.values())
    return rows, truncated


def _statement_to_dict(s: Statement) -> dict:
    def snak(v: SnakValue) -> dict:
        return {"snaktype": v.snaktype, "datatype": v.datatype,
                "id": v.id, "label": v.label, "raw": v.raw,
                "unit": v.unit, "precision": v.precision}
    return {
        "subject_id": s.subject_id, "subject_label": s.subject_label,
        "property_id": s.property_id, "property_label": s.property_label,
        "value": snak(s.value), "rank": s.rank, "direction": s.direction,
        "source_entity_id": s.source_entity_id, "statement_id": s.statement_id,
        "qualifiers": [{"property_id": q.property_id,
                        "property_label": q.property_label,
                        "value": snak(q.value)} for q in s.qualifiers],
    }


def _statement_from_dict(d: dict) -> Statement:
    def snak(x: dict) -> SnakValue:
        return SnakValue(x["snaktype"], x["datatype"], x.get("id"),
                         x.get("label"), x.get("raw"),
                         x.get("unit"), x.get("precision"))
    return Statement(
        subject_id=d["subject_id"], subject_label=d["subject_label"],
        property_id=d["property_id"], property_label=d["property_label"],
        value=snak(d["value"]), rank=d["rank"],
        qualifiers=tuple(Qualifier(q["property_id"], q["property_label"], snak(q["value"]))
                         for q in d["qualifiers"]),
        direction=d["direction"], source_entity_id=d["source_entity_id"],
        statement_id=d["statement_id"],
    )


def cache_fingerprint() -> dict:
    """The settings that determine WHAT a fetch retrieved.

    Read-time settings (RANK_POLICY, NOISE_FILTER_MODE)
    are deliberately absent: `_clean_statements` applies those at return time,
    so changing them must NOT invalidate the cache. Only the query-shaping
    limits belong here.
    """
    return {"row_limit": STATEMENT_ROW_LIMIT,
            "label_langs": LABEL_LANGS,
            # The reverse fetch is per property now, so its cap shapes the
            # QUERY rather than filtering afterwards -- changing it changes what
            # was retrieved and must invalidate the cache.
            "reverse_per_prop_cap": REVERSE_PER_PROP_CAP,
            "reverse_discovery": "filter_exists_over_item_properties"}


def cache_entry_status(qid: str) -> dict:
    """
    Whether `qid`'s cache entry may be used, and why not when it may not.

    THE ONE VALIDATOR. `fetch_statements`, `fetch_entity_meta` and
    `tools/prewarm_statement_cache.py` all route through it, because they used
    to disagree: prewarm tested only `Path.exists()`, so after a version bump it
    reported "nothing to do" while every entry was stale -- pushing hundreds of
    live fetches into the eval run it was supposed to make unnecessary.

    state:
      missing     no file
      malformed   unreadable, or not the expected shape
      stale       written by an older cache version, or under different
                  query-shaping limits (see `cache_fingerprint`)
      incomplete  a required query did not succeed (should not occur -- such
                  results are not written -- but a partial entry must never be
                  silently trusted if one appears)
      ok          usable
    """
    cache_file = STATEMENT_CACHE_DIR / f"{qid}.json"
    raw = read_text_or_none(cache_file)
    if raw is None:
        return {"state": "missing", "qid": qid}
    try:
        cached = json.loads(raw)
    except json.JSONDecodeError:
        return {"state": "malformed", "qid": qid, "reason": "not valid JSON"}
    if not isinstance(cached, dict) or "statements" not in cached:
        return {"state": "malformed", "qid": qid, "reason": "unexpected shape"}
    if cached.get("v") != STATEMENT_CACHE_VERSION:
        return {"state": "stale", "qid": qid,
                "reason": f"cache v{cached.get('v')} != v{STATEMENT_CACHE_VERSION}"}
    if cached.get("fingerprint") != cache_fingerprint():
        return {"state": "stale", "qid": qid, "reason": "query limits changed"}
    succeeded = cached.get("query_succeeded") or {}
    missing = sorted(k for k, v in succeeded.items() if not v)
    if missing or not succeeded:
        return {"state": "incomplete", "qid": qid,
                "reason": f"failed queries: {', '.join(missing) or 'unrecorded'}"}
    return {"state": "ok", "qid": qid, "entry": cached,
            "truncated": cached.get("truncated") or {}}


def statement_cache_is_usable(qid: str) -> bool:
    """True when `qid` can be served from cache without a live fetch."""
    return cache_entry_status(qid)["state"] == "ok"


def truncated_entities(qids: list[str]) -> dict[str, list[str]]:
    """{qid: [directions whose limit bound]} over cached entries.

    Truncation is PERMITTED for a canonical run -- forbidding it would mean not
    running at all. What is not permitted is losing track of it: this is what
    lets a run record which entities were served an arbitrary slice, instead of
    a console warning that scrolls away.

    ⚠️ MEASURED 154 OF 238 DEV GOLD ENTITIES, ALL `reverse` (this said "34 of
    238" until 2026-08-16, which was the pre-2026-08-14 meaning -- the global
    4,000-row limit binding). `reverse` here means at least one property hit
    `REVERSE_PER_PROP_CAP`, a cap `_clean_statements` always applied and nothing
    ever recorded. The loss did not grow; the reporting did.
    """
    out: dict[str, list[str]] = {}
    for qid in qids:
        status = cache_entry_status(qid)
        if status["state"] != "ok":
            continue
        bound = sorted(k for k, v in (status["truncated"] or {}).items() if v)
        if bound:
            out[qid] = bound
    return out


def fetch_statements(qid: str, strict: bool | None = None) -> list[Statement]:
    """
    Statements for `qid`, both directions, cleaned.

    The cache stores RAW statements and `_clean_statements` runs at RETURN time,
    so filter or rank-policy changes take effect with no refetch.

    `strict` defaults to the module's STRICT_RETRIEVAL, resolved at CALL time. Under
    strict, a query that fails after its own retries raises
    IncompleteRetrievalError instead of returning a partial graph.
    """
    if strict is None:
        strict = STRICT_RETRIEVAL
    STATEMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = STATEMENT_CACHE_DIR / f"{qid}.json"
    status = cache_entry_status(qid)
    if status["state"] == "ok":
        return _clean_statements(
            [_statement_from_dict(d) for d in status["entry"]["statements"]])

    # ⚠️ THIS QUERY IS LOAD-BEARING FOR EVERY INCOMING STATEMENT, and it used to
    # fail silently. An incoming statement's VALUE is this entity, so `own_label`
    # is what gets rendered there: lose it and "[Fred Weasley] present in work:
    # Harry Potter" becomes "... : Q8337" — for all ~90 incoming statements at
    # once, and then it is CACHED, so one transient timeout poisons the entity
    # permanently. Observed live on Q8337 while building this.
    #
    # Hence: same retry as the statement queries, and on failure the cache write
    # is SKIPPED so a later run can recover rather than inheriting the damage.
    meta = _meta_for(qid)
    # Follow a redirect ONCE. Wikidata merges are not chained in practice, and a
    # loop here would turn a data error into a hang.
    query_qid = qid
    if meta.get("redirect"):
        query_qid = meta["redirect"]
        target_meta = _meta_for(query_qid)
        if target_meta["label"] != query_qid:
            meta = target_meta | {"redirect": query_qid}
        print(f"  [wikidata] {qid} is a redirect to {query_qid} — following it "
              f"(the stale QID carries no statements)")

    own_label = meta["label"]
    # ⚠️ RESOLVED AGAINST BOTH QIDs, NOT JUST THE ONE WE QUERY. When a redirect
    # is followed but the TARGET's metadata fetch also fails, `meta` stays the
    # stale entity's -- whose label may itself be the stale QID. Testing only
    # `own_label != query_qid` then passes ("Q4439148" != "Q315625") and caches
    # a bare-QID label, which is precisely the poisoning this guard exists to
    # stop. A QID-shaped label is never a real label.
    label_resolved = (own_label != query_qid and own_label != qid
                      and not re.fullmatch(r"Q\d+", own_label or ""))

    statements: list[Statement] = []
    truncated: dict[str, bool] = {"forward": False, "reverse": False}
    succeeded: dict[str, bool] = {"label": label_resolved,
                                  "forward": False, "reverse": False}

    rows, hit = _statement_rows(_FORWARD_STATEMENTS, query_qid)
    truncated["forward"] = hit
    succeeded["forward"] = True
    statements += _rows_to_statements(rows, qid=query_qid, own_label=own_label,
                                      direction="outgoing")
    try:
        rrows, rhit = _reverse_rows(query_qid)
        truncated["reverse"] = rhit
        succeeded["reverse"] = True
        statements += _rows_to_statements(rrows, qid=query_qid, own_label=own_label,
                                          direction="incoming")
    except Exception as exc:
        # 🔴 NEVER CACHE A FORWARD-ONLY RESULT. This used to swallow the failure
        # and cache what it had, so one exhausted reverse query permanently cost
        # the entity every INCOMING statement -- "films directed by X", "people
        # born in Y" -- with nothing on disk to show it. Same class as the label
        # poisoning above, and it had the same fix available all along.
        print(f"  [wikidata] ⚠️ reverse fetch FAILED for {qid} "
              f"({type(exc).__name__}) — NOT caching (forward-only is a "
              f"silent, permanent loss of every incoming statement)")
        if strict:
            raise IncompleteRetrievalError(
                f"{qid}: reverse statement query failed after "
                f"{_STATEMENT_QUERY_ATTEMPTS} attempts ({type(exc).__name__})"
            ) from exc

    if query_qid != qid:
        # `groups` keys the per-entity z-normalisation on the QID the QUESTION
        # named, so the redirect must not leak into it. Only the fetch follows
        # the redirect; the provenance stays the gold QID.
        statements = [replace(s, source_entity_id=qid) for s in statements]

    bound = [k for k, v in truncated.items() if v]
    if bound:
        # Loud, because an arbitrary slice is exactly what the old LIMIT 500 hid.
        # ⚠️ Also PERSISTED (below) -- a console warning scrolls away, and an
        # entry that is a known arbitrary slice must stay distinguishable from a
        # complete one for as long as the cache lives.
        # The two directions bind for different reasons, so name them: forward
        # hits the global STATEMENT_ROW_LIMIT, reverse hits REVERSE_PER_PROP_CAP
        # on at least one property. Calling both "row limit" invited reading the
        # reverse count as a regression when the cap simply became visible --
        # `_clean_statements` always applied it, nothing ever recorded it.
        why = {"forward": f"row limit {STATEMENT_ROW_LIMIT}",
               "reverse": f"per-property cap {REVERSE_PER_PROP_CAP}"}
        detail = ", ".join(f"{d} ({why[d]})" for d in bound)
        print(f"  [wikidata] ⚠️ truncated for {qid}: {detail} "
              f"— that part of the retrieved set is a slice, not the whole")

    if not all(succeeded.values()):
        # Do NOT cache. A cached entity whose own label is a bare QID renders
        # every incoming statement with an unreadable value, and a forward-only
        # entity is missing a whole direction -- and nothing downstream would
        # ever ask again.
        reasons = []
        if not succeeded["label"]:
            reasons.append("label unresolved (incoming statements would render "
                           "as bare QIDs)")
        if not succeeded["forward"]:
            reasons.append("forward query failed")
        if not succeeded["reverse"]:
            reasons.append("reverse query failed (every incoming statement lost)")
        print(f"  [wikidata] ⚠️ NOT caching {qid} — {'; '.join(reasons)}")
    else:
        try:
            atomic_write_text(cache_file, json.dumps({
                "v": STATEMENT_CACHE_VERSION,
                "meta": meta,
                # Structured and directional. `query_succeeded` says the fetch
                # ran; `truncated` says it ran but returned an arbitrary slice.
                # A reverse query can succeed AND be non-exhaustive, so the two
                # are separate facts and must not be collapsed into one flag.
                "query_succeeded": succeeded,
                "truncated": truncated,
                "fingerprint": cache_fingerprint(),
                "statements": [_statement_to_dict(s) for s in statements],
            }, ensure_ascii=False))
        except OSError:
            pass
    return _clean_statements(statements)


def _meta_for(qid: str) -> dict:
    """
    One live entity-metadata fetch. Degrades to the QID as label rather than
    raising, matching the rest of this module: a transient failure must cost the
    entity's metadata, not the question.
    """
    meta = {"label": qid, "description": None, "aliases": [], "redirect": None}
    try:
        rows = _sparql_query(_ENTITY_META.format(qid=qid), timeout=60, attempts=3)
    except Exception:
        return meta
    labels = {_val(r, "l") for r in rows} - {None}
    if labels:
        # Deterministic lexical choice among returned en/mul labels. The
        # returned values do not retain language tags for priority selection.
        meta["label"] = sorted(labels)[0]
    descs = {_val(r, "d") for r in rows} - {None}
    if descs:
        meta["description"] = sorted(descs)[0]
    meta["aliases"] = sorted({_val(r, "a") for r in rows} - {None})
    targets = {_local(_val(r, "redirect")) for r in rows} - {None, ""}
    if targets:
        meta["redirect"] = sorted(targets)[0]
    return meta


def fetch_entity_meta(qid: str) -> dict:
    """
    Label, one-line description and aliases for `qid`.

    Reads the cache written by `fetch_statements`, so calling it after (or
    alongside) a statement fetch costs nothing. Falls back to its own small
    query when the entity has not been fetched yet.

    Returns `{"label": str, "description": str | None, "aliases": list[str]}`;
    `label` degrades to the QID rather than raising, matching `fetch_statements`.
    """
    # Same validator as fetch_statements and prewarm -- three readers of one
    # cache must not disagree about whether an entry is usable.
    status = cache_entry_status(qid)
    if status["state"] == "ok":
        meta = status["entry"].get("meta")
        if meta:
            return meta

    meta = _meta_for(qid)
    if meta.get("redirect"):
        target = _meta_for(meta["redirect"])
        if target["label"] != meta["redirect"]:
            return target | {"redirect": meta["redirect"]}
    return meta


def _clean_statements(statements: list[Statement]) -> list[Statement]:
    """
    Dedup + rank filter + noise filter (statement AND qualifier properties) +
    per-property cap on reverse statements.

    ⚠️ The noise filter now runs on qualifier properties too, because qualifiers
    bring their own plumbing -- on Q2263 the raw fetch carries "Instagram numeric
    ID" as a qualifier, which is dropped by datatype like any other ExternalId.

    ⚠️ It catches LESS than a first reading suggests, and that is deliberate.
    Measured on Q2263: "statement is subject of" (15 occurrences, pointing at the
    award ceremony -- "67th Academy Awards", which itself carries the year) and
    "object of statement has role" (6, e.g. "voice actor") both SURVIVE. Neither
    is an identifier or a maintenance link, so no existing rule fires, and both
    read as facts rather than plumbing. Do not add them to the blocklist without
    measuring what they cost -- the qualifier census belongs in the context-
    inflation measurement (Step 6), not in a guess here.
    """
    cleaned: list[Statement] = []
    reverse_prop_counts: dict[str, int] = {}
    for s in dict.fromkeys(statements):
        if s.rank not in RANK_POLICY:
            continue
        if _is_noise_prop(s.property_label, s.value.datatype):
            continue
        if s.direction == "incoming":
            n = reverse_prop_counts.get(s.property_label, 0) + 1
            reverse_prop_counts[s.property_label] = n
            if n > REVERSE_PER_PROP_CAP:
                continue
        keep = tuple(q for q in s.qualifiers
                     if not _is_noise_prop(q.property_label, q.value.datatype))
        cleaned.append(s if keep == s.qualifiers else replace(s, qualifiers=keep))
    return cleaned


# ---------------------------------------------------------------------------
# Step 5 — The demo entry points: MOVED OUT (2026-08-16)
# ---------------------------------------------------------------------------
#
# `retrieve_context(entity_name)` and its formatter are in
# `src/retrieval/entity_linking.py`. This module is now purely the eval path:
# QID in, statements out.
