"""
The unit of Wikidata data: a STATEMENT, not a triple.

WHY THIS TYPE EXISTS. Configurations 3 and 4 modelled Wikidata as
`(subject, predicate, object)` 3-tuples for three months. A 3-tuple has nowhere
to put a qualifier, so the retrieval could not express *when* a fact held, *what*
it applied to, or *how many times* it happened. Tom Hanks won the Academy Award
for Best Actor twice — 1994 for *Philadelphia*, 1995 for *Forrest Gump* — and the
old representation showed one line, because Wikidata's flat `wdt:` view collapses
statements that share a property and a value.

⚠️ That collapse is NOT a deduplication bug in this repo. It happens inside
Wikidata's RDF before any of our code runs; verified against the raw disk cache,
where the award already appears once. Removing `_clean_statements()`'s dedup would change
nothing. The fix had to be the data structure.

Measured on the first 60 DEV gold QIDs (2026-08-11): 11,873 statements collapse
to 8,870 truthy edges, so **25.3% carry no distinct edge**, and **31.1% carry
qualifiers that were never read**.

SHAPE. `SnakValue` is shared between a statement's own value and its qualifiers'
values because in Wikidata they ARE the same construct — the JSON calls them
`mainsnak` and qualifier snaks and gives them identical structure. Mirroring the
source model rather than inventing a parallel one is deliberate.

The representation and its limitations are described in `docs/reproduction.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Wikidata's own UI wording for the two non-values, reused so the rendered
# context says what wikidata.org says. Parenthesised because bare "unknown value"
# reads as though the value IS the string "unknown value".
UNKNOWN_VALUE = "(unknown value)"
NO_VALUE = "(no value)"

# Statement ranks, best first. `deprecated` marks a statement Wikidata considers
# wrong or superseded -- it is never served as current.
RANKS = ("preferred", "normal", "deprecated")

Direction = Literal["outgoing", "incoming"]

# Wikidata pads every date to midnight and may prefix a sign, so a plain day
# reaches us as "+1994-03-21T00:00:00Z". Rendering that verbatim spends tokens on
# padding and makes qualifier-heavy lines much longer than they need to be --
# every `point in time` on every statement carries eleven wasted characters.
_MIDNIGHT = "T00:00:00Z"

# 🔴 PRECISION IS NOT COSMETIC -- WITHOUT IT WE ASSERT A DAY WIKIDATA DOES NOT
# CLAIM. Wikidata stores a YEAR-precision date as 1 January and flags the
# precision separately, so rendering the stored string gives "1994-01-01" for a
# fact whose content is "1994". Measured on 29 DEV gold entities: 16 of 36 time
# values are year precision, i.e. ~44% of dates were being over-specified.
#
# Values are Wikidata's own scale; only those we can render honestly are named.
_PREC_DAY, _PREC_MONTH, _PREC_YEAR, _PREC_DECADE = 11, 10, 9, 8


def _tidy_time(text: str, precision: int | None = None) -> str:
    """
    Strip Wikidata's padding and truncate to what the precision actually claims.

    With `precision` None the old behaviour is kept -- strip the midnight padding
    and nothing else -- so a value fetched before the precision join existed
    still renders, just over-specified.
    """
    if text.endswith(_MIDNIGHT):
        text = text[: -len(_MIDNIGHT)]
        if text.startswith("+"):
            text = text[1:]
    if precision is None:
        return text
    # ⚠️ BCE dates carry a leading "-" (astronomical year numbering). Until
    # 2026-08-18 that sign failed the isdigit() gate below, so a year-precision
    # BCE date skipped truncation entirely and rendered "1 January" of a year
    # Wikidata never claimed a day for — 66 cached snaks. The sign is split off
    # so the digits gate and the "-"-split see only the date body, and is
    # re-attached to whatever the precision keeps.
    sign, body = ("-", text[1:]) if text.startswith("-") else ("", text)
    if not body[:1].isdigit():
        return text

    parts = body.split("-")
    if precision >= _PREC_DAY:
        return text
    if precision == _PREC_MONTH:
        return sign + "-".join(parts[:2])
    if precision == _PREC_YEAR:
        return sign + parts[0]
    if precision == _PREC_DECADE:
        return f"{sign}{parts[0]}s"
    # Century and coarser. The stored year is the START of the span, not a claim
    # about that year, so naming it plainly would be a fabrication of the same
    # kind precision exists to prevent. "circa" is vague in the direction the
    # data is vague. Rare: 0 occurrences in the 29-entity DEV sample.
    return f"circa {sign}{parts[0]}"


@dataclass(frozen=True)
class SnakValue:
    """
    One Wikidata value — a statement's own, or a qualifier's.

    `snaktype` is the field that makes "we know there is none" different from
    "we do not know" and from "we have no data at all". Wikidata records the
    first two explicitly (`novalue`, `somevalue`); the old triple pipeline
    dropped both, so all three collapsed into the same silence. A question whose
    true answer is *zero* or *none* was unanswerable not because the graph was
    incomplete but because the retrieval could not carry the distinction.

    `raw` holds the lexical value returned by the endpoint, such as an ISO
    timestamp or quantity amount. Units and time precision have separate fields;
    calendar models are not fetched. It is a `str`, not a `dict`: a dict would make
    this dataclass unhashable, and `_clean_statements()` dedups via `dict.fromkeys`.
    Anything needing structure parses `raw` at the point of use.
    """

    snaktype: str                 # value | somevalue | novalue
    datatype: str                 # wikibase-item | time | quantity | external-id | …
    id: str | None = None         # QID when the value is an entity
    label: str | None = None
    raw: str | None = None
    # From the psv: VALUE NODE, which the flat value alone cannot carry.
    #
    # `unit` — a bare quantity is not the fact. Wikidata says *600,000,000 United
    # States dollars*; dropping the unit leaves "600000000", which a reader can
    # only guess at. 23 of 108 quantity statements in the DEV sample carry a real
    # unit (the rest are genuinely unitless — counts, ordinals).
    #
    # `precision` — see _tidy_time. Without it a year-precision date renders as
    # 1 January of that year, asserting a day that Wikidata does not claim.
    unit: str | None = None       # resolved unit LABEL, not the QID
    precision: int | None = None  # Wikidata time precision (9=year, 11=day)

    def render(self) -> str:
        """Display form. Never empty — an empty value would read as a missing field."""
        if self.snaktype == "somevalue":
            return UNKNOWN_VALUE
        if self.snaktype == "novalue":
            return NO_VALUE
        text = self.label or self.raw or UNKNOWN_VALUE
        text = _tidy_time(text, self.precision)
        if self.unit:
            text = f"{text} {self.unit}"
        return text

    @property
    def sort_key(self) -> str:
        return self.label or self.raw or self.id or self.snaktype


@dataclass(frozen=True)
class Qualifier:
    """One key–value pair scoping a statement: *when*, *for what*, *in which role*."""

    property_id: str
    property_label: str
    value: SnakValue

    def render(self) -> str:
        return f"{self.property_label}: {self.value.render()}"

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.property_id, self.value.sort_key)


@dataclass(frozen=True)
class Statement:
    """
    One Wikidata statement, including the qualifiers and typed values fetched.

    Field notes, each recording a specific failure it prevents:

    `rank`
        Fetching statements without it would surface **deprecated** values as
        though current — strictly worse than the old behaviour, which at least
        filtered them out by accident. Rank is what makes the extra statements
        safe to use, not optional metadata.

    `direction`
        Replaces inferring forward-vs-reverse by regex-testing whether the
        subject looks like a QID (`re.fullmatch(r"Q\\d+", subj)`). That test was
        load-bearing enough that a previous reverse-triple change was rejected
        for breaking it.

    `source_entity_id`
        The QID we consulted, which is the grouping the per-entity
        z-normalisation needs. Previously reconstructed rather than carried.

    `statement_id`
        Wikidata's statement GUID. **Metadata — never rendered**, since a GUID in
        the prompt is noise. Kept for two things the old shape could not do:
        detecting data drift between runs (the `gold_error` audit findings are
        drift), and recognising one fact reached by two paths, since the same
        statement fetched from two question entities produces two objects that
        differ in `direction` and `source_entity_id`. ⚠️ A GUID is stable while a
        statement lives; deleting and re-adding a fact mints a new one. That
        makes it a good CHANGE signal, not a permanent identifier.
    """

    subject_id: str
    subject_label: str
    property_id: str
    property_label: str
    value: SnakValue
    rank: str
    qualifiers: tuple[Qualifier, ...] = ()
    direction: Direction = "outgoing"
    source_entity_id: str = ""
    statement_id: str = ""

    def render(self) -> str:
        """
        The line the embedding ranker scores and the answering LLM reads.

        Format is `[Subject] property: value (qual: value; qual: value)`, chosen
        for this study's qualifier-bearing records. This is a project-specific
        rendering, not a claim to reproduce another system's prompt format.

        Qualifier property names are kept rather than dropped for brevity. A bare
        `(1994, Philadelphia)` is shorter but ambiguous — 1994 could be a birth,
        a release or a ceremony — and the ranker embeds this string, so an
        ambiguous rendering degrades retrieval as well as reading.

        ⚠️ The subject prefix is load-bearing and predates this type: bare
        `pred: obj` strings caused mass abstention, because the answering model
        could not attribute a fact to an entity.
        """
        line = f"[{self.subject_label}] {self.property_label}: {self.value.render()}"
        if self.qualifiers:
            inner = "; ".join(q.render() for q in self.qualifiers)
            line = f"{line} ({inner})"
        return line

    @property
    def is_deprecated(self) -> bool:
        return self.rank == "deprecated"


def sort_qualifiers(qualifiers: list[Qualifier] | tuple[Qualifier, ...]) -> tuple[Qualifier, ...]:
    """
    Deterministic qualifier order.

    ⚠️ NOT cosmetic. SPARQL guarantees no row order, so the same statement
    fetched twice can arrive with its qualifiers permuted. Unsorted, the two
    would render as different strings and both survive deduplication — the pool
    would carry the same fact twice, and the top-k would spend a slot on it.
    """
    return tuple(sorted(qualifiers, key=lambda q: q.sort_key))
