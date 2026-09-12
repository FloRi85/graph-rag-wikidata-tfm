"""
The question-side metadata Mintaka ships alongside each question.

WHY THIS MODULE EXISTS. `load_questions.transform_row` used to reduce
`questionEntity[]` to three parallel lists in a single `if`:

    if m.get("entityType") in keep and m.get("name"):

That one line made SEVEN decisions silently. It dropped every non-`entity` type
(ordinal, date, cardinal, quantity, money, percent, time -- 6,418 mentions), and
it dropped the 237 mentions Mintaka marks as entity-typed but could not link to
Wikidata. Nothing recorded that a choice had been made, so neither could be
reviewed, measured or argued for. This module replaces that with a registry
that is EXHAUSTIVE over all eight types: an unhandled type raises rather than
vanishing.

WHAT IS IN SCOPE HERE, AND WHAT IS NOT. A Mintaka record has exactly seven
top-level keys, and they fall into three groups:

  reaches the prompt   `question`, `questionEntity[]`
  EVALUATION ONLY      `answer.*`, incl. supportingEnt / supportingNum
  DELIBERATELY UNUSED  `category`, `complexityType`, `translations`

`supportingEnt` on a `count` question is the list of things being counted
(Angus Young, Cliff Williams, ... for "how many current members of AC/DC"), so
routing it into a prompt would be feeding the answer. This module reads
`questionEntity` and nothing else -- the boundary is enforced by what the
functions accept, not by remembering.

⭐ THE LINE IS "VISIBLE IN THE QUESTION", AND IT IS WHY THE LITERALS ARE SAFE.
"$200 million" is already in the text the model reads; Mintaka merely also
supplies it normalised as `money(200000000)`. Passing that on is normalisation
of visible information, never new information. The same test is what excludes
the other two annotations (decision 2026-08-14):

  `complexityType` is an annotator's judgement ABOUT the question, not
  something present in it. Telling the model "this is a `count` question" is
  close to telling it "reply with a number" -- `count` maps to a numerical
  answer on 2,000 of 2,000 rows, and majority-class prediction of answer_type
  from complexity is right 82.6% of the time. In deployment nothing supplies
  it; it would have to be inferred.

  `category` is milder but the same shape: a topic label assigned by a human,
  not a fact stated in the question.

  `translations` is out for the simpler reason that the evaluation is English.

⚠️ THIS IS A DIFFERENT KIND OF ORACLE FROM THE GOLD QIDs, and the distinction
is worth stating in the write-up. The QID bypass isolates retrieval from entity
linking and has direct precedent -- KAPING does exactly this on Mintaka
(Baek 2023, Appendix A.1, verified off the PDF). There is no comparable
precedent for handing a model the question's reasoning type.

Both excluded fields remain available for ANALYSIS, where they cost nothing:
`complexityType` is the right stratification for the class-anchor finding, and
`translations` makes the cross-lingual extension a concrete piece of future
work rather than an aspirational one.

⚠️ TYPES ARE BUILT ONE AT A TIME, AND `PENDING_TYPES` NAMES THE ONES THAT ARE
NOT DONE. Each type gets its own handler, its own docstring and its own tests
rather than a shared parser, because their shapes genuinely differ: `entity` is
keyed on a QID and merges across surface forms, `cardinal` is a numeric operand
whose value is cleaner than its mention, and `quantity`/`money`/`percent`/`time`
put the unit ONLY in the mention. A single parser would have to hide those
differences to work, and hiding them is what this module exists to stop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Stratification vocabularies
# ---------------------------------------------------------------------------
#
# `category` and `complexityType` are properties of the QUESTION, so they live
# here -- but they are ANALYSIS-ONLY and never reach a prompt. See the module
# docstring for why: they are annotator judgements about a question rather than
# anything visible in it.
#
# ⚠️ THREE OF THESE VOCABULARIES OVERLAP ACROSS AXES, which makes unqualified
# prose ambiguous:
#     ordinal  is both a complexityType and a questionEntity.entityType
#     date     is both a questionEntity.entityType and an answer.answerType
#     entity   is both a questionEntity.entityType and an answer.answerType
#     count    is a complexityType, but reads like an answer shape
# They are therefore listed QUALIFIED: `c_` complexity, `q_` question entity,
# `a_` answer. The qualifier is a naming convention for docs / CLI / analysis
# ONLY -- the literal Mintaka value is the part after the prefix, and that is
# what the parsers compare against. Use `raw_type()` to get back to the
# on-the-wire value. CATEGORIES needs no qualifier: nothing collides with it.
_QUALIFIERS = ("c_", "q_", "a_")

CATEGORIES = (
    "history", "movies", "music", "videogames",
    "sports", "books", "geography", "politics",
)
COMPLEXITY_TYPES = (
    "c_generic", "c_intersection", "c_count", "c_comparative", "c_yesno",
    "c_ordinal", "c_multihop", "c_difference", "c_superlative",
)
QUESTION_ENTITY_TYPES = (
    "q_entity", "q_cardinal", "q_ordinal", "q_date",
    "q_time", "q_percent", "q_quantity", "q_money",
)


def raw_type(name: str) -> str:
    """Qualified vocabulary name -> the literal Mintaka value ("q_entity" -> "entity").

    Idempotent: an already-raw value passes through unchanged, so callers may
    accept either spelling.
    """
    return name[2:] if name[:2] in _QUALIFIERS else name


# Every `questionEntity.entityType` that occurs in Mintaka, measured over all
# 20,000 raw rows (train 14,000 + dev 2,000 + test 4,000):
#
#   entity   29,360     ordinal  3,308     date     2,322     cardinal  694
#   quantity     60     money       16     percent      9     time         9
#
# `entity` is the only type carrying a QID and a label; the other seven put a
# NORMALISED literal in `name` (money 200000000 for "$200 million", percent 9 for
# "nine per cent") and leave `label` null on every single row.
ENTITY = "entity"
CARDINAL = "cardinal"
ORDINAL = "ordinal"
DATE = "date"
QUANTITY = "quantity"
MONEY = "money"
PERCENT = "percent"
TIME = "time"
LITERAL_TYPES = ("ordinal", "date", "cardinal", "quantity", "money", "percent", "time")
ALL_TYPES = (ENTITY,) + LITERAL_TYPES

# Types recognised by the registry but whose handler is not written yet.
#
# ⚠️ THIS SET EXISTS SO THE OMISSION IS DECLARED RATHER THAN SILENT. The failure
# being replaced was a single `if` that dropped six thousand mentions without a
# trace; a set that must be edited to shrink is the opposite of that. A test
# pins its contents, so removing a type from here without writing its handler
# fails the suite.
#
# ✅ EMPTY as of 2026-08-14: all eight types have handlers. It is kept rather
# than deleted because a future Mintaka release adding a type should land here
# first -- declared and inert -- rather than silently in the `continue` of every
# builder.
PENDING_TYPES: frozenset[str] = frozenset()


class UnknownEntityType(ValueError):
    """Raised when Mintaka carries an entityType this module does not declare.

    Deliberately fatal. The failure this replaces was silent: an unrecognised
    type simply failed the `in keep` test and disappeared, which is how 6,418
    literal mentions and 237 unlinked entities left the pipeline without anyone
    noticing for three months.
    """


@dataclass(frozen=True)
class EntityPackage:
    """One question entity, merged across its surface mentions.

    `name` is the Wikidata QID, or None when Mintaka marked the mention as an
    entity but could not link it (237 mentions: fictional characters and minor
    names -- "Gerry Lane", "Cooke Maroney", "Winston").

    `mentions` holds ONLY surface forms that differ from `label`; when the
    question says exactly what Wikidata calls the thing -- 57.7% of TEST anchors
    -- the list is empty and the renderer prints the label alone.

    `span` is the EARLIEST character offset among this entity's mentions, since
    a merged entity occupies several positions in the question. It exists so the
    block can be sorted into reading order -- see `build_packages`.
    """

    name: str | None
    entity_type: str
    label: str | None
    mentions: tuple[str, ...] = field(default=())
    span: tuple[int, int] | None = None
    first_mention: str | None = None

    @property
    def linked(self) -> bool:
        return self.name is not None

    @property
    def surface(self) -> str:
        """The name the QUESTION used for this entity; the label when they agree.

        ⚠️ NOT `mentions[0]`, AND THE DIFFERENCE IS LOAD-BEARING. `mentions` is a
        DISPLAY list: it holds only surface forms that differ from the label, so
        the rendered line does not repeat itself. `first_mention` is every
        mention, label-identical ones included, at its earliest span. Reading
        `mentions[0]` conflates the two and picks the wrong word whenever a
        question names an entity both ways:

            "Did the United States declare war on Japan following the
             Japanese attack on Pearl Harbor?"
            Q17 label 'Japan' · mentions 'Japan' (span 37), 'Japanese' (57)
            mentions[0]   -> 'Japanese'   (the only label-DIFFERING form)
            first_mention -> 'Japan'      (what the question says first)

        With `first_mention` this reproduces the old `transform_row` rule
        (`mention or label`) exactly on all three splits -- see
        `Question.entity_mentions`.
        """
        return self.first_mention or (self.mentions[0] if self.mentions
                                      else (self.label or ""))

    def render_line(self) -> str:
        if not self.linked:
            surface = self.mentions[0] if self.mentions else ""
            return f'- ({self.entity_type}, no Wikidata id): "{surface}"'
        line = f"- {self.name} ({self.entity_type}): {self.label}"
        if self.mentions:
            line += " — mentioned as " + ", ".join(f'"{m}"' for m in self.mentions)
        return line


# ---------------------------------------------------------------------------
# Spans — what puts the block in reading order
# ---------------------------------------------------------------------------
#
# ⚠️ MINTAKA'S ANNOTATION ORDER IS NOT THE QUESTION'S READING ORDER. Measured
# over the 12,192 questions carrying two or more spans, 7,225 (59.3%) list their
# mentions out of order:
#
#   "What is the seventh tallest mountain in North America?"
#       span=[40,53] entity  'North America'   <- listed first
#       span=[12,19] ordinal 'seventh'         <- occurs first in the question
#
# Sorting on the span puts each constraint next to the thing it constrains,
# which matters most for `ordinal` and `cardinal`: "first" attaches to "choice",
# "32" attaches to "NFL teams". `span` is usable on 19,998 of 20,000 rows.

# Sorts last, stably, when a row has no usable span (2 rows in 20,000).
_NO_SPAN = (1 << 30, 1 << 30)


def _read_span(mention_row: dict) -> tuple[int, int] | None:
    span = mention_row.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return (int(span[0]), int(span[1]))
        except (TypeError, ValueError):
            return None
    return None


def _earliest(a: tuple[int, int] | None, b: tuple[int, int] | None):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def sort_key(package) -> tuple[int, int]:
    """Reading-order key. Packages without a span sort last, preserving order."""
    return package.span or _NO_SPAN


def _surface_key(text: str) -> str:
    """Normalised form for deciding whether two spellings are the same one.

    Case-insensitive by decision (2026-08-13): 107 TEST mentions differ from the
    label by capitalisation alone ("the Beatles" / "The Beatles"), and printing
    both forms for those adds a line of prompt without adding information.
    Strict identity collapses 57.7% of TEST anchors; folding case takes it to
    59.5%.

    ⚠️ KEEP THIS IN STEP WITH `parse_answers._norm`. The question side and the
    answer side apply the SAME rule -- keep both spellings when they differ,
    keep one when they are identical -- and only the purpose differs: here the
    forms are printed into a prompt, there they are matched against. Collapsing
    internal whitespace as well as case affects 0 rows in 20,000 on either side,
    so the two are aligned at no cost rather than left almost-identical.
    """
    return " ".join(str(text).split()).casefold()


def _same_surface(label: str, mention: str) -> bool:
    """Is this mention just the label again?"""
    return _surface_key(label) == _surface_key(mention)


def build_entity_packages(question_entities: list[dict] | None) -> list[EntityPackage]:
    """`questionEntity[]` -> one EntityPackage per distinct entity, order preserved.

    Reads ONLY entityType == "entity". The seven literal types are recognised
    (so they cannot raise) and skipped, because nothing consumes them yet.

    ⚠️ MERGED PER QID, NOT PER MENTION. Mintaka tags the same entity twice on 13
    TEST and 9 DEV questions, sometimes under different surface forms -- Q41254
    appears as both "Grammy Awards" and "Grammy". One record carrying both is
    strictly better than two records: it stops `rag.py` pooling the same
    Wikipedia article twice (which wasted two of its 30 top-k slots) without
    losing either wording.

    Unlinked mentions cannot be merged -- they have no key -- so they are
    deduplicated on the mention string and appended in order of appearance.

    ⚠️ A MERGED ENTITY'S `mentions` ARE ORDERED BY SPAN, NOT BY ANNOTATION ORDER,
    for the same reason `build_packages` sorts the block that way: Mintaka's
    order is not the question's. It decides which surface form `surface` returns,
    and therefore the name prepended to every rendered statement --
    "How many U.S. states were admitted to the Union during the 1840's?" tags
    Q30 as 'Union' (span 42) and 'U.S.' (span 9), and the question says 'U.S.'
    first. Rare enough to name exactly: 1 anchor in DEV-200, 3 in dev, 7 in test.
    """
    packages: dict[str, EntityPackage] = {}
    order: list[str] = []
    # qid -> [(span, mention)], so the surface forms can be put in reading order
    # once every row has been seen. Collected separately because EntityPackage is
    # frozen and built incrementally.
    mention_spans: dict[str, list[tuple[tuple[int, int], str]]] = {}
    unlinked: list[EntityPackage] = []
    seen_unlinked: set[str] = set()

    for mention_row in question_entities or []:
        entity_type = mention_row.get("entityType")
        if entity_type not in ALL_TYPES:
            raise UnknownEntityType(
                f"unknown questionEntity.entityType {entity_type!r}; "
                f"declared types are {ALL_TYPES}"
            )
        if entity_type != ENTITY:
            continue

        qid = mention_row.get("name")
        mention = (mention_row.get("mention") or "").strip()
        span = _read_span(mention_row)

        if not qid:
            # Entity-typed but unlinked. Kept rather than dropped: the mention is
            # already in the question text, so surfacing it leaks nothing, and it
            # is the only signal that the question turns on an entity we have no
            # subgraph for. All 9 zero-anchor TEST questions are of this kind.
            if mention and mention not in seen_unlinked:
                seen_unlinked.add(mention)
                # `first_mention` set explicitly: an unlinked package has no
                # label for `surface` to fall back to, and leaving the field
                # None while `mentions` holds the answer is the kind of
                # near-duplicate state that invites reading the wrong one.
                unlinked.append(
                    EntityPackage(None, ENTITY, None, (mention,), span, mention))
            continue

        # One TRAIN row (7bb95643, "Terry Malloy") carries a QID with no label.
        # The mention is the only name we have for it, so it becomes the label.
        label = mention_row.get("label") or mention

        existing = packages.get(qid)
        if existing is None:
            packages[qid] = EntityPackage(qid, ENTITY, label, (), span)
            order.append(qid)
            mention_spans[qid] = []
            existing = packages[qid]

        # EVERY mention is recorded with its span, label-identical ones included.
        # `mentions` is filtered down to the label-differing forms below; this
        # list also has to answer "what does the question say FIRST", which the
        # filtered one cannot -- see `EntityPackage.surface`.
        if mention and mention not in [m for _, m in mention_spans[qid]]:
            mention_spans[qid].append((span or _NO_SPAN, mention))

        # A merged entity occupies several positions; the earliest is the one
        # that puts it where a reader first meets it.
        if span != existing.span:
            packages[qid] = EntityPackage(
                existing.name,
                existing.entity_type,
                existing.label,
                existing.mentions,
                _earliest(existing.span, span),
                existing.first_mention,
            )

    # Sorted, not appended: stable, so equal spans keep the order Mintaka gave.
    for qid in order:
        p = packages[qid]
        in_order = sorted(mention_spans[qid], key=lambda t: t[0])
        packages[qid] = EntityPackage(
            p.name, p.entity_type, p.label,
            tuple(m for _, m in in_order
                  if not _same_surface(p.label or "", m)),
            p.span,
            in_order[0][1] if in_order else None,
        )

    return [packages[qid] for qid in order] + unlinked


# ---------------------------------------------------------------------------
# `cardinal` — a number appearing in the question (694 mentions, 667 questions)
# ---------------------------------------------------------------------------
#
# WHAT IT IS. Measured over all 20,000 rows, `cardinal` is overwhelmingly the
# right-hand operand of a NUMERIC COMPARISON, not a thing to retrieve:
#
#   complexity   yesno 172 · difference 167 · intersection 119 · generic 77 ·
#                count 63 · superlative 28 · comparative 25 · ordinal 22 ·
#                multihop 21          (each complexity is 1/9 of every split,
#                                      so the first three are a 3x skew)
#
#   218 of 694 sit directly after an operator: "more than" 82, "over" 25,
#   "at least" 14, "than" 13, "less than" 13, "under" 6.
#       "Are there more than four different Harry Potter movies?"      -> 4
#       "Which books do not have at least 25 chapters?"                -> 25
#
# ⚠️ IT IS `cardinal`, NOT "cardinality". Cardinality is the size of a set; a
# third of these are thresholds being TESTED against, not counts of anything.
# Labelling the line "cardinality: 4" would assert the answer set has four
# members, which is false and points the model at the wrong task -- on exactly
# the aggregation questions this system is already weakest at.
#
# ⚠️ HERE THE VALUE IS CLEANER THAN THE MENTION -- the reverse of
# quantity/money/percent/time. The mention absorbs surrounding words:
#       name=1000000    mention='fewer than 1 million'
#       name=1000000000 mention='larger than one billion'
#       name=1000       mention='more then 1000'     (the typo is Mintaka's)
# The mention is still carried verbatim: it is the literal surface text, and
# trimming it to the numeral would be this module editorialising.
#
# ⚠️ IT IS NOT ALWAYS A COUNT. Eight cardinals abut an entity span and belong to
# a title ("Men in Black" + "3", "John Wick" + "4"); "No. 1" is a chart
# position and "four times" a frequency. Recorded, not handled -- 8 in 694.
#
# ⚠️ MINTAKA IS INCONSISTENT ABOUT DECADES, and no rule here can fix it:
#       'thirties' -> 30        (scalar)
#       'forties'  -> '40:49'   (colon range)
#       '20s'      -> '20:29'   (colon range)
# Ranges are `lo:hi` and there are exactly 3, all in TEST. They are carried raw
# so nothing is silently reinterpreted.

# Thousands separators to ignore when deciding whether two renderings are the
# same number. A single '.' is left alone -- it is a decimal point ('0.5' for
# "half") -- but two or more mean group separators (1.000.000).
_GROUPING = (",", " ", " ", " ")


def as_number(text) -> float | None:
    """Parse a surface string as a number, or None if it is not purely numeric.

    None is the meaningful case: it means the two renderings are genuinely
    different and both are worth showing. Word forms ("two"), phrases ("fewer
    than 1 million"), chart positions ("No. 1") and ranges ("40:49") all return
    None and therefore never collapse.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    for sep in _GROUPING:
        s = s.replace(sep, "")
    if s.count(".") > 1:            # 1.000.000 -> grouping, not a decimal point
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _same_number(name, mention: str) -> bool:
    """Do the value and the mention denote the same number?

    NUMERIC equivalence, not string equivalence, by decision (2026-08-13):
    1000000 / 1,000,000 / 1.000.000 are one value and the mention adds nothing,
    while 2 / "two" are two different renderings and both are kept. Measured
    over all 694 cardinals: 304 collapse (43.8%), 390 keep both (56.2%).
    """
    a, b = as_number(name), as_number(mention)
    return a is not None and b is not None and a == b


@dataclass(frozen=True)
class CardinalPackage:
    """One cardinal number from the question.

    `name` is Mintaka's normalised value, carried VERBATIM -- int for most,
    str for decimals ('0.5') and ranges ('40:49'). It is deliberately not
    coerced: `40:49` has no numeric form and forcing one would invent data.

    `mention` is "" when it denotes the same number as `name`, and the surface
    text otherwise. Unlike `EntityPackage` this holds a single mention rather
    than a tuple: no question in the corpus repeats a cardinal VALUE, so there
    is nothing to merge (27 questions carry two cardinals, always distinct).
    """

    name: str | int | float
    entity_type: str = CARDINAL
    mention: str = ""
    span: tuple[int, int] | None = None

    @property
    def is_range(self) -> bool:
        """`lo:hi` form. Three in the corpus, all TEST: '220:170', '40:49', '20:29'."""
        return ":" in str(self.name)

    def render_line(self) -> str:
        line = f"- ({self.entity_type}): {self.name}"
        if self.mention:
            line += f' — mentioned as "{self.mention}"'
        return line


def build_cardinal_packages(question_entities: list[dict] | None) -> list[CardinalPackage]:
    """`questionEntity[]` -> one CardinalPackage per cardinal, order preserved.

    Reads ONLY entityType == "cardinal"; every other declared type is skipped
    and an undeclared one raises, same as `build_entity_packages`.
    """
    packages: list[CardinalPackage] = []
    for mention_row in question_entities or []:
        entity_type = mention_row.get("entityType")
        if entity_type not in ALL_TYPES:
            raise UnknownEntityType(
                f"unknown questionEntity.entityType {entity_type!r}; "
                f"declared types are {ALL_TYPES}"
            )
        if entity_type != CARDINAL:
            continue
        name = mention_row.get("name")
        if name is None or str(name).strip() == "":
            continue
        mention = (mention_row.get("mention") or "").strip()
        keep = "" if (not mention or _same_number(name, mention)) else mention
        packages.append(CardinalPackage(name, CARDINAL, keep, _read_span(mention_row)))
    return packages


# ---------------------------------------------------------------------------
# `ordinal` — a position in an ordered sequence (3,308 mentions, 3,262 questions)
# ---------------------------------------------------------------------------
#
# WHAT IT IS. A positional selector, and the word AFTER it says what is being
# ordered. Measured over all 20,000 rows, it comes in two flavours:
#
#   index into a series   "the 43rd president" (245) · "the first book" (166) ·
#                         "their fourth album" (101) · "installment" (65)
#   RANK in a sort order  "second largest by area" (119) · "longest" (73) ·
#                         "tallest" (44)
#
# ⚠️ The second flavour needs a SORT, which is the same structural weakness as
# `count`: a triple store has no ordering to retrieve. Recorded here because it
# is a stratification worth making in the analysis, NOT inferred by this handler
# -- keying off the following word would be guessing at syntax.
#
# Values are heavily front-loaded: 1 accounts for 1,820 of 3,308 (55%), then
# 2 (525) and 3 (324), with a long tail to 152. Surface forms are 2,941 spelled
# out ("first", "tenth") and 366 numeric-suffix ("1st", "43rd", "121st").
#
# ⚠️ `-1` IS A SENTINEL MEANING "LAST", NOT A POSITION. Eight mentions carry it,
# and one of them is in TEST (8d00b73c, "the last year the Lakers won the
# Finals"), so it is not a train-only curiosity. It covers "last" x7 and
# "final" x1. It is rendered as the word `last` rather than the raw `-1` by
# decision (2026-08-13): printing "ordinal: -1" into a prompt invites the model
# to read a negative index, which is the one place in this type where the raw
# value is not merely opaque but wrong. This is the only interpretation this
# module performs, and it is confined to eight mentions.
#
# ⚠️ ONE MIS-ANNOTATION, recorded not handled: name=67 mention='the age of 67'
# is an age, not an ordinal. 1 in 3,308, TRAIN only.
#
# ⚠️ THE COLLAPSE NEVER FIRES FOR PLAIN ORDINALS, and that is a property of the
# data rather than a special case: "first" and "1st" both fail to parse as
# numbers, so 0 of 3,308 collapse and every plain ordinal shows both forms. The
# same rule as `cardinal` is applied unchanged; it simply resolves one way here.

# Mintaka's "last" sentinel. Stored as the string '-1' in every observed row,
# but compared numerically so an int -1 is handled too.
LAST_SENTINEL = "last"


def _ordinal_position(name) -> str:
    """The position as it should be READ: the sentinel becomes the word 'last'."""
    value = as_number(name)
    if value is not None and value == -1:
        return LAST_SENTINEL
    return str(name)


@dataclass(frozen=True)
class OrdinalPackage:
    """One ordinal position from the question.

    `name` is Mintaka's raw value, carried verbatim so the sentinel is still
    recoverable; `position` is what gets printed. `mention` is "" when it says
    the same thing as `position`, which happens only for the seven `-1`/"last"
    rows -- every other ordinal keeps both forms.
    """

    name: str | int
    entity_type: str = ORDINAL
    mention: str = ""
    span: tuple[int, int] | None = None

    @property
    def position(self) -> str:
        return _ordinal_position(self.name)

    @property
    def is_last(self) -> bool:
        return self.position == LAST_SENTINEL

    def render_line(self) -> str:
        line = f"- ({self.entity_type}): {self.position}"
        if self.mention:
            line += f' — mentioned as "{self.mention}"'
        return line


def _same_position(name, mention: str) -> bool:
    """Does the mention say the same thing as the rendered position?

    Compared against the RENDERED position, not the raw value -- the same way
    `entity` compares a mention against its label rather than its QID. So
    '-1'/"last" collapses while '-1'/"final" does not, and 7/"seventh" does not.
    Numeric equivalence is also honoured, matching `cardinal`, though no ordinal
    in the corpus reaches it.
    """
    position = _ordinal_position(name)
    if position.strip().casefold() == mention.strip().casefold():
        return True
    a, b = as_number(position), as_number(mention)
    return a is not None and b is not None and a == b


def build_ordinal_packages(question_entities: list[dict] | None) -> list[OrdinalPackage]:
    """`questionEntity[]` -> one OrdinalPackage per ordinal, order preserved.

    Nothing is merged: 46 questions carry two ordinals and they are distinct
    positions ("the first book of the second series"), so collapsing them would
    lose a constraint.
    """
    packages: list[OrdinalPackage] = []
    for mention_row in question_entities or []:
        entity_type = mention_row.get("entityType")
        if entity_type not in ALL_TYPES:
            raise UnknownEntityType(
                f"unknown questionEntity.entityType {entity_type!r}; "
                f"declared types are {ALL_TYPES}"
            )
        if entity_type != ORDINAL:
            continue
        name = mention_row.get("name")
        if name is None or str(name).strip() == "":
            continue
        mention = (mention_row.get("mention") or "").strip()
        keep = "" if (not mention or _same_position(name, mention)) else mention
        packages.append(OrdinalPackage(name, ORDINAL, keep, _read_span(mention_row)))
    return packages


# ---------------------------------------------------------------------------
# `date` — a temporal reference (2,322 mentions, 2,242 questions)
# ---------------------------------------------------------------------------
#
# ⚠️ THE TYPE NAME LIES, AND THIS IS THE EVIDENCE FOR NOT PARSING. Alongside
# real dates the bucket holds AGES and DURATIONS, and no magnitude rule can
# separate them -- 11 is an age, 814 is a year:
#
#   name=53   "Was Gerald Ford 53 when he died?"                    AGE
#   name=46   "Which president did not live past 46?"               AGE
#   name=9    "queen of England for nine days"                      DURATION
#   name=21   "21st century James Bond movie"                       CENTURY
#   name=220  "AD 220"                                              real year
#   name=814  "king of the Franks and died in 814"                  real year
#   [TEST] name=11  "didn't know he was a wizard until he was 11"   AGE
#
# Eleven such rows, one in TEST. `shape` below therefore classifies SYNTAX only
# and never claims to know what a value means.
#
# SHAPES, measured over all 20,000 rows:
#
#   year   YYYY            2,109  90.8%   2020
#   range  lo:hi             112   4.8%   '1990:1999' <- "the 90s"
#   iso    YYYY-MM-DD         58   2.5%   '1974-10-11'
#   month  YYYY-MM            19   0.8%   '2020-08'
#   short  <1000              11   0.5%   ages, durations, one century
#   recur  XXXX-MM[-DD]        8   0.3%   'XXXX-07-04'
#   bc     N BC                7   0.3%   '218 BC', '431 BC:404 BC'
#
# ⚠️ RANGES ARE 112, NOT A CURIOSITY. 109 are year:year (decades: "the 90s",
# explicit spans: "between 1959 and 1980") and 3 are ISO:ISO
# ('2003-05-15:2013-04-10'). The `lo:hi` colon is Mintaka's own convention and
# is carried RAW by decision (2026-08-14) -- rendering it as a dash would be
# this module reformatting the corpus.
#
# ⚠️ `XXXX` IS A YEAR WILDCARD, NOT A REDACTION -- "this calendar date, in any
# year". Eight mentions, and SIX of them are in DEV or TEST:
#       [dev]  'XXXX-07-04'  "the Fourth of July"
#       [dev]  'XXXX-12-25'  "Christmas Day"
#       [test] 'XXXX-07'     "July"              <- month only
# It is DECODED for rendering, on the same reasoning as the ordinal `-1`
# sentinel: printed raw, "XXXX" reads as unknown or redacted, which is the
# opposite of what it means. That is the second and last interpretation this
# module performs, and like the first it is confined to single-digit counts.
#
# ⚠️ BC VALUES ARE NON-NUMERIC STRINGS and can themselves be ranges
# ('431 BC:404 BC', '2589 BC:2566 BC'). This is why the collapse rule here must
# be STRING-or-numeric rather than numeric alone as in `cardinal`: '218 BC'
# equals its mention only as text.
#
# 🐛 ONE MINTAKA BUG, recorded not corrected: name='1980:1980' for mention
# "1980s", which should be 1980:1989. One row, TRAIN.
#
# 🟢 NO LEAK. Checked over all 2,322: the question-side date is NEVER equal to
# the gold answer. This was the highest-risk type -- a temporal question's
# answer is often a date -- and it is clean. Only 13 of these questions even
# have answerType='date'; the rest resolve to entity (1,541), boolean (524) or
# numerical (240).

_MONTHS = (
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_YEAR_WILDCARD = "XXXX"
# What a wildcard year is rendered as. Kept as a constant so the decision is
# revertible in one place: dropping the decode means returning str(name).
_ANY_YEAR = "any year"

_ISO_DAY_RE = re.compile(r"^(\d{4}|XXXX)-(\d{2})-(\d{2})$")
_ISO_MONTH_RE = re.compile(r"^(\d{4}|XXXX)-(\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_SHORT_RE = re.compile(r"^\d{1,3}$")


def _date_shape(name) -> str:
    """Classify the SYNTAX of a date value. Never infers what it means."""
    s = str(name).strip()
    if ":" in s:
        return "range"
    if _YEAR_WILDCARD in s:
        return "recurring"
    if s.upper().endswith("BC"):
        return "bc"
    if _ISO_DAY_RE.match(s):
        return "iso"
    if _ISO_MONTH_RE.match(s):
        return "month"
    if _YEAR_RE.match(s):
        return "year"
    if _SHORT_RE.match(s):
        return "short"
    return "other"


def _render_date_value(name) -> str:
    """The date as it should be READ. Raw, except that XXXX is spelled out."""
    s = str(name).strip()
    if _YEAR_WILDCARD not in s:
        return s
    day = _ISO_DAY_RE.match(s)
    if day:
        _, month, dom = day.groups()
        return f"{_MONTHS[int(month)]} {int(dom)} ({_ANY_YEAR})"
    month_only = _ISO_MONTH_RE.match(s)
    if month_only:
        return f"{_MONTHS[int(month_only.group(2))]} ({_ANY_YEAR})"
    return s


@dataclass(frozen=True)
class DatePackage:
    """One temporal reference from the question.

    `name` is Mintaka's raw value, carried verbatim; `value` is what gets
    printed and differs only for the eight `XXXX` rows. `shape` is a syntactic
    classification, offered so analyses can stratify on temporal form -- a
    measurement this project has never made -- without re-deriving it.
    """

    name: str | int
    entity_type: str = DATE
    mention: str = ""
    span: tuple[int, int] | None = None

    @property
    def value(self) -> str:
        return _render_date_value(self.name)

    @property
    def shape(self) -> str:
        return _date_shape(self.name)

    @property
    def is_range(self) -> bool:
        return self.shape == "range"

    def render_line(self) -> str:
        line = f"- ({self.entity_type}): {self.value}"
        if self.mention:
            line += f' — mentioned as "{self.mention}"'
        return line


def _same_date(name, mention: str) -> bool:
    """Does the mention say the same thing as the rendered value?

    String equality is the binding test here and numeric equality alone is not
    enough: '218 BC' matches its mention only as text. Compared against the
    RENDERED value, so a decoded XXXX never collapses against its surface form.
    """
    value = _render_date_value(name)
    if value.strip().casefold() == mention.strip().casefold():
        return True
    a, b = as_number(value), as_number(mention)
    return a is not None and b is not None and a == b


def build_date_packages(question_entities: list[dict] | None) -> list[DatePackage]:
    """`questionEntity[]` -> one DatePackage per date, order preserved.

    Nothing is merged: "Who was president of Argentina from 1989 to 1999?"
    carries two dates that are the two ends of a span, and collapsing them
    would destroy the constraint.
    """
    packages: list[DatePackage] = []
    for mention_row in question_entities or []:
        entity_type = mention_row.get("entityType")
        if entity_type not in ALL_TYPES:
            raise UnknownEntityType(
                f"unknown questionEntity.entityType {entity_type!r}; "
                f"declared types are {ALL_TYPES}"
            )
        if entity_type != DATE:
            continue
        name = mention_row.get("name")
        if name is None or str(name).strip() == "":
            continue
        mention = (mention_row.get("mention") or "").strip()
        keep = "" if (not mention or _same_date(name, mention)) else mention
        packages.append(DatePackage(name, DATE, keep, _read_span(mention_row)))
    return packages


# ---------------------------------------------------------------------------
# The MEASUREMENT family — quantity, money, percent, time
# ---------------------------------------------------------------------------
#
# These four are the one place where merging is justified by the data rather
# than by convenience: all four are A MAGNITUDE WHOSE UNIT EXISTS ONLY IN THE
# MENTION. `name` is a bare number and `label` is null on every row, so the
# surface text is not a nicety here -- it is the half that carries the meaning.
# Each type still gets its own class, docstring and tests; what they share is
# one render rule and one collapse rule, defined once below.
#
# ⚠️ THE MENTION CAN NEVER BE DROPPED FOR THESE. Measured: 0 of 60 quantities
# collapse, and the reason is visible in a single question:
#
#   "Who has six championships in the NBA and is 6 foot six inches?"
#       cardinal  name=6  mention='six'          <- championships
#       quantity  name=6  mention='6 foot'       <- height, feet
#       quantity  name=6  mention='six inches'   <- height, inches
#
# Three sixes, three meanings. Mintaka splits a compound measurement (6'6",
# 4'8", "37 foot 2 inch") into separate mentions with independent magnitudes and
# independent units, so discarding the mention turns a person's height into two
# unrelated integers.
#
# The collapse rule is applied unchanged from `cardinal`/`date` anyway. It is
# not special-cased to "never" -- the data simply never satisfies it, and
# leaving the rule uniform means a future corpus that does satisfy it behaves
# consistently rather than surprisingly.


def _render_measurement(pkg) -> str:
    """`- (type): value — mentioned as "surface"`, the shared family rendering.

    Value first, matching every other type in this module (decision
    2026-08-14). The mention does the disambiguating either way, and one
    convention across all eight types is worth more than a tighter line for the
    94 mentions this family covers.
    """
    line = f"- ({pkg.entity_type}): {pkg.name}"
    if pkg.mention:
        line += f' — mentioned as "{pkg.mention}"'
    return line


def _same_measurement(name, mention: str) -> bool:
    """String OR numeric equality, as for `date`. Never satisfied in practice."""
    a = str(name).strip()
    b = mention.strip()
    if a.casefold() == b.casefold():
        return True
    x, y = as_number(a), as_number(b)
    return x is not None and y is not None and x == y


def _build_measurements(question_entities, wanted_type, package_cls):
    """Shared builder for the measurement family. Order preserved, nothing merged."""
    packages = []
    for mention_row in question_entities or []:
        entity_type = mention_row.get("entityType")
        if entity_type not in ALL_TYPES:
            raise UnknownEntityType(
                f"unknown questionEntity.entityType {entity_type!r}; "
                f"declared types are {ALL_TYPES}"
            )
        if entity_type != wanted_type:
            continue
        name = mention_row.get("name")
        if name is None or str(name).strip() == "":
            continue
        mention = (mention_row.get("mention") or "").strip()
        keep = "" if (not mention or _same_measurement(name, mention)) else mention
        packages.append(package_cls(name, wanted_type, keep, _read_span(mention_row)))
    return packages


# ---------------------------------------------------------------------------
# `quantity` — a physical measurement (60 mentions, 54 questions)
# ---------------------------------------------------------------------------
#
# train 46 · dev 3 · test 11. The unit vocabulary is LENGTH and AREA only:
#
#   feet 9 · km 8 · square miles 6 · meters 4 · mile 3 · miles 3 ·
#   million square miles 3 · foot 2 · meter 2 · million-acre 2 ·
#   inches · sq km · square kilometers · yards · acres · inch
#
# ⭐ UNLIKE `cardinal`, THE NORMALISED VALUE EARNS ITS PLACE. About 15% carry a
# scale word Mintaka has genuinely resolved -- '1.5-million-acre' -> 1500000,
# '3 million square miles' -> 3000000, 'one thousand yards' -> 1000 -- against
# `cardinal`, where 91.8% of values were trivially readable off the mention.
#
# ROLE: a filter threshold, same as `cardinal`. Complexity is intersection 18 ·
# count 18 · difference 17 ("taller than 20,000 feet", "more than 3,000 miles
# long").
#
# 🟢 NO LEAK: 0 of 60 equal the gold answer.
#
# `name` is int on 57 rows and str on 3 ('2.2', '11.33', '0.78') -- the decimals.


@dataclass(frozen=True)
class QuantityPackage:
    """One physical measurement from the question.

    `mention` is effectively always populated: it holds the unit, without which
    `name` is a number with no meaning. See the family note above for the
    compound-measurement case that makes this non-negotiable.
    """

    name: str | int | float
    entity_type: str = QUANTITY
    mention: str = ""
    span: tuple[int, int] | None = None

    #: What kind of unit the mention carries. Declared per type so the reason
    #: the mention is load-bearing is stated in code, not only in a comment.
    unit_kind = "physical"

    def render_line(self) -> str:
        return _render_measurement(self)


def build_quantity_packages(question_entities: list[dict] | None) -> list[QuantityPackage]:
    """`questionEntity[]` -> one QuantityPackage per quantity, order preserved.

    Nothing is merged: "6 foot six inches" is two mentions of one height, and
    they must stay two lines because their units differ.
    """
    return _build_measurements(question_entities, QUANTITY, QuantityPackage)


# ---------------------------------------------------------------------------
# `money` — a monetary amount (16 mentions, 16 questions)
# ---------------------------------------------------------------------------
#
# train 10 · dev 3 · test 3. One per question, always.
#
# ⭐ THE NORMALISATION IS NON-TRIVIAL ON EVERY SINGLE ROW -- 16 of 16 -- which
# is the strongest case for the value of any type in this module. Mintaka has
# resolved both the symbol and the scale word, and the surface forms vary
# wildly for the same amount:
#
#   '$200 million'                 -> 200000000
#   '$1.4 billion'                 -> 1400000000
#   'US $170 million'              -> 170000000
#   '2 billion USD'                -> 2000000000
#   '1 billion dollars'            -> 1000000000
#   'a million dollars'            -> 1000000
#   'one hundred million dollars'  -> 100000000
#   '40 million dollar'            -> 40000000
#
# Six different ways of writing a currency amount, all flattened to an integer.
# `name` is int on all 16 rows.
#
# ⚠️ THE CURRENCY ITSELF IS ONLY IN THE MENTION, and `$`/USD is not a safe
# default: one row is '12,000 gold piece' (a Roman ransom), where 12000 counts
# gold pieces rather than any modern currency. So `unit_kind = "currency"` is
# the honest label for 15 of 16, and the mention is what actually says what is
# being counted. Same conclusion as `quantity`, reached from a different
# direction.
#
# ROLE: a threshold, like `cardinal` and `quantity` -- complexity is count 4 ·
# generic 4 · difference 3 ("grossed more than $1.4 billion", "net worth greater
# than 2 billion USD").
#
# 🟢 NO LEAK: 0 of 16 equal the gold answer.
# ⚠️ COLLAPSE NEVER FIRES: 0 of 16, since no mention is a bare number.


@dataclass(frozen=True)
class MoneyPackage:
    """One monetary amount from the question.

    `name` carries Mintaka's resolved integer, which is doing more work here
    than for any other type; `mention` carries the currency, without which the
    integer does not say what it counts.
    """

    name: str | int | float
    entity_type: str = MONEY
    mention: str = ""
    span: tuple[int, int] | None = None

    #: 15 of 16 are a real currency; one counts gold pieces. See the note above.
    unit_kind = "currency"

    def render_line(self) -> str:
        return _render_measurement(self)


def build_money_packages(question_entities: list[dict] | None) -> list[MoneyPackage]:
    """`questionEntity[]` -> one MoneyPackage per amount, order preserved."""
    return _build_measurements(question_entities, MONEY, MoneyPackage)


# ---------------------------------------------------------------------------
# `percent` — a proportion (9 mentions, 9 questions)
# ---------------------------------------------------------------------------
#
# train 6 · dev 0 · test 3.
#
# ⚠️ ZERO OCCURRENCES IN DEV. Anyone auditing this type on the development
# split alone would conclude it does not exist and prune it. It is real, and
# three of the nine are in TEST.
#
# ⭐ THIS TYPE KEEPS ITS MENTION FOR A DIFFERENT REASON THAN THE OTHER THREE,
# WHICH IS WHY IT IS DECLARED SEPARATELY. Its unit does not vary -- everything
# here is a percentage, and the type name already says so. What the mention
# uniquely carries is the HEDGE:
#
#   'approximately 90%'  -> 90        <- "approximately" survives only here
#   'about 98%'          -> 98
#   'approximately 29%'  -> 29
#   'nine per cent'      ->  9        <- and the spelled-out form
#   'two percent'        ->  2
#   '100%' / '70%' / '0.1%'
#
# Three of nine are hedged, and "approximately 29%" is not the same claim as
# "29%". Dropping the mention would silently sharpen a deliberately vague
# number -- a different failure from losing a unit, and worse, because the
# result still reads as well-formed.
#
# ROLE: intersection 6 · difference 2 · generic 1.
# 🟢 NO LEAK: 0 of 9. ⚠️ COLLAPSE NEVER FIRES: 0 of 9.
# `name` is int on 8 rows and str on one ('0.1').


@dataclass(frozen=True)
class PercentPackage:
    """One proportion from the question.

    The unit is constant, so `mention` is kept for the hedge rather than for a
    unit -- see the note above.
    """

    name: str | int | float
    entity_type: str = PERCENT
    mention: str = ""
    span: tuple[int, int] | None = None

    #: Constant across every row; the hedge is the part that varies.
    unit_kind = "proportion"

    def render_line(self) -> str:
        return _render_measurement(self)


def build_percent_packages(question_entities: list[dict] | None) -> list[PercentPackage]:
    """`questionEntity[]` -> one PercentPackage per proportion, order preserved."""
    return _build_measurements(question_entities, PERCENT, PercentPackage)


# ---------------------------------------------------------------------------
# `time` — a duration or scoping window (9 mentions, 9 questions)
# ---------------------------------------------------------------------------
#
# train 7 · dev 2 · TEST 0.
#
# ⚠️ ZERO OCCURRENCES IN TEST, so this type can never move a reported number.
# It gets a handler anyway: leaving one type out of an exhaustive registry is
# the exact hole this module was built to close, and "rare" is how the 237
# unlinked entities and 6,418 literals went unnoticed for three months.
#
# ⭐ THIS IS THE PUREST CASE OF THE FAMILY'S PROBLEM. Six of the nine rows have
# `name=1`, and that 1 means two different things:
#
#   name=1  mention='one night'   x5     <- a single awards ceremony
#   name=1  mention='an hour'     x1     <- sixty minutes
#
# Nothing in the value distinguishes them. Likewise name=24 is both '24-hour'
# (a period) and 'its first 24 hours' (a window from release). If any type
# justifies the family rule that the mention is never dropped, it is this one.
#
# ⚠️ "ONE NIGHT" IS NOT A DURATION, IT IS AN IDIOM for a single event -- "the
# most Grammy Awards won in one night". So `unit_kind = "duration"` is a loose
# label here, and the units observed are: hour, hours, minutes, night.
#
# ⭐ ITS ROLE DIFFERS FROM THE REST OF THE FAMILY. `cardinal`, `quantity`,
# `money` and `percent` are thresholds ("more than X"). `time` is mostly a
# SCOPING WINDOW for a superlative -- complexity is superlative 6 of 9, against
# comparative 1, difference 1, generic 1. "The most awards in one night" does
# not filter by a number, it bounds what counts as one contest.
#
# 🟢 NO LEAK: 0 of 9. ⚠️ COLLAPSE NEVER FIRES: 0 of 9. `name` is int on all 9.


@dataclass(frozen=True)
class TimePackage:
    """One duration or scoping window from the question.

    `mention` is the only thing separating "an hour" from "one night" -- see
    the note above. Six of nine rows share `name=1`.
    """

    name: str | int | float
    entity_type: str = TIME
    mention: str = ""
    span: tuple[int, int] | None = None

    #: Loose: the observed units are hour, hours, minutes and "night", and the
    #: last of those is an idiom for a single event rather than a duration.
    unit_kind = "duration"

    def render_line(self) -> str:
        return _render_measurement(self)


def build_time_packages(question_entities: list[dict] | None) -> list[TimePackage]:
    """`questionEntity[]` -> one TimePackage per duration, order preserved."""
    return _build_measurements(question_entities, TIME, TimePackage)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

ENTITY_BLOCK_HEADER = "Question entities:"

# Which question-entity types get a line in the prompt block. A DECLARED
# EXPERIMENTAL ARM, not a setting: the run's value lives in
# `prompts.ENTITY_BLOCK_MODE` and is recorded by `experiment_spec`, so every
# result file states which prompt produced it. See `Question.prompt_block`.
BLOCK_MODE_ALL = "all"        # linked entities + the normalised literals
BLOCK_MODE_ENTITY = "entity"  # linked entities only
BLOCK_MODES = (BLOCK_MODE_ALL, BLOCK_MODE_ENTITY)


def render_entity_block(packages: list[EntityPackage]) -> str:
    """The entity-only block, in the order given. Empty string when there is none.

    QID first, because the identifier is the primary key of the thing being
    described and the label is a gloss on it. Returning "" rather than a header
    with no rows matters: a config must never be handed an empty section that
    reads as "we looked and found nothing".
    """
    if not packages:
        return ""
    return ENTITY_BLOCK_HEADER + "\n" + "\n".join(p.render_line() for p in packages)


def build_packages(question_entities: list[dict] | None) -> list:
    """Every handled type for one question, sorted into READING ORDER.

    The single entry point a caller should use. Sorting on `span` rather than on
    Mintaka's annotation order is a decision (2026-08-13) resting on a
    measurement: 7,225 of 12,192 multi-mention questions (59.3%) are annotated
    out of order, and the block is easier to read -- and each constraint sits
    beside what it constrains -- when it tracks the sentence.

    ⚠️ Types in `PENDING_TYPES` contribute nothing yet. They are recognised, so
    they cannot raise, and they are named in that set, so the omission is
    declared rather than silent.
    """
    packages = (
        build_entity_packages(question_entities)
        + build_ordinal_packages(question_entities)
        + build_cardinal_packages(question_entities)
        + build_date_packages(question_entities)
        + build_quantity_packages(question_entities)
        + build_money_packages(question_entities)
        + build_percent_packages(question_entities)
        + build_time_packages(question_entities)
    )
    # Stable, so equal spans keep the order the builders produced.
    return sorted(packages, key=sort_key)


def render_question_block(question_entities: list[dict] | None) -> str:
    """`questionEntity[]` -> the prompt block, reading-ordered. "" when empty."""
    packages = build_packages(question_entities)
    if not packages:
        return ""
    return ENTITY_BLOCK_HEADER + "\n" + "\n".join(p.render_line() for p in packages)


def render_cardinal_line(pkg: CardinalPackage) -> str:
    """One `- (cardinal): ...` row.

    The normalised value is what is printed; when the mention is a different
    rendering it follows in the same `— mentioned as "..."` form the entity
    lines use, so a reader learns one convention rather than two.
    """
    line = f"- ({pkg.entity_type}): {pkg.name}"
    if pkg.mention:
        line += f' — mentioned as "{pkg.mention}"'
    return line


def render_ordinal_line(pkg: OrdinalPackage) -> str:
    """One `- (ordinal): ...` row, printing the position rather than the raw value."""
    line = f"- ({pkg.entity_type}): {pkg.position}"
    if pkg.mention:
        line += f' — mentioned as "{pkg.mention}"'
    return line


def packages_to_rows(packages: list[EntityPackage]) -> list[dict]:
    """JSON-serialisable form, for storing on a flattened question row."""
    return [
        {
            "name": p.name,
            "entity_type": p.entity_type,
            "label": p.label,
            "mentions": list(p.mentions),
            "span": list(p.span) if p.span else None,
            "first_mention": p.first_mention,
        }
        for p in packages
    ]


def rows_to_packages(rows: list[dict] | None) -> list[EntityPackage]:
    """Inverse of `packages_to_rows`, for reading a flattened question row."""
    return [
        EntityPackage(
            r.get("name"),
            r.get("entity_type", ENTITY),
            r.get("label"),
            tuple(r.get("mentions") or ()),
            tuple(r["span"]) if r.get("span") else None,
            r.get("first_mention"),
        )
        for r in rows or []
    ]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
#
# 🔴 THIS MODULE LOADS THE QUESTION SIDE ONLY, AND THAT IS THE POINT. The gold
# answer is produced by `parse_answers.py`, which nothing in `src/pipelines/`
# imports. A pipeline handed the output of this loader therefore CANNOT reach
# the answer key -- the separation is a property of the import graph rather than
# a convention someone has to keep.


@dataclass(frozen=True)
class Question:
    """One Mintaka question, question-side only.

    `entities` holds the parsed packages for all eight questionEntity types,
    already sorted into reading order. `category` and `complexity` are carried
    for STRATIFICATION and must not be rendered into a prompt.
    """

    id: str
    text: str
    entities: tuple = field(default=())
    category: str = ""
    complexity: str = ""

    @property
    def qids(self) -> list[str]:
        """Distinct Wikidata ids anchoring this question, in reading order.

        This is what `rag.py`, `graph_rag.py` and `graph_rag_rerank.py` receive
        as `qids`. Unlinked entity mentions contribute nothing here -- they have
        no id to fetch -- but they still appear in `entity_block`.
        """
        return [p.name for p in self.entities
                if p.entity_type == ENTITY and getattr(p, "linked", False)]

    @property
    def entity_names(self) -> list[str]:
        """The canonical Wikidata label per QID, parallel to `qids`."""
        return [p.label for p in self.entities
                if p.entity_type == ENTITY and getattr(p, "linked", False)]

    @property
    def entity_mentions(self) -> list[str]:
        """The name the QUESTION used per QID, parallel to `qids`.

        ⭐ THIS, NOT `entity_names`, IS WHAT THE PIPELINES RECEIVE (decision
        2026-08-16). They zip it with `qids` into `qid_names`, which overrides
        `Statement.subject_label` and so becomes the string in brackets on every
        rendered line: `[Academy Award] winner: ...`.

        The two differ on 122 of 301 DEV-200 anchors (40.5%) -- `mention`
        'Academy Award' against `label` 'Academy Awards'. The mention is chosen
        for the reason `wikidata_pool.build_pool` already documents: the prepend
        exists so the ranker sees the question's own phrasing echoed in the
        candidate line, and substituting a differently-worded label weakens the
        very match it was added to create.

        ⚠️ IT ALSO KEEPS EVERY RECORDED RUN COMPARABLE ON THIS AXIS. The old
        `load_questions.transform_row` used `mention or label`, and `surface`
        reproduces that rule exactly, so deleting that module changes what the
        models read in no respect at all.

        ⚠️ The cost is real and is accepted: annotator artifacts survive --
        'Academy Awards for Best Actor,' keeps its trailing comma, and '2021
        Olympic games' names an entity Wikidata calls '2020 Summer Olympics'.
        The canonical label is not lost, only moved: the entity block prints it
        as the primary name with the mention beside it.
        """
        return [p.surface for p in self.entities
                if p.entity_type == ENTITY and getattr(p, "linked", False)]

    def prompt_block(self, mode: str = BLOCK_MODE_ALL) -> str:
        """The `Question entities:` block for a prompt. "" when there is nothing.

        Reaches C2, C3 and both of C4's calls; C1 never receives it (decisions
        of 2026-08-16). Reading-ordered, like everything `build_packages`
        returns.

        `mode` selects WHICH types get a line, and is a declared experimental
        arm rather than a preference -- `prompts.ENTITY_BLOCK_MODE` holds the
        run's value and `experiment_spec` records it:

          "all"     every handled type: the linked entities plus the
                    normalised literals (date, ordinal, cardinal, quantity,
                    money, percent, time). 66 of 200 DEV questions carry at
                    least one literal line.
          "entity"  linked entities only -- the things a QID was fetched for,
                    which is exactly what the `Context:` block below is built
                    from.

        ⚠️ UNLINKED MENTIONS ARE EXCLUDED IN BOTH MODES (decision 3b). Mintaka
        marks 237 mentions as entities without linking them, so they have no
        QID, no label, and nothing was fetched for them. A line naming one
        would report OUR coverage gap rather than describe the question, and
        the effect would land on abstention -- a headline metric. They remain
        available on `self.entities` for analysis.

        ⚠️ Literals print Mintaka's normalised value, which is not always
        self-explanatory: "a 4-minute mile" becomes `(quantity): 4` and "the
        1900s" becomes `(date): 1900:1999`. The mention is printed beside it
        whenever it differs, so the value is never the only thing shown.
        """
        if mode not in BLOCK_MODES:
            raise ValueError(f"unknown block mode {mode!r}; expected one of {BLOCK_MODES}")

        packages = [p for p in self.entities if getattr(p, "linked", True)]
        if mode == BLOCK_MODE_ENTITY:
            packages = [p for p in packages if p.entity_type == ENTITY]
        if not packages:
            return ""
        return ENTITY_BLOCK_HEADER + "\n" + "\n".join(
            p.render_line() for p in packages)


class NotRawMintakaError(ValueError):
    """Raised when a questions file is not in the original Mintaka shape.

    Deliberately fatal rather than a fallback. The loader this replaces
    auto-detected raw-vs-flattened on `"entity_qids" not in row[0]`, and that
    test misfires: a hand-written question file whose records carried
    `entity_qid` (singular) was read as raw Mintaka, produced twelve rows with
    no entities and an empty gold, and raised nothing. A loud failure is the
    cheapest possible outcome.
    """


def _is_raw(row: dict) -> bool:
    """Raw Mintaka records carry the two nested structures; nothing else does."""
    return isinstance(row, dict) and "questionEntity" in row and "answer" in row


def parse_question(raw: dict) -> Question:
    """One raw Mintaka record -> its Question."""
    return Question(
        id=raw["id"],
        text=raw["question"],
        entities=tuple(build_packages(raw.get("questionEntity"))),
        category=raw.get("category", ""),
        complexity=raw.get("complexityType", ""),
    )


def load_questions(path, *, ids: set[str] | None = None) -> list[Question]:
    """Load the question side of a raw Mintaka file.

    `ids` restricts the result to those question ids, preserving file order --
    which is how a stratified SAMPLE should be expressed. A sample that stores
    only ids cannot drift from the parser, whereas the flattened sample files
    on disk froze the OLD schema at the moment they were written and would
    silently deny a run every improvement made here since.

    Raises `NotRawMintakaError` on anything that is not original Mintaka.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows:
        return []
    if not _is_raw(rows[0]):
        raise NotRawMintakaError(
            f"{path} is not raw Mintaka (a record needs 'questionEntity' and "
            f"'answer'); got keys {sorted(rows[0])[:8]}"
        )
    out = [parse_question(r) for r in rows]
    if ids is not None:
        out = [q for q in out if q.id in ids]
    return out


def load_question_ids(path) -> set[str]:
    """The question ids named by a sample file, whatever shape it is in.

    Accepts a bare list of ids, or a list of records carrying an `id` -- so an
    existing flattened sample can still say WHICH questions it selected even
    though its other fields are stale.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows:
        return set()
    if isinstance(rows[0], str):
        return set(rows)
    return {r["id"] for r in rows if isinstance(r, dict) and r.get("id")}
