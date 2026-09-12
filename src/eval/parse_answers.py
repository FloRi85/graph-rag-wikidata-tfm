"""
The gold answer Mintaka ships with each question, one handler per `answerType`.

WHY THIS MODULE EXISTS. `metrics.gold_forms()` assembles the acceptable gold
strings inline, and it grew one special case at a time -- dates got Mintaka's
typed `answer_value` bolted on after scorer fix #3, entity answers get their
labels, and `supportingEnt` / `supportingNum` are read by nothing at all. This
gives each answer type one place that says what its gold IS, so the next
special case has somewhere to go and the unused fields stop being invisible.

⚠️ THIS IS THE MIRROR OF `parse_questions.py`, AND IT USES THE SAME COLLAPSE
RULE: keep both spellings when they differ, keep one when they are identical.
What differs is only the PURPOSE, and that is worth stating because it is easy
to assume the rules must diverge. On the question side the forms are PRINTED,
so a duplicate is noise in a prompt. Here the forms are MATCHED AGAINST, so a
duplicate is a redundant comparison. Same mechanic either way.

Measured over the 11,668 single-entity answers, label and mention differ on
23.3%, and those are the rows that matter: a model answering "The Rock" is
right, and one answering "Dwayne Johnson" is right, so dropping either would
mark a correct answer wrong.

    label='Dwayne Johnson'      mention='The Rock'
    label='The Hurt Locker'     mention='Hurt Locker'
    label='Avengers: Endgame'   mention='Avenger: Endgame'   (Mintaka's typo)

🔴 EVALUATION ONLY. Nothing here may ever reach a prompt. `supportingEnt` on a
count question IS the list being counted (Angus Young, Cliff Williams, ... for
"how many current members of AC/DC"), and `mention` is the gold string itself.
There is deliberately no `render_line()` anywhere in this module -- the
question-side packages have one because they are prompt-bound, and the absence
here is the structural difference between the two halves. A test asserts that
`src/pipelines/` and `src/prompts.py` never import this module.

⚠️ IT DOES NOT SCORE ANYTHING. Building the gold and comparing a prediction
against it are separate jobs; `metrics.py` keeps the matchers. This module is
the noun, not the verb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Every `answer.answerType` in Mintaka, measured over all 20,000 raw rows
# (train / dev / test):
#
#   entity     12,524   8,777 / 1,248 / 2,499
#   numerical   3,306   2,316 /   335 /   655
#   boolean     2,867   2,010 /   284 /   573
#   date        1,275     878 /   132 /   265
#   string         28      19 /     1 /     8
ENTITY = "entity"
NUMERICAL = "numerical"
BOOLEAN = "boolean"
DATE = "date"
STRING = "string"
ALL_ANSWER_TYPES = (ENTITY, NUMERICAL, BOOLEAN, DATE, STRING)

# Answer types with no handler yet. Same device as `parse_questions`: the
# omission is declared rather than silent, and a test pins the contents.
PENDING_ANSWER_TYPES: frozenset[str] = frozenset()

# The five types fall into three groups, and the grouping is a measured property
# of the corpus rather than a convenience:
#
#   entity              carries QIDs, multilingual labels and set members
#   boolean, string     the payload and the mention, nothing else -- 0 of 2,895
#                       rows ever carry a supporting field
#   date, numerical     may carry supporting evidence: numerical on 1,889 of
#                       3,306 rows (mostly supportingEnt, the count questions),
#                       date on exactly 1 of 1,275
SIMPLE_TYPES = (BOOLEAN, STRING)
SUPPORTED_TYPES = (DATE, NUMERICAL)


class UnknownAnswerType(ValueError):
    """Raised when Mintaka carries an answerType this module does not declare."""


@dataclass(frozen=True)
class GoldAnswer:
    """What counts as the right answer to one question.

    `forms`
        Every acceptable surface spelling, deduplicated. A prediction matching
        ANY of them is right.

        ⭐ THE CANONICAL LABEL LEADS AND THE MENTION FOLLOWS (decision
        2026-08-14). Both are kept when they are different spellings, because
        each rescues answers the other loses -- a model saying "Star Trek" fails
        against mention 'Star Trek (2009)' but matches label 'Star Trek', while
        one saying "Tiger King" fails against label 'Tiger King: Murder, Mayhem
        and Madness' and matches the mention. When they are the SAME spelling
        only the label survives, and on the 306 rows where that choice is
        visible the label is the better-formed string:

            label='Gone with the Wind'  mention='Gone With The Wind'
            label='Emma Stone'          mention='Emma stone'
            label='mother!'             mention='Mother!'

        Scoring normalises case anyway, so this affects what a report DISPLAYS
        rather than what matches; it is still the right way round.

    `members`
        The individual entities when the gold is a SET of two or more. Kept
        separate from `forms` because the distinction changes the scoring rule:
        set answers score precision/recall/F1 over members instead of
        best-of-forms, so naming one of three correct films earns partial
        credit (F1 0.5) rather than the perfect 1.0 the old best-single-match
        rule gave it. ⚠️ Partial credit still meets the shared >=0.5 CORRECT
        threshold for small sets (1-of-2 = F1 0.667, 1-of-3 = 0.5) — that is
        the stated policy, not an accident: the outcome threshold is one rule
        for every answer type, and §4 must state it for sets explicitly. 561
        entity answers are sets (train 391 / dev 58 / test 112).

    `qids`
        The gold Wikidata ids. Carried but unused today -- `metrics.py` records
        QID-structural matching as future work, and this is where the input for
        it will come from. 295 entity answers have none (the payload is null),
        and for those the English mention is all that survives.

    `supporting_number`
        The quantity that makes a superlative true -- "43 years, 29 days" for
        "the youngest current US governor". Attached to 1,979 ENTITY answers.
        `None` when absent, never 0: "not annotated" and "the quantity is zero"
        are different findings, and 5 rows carry the key set explicitly to null.

    `supporting_entities`
        The things being counted, for count questions. Attached to 1,886
        NUMERICAL answers and nothing else, so it is always empty for `entity`.
    """

    answer_type: str
    payload: object = None
    mention: str = ""
    forms: tuple[str, ...] = field(default=())
    members: tuple[str, ...] = field(default=())
    qids: tuple[str, ...] = field(default=())
    supporting_number: object = None
    supporting_entities: tuple[str, ...] = field(default=())

    @property
    def is_set(self) -> bool:
        """Two or more distinct members, i.e. scored as a set rather than best-of."""
        return len(self.members) >= 2

    @property
    def has_qid(self) -> bool:
        return bool(self.qids)


def _norm(text: str) -> str:
    """Casefolded, whitespace-collapsed key for de-duplicating surface forms.

    ⚠️ Deliberately identical to `parse_questions._surface_key`. The two sides
    apply one rule and a test pins that they agree, so a change to either must
    be a change to both.
    """
    return " ".join(str(text).split()).casefold()


def _dedup(values) -> tuple[str, ...]:
    """Distinct non-empty strings, first spelling wins, order preserved."""
    out, seen = [], set()
    for v in values:
        text = str(v).strip() if v is not None else ""
        if not text:
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def _english_labels(payload) -> list[str]:
    """The `en` label of each answer entity, in payload order."""
    labels = []
    for entry in payload or []:
        if isinstance(entry, dict):
            label = entry.get("label") or {}
            english = label.get("en") if isinstance(label, dict) else label
            if english:
                labels.append(english)
    return labels


def _qids(payload) -> list[str]:
    return [e["name"] for e in (payload or [])
            if isinstance(e, dict) and e.get("name")]


def _supporting_entity_labels(answer: dict) -> tuple[str, ...]:
    return _dedup(_english_labels(answer.get("supportingEnt")))


def supporting_evidence(answer: dict):
    """Whichever supporting field this answer carries, else None.

    Lookup order (decision 2026-08-14): `supportingNum` if populated, else
    `supportingEnt` if populated, else None -- NEVER 0 and never an empty
    tuple, because "the annotator recorded nothing" and "the quantity is zero"
    must stay distinguishable.

    ⚠️ THE ORDER IS SAFE ONLY BECAUSE THE TWO NEVER CO-OCCUR. Measured over all
    20,000 rows: 1,988 carry supportingNum, 1,886 carry supportingEnt, and the
    intersection is EXACTLY ZERO. They partition by answer type -- supportingNum
    rides on entity answers (superlatives), supportingEnt on numerical answers
    (counts) -- so neither can mask the other. If a future release breaks that,
    this function silently starts preferring the number; the test that pins the
    disjointness is the guard.

    ⚠️ Returns EITHER a scalar OR a tuple of labels. Callers that need to know
    which read `supporting_number` / `supporting_entities` on the GoldAnswer
    instead; this accessor exists for the "is there any supporting evidence at
    all" question.
    """
    number = answer.get("supportingNum")
    if number is not None and str(number).strip() != "":
        return number
    entities = _supporting_entity_labels(answer)
    return entities or None


# ---------------------------------------------------------------------------
# `entity` — the answer is one or more Wikidata items (12,524 answers)
# ---------------------------------------------------------------------------
#
# SHAPES, over all 20,000 rows:
#
#   one item          11,668     Q1153188 "Mount Lucania"
#   several items        561     3 films, mention comma-joined
#   no payload at all    295     mention survives, QIDs do not
#
# ⭐ FORMS ARE ALTERNATIVES, NOT DUPLICATES. label and mention differ on 23.3%
# of single-entity answers, and collapsing them would mark correct answers
# wrong -- see the module docstring for the measured examples.
#
# ⚠️ THE SEVERAL-ITEM CASE IS NOT A LIST OF SPELLINGS. Mintaka's `mention` for
# those is the members comma-joined ("Bad Boys for Life, Sonic the Hedgehog,
# Birds of Prey"), so treating the members as alternative forms would score
# naming ONE of three as a perfect answer -- a defect the scorer already fixed
# once. `members` is therefore populated separately and `is_set` says which
# rule applies.
#
# ⚠️ 295 ANSWERS HAVE NO QID (train 209 / dev 19 / test 67). All of them still
# carry a populated mention, so they remain scoreable by surface form; what is
# lost is only the QID, which matters solely for the QID-structural matching
# that `metrics.py` lists as future work. The long-standing "295 empty gold
# answers" note overstates this -- the gold TEXT is intact.
#
# `supportingNum` rides on 1,979 of these (the superlative questions) and is
# carried through rather than dropped.


def build_entity_answer(answer: dict) -> GoldAnswer:
    """One `answer` object with answerType == 'entity' -> its GoldAnswer."""
    answer_type = answer.get("answerType")
    if answer_type not in ALL_ANSWER_TYPES:
        raise UnknownAnswerType(
            f"unknown answer.answerType {answer_type!r}; "
            f"declared types are {ALL_ANSWER_TYPES}"
        )
    payload = answer.get("answer")
    mention = (answer.get("mention") or "").strip()
    labels = _english_labels(payload)

    # Members only when the gold genuinely has several DISTINCT entities. Five
    # rows list one QID twice (Q34166 "Slash" x2); deduplicating means they are
    # correctly NOT set answers, which is also what `metrics.py` already does.
    distinct_labels = _dedup(labels)
    members = distinct_labels if len(distinct_labels) >= 2 else ()

    # Acceptable spellings, canonical labels FIRST. When label and mention are
    # the same spelling only the label survives; when they differ both are kept.
    # On a set answer the mention is the comma-joined whole, which is still a
    # legitimate way to answer, so it stays in `forms` while `members` drives
    # set scoring.
    forms = _dedup([*labels, mention])

    number = answer.get("supportingNum")
    if number is not None and str(number).strip() == "":
        number = None

    return GoldAnswer(
        answer_type=ENTITY,
        payload=payload,
        mention=mention,
        forms=forms,
        members=members,
        qids=tuple(_qids(payload)),
        supporting_number=number,
        supporting_entities=_supporting_entity_labels(answer),
    )


# ---------------------------------------------------------------------------
# `boolean` and `string` — the payload and the mention, nothing else
# ---------------------------------------------------------------------------
#
# ONE BUILDER FOR BOTH, because the corpus says they are the same shape:
# neither type EVER carries `supportingNum` or `supportingEnt` (0 of 2,867
# boolean and 0 of 28 string rows), neither has QIDs, and neither has a set
# answer. The payload is a bare scalar in a one-item list.
#
#   boolean  2,867 rows   payload list[bool]   mention 'Yes' / 'No'
#   string      28 rows   payload list[str]    (+1 row where it is a bare str)
#
# ⚠️ THE BOOLEAN PAYLOAD AND MENTION NEVER AGREE, and that is by construction
# rather than a defect: the payload is `True`/`False` and the mention is
# 'Yes'/'No', so all 2,867 rows "differ". Both are kept as acceptable forms,
# which is what lets a model answering either way be scored right.
#
# ⚠️ `string` IS TWO POPULATIONS WEARING ONE LABEL, measured over all 28:
#
#   comparative operators (13)   'Before' 'After' 'Same' 'Less' 'Both'
#   names and aliases     (15)   'Currer Bell' 'Robert Galbraith' 'Jazzy'
#
# The names behave like entity answers. The operators do NOT, and they carry a
# live scoring hazard that this module can only FLAG, not fix: `metrics.py` has
# no `string` branch, so these fall through to the entity branch, where
# `entity_match` is a token-CONTAINMENT test. Gold 'After' therefore matches any
# prediction containing the word "after" -- including "it did not come after; it
# came before", which scores 1.0 while asserting the opposite. Verified against
# the live scorer 2026-08-14.
#
# Exposure is small and worth stating precisely: 0 string questions in DEV-200,
# 0 in the TEST-100 sample, and 8 of 4,000 in TEST of which 3 are operators --
# 0.075% of the split, so it cannot move a headline. `is_comparative_operator`
# exists so a future fix in `metrics.py` has a hook rather than a hard-coded
# word list.

# The closed vocabulary of comparative answers, taken from the corpus rather
# than imagined: these are every distinct operator gold in all 20,000 rows.
COMPARATIVE_OPERATORS = frozenset({"before", "after", "same", "less", "both", "more"})


def _scalar_payload(payload):
    """The single value out of a one-item list, or the bare value as given.

    One TRAIN row (c57c3047, "Wonderboy") stores the payload as a bare string
    rather than a list. Handled explicitly rather than by accident.
    """
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload


def build_simple_answer(answer: dict) -> GoldAnswer:
    """A `boolean` or `string` answer -> its GoldAnswer.

    Both types share this builder because they share a shape, not to save
    lines: no QIDs, no set members, and no supporting evidence on any row.
    """
    answer_type = answer.get("answerType")
    if answer_type not in ALL_ANSWER_TYPES:
        raise UnknownAnswerType(
            f"unknown answer.answerType {answer_type!r}; "
            f"declared types are {ALL_ANSWER_TYPES}"
        )
    if answer_type not in SIMPLE_TYPES:
        raise UnknownAnswerType(
            f"build_simple_answer handles {SIMPLE_TYPES}, not {answer_type!r}"
        )

    payload = answer.get("answer")
    mention = (answer.get("mention") or "").strip()
    value = _scalar_payload(payload)

    # Booleans are rendered 'Yes'/'No' rather than 'True'/'False': the mention
    # is the English a model would produce, and str(True) is not it.
    if answer_type == BOOLEAN and isinstance(value, bool):
        rendered = "Yes" if value else "No"
    else:
        rendered = "" if value is None else str(value)

    forms = _dedup([rendered, mention])

    return GoldAnswer(
        answer_type=answer_type,
        payload=payload,
        mention=mention,
        forms=forms,
    )


def is_comparative_operator(gold: GoldAnswer) -> bool:
    """Is this a `string` gold from the before/after family?

    Exists so a scoring fix can target these 13 rows structurally instead of
    hard-coding a word list. See the note above for why they need one: they are
    matched by token containment today, so "it did not come after" scores 1.0
    against gold 'After'.
    """
    return gold.answer_type == STRING and any(
        _norm(f) in COMPARATIVE_OPERATORS for f in gold.forms
    )


# ---------------------------------------------------------------------------
# `date` and `numerical` — a value, plus possible supporting evidence
# ---------------------------------------------------------------------------
#
# ONE BUILDER FOR BOTH, because both may carry supporting evidence while
# `boolean` and `string` never can:
#
#   numerical  3,306 rows   supportingEnt on 1,886 (57.0%)  supportingNum on 3
#   date       1,275 rows   supportingNum on 1
#
# ⭐ `supportingEnt` ON A NUMERICAL ANSWER IS THE LIST BEING COUNTED. These are
# the count questions -- gold 5, supportingEnt ['Angus Young', 'Cliff Williams',
# 'Brian Johnson', 'Phil Rudd', 'Stevie Young'] for "how many current members of
# AC/DC". Carrying it is what makes it possible to ask whether a retrieval
# config had the members in its context and still failed to count them, which
# the project has so far only ARGUED rather than measured.
#
# ⚠️ VALUES ARE CARRIED VERBATIM, NEVER NORMALISED (decision 2026-08-14), and
# the corpus shows why. `numerical` is not reliably a number:
#
#   199 payloads are STRINGS   '5\'11"' · '6\'2"'   <- feet-and-inches heights
#     5 payloads are FLOATS    18.99 · 37.4 · 82.8 · 0.04 · 20.5
#                              20.5 is mention '20 years, 6 months'
#
# ⚠️ AND `supportingNum` IS NOT RELIABLY A NUMBER EITHER. One row carries
# 'New York, NY' -- a place name in a field named "Num" -- and two duplicate the
# answer itself (gold 151, supportingNum 151). The date row carries
# '7,052,770 votes.'. Nothing here may assume a numeric type.


def build_supported_answer(answer: dict) -> GoldAnswer:
    """A `date` or `numerical` answer -> its GoldAnswer, supporting evidence kept.

    The supporting lookup is `supportingNum` if populated, else `supportingEnt`
    if populated, else nothing -- and the two are stored on separate fields so a
    caller can tell which it got. `supporting_evidence()` implements the same
    order for callers that only need "is there any".
    """
    answer_type = answer.get("answerType")
    if answer_type not in ALL_ANSWER_TYPES:
        raise UnknownAnswerType(
            f"unknown answer.answerType {answer_type!r}; "
            f"declared types are {ALL_ANSWER_TYPES}"
        )
    if answer_type not in SUPPORTED_TYPES:
        raise UnknownAnswerType(
            f"build_supported_answer handles {SUPPORTED_TYPES}, not {answer_type!r}"
        )

    payload = answer.get("answer")
    mention = (answer.get("mention") or "").strip()
    value = _scalar_payload(payload)
    rendered = "" if value is None else str(value)

    # The typed value leads, the mention follows when it is a different string.
    # This is the pairing scorer fix #3 had to bolt on for dates: the mention is
    # often an unparseable abbreviation ('18-Dec-46') while the payload is ISO
    # ('1946-12-18'), and 286 of 1,275 date rows differ this way.
    forms = _dedup([rendered, mention])

    number = answer.get("supportingNum")
    if number is not None and str(number).strip() == "":
        number = None

    return GoldAnswer(
        answer_type=answer_type,
        payload=payload,
        mention=mention,
        forms=forms,
        supporting_number=number,
        supporting_entities=_supporting_entity_labels(answer),
    )


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------

def build_gold_answer(answer: dict) -> GoldAnswer:
    """Any Mintaka `answer` object -> its GoldAnswer, whatever the type.

    The one function a caller should use. An undeclared answerType raises
    rather than falling through to a default, for the same reason the question
    side does: a silently skipped type is how six thousand mentions went
    unnoticed for three months.
    """
    answer_type = answer.get("answerType")
    if answer_type == ENTITY:
        return build_entity_answer(answer)
    if answer_type in SIMPLE_TYPES:
        return build_simple_answer(answer)
    if answer_type in SUPPORTED_TYPES:
        return build_supported_answer(answer)
    raise UnknownAnswerType(
        f"unknown answer.answerType {answer_type!r}; "
        f"declared types are {ALL_ANSWER_TYPES}"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
#
# 🔴 SEPARATE FROM `parse_questions.load_questions` ON PURPOSE. Scoring imports
# this module; `src/pipelines/` imports the other one. Because the gold answer
# is only ever produced here, a pipeline cannot obtain it -- not by convention,
# but because it never imports the code that builds it. A test asserts that no
# file under `src/pipelines/` and not `src/prompts.py` names this module.
#
# The two loaders read the same file, which costs one extra JSON parse per run.
# At 20,000 rows that is a few hundred milliseconds against hours of API calls.

# The answer-type vocabulary, qualified for docs and CLI use. See
# `parse_questions.raw_type` for the convention and why the axes need
# disambiguating -- `entity` and `date` each name two different things.
ANSWER_TYPES = ("a_entity", "a_numerical", "a_boolean", "a_date", "a_string")


def load_answers(path) -> dict[str, GoldAnswer]:
    """Load the gold answers of a raw Mintaka file, keyed by question id.

    A dict rather than a list because there is only one sane use: looking up
    the gold for a prediction. `parse_questions.load_questions` returns a LIST
    because evaluation order matters there.

    Raises `NotRawMintakaError` (imported from the question side, so there is
    one definition of "this file is not Mintaka") on anything else.
    """
    from src.eval.parse_questions import NotRawMintakaError, _is_raw

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows:
        return {}
    if not _is_raw(rows[0]):
        raise NotRawMintakaError(
            f"{path} is not raw Mintaka (a record needs 'questionEntity' and "
            f"'answer'); got keys {sorted(rows[0])[:8]}"
        )
    return {r["id"]: build_gold_answer(r["answer"]) for r in rows}
