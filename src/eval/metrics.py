"""
Evaluation metrics for the Graph-RAG thesis.

Dispatcher on Mintaka answer type (answer.answerType — the TRUE answer shape,
not the reasoning-complexity proxy):
  boolean              → boolean_match (primary score)
  numerical            → numeric_match (digit + spelled-out; set-membership)
  date                 → date_match    (year-aware; full-date exactness)
  entity, string       → max(token_f1, entity_match), or set F1 for multiple members

Why answer_type and not complexityType: complexity describes the REASONING
(e.g. "comparative"), which mixes answer shapes — comparative questions yield
boolean, entity AND string answers. Dispatching on complexity forced ~228
non-boolean comparative answers through the boolean matcher, and scored
numerical answers in generic/multihop/ordinal questions with token-F1 instead
of numeric equality. answer_type gives each answer its correct matcher.
Results are still grouped by complexity for the per-complexity report.

Each scorer returns: {abstained, em, f1, score}
  - score is the primary metric for the given answer type
  - abstentions score 0 everywhere and are counted separately

Multi-gold scoring (SINGLE-answer questions): predictions are matched against
EVERY acceptable surface form of the answer — the surface mention (`expected`)
plus every canonical Wikidata label (`answer_entities`) — and the best score is
kept. This corrects the ~26% of single-answer entity questions where label and
mention diverge ("Grammy" vs "Grammy Award"), which single-gold scoring
mis-penalised.

Set scoring (MULTI-answer questions): when `answer_entities` holds two or more
distinct labels, those are members of one set answer, NOT surface variants — so
best-of-forms is wrong there and the scorer uses set precision/recall/F1 with
greedy one-to-one member matching instead. Previously, naming one of three
actors scored a perfect 1.0; affects 112/4000 test questions (5 in the
100-sample). Boolean and numerical answers are never treated as sets.

Abstention detection tolerates up to two words between a negation and its
object ("not *explicitly* stated", "no *relevant* information"). The earlier
adjacent-only patterns silently scored those as attempted answers, inflating
coverage and depressing F1@attempted. This is distinct from context faithfulness.

Future work:
  - QID-structural matching: resolve predicted entity names to Wikidata QIDs and
    compare on QID identity rather than surface form. The `answer_qids` field in
    result rows is reserved for this and is not yet used by the scorer.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# The exact string the prompt instructs every config to emit when declining.
# Imported rather than duplicated so the clause and the detector cannot drift:
# if one is edited, the other follows. src.prompts is deliberately import-light
# (no openai), so this does not pull a client library into scoring.
from src.prompts import ABSTAIN_SENTINEL


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop leading articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Abstention detection
# ---------------------------------------------------------------------------

# Each pattern tolerates up to two intervening words (`_GAP`) between the
# negation and its object. Without it, only the exact adjacent phrasing matched:
# "not stated" was detected but "not *explicitly* stated" was not, and "no
# information" but not "no *relevant* information". Those near-misses were
# scored as attempted answers, which inflated coverage and depressed
# F1@attempted; context faithfulness is a separate metric.
_GAP = r"(?:\w+\s+){0,2}"

_ABSTENTION_PATTERNS = [
    rf"not {_GAP}(?:in|provided in|found in|contained in|present in) the context",
    rf"answer is not {_GAP}(?:in|available|provided|present|given)",
    rf"(?:cannot|can't|can not|could not|couldn't) {_GAP}"
    rf"(?:answer|determine|provide|find|tell|say|confirm|identify|be determined|be found)",
    rf"(?:unable|not able) to {_GAP}"
    rf"(?:answer|determine|find|provide|tell|say|identify|confirm|verify)",
    rf"not possible to {_GAP}(?:answer|determine|tell|say|know|identify)",
    rf"(?:no|not enough|not sufficient|insufficient) {_GAP}"
    rf"(?:information|context|data|details|evidence|facts)",
    # Contractions are listed explicitly. "cannot|can't|could not|couldn't"
    # above always covered both forms; this family did not, so "the context
    # doesn't provide" was scored as an ATTEMPTED answer while "does not
    # provide" was scored as an abstention. Verified 2026-08-05 to change
    # nothing already measured -- 0 flips across 2,765 stored gpt-4o-mini
    # answers, which never use contractions here -- so this is forward
    # protection for conversational models, not a re-scoring event.
    rf"(?:do not|does not|did not|don't|doesn't|didn't) {_GAP}"
    rf"(?:mention|contain|include|provide|specify|state|indicate|allow|permit)",
    rf"(?:is not|are not|was not|isn't|aren't|wasn't) {_GAP}"
    rf"(?:mentioned|stated|specified|provided|available|given|listed|included|found)",
    rf"(?:lack|lacks|lacking) {_GAP}(?:information|context|data|details)",
    rf"not {_GAP}(?:mentioned|stated|specified|provided|available|given|listed|included|found)",
    r"information provided does not",
    r"i (?:don't|do not) (?:know|have)",
]

_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


def is_abstention(answer: str | None) -> bool:
    # An empty or missing answer is NOT an abstention. The model declining to
    # answer and the provider returning nothing are different events, and only
    # the first is the calibrated-refusal behaviour the thesis measures. Guarded
    # here because llm_config.extract_text can legitimately return "" when a
    # response carried no answer text (see its docstring).
    if not answer:
        return False
    return bool(_ABSTENTION_RE.search(answer))


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 (SQuAD-standard partial credit)."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# Spelled-out numbers the model commonly uses instead of digits.
_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    # ordinals occasionally appear as answers ("the first", "second")
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6,
                "billion": 10 ** 9, "trillion": 10 ** 12}

# Scales that OPEN a magnitude group. "hundred" is deliberately excluded: it
# MULTIPLIES the pending value ("two hundred thousand" = 200,000) instead of
# opening a group, so it must not take part in the descending-scale guard below.
_GROUP_SCALES = {k: v for k, v in _WORD_SCALES.items() if v >= 1000}

# One pass, two token kinds. Punctuation and currency symbols are NOT tokens --
# they are inspected as inter-token gaps (see `_parse_numbers`), because "2
# million, 3 thousand" must split while "twenty-three" must not.
_NUMBER_TOKEN_RE = re.compile(r"(?P<num>-?\d[\d,]*(?:\.\d+)?)|(?P<word>[a-z]+)")

# Characters allowed BETWEEN two tokens without breaking a numeric phrase.
# A hyphen is included so "twenty-three" stays one number.
_PHRASE_GAP_CHARS = set(" \t\r\n-")


def _round_scaled(v: float) -> float:
    """Snap a composed value to 6 decimals.

    ⚠️ A DATASET-SCOPED ABSOLUTE TOLERANCE, NOT LOSSLESS ARITHMETIC. Composing a
    decimal mantissa with a scale ("3.967 million") is float multiplication and
    can leave a representation residue that breaks set equality against the
    integer form of the same quantity. Measured across every scale Mintaka
    reaches -- 2.873e6, 8.8e6, 1.5e9, 1.1e12, 3.7e12 -- the products are already
    exactly integer-valued in float64, so this is insurance rather than a
    correction. Decimal was considered and rejected: it changes the arithmetic
    model of the whole matcher to fix a class that does not occur in this data.
    """
    return round(v, 6)


def _parse_numbers(text: str) -> list[float]:
    """Every numeric quantity in `text`, in order, as composed values.

    ⚠️ ONE GRAMMAR, NOT TWO EXTRACTORS. This replaced a digit regex unioned with
    a separate spelled-out-word scanner (fixed 2026-08-13). That union emitted a
    phrase's COMPONENTS instead of its value: "$1 million" produced
    {1.0, 1000000.0} because the word scanner read a bare "million" as 10^6, so
    it MATCHED gold "$2 million" (they collide on the stray 10^6) while
    "2000000" FAILED to match it (gold never composed to 2,000,000). Six full-
    TEST golds carry digit-plus-scale expressions, so both directions were live.

    A phrase is consumed whole and only its composed value is emitted:
    "1.5 million" -> [1500000.0], never [1.5, 1000000.0, 1500000.0].

    Phrase boundaries, each pinned by a test:
      descending scales COMBINE      "2 million 300 thousand" -> [2300000]
      equal/rising scales SPLIT      "2 million and 3 million" -> [2e6, 3e6]
      two bare quantities SPLIT      "Three 6 Mafia" -> [3, 6]   (never 9)
      tens + unit COMBINE            "twenty three" -> [23]
      any other word breaks the run  "260 million to 325 million" -> two values
      punctuation in the gap breaks  "2 million, 3 thousand" -> two values

    ⚠️ The run also breaks on any non-number word, which is what keeps a
    SENTENCE from composing: "the population is 2 million and the area is 300
    thousand" breaks at "the" and yields both quantities separately, rather than
    silently folding them into 2,300,000. That property is load-bearing for the
    verbose configs (C1/C2 answer in prose) and is pinned by a test.
    """
    values: list[float] = []
    current = 0.0                   # bare value accumulating in this group
    result = 0.0                    # composed value across descending groups
    active = False                  # inside a numeric run?
    last_scale = float("inf")       # smallest scale applied so far in this run
    last_mag = 0.0                  # magnitude of the last bare term added

    def flush() -> None:
        nonlocal current, result, active, last_scale, last_mag
        if active:
            values.append(_round_scaled(result + current))
        current = result = 0.0
        active = False
        last_scale = float("inf")
        last_mag = 0.0

    lowered = text.lower()
    prev_end = 0
    for m in _NUMBER_TOKEN_RE.finditer(lowered):
        gap = lowered[prev_end:m.start()]
        prev_end = m.end()
        if any(ch not in _PHRASE_GAP_CHARS for ch in gap):
            flush()

        num, word = m.group("num"), m.group("word")

        if num is not None:
            try:
                v = float(num.replace(",", ""))
            except ValueError:
                flush()
                continue
            # ⚠️ A DIGIT NEVER COMBINES ADDITIVELY. Numeral grammar would allow
            # "20 3" -> 23 the way it allows "twenty three", but digits in a
            # model's prose are usually separate facts ("ranked 20, 3 times"),
            # and composing them would DESTROY both gold matches. Words are
            # written out deliberately; digits are not. So: flush first.
            if current:
                flush()
            current += v
            active, last_mag = True, 1.0
            continue

        if word in _WORD_UNITS:
            # Additive only when this term is SMALLER than the last one --
            # "twenty three" -> 23 and "one hundred and five" -> 105, but
            # "three six" -> two numbers and "nineteen eighty" -> two numbers.
            mag = 1.0 if _WORD_UNITS[word] < 10 else 10.0
            if current and mag >= last_mag:
                flush()
            current += _WORD_UNITS[word]
            active, last_mag = True, mag
        elif word in _WORD_TENS:
            if current and last_mag <= 10.0:
                flush()
            current += _WORD_TENS[word]
            active, last_mag = True, 10.0
        elif word == "hundred":
            current = (current or 1.0) * 100.0
            active, last_mag = True, 100.0
        elif word in _GROUP_SCALES:
            scale = float(_GROUP_SCALES[word])
            if scale >= last_scale:
                # Numerals descend ("two million three hundred thousand"). A
                # scale that repeats or rises therefore opens a NEW quantity:
                # "2 million and 3 million" is two values, not five million.
                # `current` holds the new mantissa, so it must NOT be emitted
                # with the value being closed.
                if result:
                    values.append(_round_scaled(result))
                result, last_scale = 0.0, float("inf")
            result += (current or 1.0) * scale
            current, last_scale = 0.0, scale
            active, last_mag = True, 0.0
        elif word == "and" and active:
            continue                # "one hundred and five" — keep the run open
        else:
            flush()
    flush()
    return values


def _extract_numbers(text: str) -> set[float]:
    """All numeric quantities in `text`, composed (see `_parse_numbers`)."""
    return set(_parse_numbers(text))


def numeric_match(pred: str, gold: str) -> float:
    """1.0 if EVERY number in this gold form appears in the prediction, else 0.0.

    Both digit and spelled-out numbers are recognised, so 'one' matches gold '1'.
    Matching against the SET of numbers in the prediction (rather than the first
    one) avoids grabbing an incidental figure — e.g. the '6' in 'Three 6 Mafia'
    or a year in the sentence — when the real answer sits elsewhere. This is a
    deliberately recall-oriented rule appropriate for short factual answers.

    ⚠️ Subset, not intersection (fixed 2026-08-18). A multi-number gold like
    5'7" extracts {5, 7}; under the old any-shared-number rule the prediction
    5'10" matched it on the stray 5. Requiring every gold number keeps all
    single-number behaviour identical (most golds, incl. digit+scale phrases,
    compose to ONE value via _parse_numbers) while closing the false positive.
    The caller takes max() over gold forms, so a stricter per-form rule can
    only be satisfied by the form the prediction actually expresses.
    """
    golds = _extract_numbers(gold)
    if not golds:
        return 0.0
    return 1.0 if golds <= _extract_numbers(pred) else 0.0


_YES = {"yes", "true", "correct", "right", "affirmative"}
_NO  = {"no", "not", "false", "incorrect", "wrong", "negative"}


def _parse_boolean(text: str) -> bool | None:
    """Yes/no surface form -> bool, or None when neither is asserted.

    Module-level so diagnostic tools can ask "does this text contain a
    boolean at all" against the SAME vocabulary the scorer matches with.
    """
    tokens = normalize(text).split()
    if not tokens:
        return None
    # first-word wins (handles "Yes, because ..." and "No, ...")
    if tokens[0] in _YES:
        return True
    if tokens[0] in _NO:
        return False
    # fallback: scan all tokens
    if any(t in _YES for t in tokens):
        return True
    if any(t in _NO for t in tokens):
        return False
    return None


def boolean_match(pred: str, gold: str) -> float:
    """Map yes/no surface forms to bool and compare."""
    p, g = _parse_boolean(pred), _parse_boolean(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p == g else 0.0


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_years(text: str) -> set[int]:
    """Every plausible 4-digit year (1000–2099) in `text`."""
    return {int(y) for y in re.findall(r"\b(1\d{3}|20\d{2})\b", text)}


def _extract_full_dates(text: str) -> set[tuple[int, int, int]]:
    """(year, month, day) triples from ISO, 'Month D, YYYY' and 'D Month YYYY'."""
    dates: set[tuple[int, int, int]] = set()
    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        dates.add((int(y), int(m), int(d)))
    low = text.lower()
    months = "|".join(_MONTHS)
    for mon, d, y in re.findall(rf"\b({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", low):
        dates.add((int(y), _MONTHS[mon], int(d)))
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\.?,?\s+(\d{{4}})\b", low):
        dates.add((int(y), _MONTHS[mon], int(d)))
    return dates


def date_match(pred: str, gold: str) -> float:
    """Date-aware match. Gold is 'YYYY' or 'YYYY-MM-DD' (Mintaka answer_value).

    Rule (year is the semantically salient unit for Mintaka 'when' questions):
      - exact full-date gold reproduced in pred            -> 1.0
      - gold year present in pred, no *conflicting* full   -> 1.0
        date committed for that year in pred
      - pred commits to a DIFFERENT full date in the gold  -> 0.0
        year (right year, wrong day/month)
      - gold year absent                                   -> 0.0
    Replaces token-F1, which scored '1970-08-02' vs a prose date on loose word
    overlap and gave near-zero credit for a correct year.
    """
    m = re.match(r"^\s*(\d{4})(?:-(\d{2})-(\d{2}))?\s*$", gold)
    if not m:
        # Non-standard gold — fall back to token overlap rather than mis-score.
        return token_f1(pred, gold)
    gold_year = int(m.group(1))
    gold_full = (gold_year, int(m.group(2)), int(m.group(3))) if m.group(2) else None

    pred_years = _extract_years(pred)
    pred_fulls = _extract_full_dates(pred)

    if gold_full and gold_full in pred_fulls:
        return 1.0
    if gold_year in pred_years:
        # If pred asserts a full date in the gold year that disagrees, it's wrong.
        if gold_full and any(fd[0] == gold_year for fd in pred_fulls) \
                and gold_full not in pred_fulls:
            return 0.0
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------
#
# token_f1 divides by the LENGTH OF THE PREDICTION, so a correct entity wrapped
# in a sentence fails: "Bono" scores 1.00, "The singer of U2 born in Ireland is
# Bono" scores 0.22 and is graded a hallucination. That penalty is not neutral
# across configurations -- it falls hardest on the verbose ones (measured mean
# answer length on entity questions: C1 10.2 tokens, C2 11.8, C3 7.8, C4 6.3),
# i.e. on exactly the baselines the thesis argues against. This is the same bug
# class already fixed for numerical and date answers; entity is the largest
# answer type and was the last to still carry it.
#
# The fix mirrors date_match: a RECALL-ORIENTED rule plus a CONFLICT GUARD.
#
#   recall  -- if a gold form appears in the prediction as a contiguous token
#              span, the answer is correct regardless of what surrounds it.
#   guard   -- containment alone over-credits, because a wrong answer can
#              mention the gold in passing: "The Wheel of Time has more books"
#              contains the gold "The Southern Vampire Mysteries" later in the
#              same sentence. The guard fires only for EITHER-OR questions,
#              detected structurally as "the gold answer is itself one of the
#              question's entities" -- true for comparatives ("Which series has
#              more books, A or B?"), false for ordinary lookups ("Where was the
#              director of Metropolis born?", gold Vienna). When it fires,
#              whichever candidate is named FIRST is taken as the asserted
#              answer, so naming the distractor first scores 0.
#
# The guard needs the question's entity labels, which live in the QUESTION file,
# not the result row -- so pass `questions_by_id` when scoring. Without it the
# guard is inactive and either-or questions are scored leniently; that is a
# documented degradation, not silent.
#
# Combined with token_f1 via max() in score_answer, so this can only ever RAISE
# a score. No previously-correct answer can become wrong.


def _token_span_index(hay: list[str], needle: list[str]) -> int:
    """First index at which `needle` occurs in `hay` as a contiguous run, else -1."""
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def entity_match(pred: str, gold: str, competitors: list[str] | None = None) -> float:
    """1.0 if `gold` is asserted in `pred`, else 0.0. See the note above.

    `competitors` are rival candidates named in the question. Supply them ONLY
    for either-or questions; a competitor appearing before the gold means the
    prediction asserted the other option.
    """
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_at = _token_span_index(pred_tokens, gold_tokens)
    if gold_at < 0:
        return 0.0

    for c in (competitors or []):
        rival = normalize(c).split()
        if not rival or rival == gold_tokens:
            continue
        rival_at = _token_span_index(pred_tokens, rival)
        if 0 <= rival_at < gold_at:
            return 0.0
    return 1.0


def rival_candidates(golds: list[str], question_entities: list[str] | None) -> list[str]:
    """Competing candidates for entity_match, or [] when the guard should not fire.

    Returns [] unless the gold is itself one of the question's entities, which is
    the structural signature of an either-or question. Gold forms are removed
    from the result so a label variant of the answer is never its own rival.
    """
    if not question_entities:
        return []
    gold_keys = {normalize(g) for g in golds if normalize(g)}
    entity_keys = {normalize(e) for e in question_entities if normalize(e)}
    if not (gold_keys & entity_keys):
        return []
    return [e for e in question_entities if normalize(e) and normalize(e) not in gold_keys]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BOOLEAN_ANSWER_TYPE = "boolean"
_NUMERIC_ANSWER_TYPE = "numerical"
_DATE_ANSWER_TYPE = "date"


# ---------------------------------------------------------------------------
# Set-answer scoring (multi-entity gold)
# ---------------------------------------------------------------------------

# A predicted item counts as the same entity as a gold member at or above this
# token-F1. 0.5 accepts partial names ("Damon" -> "Matt Damon", F1 0.67) and a
# member still embedded in carrier text ("the actors are Matt Damon", F1 0.57)
# while rejecting different entities ("Ben Affleck" vs "Matt Damon", F1 0.0).
_SET_MATCH_THRESHOLD = 0.5

# Split a free-text answer into candidate items: commas, semicolons, newlines,
# " and "/" & ", and leading list markers ("-", "*", "1.").
_SET_SPLIT_RE = re.compile(r"[,;\n]|\band\b|&", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def split_set_prediction(pred: str) -> list[str]:
    """Split a predicted answer into candidate set members."""
    items = []
    for part in _SET_SPLIT_RE.split(pred):
        part = _LIST_MARKER_RE.sub("", part).strip()
        if normalize(part):
            items.append(part)
    return items


def _member_span_pattern(member: str) -> re.Pattern:
    """Case-insensitive whole-member pattern, tolerant of whitespace runs."""
    return re.compile(r"\s+".join(re.escape(w) for w in member.split()),
                      re.IGNORECASE)


def set_match(pred: str, members: list[str]) -> dict:
    """
    Score a prediction against a MULTI-ENTITY gold answer.

    Returns precision/recall/F1 over set members, using greedy one-to-one
    matching so that repeating one member cannot satisfy several golds.

    This replaces the previous behaviour, where every member was treated as an
    alternative *complete* gold answer and the best single match was kept — so
    naming one of three actors scored a perfect 1.0. Affects 112/4000 test
    questions (5 in the 100-sample).

    Two guards added 2026-08-18:
    - A member whose own name contains a split delimiter ("Bosnia and
      Herzegovina", "Law & Order") is matched as a WHOLE SPAN against the
      prediction first and the span removed before splitting — otherwise a
      fully correct answer is shredded into fragments and can never reach
      exact 1.0 (5 TEST golds).
    - A DUPLICATE prediction item (by `normalize`) can never match a member,
      but still counts against precision. Without this, "Iron Man, Iron Man"
      scored exact 1.0 against {Iron Man, Iron Man 2}: the second copy
      fuzzy-matched the other member at the 0.5 threshold. A repeated item is
      one claim said twice — it must not fake coverage, and (per the pinned
      repetition test) it still dilutes precision like any spurious item.
    """
    if not members:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}

    # Pass 1: whole-span matches for members the splitter would break apart.
    remaining_pred = pred
    unmatched = list(range(len(members)))
    n_matched = 0
    n_span_items = 0
    n_dupes = 0
    for i, m in enumerate(members):
        if len(_SET_SPLIT_RE.split(m)) <= 1:
            continue                      # splitter leaves this member intact
        replaced, hits = _member_span_pattern(m).subn(" ; ", remaining_pred)
        if hits:
            remaining_pred = replaced
            unmatched.remove(i)
            n_matched += 1
            n_span_items += 1
            n_dupes += hits - 1           # extra copies of the span are repeats

    # Pass 2: split the remainder and greedily match, one item per member.
    items, seen = [], set()
    for item in split_set_prediction(remaining_pred):
        key = normalize(item)
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        items.append(item)

    if not items and not n_span_items and not n_dupes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}

    for item in items:
        best_i, best_score = None, _SET_MATCH_THRESHOLD
        for i in unmatched:
            sc = token_f1(item, members[i])
            if sc >= best_score:
                best_i, best_score = i, sc
        if best_i is not None:
            unmatched.remove(best_i)
            n_matched += 1

    # Every claim the model made: distinct items, span matches, and repeats.
    n_items = len(items) + n_span_items + n_dupes
    precision = n_matched / n_items
    recall = n_matched / len(members)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact = 1.0 if (precision == 1.0 and recall == 1.0) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "exact": exact}


def score_answer(
    pred: str,
    gold,
    answer_type: str = "",
    answer_entities: list[str] | None = None,
    question_entities: list[str] | None = None,
    assume_attempted: bool = False,
) -> dict:
    """
    Score one prediction against the gold answer.

    `gold` may be a single string OR a list of acceptable gold surface forms
    (e.g. the answer's canonical Wikidata label AND its surface mention, which
    differ in ~26% of single-answer entity questions). The prediction is scored
    against every form and the BEST score is kept — so a model answering in the
    canonical form is not penalised when the gold stored the mention (or vice
    versa). For boolean/numeric answers the forms collapse to one canonical value.

    `answer_entities` carries the gold's per-entity labels. TWO OR MORE distinct
    labels mean a genuine SET answer, not surface variants of one answer, and the
    prediction is scored with set precision/recall/F1 over members instead of
    best-of-forms. Without this distinction, naming one of three actors scored
    1.0. Boolean and numerical answers are never treated as sets.

    `question_entities` are the labels of the entities named in the QUESTION.
    They are used only to guard entity scoring on either-or questions (see
    entity_match); omitting them scores those questions leniently.

    Matcher is selected by `answer_type` (Mintaka answer.answerType):
      boolean   -> boolean_match
      numerical -> numeric_match
      date      -> date_match
      entity / string (and unknown) -> token_f1, raised to 1.0 when a gold form
      is asserted verbatim inside a longer answer (entity_match); or set_match
      when the gold has multiple members

    Returns:
      abstained (bool), em (float), f1 (float), score (float), set_answer (bool)
      where `score` is the primary metric for this answer type and drives
      correct/wrong classification. `f1` is token overlap for single entity/string
      answers, the typed match for Boolean/numerical/date answers, and set F1
      for multi-member answers.
    """
    golds = [gold] if isinstance(gold, str) else list(gold)
    golds = [g for g in golds if g] or [""]

    # `assume_attempted` skips this short-circuit so the text is scored on its
    # merits and the CALLER decides what it was. Required by the four-way
    # classifier: its rule "a correct answer is correct even if it hedges"
    # cannot fire if the score was already zeroed here. The C3 answer that
    # listed three Neil Breen films while noting the context "does not specify
    # 2021" is scored, not discarded, because of this flag.
    if not assume_attempted and is_abstention(pred):
        return {"abstained": True, "em": 0.0, "f1": 0.0, "score": 0.0,
                "set_answer": False}

    # Set answer: >=2 distinct members, and a type that can have members.
    members, seen = [], set()
    for m in (answer_entities or []):
        key = normalize(m)
        if key and key not in seen:
            seen.add(key)
            members.append(m)

    if len(members) >= 2 and answer_type not in (
        _BOOLEAN_ANSWER_TYPE, _NUMERIC_ANSWER_TYPE
    ):
        r = set_match(pred, members)
        return {"abstained": False, "em": r["exact"], "f1": r["f1"],
                "score": r["f1"], "set_answer": True,
                "precision": r["precision"], "recall": r["recall"]}

    em = max(exact_match(pred, g) for g in golds)

    if answer_type == _BOOLEAN_ANSWER_TYPE:
        s = max(boolean_match(pred, g) for g in golds)
        return {"abstained": False, "em": em, "f1": s, "score": s,
                "set_answer": False}

    if answer_type == _NUMERIC_ANSWER_TYPE:
        s = max(numeric_match(pred, g) for g in golds)
        return {"abstained": False, "em": em, "f1": s, "score": s,
                "set_answer": False}

    if answer_type == _DATE_ANSWER_TYPE:
        s = max(date_match(pred, g) for g in golds)
        return {"abstained": False, "em": em, "f1": s, "score": s,
                "set_answer": False}

    # entity / string. `f1` stays token overlap so the reported F1 column keeps
    # its meaning and remains comparable with earlier result files; `score` --
    # which drives correct/wrong classification and therefore the hallucination
    # rate -- additionally credits a gold form asserted inside a longer sentence.
    # max() makes this monotone: it can only raise a score, never lower one.
    f1 = max(token_f1(pred, g) for g in golds)
    rivals = rival_candidates(golds, question_entities)
    contained = max(entity_match(pred, g, rivals) for g in golds)
    return {"abstained": False, "em": em, "f1": f1, "score": max(f1, contained),
            "set_answer": False}


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

# `base_llm_abstain` is Config 1. The suffix is historical — see run_eval.CONFIGS.
# The bare `base_llm` key found in pre-2026-08-04 result files is a dropped
# no-abstention variant and is deliberately absent here, so those columns are
# ignored rather than re-scored as the current baseline.
CONFIGS = ["base_llm_abstain", "rag", "graph_rag", "rerank"]


def present_configs(results: list[dict]) -> list[str]:
    """
    Configs actually present in a result set, in CONFIGS order.

    Result files predating a config must not gain a phantom all-zero row for it.
    Retrieval sweeps likewise store only graph_rag/rerank.
    """
    seen = set()
    for row in results:
        seen |= set(row.get("answers") or {})
        seen |= set(row.get("errors") or {})
    return [c for c in CONFIGS if c in seen]


def failed_configs(row: dict) -> set[str]:
    """
    Configs whose answer for this question is an infrastructure failure.

    Two encodings are recognised:
      - `errors` dict written by run_eval.py (current)
      - an answer string starting with "ERROR:" (legacy result files, incl. the
        2026-07-25 sweep, where 9 API failures were stored this way)
    """
    failed = {c for c in (row.get("errors") or {}) if c in CONFIGS}
    for c, v in (row.get("answers") or {}).items():
        if isinstance(v, str) and v.startswith("ERROR:"):
            failed.add(c)
    return failed


# ---------------------------------------------------------------------------
# Outcome classification — the basis of the hallucination rate
#
# Every answer falls in exactly one of three buckets:
#
#   abstained  the model declined ("not in the context")
#   correct    it answered and the answer scored >= WRONG_THRESHOLD
#   wrong      it answered and the answer scored below that
#
# "wrong" is the HALLUCINATION bucket: a confident assertion that is not
# supported. It is the quantity this thesis claims KG retrieval reduces, and it
# is not recoverable from F1 alone — F1 averages abstentions (scored 0) together
# with confident errors (also 0), which is exactly the distinction that matters.
# A config can lower F1 while becoming more reliable, by converting confident
# errors into abstentions; C3 does precisely that.
#
# The threshold is a judgement call on partial credit, so report the headline
# alongside a sensitivity sweep (tools/report.py --thresholds) rather than
# quoting one number. On DEV-200 the ordering of configs is stable across
# 0.3/0.5/0.7 — say so, and the choice stops being interesting.
# ---------------------------------------------------------------------------

WRONG_THRESHOLD = 0.5

ABSTAINED = "abstained"
CORRECT = "correct"
WRONG = "wrong"


def classify(scored: dict, threshold: float = WRONG_THRESHOLD) -> str:
    """Bucket one score_answer() result as abstained / correct / wrong."""
    if scored["abstained"]:
        return ABSTAINED
    return CORRECT if scored["score"] >= threshold else WRONG


# ---------------------------------------------------------------------------
# Four-way outcome classification
# ---------------------------------------------------------------------------
#
# The three-way split above forces every answer into abstained/correct/wrong,
# which means every ambiguous case silently lands in one of them -- and each
# mistake moves the headline in a direction the thesis cares about. Measured on
# the 2026-08-06 reference run, the paraphrase regex made BOTH errors at once:
#
#   MISSES     "Not in context.", "Unknown", "None" were graded as confident
#              errors (25 of them on the 2026-08-05 run, concentrated in C3/C4)
#              -> hallucination rate INFLATED.
#   FALSE HITS a real answer carrying a hedge ("...(Note: if there are other
#              instances not mentioned in the context...)") was graded as a
#              refusal -- including one WRONG C2 answer, which therefore
#              vanished from the hallucination count -> rate DEFLATED.
#
# The fix is not a better regex. It is a fourth bucket, so the ambiguity is
# REPORTED instead of guessed at:
#
#   ABSTENTION      the model did what it was told: replied with the exact
#                   sentinel (optionally followed by its own commentary).
#                   An exact string test -- no judgement, no regex.
#   CORRECT         scored at or above threshold, whatever else it said.
#   HALLUCINATION   a clean attempt, scored below threshold. A CONFIDENT error.
#   OTHER           everything that cannot be honestly called any of the above:
#                   empty responses, truncated answers, and refusal-shaped
#                   replies that ignored the required wording ("Not specified
#                   in the provided facts", "Cannot be determined").
#
# OTHER is a reported column, not a bin to hide things in. If it is small the
# headline is unaffected; if it grows, that is itself the finding -- a model
# that will not follow the output contract cannot be scored as if it had.
#
# ORDER MATTERS and is deliberate: a correct answer is correct even if it
# hedges, so scoring is checked BEFORE the refusal-shape test. Otherwise the
# C3 answer that listed three Neil Breen films while noting the context "does
# not specify 2021" would be discarded as unclassifiable rather than graded.

HALLUCINATION = "hallucination"
ABSTENTION = "abstention"
OTHER = "other"

# The refusal-shape test reads only the OPENING of an answer, because a refusal
# announces itself first -- "Not specified in the provided facts." A hedge that
# appears later is commentary attached to a real answer, not a decline.
#
# Measured on the reference run: a WRONG C2 answer ("Inglourious Basterds",
# gold Pulp Fiction) carried the footnote "(Note: ... other instances not
# mentioned in the context ...)". Searching the whole string filed it as
# unclassifiable and quietly removed a hallucination from the count. Reading
# only the opening keeps it where it belongs, and still catches every genuine
# C4 refusal, all of which lead with the decline.
_REFUSAL_HEAD = 90

# A refusal that OPENS the answer is a decline; the same words further in are a
# caveat attached to a real answer. This constant is the boundary, and it exists
# because the CORRECT test used to run first and let refusals score as answers.
#
# THE BUG IT CLOSES (audited 2026-08-10 over 5,989 stored answers). A refusal
# names no fact, but it does contain words:
#
#   gold "No" · "The context does not provide information about their heights."
#     -> boolean_match sees "not", reads it as No, matches the gold -> CORRECT
#   gold "3"  · "The context does not provide ... ships Columbus set sail with
#                on August 3, 1492."   -> numeric_match finds the 3 -> CORRECT
#
# 54 answers were graded CORRECT this way. It inflates correctness and deflates
# abstention, in the retrieval configs specifically, because they are the ones
# that refuse. ~40 are boolean questions whose gold is "No" colliding with the
# "not" inside the refusal.
#
# WHY A POSITION TEST AND NOT A PLAIN REORDER. Simply testing refusal-shape
# before CORRECT also demotes genuine answers that carry a trailing hedge --
# measured: "**Answer:** Joe Biden **Fact (implied from context, though not
# directly stated)**" (refusal phrase at offset 60) and "All except possibly
# Ronald Reagan (insufficient data to confirm ...)" (offset 35). Both ANSWER the
# question. Requiring the refusal to LEAD keeps them CORRECT.
#
# WHY 30. Measured sensitivity, refusals-scored-CORRECT fixed vs real answers
# damaged: cut 5 -> 25/0 · 12 -> 43/0 · 20 -> 45/0 · 30 -> 46/0 · 40 -> 47/1 ·
# 90 -> 53/1. Zero damage up to 30, and the first real answer is lost at 40.
# ⚠️ 30 sits at the end of that plateau and was therefore chosen ON THIS DATA --
# say so rather than presenting it as derived. It LEAKS 6 genuine refusals whose
# decline starts later ("I'm sorry, but after reviewing the provided context, I
# couldn't find ..." begins its refusal at offset 55); those still score CORRECT
# if they happen to contain the gold. The leak is the deliberate price of not
# demoting real answers, and the residual is in the same direction as before.
_REFUSAL_LEAD = 30

OUTCOMES = (CORRECT, HALLUCINATION, ABSTENTION, OTHER)

# THE ATTEMPT CONTRACT, IN ONE PLACE. An answer is an ATTEMPT when the model
# committed to a claim about the world -- it either got it right (CORRECT) or it
# got it wrong (HALLUCINATION). An ABSTENTION declines to claim anything, and an
# OTHER reply (empty, truncated, or refusal-shaped in the model's own words) is
# not evidence about accuracy either.
#
# ⚠️ THIS LIVED IN THREE PLACES AND THEY DISAGREED (fixed 2026-08-13).
# `metrics._aggregate` used {CORRECT, HALLUCINATION}; `tools/report.py`'s
# selection-effect check used `not abstained`, which silently counts OTHER as an
# attempt; `tools/score_faithfulness.py` kept a third copy. On the
# statement-model-v1 run that put C4's ten OTHER replies inside the attempted
# subset and flipped the reported selection effect from +2.7 to -2.9 -- the
# sample, the direction and the CI were all wrong. Import this; do not re-derive
# it, and do not test `abstained` as a proxy for it.
ATTEMPTED_OUTCOMES = frozenset((CORRECT, HALLUCINATION))


def is_attempted(record: dict) -> bool:
    """True when this scored record is an attempt (see ATTEMPTED_OUTCOMES)."""
    return record.get("outcome") in ATTEMPTED_OUTCOMES


def opens_with_refusal(answer: str | None) -> bool:
    """True when the answer LEADS with refusal language in the model's own words.

    Distinct from `is_compliant_abstention`, which demands the exact instructed
    sentinel. This catches the non-compliant decline -- the C4 behaviour that
    lands in OTHER -- and does so before correctness is considered, so a refusal
    cannot borrow a number or a "not" from its own prose and score as an answer.
    """
    text = (answer or "").strip()
    if not text:
        return False
    m = _ABSTENTION_RE.search(text[:_REFUSAL_HEAD])
    return bool(m) and m.start() <= _REFUSAL_LEAD


def is_compliant_abstention(answer: str | None) -> bool:
    """
    True when the model used the exact refusal string it was instructed to use.

    Trailing commentary is allowed -- five C2 answers on the reference run gave
    the sentinel and then explained what the context DID contain, which is
    compliance plus helpfulness, not evasion. Leading commentary is not: an
    answer that reaches the sentinel only after arguing with itself has not
    followed the contract, and lands in OTHER where it can be counted.
    """
    if not answer:
        return False
    return answer.strip().startswith(ABSTAIN_SENTINEL)


def classify_outcome(
    answer: str | None,
    scored: dict,
    finish_reason: str | None = None,
    threshold: float = WRONG_THRESHOLD,
) -> str:
    """
    One of CORRECT / HALLUCINATION / ABSTENTION / OTHER.

    Order, and every step earns its place:
      empty                     -> OTHER       provider returned nothing
      exact sentinel            -> ABSTENTION  the instructed refusal
      OPENS with a refusal      -> OTHER       a decline in the model's own words
      scores >= threshold       -> CORRECT     correct despite a TRAILING hedge
      finish_reason == length   -> OTHER       cut off, so not evidence
      refusal later in the head -> OTHER       refusal-shaped, wrong wording
      otherwise                 -> HALLUCINATION

    ⚠️ The third step was added 2026-08-10 and moved AHEAD of the CORRECT test.
    Before that, a refusal could score CORRECT by borrowing its own words -- the
    "not" in "the context does not provide ..." matches a gold of "No", and a
    number quoted back from the question matches a numeric gold. See
    `_REFUSAL_LEAD` for the audit, the measured cost of the alternatives, and
    the residual leak.

    `scored` is a score_answer() result for the same answer. `finish_reason`
    comes from row["finish_reasons"][config]; pass it so truncated answers are
    not graded as fabrications -- they are unusable, which is a different claim.
    """
    text = (answer or "").strip()
    if not text:
        return OTHER                       # provider returned nothing
    if is_compliant_abstention(text):
        return ABSTENTION
    if opens_with_refusal(text):
        return OTHER                       # a decline, whatever words it borrows
    if scored.get("score", 0.0) >= threshold:
        return CORRECT                     # correct despite a TRAILING hedge
    if finish_reason == "length":
        return OTHER                       # cut off mid-answer, not a fabrication
    if _ABSTENTION_RE.search(text[:_REFUSAL_HEAD]):
        return OTHER                       # refusal-shaped, wrong wording
    return HALLUCINATION                   # a clean attempt, and it is wrong


def parsed_gold_forms(gold) -> tuple[list[str], list[str]]:
    """
    Acceptable gold surface forms from a `parse_answers.GoldAnswer`.

    ⭐ THE SUCCESSOR TO `gold_forms` BELOW, AND THE DIFFERENCE IS WHERE THE GOLD
    COMES FROM RATHER THAN WHAT IT CONTAINS. `gold_forms` assembles the forms
    from fields FROZEN INTO A RESULT ROW when that run was written, so a run
    carries whatever definition of "gold" was current on the day. This reads the
    gold from raw Mintaka by question id, so every run — including runs recorded
    a month ago — is scored against one definition that lives in one module.

    ⚠️ ONE SUBSTANTIVE DIFFERENCE, and it is worth stating precisely because the
    delta check exists to measure exactly this. `gold_forms` deliberately
    withheld Mintaka's typed payload on NUMERICAL answers (value `2000000` vs
    mention `'$2 million'`), on the reasoning that the mention is the better
    gold there. `GoldAnswer.forms` carries both. Since `score_answer` takes the
    MAX over forms, an extra form can only raise a score, never lower one — so
    any movement is wrong→right, never the reverse. Whether that movement is a
    correction or an over-match requires checking the additional gold form.
    This distinction motivated the historical migration comparison.

    Everything else is equivalent, not merely similar:
      - ORDER differs (canonical label leads here, the mention led before) and
        cannot matter: every matcher is a max over forms.
      - SET DETECTION is the same rule on both sides — two or more DISTINCT
        members — so `members` being empty for a single-label answer and
        `answer_entities` holding that one label select the same branch.
    """
    return (list(gold.forms) or [""]), list(gold.members)


def gold_forms(row: dict, questions_by_id: dict[str, dict] | None = None) -> tuple[list[str], list[str]]:
    """
    Acceptable gold surface forms for a result row, plus its answer_entities.

    LEGACY — THE PRE-2026-08-16 DEFINITION, retained as a fallback for callers
    that provide result-row metadata without parsed benchmark answers.
    Current evaluation uses `parsed_gold_forms`. Do not add a special case here; add it to
    `src/eval/parse_answers.py`, which is where an answer type's gold is defined.

    Forms are the answer mention plus every canonical Wikidata label; the scorer
    keeps the best match across them (see score_answer). `answer_entities` is
    returned separately because set-answer detection needs the members, not the
    deduplicated form list.

    DATE answers additionally get Mintaka's typed `answer_value` (ISO
    'YYYY[-MM-DD]') as an extra form. The mention is often a locale-style
    abbreviation ('13-Mar-07') which `date_match` cannot parse, so it silently
    degraded to token_f1 and scored a correct 'March 13, 2007' as 0.0 — ~22% of
    test and ~29% of dev date questions. Added rather than substituted: the
    scorer takes the max over forms, so the ISO value catches full-date
    phrasings while the mention still catches anything matching the surface text.

    Deliberately NOT applied to numerical answers, where the mention is the
    BETTER gold: value 2000000 vs mention '$2 million' — a model answering
    '2 million' matches the mention and would fail against the raw value.
    """
    qid = row.get("id", "")
    q = (questions_by_id or {}).get(qid) or {}
    answer_entities = [e for e in (row.get("answer_entities") or []) if e]
    if questions_by_id and not answer_entities:
        answer_entities = [e for e in (q.get("answer_entities") or []) if e]

    extra: list[str] = []
    if (row.get("answer_type") or q.get("answer_type", "")) == "date":
        value = row.get("answer_value", q.get("answer_value"))
        if value not in (None, ""):
            extra.append(str(value))

    forms: list[str] = []
    for g in [row.get("expected", ""), *answer_entities, *extra]:
        if g and g not in forms:
            forms.append(g)
    return (forms or [""]), answer_entities


def question_entity_labels(row: dict, questions_by_id: dict[str, dict] | None = None) -> list[str]:
    """
    Surface forms of the entities named in the question (mentions and labels).

    Needed by the either-or guard in entity scoring. Result rows written by
    run_eval.py do not carry these fields, so they normally come from the
    question file via `questions_by_id`; without it the list is empty and the
    guard stays off.
    """
    q = (questions_by_id or {}).get(row.get("id", "")) or {}
    out: list[str] = []
    for key in ("entity_names", "entity_labels"):
        for e in (row.get(key) or q.get(key) or []):
            if e and e not in out:
                out.append(e)
    return out


def excluded_question_ids(results: list[dict]) -> set[str]:
    """
    Ids to drop from EVERY config because some config hit an infrastructure
    failure there. See score_results for why this is all-or-nothing.
    """
    return {row.get("id", "") for row in results if failed_configs(row)}


def context_words(row: dict, config_id: str) -> int | None:
    """
    Words of retrieved context handed to the answering LLM, or None if unknown.

    Reads the `context_words` field written by run_eval.py, falling back to
    counting `contexts[config]["context"]` so result files written before that
    field existed still report a figure.

    Returns None — NOT 0 — when neither is present. "No measurement" and "empty
    context" are different claims, and a config with no retrieval step at all
    (C1) legitimately has neither. Printing 0 for an old file would read as
    "this config got nothing", which is a different and false statement.
    """
    recorded = (row.get("context_words") or {}).get(config_id)
    if isinstance(recorded, int):
        return recorded

    capture = (row.get("contexts") or {}).get(config_id) or {}
    ctx = capture.get("context")
    if isinstance(ctx, str):
        return len(ctx.split())
    return None


def per_question_scores(
    results: list[dict],
    questions_by_id: dict[str, dict] | None = None,
    threshold: float = WRONG_THRESHOLD,
    golds: dict | None = None,
    *,
    per_config_exclusion: bool = False,
) -> dict[str, dict[str, dict]]:
    """
    Score every (config, question) pair individually: {config: {qid: record}}.

    `golds` maps question id → `parse_answers.GoldAnswer`. When supplied it is
    the sole source of the gold forms AND of `answer_type`, so the matcher
    dispatch and the strings it dispatches on can no longer come from different
    places. Omit it to score from the result rows themselves — the legacy path,
    retained only until every caller supplies `golds`.

    ⚠️ A SUPPLIED `golds` MUST COVER EVERY QUESTION SCORED. A missing id raises
    rather than falling back to the row: silently scoring some questions under
    one gold definition and the rest under another is the "contract lived in
    three places" failure that cost two reported columns in August.

    Each record carries the scorer output plus `outcome` (abstained/correct/
    wrong), `answer_type` and `complexity`, so downstream reporting can
    stratify and run paired tests without re-scoring or re-deriving gold forms.

    Aggregates in this module are built on top of this, so a per-question view
    and the headline table can never disagree.

    ⚠️ TWO EXCLUSION SHAPES EXIST, AND WHICH ONE IS RIGHT DEPENDS ON THE
    COMPARISON (2026-08-18):
    - `per_config_exclusion=False` (default): a question any config failed on
      is absent from EVERY config. Correct for within-run reporting, where the
      four columns of a table must share one denominator.
    - `per_config_exclusion=True`: only the config that actually failed loses
      the row. Correct for cross-run pairing of ONE config (compare_runs),
      where the all-or-nothing rule silently dropped a healthy C3 row because
      an unrelated C4 call had 500'd — measured on the k-sweep, the corrected
      cohort moved C3 k30−k10 from +9.09 to +9.23 F1 (decision unchanged).
    """
    configs = present_configs(results)
    excluded = set() if per_config_exclusion else excluded_question_ids(results)

    out: dict[str, dict[str, dict]] = {c: {} for c in configs}
    for row in results:
        qid = row.get("id", "")
        if qid in excluded:
            continue
        failed = failed_configs(row) if per_config_exclusion else frozenset()

        complexity = row.get("complexity", "")
        answer_type = row.get("answer_type", "")
        if questions_by_id:
            q = questions_by_id.get(qid) or {}
            complexity = complexity or q.get("complexity", "")
            answer_type = answer_type or q.get("answer_type", "")

        if golds is not None:
            if qid not in golds:
                raise KeyError(
                    f"no gold answer for question id {qid!r}. `golds` must cover "
                    f"every scored question — scoring part of a run against one "
                    f"gold definition and the rest against another is not a "
                    f"fallback, it is two experiments in one table."
                )
            g = golds[qid]
            gold, answer_entities = parsed_gold_forms(g)
            answer_type = g.answer_type
        else:
            gold, answer_entities = gold_forms(row, questions_by_id)
        q_entities = question_entity_labels(row, questions_by_id)

        for config_id in configs:
            if config_id in failed:
                continue                  # per-config mode: this call errored
            pred = (row.get("answers") or {}).get(config_id, "")
            # Scored as an attempt, then classified: the four-way classifier
            # owns the abstained/other decision, so the scorer must not
            # pre-empt it by zeroing anything that looks like a refusal.
            s = dict(score_answer(pred, gold, answer_type, answer_entities,
                                  q_entities, assume_attempted=True))
            finish_reason = (row.get("finish_reasons") or {}).get(config_id)
            s["outcome"] = classify_outcome(pred, s, finish_reason, threshold)
            # `abstained` means EXACTLY "outcome == ABSTENTION" and nothing else.
            # It is NOT a proxy for "not attempted" -- OTHER is also not an
            # attempt, and reading this field as if it were is the bug that
            # corrupted the selection-effect check. Use is_attempted() for that.
            s["abstained"] = s["outcome"] == ABSTENTION

            # The matcher's verdict, preserved before the outcome zeroes it.
            # Without this the evidence disappears: `raw_f1 > 0` on an ABSTENTION
            # or OTHER row is exactly how we detect a refusal accidentally
            # matching its gold ("the context does not provide..." vs gold "No"),
            # which is the defect the 2026-08-10 scorer audit found 54 of.
            s["raw_score"], s["raw_em"], s["raw_f1"] = s["score"], s["em"], s["f1"]
            if s.get("set_answer"):
                s["raw_precision"] = s.get("precision", 0.0)
                s["raw_recall"] = s.get("recall", 0.0)

            # ⚠️ ZEROED FOR REPORTING, and em/f1 must be zeroed too -- not just
            # `score`. `_aggregate` averages em/f1 over ALL questions, so a
            # refusal that borrows a token from its own prose was inflating the
            # headline F1 column: C2 51.8 -> 49.8, C3 42.3 -> 40.3 on
            # statement-model-v1, i.e. biased towards the configs that refuse.
            # f1_attempted is computed on the attempted subset and does not move.
            if not is_attempted(s):
                s["score"] = s["em"] = s["f1"] = 0.0
                if s.get("set_answer"):
                    s["precision"] = s["recall"] = 0.0
            s["answer_type"] = answer_type
            s["complexity"] = complexity
            s["context_words"] = context_words(row, config_id)
            out[config_id][qid] = s
    return out


def score_results(
    results: list[dict],
    questions_by_id: dict[str, dict] | None = None,
    threshold: float = WRONG_THRESHOLD,
    golds: dict | None = None,
) -> dict:
    """
    Score a full result list and return aggregate stats.

    `questions_by_id` maps question id → question dict. Recovers `complexity`
    (grouping) and `answer_type` (matcher dispatch) when a result row pre-dates
    those fields being written by run_eval.py.

    `golds` maps question id → `parse_answers.GoldAnswer` and, when supplied,
    replaces the row-derived gold entirely — see `per_question_scores`.

    Questions where ANY config hit an infrastructure failure (see
    `failed_configs`) are excluded from EVERY config, and reported under
    "errors". An API 500 is not a prediction: grading it as a wrong answer
    inflates that config's error rate, and grading it for some configs but not
    others breaks the paired comparison.

    Returns:
      {
        "per_config": {
          config_id: {
            n, n_attempted,
            abstention_rate, coverage,
            correct_rate, hallucination_rate,   # outcome buckets, see classify()
            em, f1,                  # overall (abstentions = 0)
            em_attempted, f1_attempted,  # on non-abstaining subset only
          }
        },
        "per_complexity":  { complexity:  { config_id: {...same fields...} } },
        "per_answer_type": { answer_type: { config_id: {...same fields...} } },
      }

    Grouping by BOTH complexity and answer_type is deliberate: complexity
    describes the reasoning a question demands, answer_type the shape of its
    answer, and the failure modes separate on answer_type (C3 collapses on
    numerical, nearly holds on boolean) in a way the complexity view blurs.
    """
    scores = per_question_scores(results, questions_by_id, threshold, golds)

    # Reported alongside the table: which configs failed, and how often. The
    # questions themselves are already absent from `scores` — dropped for every
    # config, so the sets stay matched for paired comparisons.
    excluded_ids = excluded_question_ids(results)
    n_excluded_by_config: dict[str, int] = {c: 0 for c in scores}
    for row in results:
        for c in failed_configs(row):
            if c in n_excluded_by_config:
                n_excluded_by_config[c] += 1

    def _aggregate(records: list[dict]) -> dict:
        n = len(records)
        if not n:
            return {"n": 0, "n_attempted": 0, "abstention_rate": 0.0, "coverage": 0.0,
                    "correct_rate": 0.0, "hallucination_rate": 0.0, "other_rate": 0.0,
                    "em": 0.0, "f1": 0.0, "score": 0.0,
                    "em_attempted": 0.0, "f1_attempted": 0.0,
                    "correct_rate_attempted": None,
                    "hallucination_rate_attempted": None,
                    "context_words": None}
        # The four outcome rates below must PARTITION the records: today
        # `classify_outcome` can only return the four members of OUTCOMES, but
        # a fifth outcome added there without extending this aggregation would
        # leak out of every table silently — the rates would just stop summing
        # to 1, with no error anywhere. Fail here instead: this is the one
        # choke point both tools/report.py and tools/document_run.py read
        # their rates through.
        stray = {r["outcome"] for r in records} - set(OUTCOMES)
        if stray:
            raise ValueError(
                f"outcome(s) {sorted(stray)} are not in OUTCOMES — the four "
                f"reported rates would no longer sum to 100%")
        # Attempted = the model committed to an answer. OTHER is not an attempt:
        # an empty, truncated or non-compliant reply is not evidence about
        # accuracy, and averaging it into f1_attempted would understate every
        # config that produces them (in practice, C4).
        att = [r for r in records if is_attempted(r)]
        na = len(att)
        cw = [r["context_words"] for r in records if r.get("context_words") is not None]
        return {
            "n": n,
            "n_attempted": na,
            "abstention_rate": sum(r["outcome"] == ABSTENTION for r in records) / n,
            "coverage": na / n,
            "correct_rate": sum(r["outcome"] == CORRECT for r in records) / n,
            "hallucination_rate": sum(r["outcome"] == HALLUCINATION for r in records) / n,
            "other_rate": sum(r["outcome"] == OTHER for r in records) / n,
            "em": sum(r["em"] for r in records) / n,
            "f1": sum(r["f1"] for r in records) / n,
            "score": sum(r["score"] for r in records) / n,
            "em_attempted": (sum(r["em"] for r in att) / na) if na else 0.0,
            "f1_attempted": (sum(r["f1"] for r in att) / na) if na else 0.0,
            # ⭐ THE OUTCOME RATES OVER THE ATTEMPTED SUBSET (added 2026-08-17).
            # "When it answers, how often is it right / confidently wrong?" --
            # a question the all-questions rates above cannot express, because
            # abstaining lowers both of them at once.
            #
            # ⚠️ REPORTED BESIDE THE ALL-QUESTIONS RATES, NEVER INSTEAD OF THEM.
            # This denominator is gameable in exactly the direction the thesis
            # argues: a config that abstains on 199 of 200 questions and answers
            # the one easy one correctly scores 100% correct and 0%
            # hallucination here, while the all-questions rates put it at 0.5%
            # and 0.0%. Only the all-questions view makes abstention COST
            # something, and C3 already abstains on half the split.
            #
            # ⚠️ None, NOT 0.0, when nothing was attempted. "It never answered"
            # and "it answered and was never wrong" are opposite findings, and
            # 0.0 would read as the second. Same rule `faithfulness.score`
            # follows for an unmeasurable answer.
            "correct_rate_attempted":
                (sum(r["outcome"] == CORRECT for r in att) / na) if na else None,
            "hallucination_rate_attempted":
                (sum(r["outcome"] == HALLUCINATION for r in att) / na) if na else None,
            # Mean over the questions that HAVE a measurement; None if none do.
            # Averaged across all questions, not just attempted ones: the size
            # of the context is a property of retrieval, independent of whether
            # the answering LLM chose to use it.
            "context_words": (sum(cw) / len(cw)) if cw else None,
        }

    def _grouped(field: str) -> dict:
        groups = {r[field] for recs in scores.values() for r in recs.values()}
        return {
            g: {c: _aggregate([r for r in recs.values() if r[field] == g])
                for c, recs in scores.items()}
            for g in groups
        }

    return {
        "per_config": {c: _aggregate(list(recs.values())) for c, recs in scores.items()},
        "per_complexity": _grouped("complexity"),
        "per_answer_type": _grouped("answer_type"),
        "threshold": threshold,
        "errors": {
            "n_questions_excluded": len(excluded_ids),
            "excluded_ids": sorted(excluded_ids),
            "n_failures_by_config": {c: n for c, n in n_excluded_by_config.items() if n},
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_CONFIG_LABELS = {
    "base_llm_abstain": "C1 Base LLM",
    "rag":              "C2 RAG",
    "graph_rag":        "C3 Graph-RAG",
    # The result KEY stays `rerank` for backward compatibility with every stored
    # result file; the LABEL says condensing, because that is what C4 adds.
    # Both C3 and C4 rerank by embedding — the condensing LLM call is C4's only
    # distinctive step, and this label feeds tools/report.py --latex, i.e. it
    # lands in the thesis next to prose that names the config "Graph-RAG +
    # condensing" and a section titled "Ranking is not condensing".
    "rerank":           "C4 Graph+Condense",
}


def print_summary(scored: dict) -> None:
    """Print a human-readable summary of score_results() output."""
    pc = scored["per_config"]
    cpx = scored["per_complexity"]

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)

    err = scored.get("errors") or {}
    if err.get("n_questions_excluded"):
        by_cfg = ", ".join(f"{_CONFIG_LABELS.get(c, c)} {n}"
                           for c, n in (err.get("n_failures_by_config") or {}).items())
        print(f"⚠️  {err['n_questions_excluded']} question(s) EXCLUDED — infrastructure "
              f"failure, not a prediction.")
        print(f"    Failures by config: {by_cfg}")
        print(f"    Dropped for ALL configs to keep the question sets matched.")
        print("-" * 72)

    # Overall table. Halluc% sits next to Abst% on purpose: the pair is the
    # actual reliability story, and F1 alone cannot tell them apart (it averages
    # abstentions and confident errors together as zeros).
    # Other% is a COLUMN, not a footnote. The outcome buckets became four when
    # classify_outcome shipped, and this table kept three: a row could then be
    # short of 100 with the missing share nowhere on screen -- on the reference
    # DEV run C4 printed 45.5 + 42.5 + 7.5 = 95.5 beneath a legend asserting the
    # three summed to 100. The 4.5% it hid was nine refusals phrased in the
    # model's own words instead of the instructed sentinel, which is a finding
    # about instruction-following rather than a rounding detail. This function
    # prints automatically after every run and is therefore the view most likely
    # to be read; tools/report.py already showed the column.
    thr = scored.get("threshold", WRONG_THRESHOLD)
    print(f"{'Config':<18} {'N':>4} {'Cov%':>6} {'Abst%':>6} {'Corr%':>6} "
          f"{'Hall%':>6} {'Oth%':>6} {'EM':>6} {'F1':>6} {'EM@att':>8} {'F1@att':>8} {'CtxW':>8}")
    print("-" * 103)
    for c in pc:
        d = pc[c]
        cw = d.get("context_words")
        print(
            f"{_CONFIG_LABELS[c]:<18}"
            f"{d['n']:>4}"
            f"{d['coverage']*100:>6.1f}"
            f"{d['abstention_rate']*100:>6.1f}"
            f"{d.get('correct_rate', 0.0)*100:>6.1f}"
            f"{d.get('hallucination_rate', 0.0)*100:>6.1f}"
            f"{d.get('other_rate', 0.0)*100:>6.1f}"
            f"{d['em']*100:>6.1f}"
            f"{d['f1']*100:>6.1f}"
            f"{d['em_attempted']*100:>8.1f}"
            f"{d['f1_attempted']*100:>8.1f}"
            + (f"{cw:>8.0f}" if cw is not None else f"{'-':>8}")
        )
    print(f"\nHall% = answered AND scored < {thr:.2f} (confident error), as a share of all "
          f"questions.\nOth% = neither a clean answer nor the instructed refusal: empty, "
          f"truncated, or a\nrefusal in the model's own words. Abst% + Corr% + Hall% + Oth% "
          f"= 100.\nFor CIs and per-type breakdowns: tools/report.py")
    # Stated, not implied: matched top_k does not mean matched context. A
    # C2-vs-C3 comparison "at equal retrieval depth" is really comparing
    # thousands of words of prose against tens of words of statements, and that
    # belongs in the table rather than in a footnote of the write-up.
    # Labelled MEAN explicitly: tools/compare_context_sizes.py reports MEDIANS
    # over the same stored contexts, and the two differ enough on C2's skewed
    # chunk lengths (7469 vs 8806 on the k30 cell) to look like a contradiction
    # if neither is named.
    print(f"CtxW = MEAN words of retrieved context per question ('-' = not recorded; "
          f"C1 has no\nretrieval step). Equal top_k does NOT mean equal context: "
          f"compare configs on this too.")

    # Per-complexity table
    print("\nPer-complexity score (matcher by answer_type: Acc for boolean/numerical, F1 otherwise)")
    print(f"{'Complexity':<16}" + "".join(f"{_CONFIG_LABELS[c]:>16}" for c in pc))
    print("-" * 80)
    for ctype in sorted(cpx.keys()):
        row_str = f"{ctype:<16}"
        for c in pc:
            d = cpx[ctype].get(c, {})
            n = d.get("n", 0)
            score = d.get("score", 0.0)
            row_str += f"{score*100:>12.1f}%({n:2d})"
        print(row_str)

    print("=" * 72)
