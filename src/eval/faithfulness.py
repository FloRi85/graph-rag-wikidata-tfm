"""
Faithfulness: is an answer supported by the context that was actually retrieved?

The hallucination classifier compares answers with Mintaka gold answers without
using context. This metric separately estimates whether an answer's claims are
supported by its supplied context. Support does not establish correctness, and
lack of support does not identify where the model obtained the answer.

The metric is the faithfulness score of RAGAS (Es et al., 2023): decompose the
answer into atomic claims, check each for entailment against the context, and
report the supported fraction.

The claim-decomposition and entailment procedure is implemented directly
through `llm_config.complete()`, without importing the RAGAS package or LangChain.

Three properties govern how the output is read
----------------------------------------------
1. **Undefined for Configuration 1.** It retrieves nothing, so there is no
   context to be faithful to. `RETRIEVAL_CONFIGS` excludes it by construction
   rather than reporting a misleading zero.
2. **Not a correctness score.** An answer fully supported by an incorrect context
   can score 1.0 while disagreeing with the gold answer. Report faithfulness
   beside correctness, never instead of it.
3. **Short answers degenerate to a single claim.** The shipped prompt instructs
   brevity and median answers run 4-7 words, so decomposition usually yields one
   claim and the score becomes near-binary. That is a property of the answers,
   not a defect here, but it means the metric carries less information on this
   task than on the long-form generation RAGAS was designed for. Say so.

The retained judge validation uses a blind, stratified DEV sample with manually
annotated claim support. `tools/emit_judge_validation.py` constructs the sample
and reports agreement; `tools/review_judge_validation.py` records annotations.
"""

from __future__ import annotations

import re
import time

from src import llm_config, token_counter

# Transient-failure retry, mirroring run_eval's. DELIBERATELY DUPLICATED rather
# than imported: `run_eval` pulls in all four pipelines and the retrieval stack,
# and a scorer that reads a finished run file has no business importing those.
# Hoisting it into `llm_config.complete()` would be the tidier home -- the same
# argument that put the rate limiter there -- but that would change the retry
# behaviour of the four experimental pipelines, which are frozen. Keep the two
# in step by hand; they are eight lines each.
_TRANSIENT_MARKERS = (
    "500", "502", "503", "504", "429",
    "timeout", "timed out", "too many concurrent", "rate limit",
    "overloaded", "connection", "temporarily unavailable", "bad gateway",
)
MAX_ATTEMPTS = 4
BACKOFF_BASE = 2.0   # seconds: 2, 4, 8

# Configuration 1 has no retrieval step. Faithfulness is undefined for it, and
# emitting 0.0 would read as "unfaithful" rather than "not applicable".
RETRIEVAL_CONFIGS = ("rag", "graph_rag", "rerank")

# Evaluation-side budget, deliberately independent of llm_config.MAX_TOKENS.
# That cap is 128 because Mintaka ANSWERS are short, and it is held identical
# across the four configs so it cannot become a confound. Neither consideration
# applies to a scoring call that must emit a claim list, and reusing 128 here
# would truncate the judgement rather than the answer -- the same defect the
# condensing call shipped with for weeks.
DECOMPOSE_MAX_TOKENS = 384
VERIFY_MAX_TOKENS = 384

# Verdict tokens the verifier is instructed to emit.
_SUPPORTED = "SUPPORTED"
_UNSUPPORTED = "UNSUPPORTED"

# Shared by verdict parsing and the out-of-range-index diagnostic.
_VERDICT_LINE_RE = re.compile(
    rf"^\s*(\d+)\s*[.)]?\s*.*?\b({_SUPPORTED}|{_UNSUPPORTED})\b",
    re.IGNORECASE)

# Stamped into every serialized faithfulness record, on every path (blank
# no_context/no_answer/no_claims records and error records included). Version 1
# is every record written before the stamp existed (the DEV and TEST files of
# 2026-08). Bump on any change to the prompts, the parser, or these fields.
INSTRUMENT_VERSION = 2


def out_of_range_indices(raw: str | None, n: int) -> list[int]:
    """
    DISTINCT verdict-line indices outside 1..n in one raw judge text, sorted.

    Contract diagnostic, not a parser change: `_parse_verdicts` silently
    ignores these indices (they never enter the verdict list), so an
    out-of-range line is evidence of a degenerate judge generation, not of a
    wrong verdict. Counting is per distinct index, not per line.
    """
    if not raw:
        return []
    found: set[int] = set()
    for line in raw.splitlines():
        m = _VERDICT_LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            if not 1 <= idx <= n:
                found.add(idx)
    return sorted(found)

_DECOMPOSE_PROMPT = """\
Break the following answer into its atomic factual claims.

Rules:
- One claim per line, each beginning with "- ".
- Each claim must stand alone and be checkable on its own.
- Use only what the answer states. Do not add, infer or correct anything.
- A short answer is usually a single claim. That is fine; do not invent more.
- Write the claim as a full statement, resolving what the question refers to.

Question: {question}
Answer: {answer}

Claims:"""

_VERIFY_PROMPT = """\
Decide whether each claim is supported by the context below.

Rules:
- A claim is {supported} only if the context states it or directly entails it.
- If the context is silent on the claim, it is {unsupported}, even if you
  believe the claim is true. You are judging support by the context, not truth.
- Answer for every claim, in order, one per line, as "<number>. {supported}" or
  "<number>. {unsupported}". Output nothing else.

Context:
{context}

Claims:
{claims}

Verdicts:"""

# 🔴 REFORMAT ONLY — IT MUST NOT JUDGE. The reply is given back verbatim and the
# model is asked to restate what is already in it. Asking it to rule on a claim
# it skipped would produce a SECOND measurement and silently substitute it for a
# missing first one, which is not a recovery: the whole point of recording
# `n_unparsed` is that a verdict the judge never gave stays absent.
_REPAIR_PROMPT = """\
Below is a reply that was supposed to list one verdict per claim, numbered 1 to
{n}, each either {supported} or {unsupported}. It was not formatted correctly.

Rewrite it in the required format. Do NOT make any new judgements: only restate
verdicts that are already present in the reply. If the reply gives no verdict
for a number, omit that number entirely rather than guessing.

Output one line per verdict found, as "<number>. {supported}" or
"<number>. {unsupported}". Output nothing else.

Reply:
{reply}

Reformatted:"""


def _is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


def _complete_with_retry(messages: list[dict], max_tokens: int):
    """
    One paced call, retrying transient provider failures.

    The limiter reduces 429s but does not eliminate them: it paces to a fixed
    rate, and the provider's own accounting is bursty. Scoring calls are short
    and therefore arrive faster than the eval's, so they press the cap harder —
    a full run died on a single unretried 429 after several hundred successful
    calls, which is the same "lose completed work to one transient" failure the
    eval's partial-write sidecar exists to prevent.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return llm_config.complete(
                model=llm_config.MODEL,
                temperature=llm_config.TEMPERATURE,
                max_tokens=max_tokens,
                messages=messages,
            )
        except Exception as e:                                    # noqa: BLE001
            last = e
            if not _is_transient(e) or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_BASE ** attempt)
    raise last                                                    # unreachable


def _complete(user_body: str, max_tokens: int) -> str:
    """
    One scoring call.

    Routed through `llm_config.complete()` like every other call in the project,
    so request pacing, token accounting and the provider seam all apply — a
    scorer that drove its own client would burst straight through the rate limit
    on a 650-call run.

    The system slot borrows the CONDENSE role's framing rather than the ANSWER
    role's. On Nemotron that supplies the reasoning-mode switch (without which
    a `<think>` block eats the token budget and the reply comes back empty) but
    not the brevity line, which would sabotage a call that must emit a list.
    """
    system = llm_config.prompt_style(role="condense")["system"]
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_body})

    response = _complete_with_retry(messages, max_tokens)
    # Scoring calls are billed like any other. `record` is None-safe, so an
    # endpoint that omits `usage` leaves the counter at zero rather than
    # erroring -- which means forgetting this line reports a free run instead of
    # failing. It was forgotten once here already, caught only because the smoke
    # output read `calls=0` after twelve live calls.
    token_counter.record(getattr(response, "usage", None))
    return llm_config.extract_text(response)


def _strip_emphasis(text: str) -> str:
    """
    Drop markdown emphasis the model wraps around a claim.

    Cosmetic in the score but not in the record: the claims are quoted in the
    write-up and read back during validation, and `**There are 7 continents.**`
    is the condenser's markdown scaffolding leaking into evidence.
    """
    return re.sub(r"(\*\*|__|\*|_)", "", text).strip()


def decompose(question: str, answer: str) -> list[str]:
    """Atomic claims stated by `answer`. Empty list if it states none."""
    if not (answer or "").strip():
        return []
    raw = _complete(
        _DECOMPOSE_PROMPT.format(question=question, answer=answer),
        DECOMPOSE_MAX_TOKENS,
    )
    claims = []
    for line in raw.splitlines():
        line = line.strip()
        # Accept "- x", "* x", "1. x" -- models drift between them regardless
        # of instruction, and a strict parser would silently score 0 claims.
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if m and m.group(1).strip():
            claims.append(_strip_emphasis(m.group(1)))
    # A model that ignored the format but returned one sensible line is more
    # useful than an empty list, which would make the question unscoreable.
    if not claims and raw.strip() and len(raw.strip().splitlines()) == 1:
        claims = [_strip_emphasis(raw)]
    return claims


def _parse_verdicts(raw: str, n: int) -> list[bool | None]:
    """Verdicts 1..n parsed from the judge's reply. `None` = not ruled on.

    ⚠️ A CONFLICTING DUPLICATE IS `None`, NOT A LAST-WINS GUESS. If the judge
    emits index 2 twice with different verdicts it has not ruled on claim 2; a
    dict assignment silently kept whichever came last and presented it as a
    reading.
    """
    seen: dict[int, bool] = {}
    conflicted: set[int] = set()
    for line in raw.splitlines():
        m = _VERDICT_LINE_RE.match(line.strip())
        if not m:
            continue
        idx, val = int(m.group(1)), m.group(2).upper() == _SUPPORTED
        if idx in seen and seen[idx] != val:
            conflicted.add(idx)
        seen[idx] = val
    return [None if i in conflicted else seen.get(i)
            for i in range(1, n + 1)]


def verify(claims: list[str], context: str) -> dict:
    """
    One entailment verdict per claim, in order, plus what could not be read.

    Returns `{verdicts, n_parsed, n_unparsed, parse_coverage, repair_used,
    raw, raw_repair}` where a verdict is True / False / **None**.

    🔴 `None` IS NOT `False`, AND THIS USED TO BE WRONG. An unreadable verdict
    was coerced to False and folded into the score, so an answer the judge
    failed to rule on came back as a confident, valid-looking 0.0 with
    `status: "ok"` -- "we could not measure this" presented as "the model made
    it up". That is the exact conflation the None-not-0.0 discipline everywhere
    else in this module exists to end; it simply had not been applied here.

    One repair attempt is made, and it is FORMAT-ONLY: the judge is handed its
    own previous reply and asked to reformat it. It is never asked to judge
    again, because a second opinion is a different measurement, not a recovery
    of the first. A verdict genuinely absent from the original reply stays None.
    """
    if not claims:
        return {"verdicts": [], "n_parsed": 0, "n_unparsed": 0,
                "parse_coverage": 1.0, "repair_used": False,
                "raw": "", "raw_repair": None,
                "n_out_of_range_initial": 0, "n_out_of_range_repair": None}

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    raw = _complete(
        _VERIFY_PROMPT.format(context=context, claims=numbered,
                              supported=_SUPPORTED, unsupported=_UNSUPPORTED),
        VERIFY_MAX_TOKENS,
    )
    verdicts = _parse_verdicts(raw, len(claims))

    raw_repair = None
    repair_used = False
    if any(v is None for v in verdicts):
        repair_used = True
        raw_repair = _complete(
            _REPAIR_PROMPT.format(reply=raw, n=len(claims),
                                  supported=_SUPPORTED, unsupported=_UNSUPPORTED),
            VERIFY_MAX_TOKENS,
        )
        repaired = _parse_verdicts(raw_repair, len(claims))
        # Only FILL GAPS. A repair that contradicts a verdict already read is
        # re-judging, which this call is explicitly not allowed to do.
        verdicts = [old if old is not None else new
                    for old, new in zip(verdicts, repaired)]

    n_parsed = sum(v is not None for v in verdicts)
    return {
        "verdicts": verdicts,
        "n_parsed": n_parsed,
        "n_unparsed": len(claims) - n_parsed,
        "parse_coverage": n_parsed / len(claims),
        "repair_used": repair_used,
        "raw": raw,
        "raw_repair": raw_repair,
        # Contract instrumentation, not a parse change: out-of-range indices
        # never enter `verdicts` (see out_of_range_indices), but a generation
        # that emits them is degenerate and future runs should carry the count
        # rather than leave it to offline archaeology. `None` = no repair call
        # was made, which is a different fact from "repair emitted none".
        "n_out_of_range_initial": len(out_of_range_indices(raw, len(claims))),
        "n_out_of_range_repair": (len(out_of_range_indices(raw_repair, len(claims)))
                                  if repair_used else None),
    }


def score(question: str, answer: str, context: str) -> dict:
    """
    Faithfulness of one answer against one context.

    Returns {score, n_claims, n_supported, claims, verdicts, status}. `score` is
    None whenever no score is meaningful, never 0.0 — a missing context and a
    fully unsupported answer are different findings, and collapsing them would
    put "we could not measure this" into the same bucket as "the model made it
    up", which is the exact conflation this module exists to end.
    """
    blank = {"score": None, "n_claims": 0, "n_supported": 0,
             "claims": [], "verdicts": [],
             # Stamped on EVERY path: a record without the stamp is version 1
             # by definition, and letting blank records fall through unstamped
             # would make the version unreadable exactly where parsing failed.
             "instrument_version": INSTRUMENT_VERSION,
             "n_out_of_range_initial": None, "n_out_of_range_repair": None}
    if not (context or "").strip():
        return {**blank, "status": "no_context"}
    if not (answer or "").strip():
        return {**blank, "status": "no_answer"}

    claims = decompose(question, answer)
    if not claims:
        return {**blank, "status": "no_claims"}

    v = verify(claims, context)
    verdicts = v["verdicts"]
    # `v is True`, explicitly: None is not falsy-by-accident here, it is a
    # different finding, and `sum()` over a list containing None would raise.
    supported = sum(1 for x in verdicts if x is True)

    common = {
        "n_claims": len(claims),
        "n_supported": supported,
        "n_parsed": v["n_parsed"],
        "n_unparsed": v["n_unparsed"],
        "parse_coverage": v["parse_coverage"],
        "repair_used": v["repair_used"],
        "claims": claims,
        "verdicts": verdicts,
        "raw_verification": v["raw"],
        "raw_repair": v["raw_repair"],
        "instrument_version": INSTRUMENT_VERSION,
        "n_out_of_range_initial": v["n_out_of_range_initial"],
        "n_out_of_range_repair": v["n_out_of_range_repair"],
    }

    if v["n_unparsed"]:
        # ⚠️ NO PARTIAL SCORE IN THE HEADLINE. Scoring the parsed subset changes
        # the denominator per answer, so answers the judge found hard would
        # quietly contribute on different terms from the rest -- denominator
        # bias, in the direction of whichever answers were easiest to read.
        # `partial_score` is kept for diagnosis and is never aggregated.
        return {**common, "score": None, "status": "unparsed_verdicts",
                "partial_score": (supported / v["n_parsed"]) if v["n_parsed"] else None}

    return {**common, "score": supported / len(claims), "status": "ok"}
