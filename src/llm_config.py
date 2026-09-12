"""Central LLM config + client factory — the provider-swap surface.

One source of truth for the answering/condensing LLM across all four
pipelines — they never touch the client or model name directly.
OpenAI-compatible endpoints (NVIDIA Build, NIM, vLLM) swap via .env alone:

    LLM_MODEL     (default: gpt-4o-mini)
    LLM_BASE_URL  e.g. "https://integrate.api.nvidia.com/v1"
    LLM_API_KEY   provider key

Unset base URL/key = OpenAI default endpoint + OPENAI_API_KEY.
"""

from __future__ import annotations

import os
import re
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Model + shared generation parameters
# Identical across all configs — required for fair comparison.
# See docs/reproduction.md for the frozen configuration.
# ---------------------------------------------------------------------------

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = 0  # Greedy decoding reduces variance but does not ensure determinism.
# Shared answer budget. C4 condensing and faithfulness have separate budgets.
# Truncation is recorded per call; the run snapshot records these settings.
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))

# ---------------------------------------------------------------------------
# Prompt structure — placement is model-specific, wrong slot silently destroys
# answers.
# OpenAI-style: persona in the system slot.
# Nemotron: all instructions in the user prompt, system slot reserved
# for "detailed thinking on/off" (model card).
# Measured (2026-08-05, 8 DEV q): thinking-on truncates 8/8 answers to EMPTY
# (<think> eats the 128 budget); the mode switch alone doesn't shorten output;
# the brevity line does (C1 median 25 → 2 words, truncation 3/8 → 0/8, no
# score cost) — what makes MAX_TOKENS=128 viable on this model.
# The brevity line also SUPPRESSES abstention (2026-08-06, paired on C3):
# without it 22.0% → 76.0% but 149/200 truncated (variant B); refusal-permitting
# wording lost 38 correct→abstain (variant E). Kept — it trades abstention
# against truncation; changing it re-opens the variant comparison.
# The control: all four configs share ONE structure within a run (C4 condense
# included) — not one prompt across models. Non-default runs say so.
# The retained DEV prompt-variant runs document this sensitivity.
# ---------------------------------------------------------------------------

_PERSONA = "You are a factual question-answering assistant. Answer questions concisely and accurately."
_BREVITY = "Answer with the fact only - no explanation, no restatement of the question."


def _build_styles() -> None:
    """(Re)derive the style dicts from the live _PERSONA/_BREVITY values.

    Extracted from module-level literals (2026-08-18) so the prompt-abstention
    study's persona/brevity arms can vary the two lines through
    apply_prompt_overrides() below. With the shipped values the dicts are
    byte-identical to what the literals produced -- pinned by
    tests/unit/test_prompt_variants.py.

    The joins guard against an EMPTY persona or brevity (declared study arms):
    a bare `_PERSONA + " " + _BREVITY` would leave a leading/trailing space and
    a "\\n\\n"-only preamble where the arm means "no preamble at all".
    """
    global _DEFAULT_STYLE, _NEMOTRON_STYLE, _PROMPT_STYLES
    answer_preamble = " ".join(part for part in (_PERSONA, _BREVITY) if part)
    # system=None means "send no system message at all", which is what the C4
    # condensing call has always done. Preserved exactly so every existing
    # gpt-4o-mini result stays reproducible.
    _DEFAULT_STYLE = {
        "answer": {"system": _PERSONA or None, "preamble": ""},
        "condense": {"system": None, "preamble": ""},
    }
    _NEMOTRON_STYLE = {
        # The mode switch must ride on EVERY call, condensing included — leaving
        # the system slot empty hands the mode back to the provider default.
        "answer": {
            "system": "detailed thinking off",
            "preamble": answer_preamble + "\n\n" if answer_preamble else "",
        },
        # No brevity line here: condensing is asked for one or two sentences of
        # prose, so ordering it to emit a bare fact would defeat the step
        # Config 4 exists to measure. The PERSONA does reach it — like the
        # entity block, framing shared by all calls within a run reaches both
        # of C4's calls, so a persona arm moves them together.
        "condense": {
            "system": "detailed thinking off",
            "preamble": _PERSONA + "\n\n" if _PERSONA else "",
        },
    }
    _PROMPT_STYLES = {
        "nvidia/llama-3.3-nemotron-super-49b-v1": _NEMOTRON_STYLE,
        # nemotron-3 keeps the 49B's placement (persona + brevity in the user
        # turn) but its reasoning switch is NOT a prompt string. The model card
        # documents exactly one mechanism — the chat-template flag
        # enable_thinking=True/False, default ON — and nothing else. The
        # "/no_think" system line used 2026-08-31 is undocumented and was probed
        # 2026-09-04 to make things WORSE: the model still reasons and the
        # server copies the trace into `content`, where the 128-token cap cuts
        # it off before any answer. The flag rides in _REQUEST_EXTRAS below and
        # is applied by complete(); this style therefore sends no system line.
        # (The 49B itself was EOL'd by NVIDIA Build 2026-08-26.) Non-reference
        # model — nothing run on it is comparable to the sealed results.
        "nvidia/nemotron-3-super-120b-a12b": {
            role: {**spec, "system": None}
            for role, spec in _NEMOTRON_STYLE.items()
        },
    }


_build_styles()


def apply_prompt_overrides(
    *, persona: str | None = None, brevity: str | None = None
) -> None:
    """Set persona/brevity for a declared prompt-variant arm and rebuild styles.

    None = keep the shipped line; "" = remove it. The two keys live here rather
    than in prompts.py because WHERE they are placed is a per-model convention
    this module owns (Nemotron carries them in the user preamble; the OpenAI
    style carries the persona in the system slot and has never had a brevity
    line -- an override changes the LINE, never the placement).

    One sanctioned caller: run_eval's --prompt-variant handling, before the
    specs snapshot and before any worker starts. The snapshot records the
    result via prompt_style(), so the arm's meta.json shows the real framing.
    """
    global _PERSONA, _BREVITY
    if persona is not None:
        _PERSONA = persona
    if brevity is not None:
        _BREVITY = brevity
    _build_styles()


# The two call roles. Named so prompt_style() can reject one arriving in the
# model slot -- see its docstring.
_ROLES = frozenset(_DEFAULT_STYLE)


def prompt_style(model: str | None = None, *, role: str = "answer") -> dict:
    """
    The {system, preamble} pair for this model and call role ('answer'|'condense').

    `role` is KEYWORD-ONLY on purpose. Both parameters take strings, so
    `prompt_style(model, role)` reads naturally and is wrong: the role lands in
    the model slot, misses _PROMPT_STYLES, and returns the default style for the
    default role. It raises nothing. tools/document_run.py made exactly that
    call and recorded the OpenAI framing into every archived Nemotron run's
    meta.json for four days.

    The `*` alone does NOT close it -- `prompt_style("answer")` still binds
    legally to `model` -- so the role names are rejected explicitly. Nothing is
    lost: no model is called "answer" or "condense", and the silent-wrong-value
    path is the one failure mode this function cannot be allowed to have.
    """
    if model in _ROLES:
        raise TypeError(
            f"prompt_style() got the role {model!r} in the model slot. "
            f"`role` is keyword-only: call prompt_style(role={model!r})."
        )
    return _PROMPT_STYLES.get(model or MODEL, _DEFAULT_STYLE)[role]


def build_messages(
    user_body: str, role: str = "answer", model: str | None = None
) -> list[dict]:
    """
    Chat messages for one call, with instructions placed where this model wants them.

    `user_body` is the config's own prompt — the part that legitimately differs
    between Configs 1-4. This function adds the same framing to all answering
    calls; the separate condensing role intentionally omits the brevity line.

    Resolved at CALL time rather than bound at import, so a tool that reassigns
    llm_config.MODEL between batches is honoured — the same reason the pipelines
    resolve top_k at call time.
    """
    style = prompt_style(model, role=role)
    messages: list[dict] = []
    if style["system"]:
        messages.append({"role": "system", "content": style["system"]})
    messages.append({"role": "user", "content": style["preamble"] + user_body})
    return messages


# Pricing (USD per 1M tokens) for the token counter's cost estimate.
#
# An ABSENT key and a key declared (0.0, 0.0) mean different things, and the
# difference matters on a metered endpoint: absent means "nobody checked", which
# on a paid provider silently reports $0.00 while credits burn. Declared zero
# means "checked, and this endpoint genuinely does not charge per token".
# Use is_priced() to tell them apart; price_per_million() collapses both to (0,0).
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    # NVIDIA Build (integrate.api.nvidia.com) meters the free tier in REQUEST
    # credits, not tokens, so no per-token price exists to record. Declared
    # explicitly so the zero is a finding rather than a lookup miss; the token
    # counts printed alongside it remain the real usage figure.
    "nvidia/llama-3.3-nemotron-super-49b-v1": (0.0, 0.0),
    # Same free tier, same request metering (demo substitute since 2026-08-31).
    "nvidia/nemotron-3-super-120b-a12b": (0.0, 0.0),
}


# Provider-side request fields a model needs beyond messages / temperature /
# max_tokens. Applied by complete() on every call, condensing and faithfulness
# included — a mode switch that skips one call hands that call back to the
# provider default. A model with NO entry sends exactly what the caller passed,
# so the reference model's requests are byte-identical to the sealed runs.
_REQUEST_EXTRAS: dict[str, dict] = {
    # Model card (build.nvidia.com, read 2026-09-04): "Reasoning Mode:
    # Configurable on/off via chat template (enable_thinking=True/False)";
    # default ON; OpenAI-client form is
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}.
    # Probed on the C1 prompt: flag off → 7-token clean answer; provider
    # default → 69 of 128 tokens spent in `reasoning_content`.
    "nvidia/nemotron-3-super-120b-a12b": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


def request_extras(model: str | None = None) -> dict:
    """extra_body fields this model needs; {} for models that need none.

    Returns a fresh copy two levels deep, so a caller merging its own keys
    (complete() does) can never mutate the table.
    """
    extras = _REQUEST_EXTRAS.get(model or MODEL, {})
    return {k: dict(v) if isinstance(v, dict) else v for k, v in extras.items()}


def price_per_million(model: str | None = None) -> tuple[float, float]:
    """(input, output) USD per 1M tokens; (0, 0) for unknown/self-hosted models."""
    return _PRICING.get(model or MODEL, (0.0, 0.0))


def is_priced(model: str | None = None) -> bool:
    """True if this model's rate was deliberately recorded (even as zero)."""
    return (model or MODEL) in _PRICING


# ---------------------------------------------------------------------------
# Response text extraction
#
# Every pipeline used to end with `return response.choices[0].message.content`,
# which assumes the provider returns a plain answer string. Two provider
# behaviours break that assumption, and both fail in ways the evaluation would
# score rather than reject:
#
#   1. REASONING TRACES. Hybrid-reasoning models emit <think>...</think> before
#      answering. MAX_TOKENS is 128 — sized for answers, not for deliberation
#      followed by an answer — so the budget is often exhausted mid-trace and no
#      answer is ever produced. The scorer would then grade the trace against the
#      gold answer, count it wrong, and record a hallucination the model never
#      committed. The trace is stripped here, including an UNTERMINATED trailing
#      <think> (the truncation case), so what reaches the scorer is only text the
#      model actually offered as its answer.
#
#   2. content=None. Some OpenAI-compatible servers route reasoning to a separate
#      field and leave `content` null. `.strip()` on that raises AttributeError
#      mid-run. Returning "" instead is deliberate: an empty answer is a missing
#      answer, and it must not be back-filled from the reasoning field — doing so
#      would reintroduce failure mode 1 through the side door.
#
# On a non-reasoning model (gpt-4o-mini, and the Nemotron-Super default measured
# 2026-08-04) neither regex matches, so this is a pass-through and existing
# results stay reproducible.
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

# finish_reason of the most recent call ON THIS THREAD.
#
# Truncation is invisible downstream: a cut-off answer is a plain short string,
# and the scorer grades it as a wrong answer rather than a missing one — the same
# class of mistake the `errors` field was added to stop. MAX_TOKENS=128 truncated
# 17% of Nemotron answers before the brevity instruction was added, so whether it
# still does is a measurement the next run has to make rather than assume.
#
# Thread-local because the eval runs questions in a pool but runs each question's
# configs sequentially within one thread — so a reader in that thread sees its own
# last call, never another question's.
_last_call = threading.local()


def last_finish_reason() -> str | None:
    """finish_reason of the last chat completion on this thread ('length' = truncated)."""
    return getattr(_last_call, "finish_reason", None)


def extract_text(response) -> str:
    """Answer text from a chat completion: never None, never a reasoning trace."""
    choice = response.choices[0]
    reason = getattr(choice, "finish_reason", None)
    # Coerced: a mocked response yields a Mock here, and a Mock recorded as a
    # finish_reason would serialize into a result file as a fake value.
    _last_call.finish_reason = reason if isinstance(reason, str) else None
    text = getattr(choice.message, "content", None) or ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Client (lazy init — avoids loading API key at import time)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None
# Guards construction only; the OpenAI client itself is thread-safe to share.
_client_lock = threading.Lock()


def get_client() -> OpenAI:
    """
    Return a shared OpenAI-compatible client.

    Honours LLM_BASE_URL / LLM_API_KEY when set (NVIDIA Build, UMA-deployed
    open models, or any OpenAI-compatible endpoint); otherwise falls back to
    OpenAI's default endpoint + OPENAI_API_KEY.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # re-check: another thread may have won the race
                kwargs: dict[str, str] = {}
                base_url = os.getenv("LLM_BASE_URL")
                api_key = os.getenv("LLM_API_KEY")
                if base_url:
                    kwargs["base_url"] = base_url
                if api_key:
                    kwargs["api_key"] = api_key
                _client = OpenAI(**kwargs)
    return _client


# ---------------------------------------------------------------------------
# Request pacing
# ---------------------------------------------------------------------------
#
# A rate limit is a property of the PROVIDER, so it lives here rather than in
# the eval layer -- the demo path and the sweep tools hit the same endpoint and
# need the same pacing.
#
# THROTTLING IS FASTER THAN NOT THROTTLING, which is counterintuitive enough to
# be worth the measurement. NVIDIA Build's free tier allows ~40 requests/minute.
# Unpaced, three workers burst past that, the endpoint returns HTTP 429, and the
# retry path sleeps 2s -> 4s -> 8s. That backoff is dead time. Measured on the
# same provider on 2026-08-06, DEV-200:
#
#   run_eval, unpaced      24 req/min   47 retries
#   variant runner, paced  33 req/min    0 retries
#
# On the 4,000-question test split (20,000 requests) that is ~14 h against
# ~10 h. Pacing gives up nothing: the ceiling was never reachable anyway.
#
# 0 disables it. Default resolves from LLM_RPM_LIMIT, else a safe value for a
# recognised metered endpoint, else off -- so an OpenAI run is unaffected and an
# NVIDIA run is paced without anyone having to remember a flag.
_rpm_limit: int = 0
_permits: list[float] = []
_permit_lock = threading.Lock()


def default_rpm() -> int:
    env = os.getenv("LLM_RPM_LIMIT")
    if env is not None:
        return int(env)
    base = (os.getenv("LLM_BASE_URL") or "").lower()
    # 33 not 40: retries consume quota too, and the cap is enforced on arrival
    # rate, so a little headroom costs nothing and prevents the backoff spiral.
    return 33 if "nvidia" in base else 0


def set_rpm_limit(rpm: int) -> None:
    global _rpm_limit
    with _permit_lock:
        _rpm_limit = max(0, int(rpm))
        _permits.clear()


def acquire_slot() -> None:
    """Block until a request may be sent. No-op when the limit is 0."""
    while True:
        with _permit_lock:
            if _rpm_limit <= 0:
                return
            now = time.time()
            # Drop permits older than the window, then take one if there's room.
            _permits[:] = [t for t in _permits if now - t < 60.0]
            if len(_permits) < _rpm_limit:
                _permits.append(now)
                return
            wait = 60.0 - (now - _permits[0]) + 0.05
        time.sleep(max(wait, 0.05))


# ---------------------------------------------------------------------------
# Call telemetry
#
# ⚠️ WHY NOT JUST COUNT `token_counter`. That accumulator increments only when a
# response arrives WITH a usage field, so it counts neither failed attempts nor
# successful responses from providers that omit usage. It is a usage ledger, not
# a call count -- its `calls` key is reported as `usage_records` for that reason.
#
# ⚠️ AND WHY NOT INFER IT FROM THE RESULTS FILE. `run_eval` used to report
# `sum(len(row["answers"]))`, which is one per config per question = 800 on
# DEV-200 -- silently omitting C4's 200 condensing calls, i.e. 20% of the run.
#
# ⚠️ SCOPE: these are APPLICATION-level attempts. The OpenAI SDK retries some
# failures inside `.create()`, and those are invisible here. Recorded as
# `includes_sdk_internal_retries: false` rather than left for a reader to assume.
#
# Deliberately NOT merged with token_counter, and the pipelines' `record()` calls
# are deliberately left alone: `git diff src/pipelines/` must stay empty.
# ---------------------------------------------------------------------------

_call_stats = {
    "completion_attempts": 0,
    "completion_responses": 0,
    "completion_exceptions": 0,
    "responses_with_usage": 0,
}
_call_stats_lock = threading.Lock()


def reset_call_stats() -> None:
    """Zero the counters. Called once per run, mirroring token_counter.reset()."""
    with _call_stats_lock:
        for k in _call_stats:
            _call_stats[k] = 0


def call_stats() -> dict:
    with _call_stats_lock:
        return dict(_call_stats, includes_sdk_internal_retries=False)


def complete(**kwargs):
    """
    One chat completion, paced and counted.

    Every pipeline calls this rather than `get_client().chat.completions.create`
    directly, so the limiter cannot be bypassed by adding a new call site and
    forgetting about it. `get_client` is still called through, so existing tests
    that patch it keep working unchanged.

    Per-model request extras (see _REQUEST_EXTRAS) are merged into `extra_body`
    here, so a pipeline never has to know which provider flag its model needs;
    keys the caller passed explicitly win.
    """
    extras = request_extras(kwargs.get("model"))
    if extras:
        kwargs["extra_body"] = {**extras, **(kwargs.get("extra_body") or {})}
    acquire_slot()
    with _call_stats_lock:
        _call_stats["completion_attempts"] += 1
    try:
        response = get_client().chat.completions.create(**kwargs)
    except Exception:
        with _call_stats_lock:
            _call_stats["completion_exceptions"] += 1
        raise
    with _call_stats_lock:
        _call_stats["completion_responses"] += 1
        if getattr(response, "usage", None) is not None:
            _call_stats["responses_with_usage"] += 1
    return response
