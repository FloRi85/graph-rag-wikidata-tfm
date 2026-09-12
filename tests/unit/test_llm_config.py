"""
Unit tests for src/llm_config.py — the provider-swap surface.

extract_text() sits between every provider response and the scorer, so its job
is to make the scorer's input independent of provider quirks. The cases below
are the ones that actually occur: a plain answer (OpenAI, and Nemotron-Super
with reasoning off), a complete reasoning trace, a trace TRUNCATED by
MAX_TOKENS=128 before its closing tag, and a null content field.

The pass-through cases matter as much as the stripping ones: every published
result in this thesis was produced before extract_text existed, so a model that
emits no reasoning must come through byte-identical or those results stop being
reproducible.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_llm_config.py -v
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock

from src import llm_config


def _response(content):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


class TestExtractText:
    # --- pass-through: must not disturb existing results --------------------

    def test_plain_answer_unchanged(self):
        assert llm_config.extract_text(_response("Bono")) == "Bono"

    def test_multiline_prose_unchanged(self):
        text = "The presidents who were not elected are:\n\n1. Gerald Ford\n2. John Tyler"
        assert llm_config.extract_text(_response(text)) == text

    def test_angle_brackets_that_are_not_think_tags_survive(self):
        text = "The operator <= means less than or equal to"
        assert llm_config.extract_text(_response(text)) == text

    # --- reasoning traces ---------------------------------------------------

    def test_complete_think_block_stripped(self):
        r = _response("<think>U2's singer is Paul Hewson, stage name Bono.</think>Bono")
        assert llm_config.extract_text(r) == "Bono"

    def test_think_block_spanning_newlines_stripped(self):
        r = _response("<think>Let me work through this.\nU2 formed in 1976.\n</think>\nBono")
        assert llm_config.extract_text(r) == "Bono"

    def test_unterminated_think_yields_empty_not_the_trace(self):
        # The MAX_TOKENS=128 truncation case: budget spent deliberating, no
        # answer ever produced. Returning the trace here would be scored as a
        # wrong answer and counted as a hallucination the model never made.
        r = _response("<think>The question asks about the singer of U2. U2 formed in "
                      "Dublin in 1976 and the lead vocalist is")
        assert llm_config.extract_text(r) == ""

    def test_answer_before_unterminated_think_is_kept(self):
        r = _response("Bono\n<think>although I should double check whether")
        assert llm_config.extract_text(r) == "Bono"

    def test_think_tags_are_case_insensitive(self):
        assert llm_config.extract_text(_response("<THINK>hmm</THINK>Bono")) == "Bono"

    # --- missing content ----------------------------------------------------

    def test_none_content_returns_empty_string(self):
        assert llm_config.extract_text(_response(None)) == ""

    def test_empty_content_returns_empty_string(self):
        assert llm_config.extract_text(_response("")) == ""

    def test_reasoning_content_is_never_promoted_to_the_answer(self):
        # Some OpenAI-compatible servers put the trace in its own field. Falling
        # back to it would reintroduce the truncation failure through the side
        # door, so an absent content is reported as absent.
        msg = MagicMock(content=None, reasoning_content="The singer of U2 is")
        assert llm_config.extract_text(MagicMock(choices=[MagicMock(message=msg)])) == ""

    def test_whitespace_is_trimmed(self):
        assert llm_config.extract_text(_response("  Bono\n")) == "Bono"


class TestGenerationParams:
    def test_max_tokens_defaults_to_128(self, monkeypatch):
        # 128 is the value every reported gpt-4o-mini result was produced at.
        # Reverting .env must restore it without touching code.
        monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
        assert int(os.getenv("LLM_MAX_TOKENS", "128")) == 128

    def test_max_tokens_reads_the_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_TOKENS", "512")
        assert int(os.getenv("LLM_MAX_TOKENS", "128")) == 512

    def test_all_configs_share_one_ANSWER_cap(self):
        # The experimental control is that the cap on the ANSWER is identical
        # across configs, so answer length cannot become a confound. C4's
        # answering call is included; its condensing call is not -- see below.
        import src.pipelines.graph_rag_rerank as rerank
        assert rerank.ANSWER_MAX_TOKENS == llm_config.MAX_TOKENS

    def test_condense_cap_is_deliberately_larger_than_the_answer_cap(self):
        """
        REGRESSION. This test previously asserted CONDENSE_MAX_TOKENS ==
        MAX_TOKENS, treating "one cap everywhere" as the control. That was the
        wrong invariant and it hid a real defect: the shared cap is sized for
        ANSWERS ("Mintaka answers are short"), while the condensing call is asked
        for one or two sentences of prose over 30 triples.

        At 128 it truncated 143/200 condensations (71.5%) on the 2026-08-07
        reference run -- median 92 words, max 108, most ending mid-sentence --
        and the answering call was handed an incomplete context and then graded
        as if the context were whole. It went unnoticed because C4's answering
        call overwrites the condensing call's finish_reason in llm_config's
        single thread-local slot.

        The control that matters is the ANSWER budget, pinned above. The
        condensing budget is an internal step and must be free to differ.
        """
        import src.pipelines.graph_rag_rerank as rerank
        assert rerank.CONDENSE_MAX_TOKENS > rerank.ANSWER_MAX_TOKENS


class TestPricing:
    def test_known_model_has_a_rate(self):
        assert llm_config.price_per_million("gpt-4o-mini") == (0.150, 0.600)

    def test_unlisted_model_returns_zero(self):
        assert llm_config.price_per_million("some/unlisted-model") == (0.0, 0.0)

    def test_is_priced_separates_declared_zero_from_unlisted(self):
        # Both report $0.00; only one of them was checked. On a metered endpoint
        # that difference is the whole warning.
        assert llm_config.is_priced("nvidia/llama-3.3-nemotron-super-49b-v1") is True
        assert llm_config.is_priced("some/unlisted-model") is False


_PERSONA = ("You are a factual question-answering assistant. "
            "Answer questions concisely and accurately.")
_NEMOTRON = "nvidia/llama-3.3-nemotron-super-49b-v1"


class TestBuildMessages:
    """
    Where the instruction goes is model-specific; the guarantee is that it goes
    to the SAME place for all four configs within a run.
    """

    def test_default_style_is_byte_identical_to_the_old_hardcoded_messages(self):
        # Every reported gpt-4o-mini result was produced by pipelines that
        # hardcoded exactly this pair. If build_messages ever drifts from it,
        # those results stop being reproducible -- which is the whole reason the
        # default style exists as a separate entry rather than a shared one.
        assert llm_config.build_messages("BODY", model="gpt-4o-mini") == [
            {"role": "system", "content": _PERSONA},
            {"role": "user", "content": "BODY"},
        ]

    def test_default_condense_sends_no_system_message(self):
        # C4's condensing call has never carried one. A system=None entry and a
        # missing entry are different things, and only the first preserves that.
        assert llm_config.build_messages("BODY", role="condense", model="gpt-4o-mini") == [
            {"role": "user", "content": "BODY"},
        ]

    def test_nemotron_reserves_the_system_slot_for_the_mode_switch(self):
        # Model card: "All instructions should be contained within the user
        # prompt"; the system slot selects reasoning mode. Measured 2026-08-05:
        # "detailed thinking on" truncates 8/8 answers to empty at MAX_TOKENS=128.
        msgs = llm_config.build_messages("BODY", model=_NEMOTRON)
        assert msgs[0] == {"role": "system", "content": "detailed thinking off"}
        assert _PERSONA in msgs[1]["content"]
        assert msgs[1]["content"].endswith("BODY")

    def test_nemotron_carries_the_mode_switch_on_the_condense_call_too(self):
        # An unset system slot hands the mode back to the provider default, so
        # the switch has to ride on every call -- not just the answering one.
        msgs = llm_config.build_messages("BODY", role="condense", model=_NEMOTRON)
        assert msgs[0] == {"role": "system", "content": "detailed thinking off"}

    def test_condense_does_not_inherit_the_brevity_instruction(self):
        # Condensing is asked for one or two sentences of prose. Telling it to
        # emit a bare fact would collapse the step Config 4 exists to measure.
        brevity = "Answer with the fact only"
        assert brevity in llm_config.build_messages("BODY", model=_NEMOTRON)[1]["content"]
        assert brevity not in llm_config.build_messages(
            "BODY", role="condense", model=_NEMOTRON)[1]["content"]

    def test_model_is_resolved_at_call_time(self, monkeypatch):
        # Sweep tools reassign llm_config.MODEL between batches. A style bound at
        # import would silently ignore them and produce cells differing in label
        # only -- the same failure the pipelines' call-time top_k guards against.
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON)
        assert llm_config.build_messages("BODY")[0]["content"] == "detailed thinking off"
        monkeypatch.setattr(llm_config, "MODEL", "gpt-4o-mini")
        assert llm_config.build_messages("BODY")[0]["content"] == _PERSONA

    def test_every_config_shares_one_structure(self):
        # The control is not "one prompt for all models" but "one structure for
        # all four configs within a run". Pins that no pipeline reintroduces its
        # own system message.
        import src.pipelines.base_llm as c1
        import src.pipelines.rag as c2
        import src.pipelines.graph_rag as c3
        import src.pipelines.graph_rag_rerank as c4
        for mod in (c1, c2, c3, c4):
            assert not hasattr(mod, "SYSTEM_PROMPT"), f"{mod.__name__} holds a local system prompt"


_NEMOTRON3 = "nvidia/nemotron-3-super-120b-a12b"
_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


class TestRequestExtras:
    """
    nemotron-3's reasoning switch is a chat-template flag, not a prompt string
    (model card: "Configurable on/off via chat template (enable_thinking=
    True/False)", default ON). It travels as extra_body, added by complete()
    so no pipeline needs to know it exists -- and a model without an entry
    must send exactly what the caller passed, or the reference model's
    requests stop matching the sealed runs.
    """

    def _create_kwargs(self, monkeypatch, **call):
        monkeypatch.setattr(llm_config, "acquire_slot", lambda: None)
        client = MagicMock()
        monkeypatch.setattr(llm_config, "get_client", lambda: client)
        llm_config.complete(**call)
        return client.chat.completions.create.call_args.kwargs

    def test_nemotron3_sends_thinking_off_on_every_call(self, monkeypatch):
        kw = self._create_kwargs(monkeypatch, model=_NEMOTRON3, messages=[])
        assert kw["extra_body"] == _THINKING_OFF

    def test_reference_model_and_openai_send_no_extra_body(self, monkeypatch):
        # Byte-identical requests for the models behind every recorded result.
        for model in (_NEMOTRON, "gpt-4o-mini"):
            kw = self._create_kwargs(monkeypatch, model=model, messages=[])
            assert "extra_body" not in kw, model

    def test_caller_extra_body_is_merged_and_wins(self, monkeypatch):
        kw = self._create_kwargs(
            monkeypatch, model=_NEMOTRON3, messages=[],
            extra_body={"seed": 7, "chat_template_kwargs": {"enable_thinking": True}})
        assert kw["extra_body"] == {"seed": 7,
                                    "chat_template_kwargs": {"enable_thinking": True}}

    def test_model_falls_back_to_the_module_default(self, monkeypatch):
        # A caller that omits `model` still gets the flag for the active model.
        monkeypatch.setattr(llm_config, "MODEL", _NEMOTRON3)
        kw = self._create_kwargs(monkeypatch, messages=[])
        assert kw["extra_body"] == _THINKING_OFF
        assert llm_config.request_extras() == _THINKING_OFF
        assert llm_config.request_extras("gpt-4o-mini") == {}

    def test_request_extras_returns_a_copy(self):
        # complete() merges caller keys into what this returns; a shared dict
        # would let one call's extra_body leak into every later call.
        got = llm_config.request_extras(_NEMOTRON3)
        got["chat_template_kwargs"]["enable_thinking"] = True
        got["seed"] = 7
        assert llm_config.request_extras(_NEMOTRON3) == _THINKING_OFF

    def test_nemotron3_sends_no_system_line(self):
        # The undocumented "/no_think" system line (2026-08-31) made the model
        # copy its reasoning into `content`; the flag replaces it entirely.
        for role in ("answer", "condense"):
            msgs = llm_config.build_messages("BODY", role=role, model=_NEMOTRON3)
            assert msgs[0]["role"] == "user", role
            assert msgs[-1]["content"].endswith("BODY")
        assert _PERSONA in llm_config.build_messages("BODY", model=_NEMOTRON3)[0]["content"]


class TestRateLimiter:
    """
    Pacing exists because throttling is FASTER than not throttling on a metered
    endpoint: unpaced, workers burst past the cap, the endpoint returns 429, and
    the retry path sleeps 2/4/8s. Measured 2026-08-06 on NVIDIA Build over
    DEV-200 -- unpaced 24 req/min with 47 retries, paced 33 req/min with none.
    """

    def teardown_method(self):
        llm_config.set_rpm_limit(0)          # never leak a limit into other tests

    def test_disabled_by_default_does_not_block(self):
        llm_config.set_rpm_limit(0)
        t0 = time.time()
        for _ in range(50):
            llm_config.acquire_slot()
        assert time.time() - t0 < 0.5

    def test_limit_blocks_once_the_window_is_full(self):
        llm_config.set_rpm_limit(3)
        for _ in range(3):
            llm_config.acquire_slot()        # fills the window
        # A 4th must wait for a permit to age out. Don't actually wait ~60s --
        # assert the bucket is full rather than sleeping through the window.
        assert len(llm_config._permits) == 3

    def test_openai_endpoint_defaults_to_off(self, monkeypatch):
        # An OpenAI run must be unaffected: every gpt-4o-mini result on record
        # was produced unpaced, and adding a silent delay would change nothing
        # about the answers but would make reruns incomparable in wall-clock.
        monkeypatch.delenv("LLM_RPM_LIMIT", raising=False)
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
        assert llm_config.default_rpm() == 0

    def test_nvidia_endpoint_defaults_to_paced(self, monkeypatch):
        monkeypatch.delenv("LLM_RPM_LIMIT", raising=False)
        monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        assert llm_config.default_rpm() == 33

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        monkeypatch.setenv("LLM_RPM_LIMIT", "7")
        assert llm_config.default_rpm() == 7

    def test_complete_paces_then_calls_through(self, monkeypatch):
        # The limiter must not be bypassable by adding a call site: every
        # pipeline goes through complete(), so this is the single choke point.
        calls = []
        monkeypatch.setattr(llm_config, "acquire_slot", lambda: calls.append("paced"))
        client = MagicMock()
        monkeypatch.setattr(llm_config, "get_client", lambda: client)
        llm_config.complete(model="m", messages=[])
        assert calls == ["paced"]
        client.chat.completions.create.assert_called_once()
