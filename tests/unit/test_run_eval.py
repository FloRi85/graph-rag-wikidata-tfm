"""Unit tests for the evaluation runner's row construction.

`run_question` builds every row of every result file, and until 2026-08-16 it
had no test at all -- the run itself was the test, at ~1,000 API calls a go.
Nothing here makes a network call: `run_config` is monkeypatched.

Run:
    venv/Scripts/python -m pytest tests/unit/test_run_eval.py -v
"""

import pytest

from src.eval import run_eval
from src.eval.parse_answers import build_gold_answer
from src.eval.parse_questions import parse_question


RAW = {
    "id": "q-tom",
    "question": "Which Academy Award did Tom Hanks win for Philadelphia?",
    "category": "movies",
    "complexityType": "generic",
    "questionEntity": [
        {"entityType": "entity", "name": "Q2263", "label": "Tom Hanks",
         "mention": "Tom Hanks", "span": [24, 33]},
        {"entityType": "entity", "name": "Q19020", "label": "Academy Awards",
         "mention": "Academy Award", "span": [6, 19]},
    ],
    "answer": {
        "answerType": "entity",
        "mention": "Best Actor",
        "answer": [{"name": "Q103916",
                    "label": {"en": "Academy Award for Best Actor"}}],
    },
}


@pytest.fixture
def question():
    return parse_question(RAW)


@pytest.fixture
def gold():
    return build_gold_answer(RAW["answer"])


@pytest.fixture
def answered(monkeypatch):
    """Every config answers with its own id, so rows are traceable to a config."""
    monkeypatch.setattr(run_eval, "run_config",
                        lambda config_id, *a, **kw: f"answer-from-{config_id}")
    return None


class TestRowConstruction:

    def test_gold_fields_come_from_the_gold_answer(self, question, gold, answered):
        row = run_eval.run_question(1, question, 1, gold)
        assert row["id"] == "q-tom"
        assert row["expected"] == "Best Actor"
        assert row["answer_type"] == "entity"
        assert row["answer_qids"] == ["Q103916"]
        assert "Academy Award for Best Actor" in row["answer_forms"]
        assert "Best Actor" in row["answer_forms"]

    def test_single_label_is_not_a_set(self, question, gold, answered):
        """`answer_entities` carries SET MEMBERS, and one label is not a set."""
        row = run_eval.run_question(1, question, 1, gold)
        assert row["answer_entities"] == []

    def test_two_labels_are_set_members(self, question, answered):
        gold = build_gold_answer({
            "answerType": "entity", "mention": "XL, XLIX",
            "answer": [{"name": "Q1", "label": {"en": "Super Bowl XL"}},
                       {"name": "Q2", "label": {"en": "Super Bowl XLIX"}}],
        })
        row = run_eval.run_question(1, question, 1, gold)
        assert row["answer_entities"] == ["Super Bowl XL", "Super Bowl XLIX"]

    def test_stratification_fields_come_from_the_question(self, question, gold, answered):
        row = run_eval.run_question(1, question, 1, gold)
        assert row["type"] == "movies"          # category
        assert row["complexity"] == "generic"

    def test_qids_are_reading_ordered_and_deduplicated(self, question, gold, answered):
        """'Academy Award' is at span 6 and 'Tom Hanks' at 24, but Mintaka lists
        Tom Hanks first. The block follows the question, not the annotator."""
        row = run_eval.run_question(1, question, 1, gold)
        assert row["entity_qids"] == ["Q19020", "Q2263"]

    def test_a_run_without_a_gold_still_produces_a_row(self, question, answered):
        """A question id absent from the split must not crash the run; the row
        records that there was nothing to grade against."""
        row = run_eval.run_question(1, question, 1, None)
        assert row["expected"] == "" and row["answer_forms"] == []
        assert row["answers"]                      # the answers are still there


class TestEntityNamesHandedToThePipelines:

    def test_pipelines_receive_the_mention_not_the_label(self, question, gold,
                                                         monkeypatch):
        """The decision of 2026-08-16 — see Question.entity_mentions.

        This is the string prepended to every rendered statement, so it is worth
        pinning at the boundary where it actually crosses into a pipeline.
        """
        seen = {}

        def capture(config_id, q_text, entity_names, entity_qids, capture_dict=None):
            seen[config_id] = list(entity_names)
            return "ok"

        monkeypatch.setattr(run_eval, "run_config", capture)
        run_eval.run_question(1, question, 1, gold)

        for names in seen.values():
            assert names == ["Academy Award", "Tom Hanks"]
            assert "Academy Awards" not in names   # the canonical label


class TestAnswerValue:

    @pytest.mark.parametrize("answer_type,payload,mention,expected", [
        ("numerical", [166], "166 pounds", 166),
        ("boolean", [True], "Yes", True),
        ("date", ["2017-08-25"], "25-Aug-17", "2017-08-25"),
        ("string", ["Currer Bell"], "Currer Bell", "Currer Bell"),
    ])
    def test_non_entity_answers_keep_the_scalar_payload(self, answer_type, payload,
                                                        mention, expected):
        gold = build_gold_answer({"answerType": answer_type, "answer": payload,
                                  "mention": mention})
        assert run_eval._row_answer_value(gold) == expected

    def test_entity_answers_have_none(self, gold):
        """Their payload is {QID, labels}; answer_qids/answer_forms carry it."""
        assert run_eval._row_answer_value(gold) is None

    def test_no_gold_is_none_not_an_error(self):
        assert run_eval._row_answer_value(None) is None


class TestArticlePreflight:
    """The text arm's half. Added 2026-08-16 — before it, only the KG arm was
    checked, so C2 could discover a missing article at question 140 and
    substitute an empty context for it, silently."""

    def test_a_usable_cache_passes_and_counts_entities(self, question, monkeypatch):
        monkeypatch.setattr(run_eval.wikipedia, "cache_entry_status",
                            lambda qid: {"state": "ok"})
        monkeypatch.setattr(run_eval.wikipedia, "empty_articles", lambda qids: {})
        report = run_eval.preflight_article_cache([question], "some.json")
        assert report == {"n_entities": 2, "no_article": {}}

    def test_an_entity_with_no_article_passes_and_is_RECORDED(self, question,
                                                              monkeypatch):
        """Absence is legal. It is not a failure — many Wikidata items have no
        enwiki page — but it changes what C2 saw, so it goes in meta.json."""
        monkeypatch.setattr(run_eval.wikipedia, "cache_entry_status",
                            lambda qid: {"state": "ok"})
        monkeypatch.setattr(run_eval.wikipedia, "empty_articles",
                            lambda qids: {"Q19020": "no_sitelink"})
        report = run_eval.preflight_article_cache([question], "some.json")
        assert report["no_article"] == {"Q19020": "no_sitelink"}

    def test_a_stale_entry_aborts_before_any_call(self, question, monkeypatch):
        monkeypatch.setattr(run_eval.wikipedia, "cache_entry_status",
                            lambda qid: {"state": "stale"})
        with pytest.raises(SystemExit, match="ARTICLE CACHE PREFLIGHT FAILED"):
            run_eval.preflight_article_cache([question], "some.json")

    def test_the_abort_names_the_tool_that_fixes_it(self, question, monkeypatch):
        monkeypatch.setattr(run_eval.wikipedia, "cache_entry_status",
                            lambda qid: {"state": "missing"})
        with pytest.raises(SystemExit, match="prewarm_article_cache.py"):
            run_eval.preflight_article_cache([question], "some.json")

    def test_both_arms_use_the_same_qid_expression(self, question, monkeypatch):
        """Prewarm, the KG preflight and this one must agree on WHICH entities a
        run needs, or 'nothing to do' precedes a run that fetches hundreds."""
        seen_kg, seen_text = [], []
        monkeypatch.setattr(run_eval, "cache_entry_status",
                            lambda qid: seen_kg.append(qid) or {"state": "ok"})
        monkeypatch.setattr(run_eval, "truncated_entities", lambda qids: {})
        monkeypatch.setattr(run_eval.wikipedia, "cache_entry_status",
                            lambda qid: seen_text.append(qid) or {"state": "ok"})
        monkeypatch.setattr(run_eval.wikipedia, "empty_articles", lambda qids: {})
        run_eval.preflight_statement_cache([question], "f.json")
        run_eval.preflight_article_cache([question], "f.json")
        assert sorted(seen_kg) == sorted(seen_text) == ["Q19020", "Q2263"]


class TestPreflightReadsTheParsedQuestions:

    def test_it_collects_qids_from_question_objects(self, question, monkeypatch):
        seen = []
        monkeypatch.setattr(run_eval, "cache_entry_status",
                            lambda qid: seen.append(qid) or {"state": "ok"})
        monkeypatch.setattr(run_eval, "truncated_entities", lambda qids: {})
        report = run_eval.preflight_statement_cache([question], "some.json")
        assert sorted(seen) == ["Q19020", "Q2263"]
        assert report["n_entities"] == 2

    def test_an_unusable_entity_aborts_before_any_call(self, question, monkeypatch):
        monkeypatch.setattr(run_eval, "cache_entry_status",
                            lambda qid: {"state": "stale"})
        with pytest.raises(SystemExit, match="PREFLIGHT FAILED"):
            run_eval.preflight_statement_cache([question], "some.json")
