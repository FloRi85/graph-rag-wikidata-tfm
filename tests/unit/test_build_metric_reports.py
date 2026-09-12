"""
Unit tests for tools/build_metric_reports.py — the TEST-4,000 metric-report supplement.

Offline by construction: the tool scores a stored run and joins a stored judge
file, so the tests run it end-to-end on small synthetic inputs. What is pinned:
the three populations are built the way the action plan defines them (attempted
= CORRECT or HALLUCINATION, never "not abstention"; pairs on the INTERSECTION;
faithfulness conditional on a scoreable attempt with C1 N/A and unscored
records EXCLUDED rather than zeroed), the numerical identities hold, the pair
orientation is A − B in the declared order, an empty group is "n/a" and a tiny
one carries †, malformed judge files are fatal, the shared rounding rule and
grounded threshold are the project's and not local copies, two
generations of the same input are byte-identical, and the explorer prints
only number strings that Python rendered (2026-09-12 audit: the retired
JavaScript formatter disagreed with the Markdown in 135 cells by 0.1 point).
"""

import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ABSTAIN = "The answer is not in the context."
CONFIGS = ("base_llm_abstain", "rag", "graph_rag", "rerank")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load("build_metric_reports")


# ---------------------------------------------------------------------------
# Synthetic inputs: 8 questions, three answer types, three complexities, two topics.
# ---------------------------------------------------------------------------

def _q(qid, kind, gold, mention, complexity, category):
    if kind == "entity":
        answer = {"answerType": "entity", "answer": [{"name": gold, "label": {"en": mention}}], "mention": mention}
    elif kind == "boolean":
        answer = {"answerType": "boolean", "answer": [gold], "mention": mention}
    else:
        answer = {"answerType": "numerical", "answer": [gold], "mention": mention}
    return {"id": qid, "question": f"Question {qid}?", "translations": {},
            "questionEntity": [{"name": "Q142", "entityType": "entity", "label": "France", "mention": "France",
                                "span": [0, 6]}],
            "answer": answer, "category": category, "complexityType": complexity}


QUESTIONS = [
    _q("q1", "entity", "Q90", "Paris", "generic", "geography"),
    _q("q2", "entity", "Q64", "Berlin", "generic", "geography"),
    _q("q3", "entity", "Q220", "Rome", "generic", "history"),
    _q("q4", "entity", "Q2807", "Madrid", "generic", "history"),
    _q("q5", "boolean", True, "Yes", "yesno", "geography"),
    _q("q6", "boolean", False, "No", "yesno", "history"),
    _q("q7", "numerical", 3, "3", "count", "geography"),
    _q("q8", "numerical", 12, "12", "count", "history"),
]
META = {q["id"]: (q["answer"]["answerType"], q["complexityType"]) for q in QUESTIONS}

# Designed outcomes (C = correct, H = hallucination, A = abstention, O = other/empty):
#          q1 q2 q3 q4 q5 q6 q7 q8
#   C1      C  H  C  H  C  C  H  C      attempted 8
#   C2      C  A  C  A  C  H  A  C      attempted 5
#   C3      C  C  A  A  O  H  C  A      attempted 4  (one Other, never an attempt)
#   C4      H  C  C  A  C  H  H  C      attempted 7
ANSWERS = {
    "base_llm_abstain": ["Paris", "Munich", "Rome", "Lisbon", "Yes", "No", "7", "12"],
    "rag":              ["Paris", ABSTAIN, "Rome", ABSTAIN, "Yes", "Yes", ABSTAIN, "12"],
    "graph_rag":        ["Paris", "Berlin", ABSTAIN, ABSTAIN, "", "Yes", "3", ABSTAIN],
    "rerank":           ["Lyon", "Berlin", "Rome", ABSTAIN, "Yes", "Yes", "9", "12"],
}
EXPECTED_ATTEMPTED = {"base_llm_abstain": 8, "rag": 5, "graph_rag": 4, "rerank": 7}
EXPECTED_HALL = {"base_llm_abstain": 3, "rag": 1, "graph_rag": 1, "rerank": 3}


def _rows():
    rows = []
    for i, q in enumerate(QUESTIONS):
        rows.append({"id": q["id"], "question": q["question"], "answer_type": META[q["id"]][0],
                     "complexity": META[q["id"]][1],
                     "answers": {c: ANSWERS[c][i] for c in CONFIGS}, "finish_reasons": {}, "errors": {}})
    return rows


def _outcome(config, i):
    """The designed outcome table above, as the scorer's labels."""
    table = {"base_llm_abstain": "CHCHCCHC", "rag": "CACACHAC", "graph_rag": "CCAAOHCA", "rerank": "HCCACHHC"}
    return {"C": "correct", "H": "hallucination", "A": "abstention", "O": "other"}[table[config][i]]


def _faith_records(unscored=(("graph_rag", "q7"),)):
    recs = []
    for c in CONFIGS[1:]:
        for i, q in enumerate(QUESTIONS):
            oc = _outcome(c, i)
            if oc not in ("correct", "hallucination"):
                continue
            score = 1.0 if oc == "correct" else 0.25
            if (c, q["id"]) in unscored:
                recs.append({"id": q["id"], "config": c, "outcome": oc, "score": None, "partial_score": 0.5,
                             "status": "error", "n_claims": 0})
            else:
                recs.append({"id": q["id"], "config": c, "outcome": oc, "score": score, "status": "ok", "n_claims": 2})
    return recs


@pytest.fixture()
def inputs(tmp_path):
    raw = tmp_path / "synthetic_raw.json"
    gold = tmp_path / "questions_raw.json"
    faith = tmp_path / "faithfulness_synthetic.json"
    raw.write_text(json.dumps(_rows()), encoding="utf-8")
    gold.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    faith.write_text(json.dumps(_faith_records()), encoding="utf-8")
    return raw, gold, faith


@pytest.fixture()
def results(tool, inputs):
    raw, gold, faith = inputs
    inp = tool.load_inputs(raw, gold, faith)
    return tool.compute(inp, resamples=40), inp


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

class TestPopulations:
    def test_outcomes_partition_and_attempted_excludes_other(self, results):
        res, _ = results
        for c in CONFIGS:
            agg = res["full"]["overall"][c]
            assert sum(agg["counts"].values()) == 8
            assert agg["counts"]["hallucination"] == EXPECTED_HALL[c]
            assert agg["n_attempted"] == EXPECTED_ATTEMPTED[c]
        # C3's empty reply is OTHER: not an attempt, not an abstention.
        assert res["full"]["overall"]["graph_rag"]["counts"]["other"] == 1
        assert res["attempted"]["overall"]["graph_rag"]["n"] == 4
        assert res["attempted"]["overall"]["graph_rag"]["excluded"] == {"abstention": 3, "other": 1}

    def test_pair_common_is_intersection_in_declared_order(self, results, tool):
        res, _ = results
        assert [p["key"] for p in res["pairs"]] == ["C2-C1", "C3-C1", "C4-C1", "C3-C2", "C4-C2", "C4-C3"]
        # C3 attempts {q1,q2,q6,q7}; C2 attempts {q1,q3,q5,q6,q8}: common = {q1,q6}
        assert res["attempted"]["pairs"]["C3-C2"]["overall"]["n_common"] == 2
        assert res["cohorts"]["common_attempted:C3-C2"]["n"] == 2
        assert res["cohorts"]["common_attempted:all_four"]["n"] == 2   # q1 and q6
        # Full-folder pairs never filter.
        for p in res["pairs"]:
            assert res["full"]["pairs"][p["key"]]["overall"]["n"] == 8

    def test_pair_delta_is_a_minus_b(self, results):
        res, _ = results
        p = res["full"]["pairs"]["C4-C3"]["overall"]
        assert p["hallucination"]["delta"] == pytest.approx(3 / 8 - 1 / 8)
        assert p["hallucination"]["bootstrap"]["delta"] == pytest.approx(p["hallucination"]["delta"])
        assert p["hallucination"]["mcnemar"]["delta"] == pytest.approx(p["hallucination"]["delta"])
        q = res["attempted"]["pairs"]["C3-C2"]["overall"]     # common {q1,q6}: C3 H on q6, C2 H on q6
        assert q["hallucination"]["delta"] == pytest.approx(0.0)
        assert q["hallucination"]["table"] == {"both_hallucinate": 1, "a_only": 0, "b_only": 0, "neither": 1}

    def test_attempted_complement_and_f1_identity(self, results):
        res, _ = results
        for c in CONFIGS:
            own = res["attempted"]["overall"][c]
            assert own["rates"]["correct"] + own["rates"]["hallucination"] == pytest.approx(1.0)
            full = res["full"]["overall"][c]
            assert full["f1_mean"] == pytest.approx(full["coverage"] * full["f1_attempted"])
        pc = res["attempted"]["pairs"]["C4-C1"]["overall"]
        assert pc["correct"]["delta"] == pytest.approx(-pc["hallucination"]["delta"])
        assert pc["correct"]["complement_of"] == "hallucination"

    def test_axes_partition_every_cohort_and_empty_group_is_zero_n(self, results):
        res, _ = results
        for axis in ("answer_type", "complexity", "category"):
            assert sum(res["axis_counts"][axis].values()) == 8
            for pk in res["full"]["pairs"]:
                assert sum(v["n"] for v in res["full"]["pairs"][pk]["by_axis"][axis].values()) == 8
        assert res["axis_counts"]["answer_type"]["date"] == 0
        assert res["full"]["pairs"]["C2-C1"]["by_axis"]["answer_type"]["date"]["available"] is False
        assert res["axis_counts"]["complexity"]["multihop"] == 0
        assert res["axis_counts"]["category"]["movies"] == 0

    def test_transition_matrix_rows_are_reference_config(self, results):
        res, _ = results
        t = res["full"]["transitions"]["C2-C1"]
        assert t["rows"] == "base_llm_abstain" and t["columns"] == "rag"
        assert sum(sum(r.values()) for r in t["matrix"].values()) == 8
        # C1 hallucinated on q2, q4, q7; C2 abstained on all three.
        assert t["matrix"]["hallucination"]["abstention"] == 3
        assert sum(res["full"]["attempt_patterns"].values()) == 8


# ---------------------------------------------------------------------------
# Faithfulness: conditional, C1 N/A, unscored excluded
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_unscored_excluded_not_zeroed(self, results):
        res, _ = results
        own = res["faithfulness"]["own"]["graph_rag"]
        assert own["n_attempted"] == 4 and own["n_scoreable"] == 3 and own["n_unscored_attempts"] == 1
        assert own["unscored_reasons"] == {"error": 1}
        # q1 correct 1.0, q2 correct 1.0, q6 hallucination 0.25; q7 unscored (partial_score ignored)
        assert own["mean"] == pytest.approx((1.0 + 1.0 + 0.25) / 3)
        acc = res["faithfulness"]["coverage_accounting"]["graph_rag"]
        assert acc == {"abstention": 3, "other": 1, "unscored_attempt": 1, "scoreable": 3, "total": 8,
                       "partition_ok": True}

    def test_c1_is_na_with_reason(self, results):
        res, _ = results
        assert "base_llm_abstain" not in res["faithfulness"]["own"]
        for pk in ("C2-C1", "C3-C1", "C4-C1"):
            assert res["faithfulness"]["pairs"][pk]["available"] is False
            assert "C1" in res["faithfulness"]["pairs"][pk]["reason"]

    def test_common_scoreable_needs_both_scored(self, results):
        res, _ = results
        # C3 scoreable {q1,q2,q6}; C4 scoreable {q1,q2,q3,q5,q6,q7,q8}: common {q1,q2,q6}
        assert res["faithfulness"]["pairs"]["C4-C3"]["overall"]["n_common_scoreable"] == 3
        assert res["cohorts"]["common_scoreable:all_three"]["n"] == 2   # q1, q6
        by = res["faithfulness"]["by_outcome"]["rerank"]
        assert by["correct"]["n"] == 4 and by["hallucination"]["n"] == 3
        assert by["grounded"]["hallucination"] == {"grounded": 0, "ungrounded": 3}

    def test_duplicate_record_is_fatal(self, tool, inputs):
        raw, gold, faith = inputs
        recs = _faith_records()
        recs.append(dict(recs[0]))
        faith.write_text(json.dumps(recs), encoding="utf-8")
        with pytest.raises(SystemExit, match="duplicate"):
            tool.load_inputs(raw, gold, faith)

    def test_outcome_disagreement_is_fatal(self, tool, inputs):
        raw, gold, faith = inputs
        recs = _faith_records()
        recs[0]["outcome"] = "hallucination" if recs[0]["outcome"] == "correct" else "correct"
        faith.write_text(json.dumps(recs), encoding="utf-8")
        with pytest.raises(SystemExit, match="scorer disagrees"):
            tool.load_inputs(raw, gold, faith)

    def test_missing_attempt_record_is_fatal(self, tool, inputs):
        raw, gold, faith = inputs
        faith.write_text(json.dumps(_faith_records()[1:]), encoding="utf-8")
        with pytest.raises(SystemExit, match="cover"):
            tool.load_inputs(raw, gold, faith)

    def test_c1_record_is_fatal(self, tool, inputs):
        raw, gold, faith = inputs
        recs = _faith_records() + [{"id": "q1", "config": "base_llm_abstain", "outcome": "correct", "score": 1.0}]
        faith.write_text(json.dumps(recs), encoding="utf-8")
        with pytest.raises(SystemExit, match="non-retrieval"):
            tool.load_inputs(raw, gold, faith)


# ---------------------------------------------------------------------------
# Shared rules, not local copies
# ---------------------------------------------------------------------------

class TestSharedRules:
    def test_pct_is_document_run_rule(self, tool):
        doc = _load("document_run")
        src = open(os.path.join(ROOT, "tools", "build_metric_reports.py"), encoding="utf-8").read()
        assert "from tools.document_run import _pct1" in src     # imported, not copied
        assert tool.pct(338, 4000) == "8.5"       # the exact-tie case the rule exists for
        assert tool.pct(338, 4000) == f"{doc._pct1(338, 4000):.1f}"
        assert tool.pct(0, 0) == "—"

    def test_grounded_threshold_matches_matrix_tool(self, tool):
        src = open(os.path.join(ROOT, "tools", "grounding_outcome_matrix.py"), encoding="utf-8").read()
        assert f"GROUNDED_THRESHOLD = {tool.GROUNDED_THRESHOLD}" in src

    def test_axes_are_parser_vocabulary(self, tool):
        from src.eval.parse_answers import ALL_ANSWER_TYPES
        from src.eval.parse_questions import CATEGORIES, COMPLEXITY_TYPES, raw_type
        assert tool.AXES[0][2] == tuple(ALL_ANSWER_TYPES)
        assert tool.AXES[1][2] == tuple(raw_type(x) for x in COMPLEXITY_TYPES)
        assert tool.AXES[2][2] == tuple(CATEGORIES)

    def test_precedent_checks_not_applicable_on_other_input(self, results):
        res, _ = results
        assert not any(c.get("match") is False for c in res["precedent_checks"])


# ---------------------------------------------------------------------------
# Rendering and determinism
# ---------------------------------------------------------------------------

class TestRendering:
    def test_markers_and_na(self, results, tool):
        res, _ = results
        files = tool.render_all(res)
        h = files["full_4000/hallucination_abstention.md"]
        assert "| date | n/a |" in h                      # empty group: unavailable, not zero
        assert "| entity | 4† |" in h                     # tiny cell
        f = files["full_4000/faithfulness.md"]
        assert "N/A — no context" in f
        assert "involves C1" in f
        assert "0.000" not in f.split("## 2. Overview")[1].split("## 3.")[0]   # no zero stands in for N/A
        import re
        for rel in files:
            assert not re.search(r"\bnan\b|\bNone\b", files[rel]), rel   # no float NaN / None leaks into prose

    def test_all_six_pair_headings_everywhere(self, results, tool):
        res, _ = results
        for rel, text in tool.render_all(res).items():
            if rel.endswith(".md") and rel != "README.md":
                for pk in ("C2 − C1", "C3 − C1", "C4 − C1", "C3 − C2", "C4 − C2", "C4 − C3"):
                    assert pk in text, (rel, pk)

    def test_explorer_is_self_contained(self, results, tool):
        res, _ = results
        html = tool.render_all(res)["explorer.html"]
        assert "http://" not in html and "https://" not in html
        assert json.dumps(res["cohorts"]["full"]["sha256"]) in html
        assert html.startswith("<title>")

    def test_deterministic(self, tool, inputs):
        raw, gold, faith = inputs
        a = tool.compute(tool.load_inputs(raw, gold, faith), resamples=40)
        b = tool.compute(tool.load_inputs(raw, gold, faith), resamples=40)
        assert tool.results_text(a) == tool.results_text(b)
        assert tool.render_all(a) == tool.render_all(b)
        assert "generated_at" not in tool.results_text(a)

    def test_main_writes_then_check_detects_drift(self, tool, inputs, tmp_path, capsys):
        raw, gold, faith = inputs
        out = tmp_path / "reports"
        argv = ["--raw", str(raw), "--gold", str(gold), "--faithfulness", str(faith), "--out", str(out),
                "--resamples", "40"]
        assert tool.main(argv) == 0
        for rel in ("README.md", "manifest.json", "results.json", "explorer.html",
                    "full_4000/hallucination_abstention.md", "full_4000/f1_correctness.md", "full_4000/faithfulness.md",
                    "attempted/hallucination_abstention.md", "attempted/f1_correctness.md", "attempted/faithfulness.md"):
            assert (out / rel).exists(), rel
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert set(manifest["outputs_sha256"]) == {"results.json", "README.md", "explorer.html",
                                                   "full_4000/hallucination_abstention.md", "full_4000/f1_correctness.md",
                                                   "full_4000/faithfulness.md", "attempted/hallucination_abstention.md",
                                                   "attempted/f1_correctness.md", "attempted/faithfulness.md"}
        assert tool.main(argv + ["--check"]) == 0
        (out / "full_4000" / "faithfulness.md").write_text("drifted", encoding="utf-8")
        assert tool.main(argv + ["--check"]) == 1
        assert "faithfulness.md" in capsys.readouterr().out

    def test_dump_ids_matches_hashes(self, tool, inputs, tmp_path):
        raw, gold, faith = inputs
        out = tmp_path / "reports"
        dump = tmp_path / "ids.json"
        argv = ["--raw", str(raw), "--gold", str(gold), "--faithfulness", str(faith), "--out", str(out),
                "--resamples", "40", "--dump-ids", str(dump)]
        assert tool.main(argv) == 0
        ids = json.loads(dump.read_text(encoding="utf-8"))
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        for name, entry in res["cohorts"].items():
            assert ids[name]["n"] == entry["n"]
            assert tool.sha256_ids(ids[name]["ids"]) == entry["sha256"]


# ---------------------------------------------------------------------------
# Explorer display strings — Python renders, JavaScript only selects
# ---------------------------------------------------------------------------

class TestExplorerDisplayStrings:
    """The 2026-09-12 external audit found the explorer's own JavaScript rounding
    disagreeing with the shared Python rule in 135 cells (exact ties such as
    21/400 printed 5.3 instead of 5.2; the Math.round(x*10)/10 trick printed
    1238/4000 as 31.0 instead of 30.9). The repair moves every number string
    into Python; these tests pin that the page has nothing left to round."""

    def test_javascript_formats_no_numbers(self, tool):
        js = tool.EXPLORER_TEMPLATE.split("<script>")[1]
        for token in ("toFixed", "EPSILON", "pct1(", "rate1(", "fmtMetric("):
            assert token not in js, token
        assert js.count("Math.round(") == 1 and "function mix(" in js   # colour interpolation only

    def test_both_blobs_embedded_results_verbatim(self, results, tool):
        res, _ = results
        html = tool.render_all(res)["explorer.html"]
        r_blob = html.split("const R = ", 1)[1].split(";\nconst D = ", 1)[0]
        d_blob = html.split("const D = ", 1)[1].split(";   // display strings", 1)[0]
        assert json.loads(r_blob.replace("<\\/", "</")) == res                       # the page shows results.json itself
        assert json.loads(d_blob.replace("<\\/", "</")) == tool.display_strings(res)

    def test_display_strings_are_the_markdown_formatters(self, results, tool):
        res, _ = results
        D = tool.display_strings(res)
        for c, agg in res["full"]["overall"].items():
            for m in res["outcomes"]:
                assert D["full"]["overall"][c][m] == tool.cp(agg, m)
            assert D["full"]["overall"][c]["f1"] == tool.f1s(agg["f1_mean"])
        for pk, p in res["full"]["pairs"].items():
            o, s = p["overall"], D["full"]["pairs"][pk]["overall"]
            ent = o["hallucination"]
            assert s["stats"]["hallucination"]["delta"] == tool.pp(ent["delta"])
            assert s["stats"]["hallucination"]["ci"] == tool.ci_pp(ent["bootstrap"])
            assert s["stats"]["hallucination"]["p"] == tool.fmt_p(ent["bootstrap"]["p_value"])
            assert s["stats"]["hallucination"]["mcnemar_p"] == tool.fmt_p(ent["mcnemar"]["p_value"])
        for pk, p in res["attempted"]["pairs"].items():
            if p["overall"]["available"]:
                assert D["attempted"]["pairs"][pk]["overall"]["retained_share"] == tool.share(p["overall"]["retained_share"])
        n = res["cohorts"]["full"]["n"]
        for k, v in res["full"]["attempt_patterns"].items():
            assert D["full"]["attempt_patterns"][k] == tool.pct(v, n)
        for pk, t in res["full"]["transitions"].items():
            for ob, row in t["matrix"].items():
                for oa in res["outcomes"]:
                    assert D["full"]["transitions"][pk]["matrix"][ob][oa] == tool.pct(row[oa], sum(row.values()))
        for c, o in res["faithfulness"]["by_outcome"].items():
            gc = o["grounded"]["correct"]
            assert D["faithfulness"]["by_outcome"][c]["correct_mean"] == tool.fs(o["correct"]["mean"])
            assert D["faithfulness"]["by_outcome"][c]["ungrounded_correct_pct"] == tool.pct(gc["ungrounded"], gc["grounded"] + gc["ungrounded"])
        for c, v in res["faithfulness"]["own"].items():
            expect = {"mean": tool.fs(v["mean"])} if v["n_scoreable"] else None
            assert D["faithfulness"]["own"][c] == expect

    def test_display_tree_mirrors_every_path_the_page_reads(self, results, tool):
        res, _ = results
        D = tool.display_strings(res)
        for section in ("full", "attempted", "faithfulness"):
            key = "own" if section == "faithfulness" else "overall"
            assert set(D[section][key]) == set(res[section][key])
            assert set(D[section]["by_axis"]) == set(res[section]["by_axis"])
            for ax, groups in res[section]["by_axis"].items():
                assert set(D[section]["by_axis"][ax]) == set(groups)
                for g, cell in groups.items():
                    assert set(D[section]["by_axis"][ax][g]["configs"]) == set(cell["configs"])
            assert set(D[section]["pairs"]) == set(res[section]["pairs"])
        for pk, p in res["full"]["pairs"].items():
            for ax, groups in p["by_axis"].items():
                assert set(D["full"]["pairs"][pk]["by_axis"][ax]) == set(groups)
        assert set(D["full"]["transitions"]) == set(res["full"]["transitions"])
        assert set(D["full"]["attempt_patterns"]) == set(res["full"]["attempt_patterns"])
        assert set(D["attempted"]["all_four"]["by_axis"]) == set(res["attempted"]["all_four"]["by_axis"])
        assert set(D["faithfulness"]["all_three"]["by_axis"]) == set(res["faithfulness"]["all_three"]["by_axis"])
        for pk, p in res["faithfulness"]["pairs"].items():
            assert (D["faithfulness"]["pairs"][pk] is None) == (not p.get("available", True))

        def leaves(x):
            if isinstance(x, dict):
                for v in x.values():
                    yield from leaves(v)
            else:
                yield x
        assert all(v is None or isinstance(v, str) for v in leaves(D))   # nothing numeric is left for the page

    def test_every_display_string_equals_the_python_formatter(self, results, tool):
        """Exhaustive: walk results and the display tree in parallel and recompute every leaf."""
        res, _ = results
        D = tool.display_strings(res)
        outs = res["outcomes"]
        checked = 0

        def agg(a, s):
            nonlocal checked
            if not a or not a.get("n"):
                assert s is None
                return
            for m in outs:
                assert s[m] == tool.cp(a, m)
            assert s["f1"] == tool.f1s(a["f1_mean"])
            checked += len(outs) + 1

        def pair(o, s, full):
            nonlocal checked
            if not o.get("available", True):
                assert s is None
                return
            agg(o["a"], s["a"]); agg(o["b"], s["b"])
            if not full:
                assert s["retained_share"] == tool.share(o["retained_share"]); checked += 1
            for k in ("hallucination", "correct", "abstention", "f1"):
                if not o.get(k):
                    assert k not in s["stats"]
                    continue
                ent, st = o[k], s["stats"][k]
                bs, mc = ent.get("bootstrap"), ent.get("mcnemar")
                assert st["delta"] == tool.pp(ent["delta"])
                assert st["ci"] == tool.ci_pp(bs)
                assert st["p"] == (tool.fmt_p(bs["p_value"]) if bs else "—")
                assert st["mcnemar_p"] == (tool.fmt_p(mc["p_value"]) if mc else "—")
                checked += 4

        def fmean(v, s):
            nonlocal checked
            ns = v.get("n_scoreable", v.get("n")) if v else 0
            assert s == ({"mean": tool.fs(v["mean"])} if ns else None); checked += 1

        def fpair(o, s):
            nonlocal checked
            if not o.get("available", True):
                assert s is None
                return
            bs = o.get("bootstrap")
            assert s == {"mean_a": tool.fs(o["mean_a"]), "mean_b": tool.fs(o["mean_b"]), "delta": tool.dfs(o["delta"]),
                         "ci": tool.ci_f(bs), "p": tool.fmt_p(bs["p_value"]) if bs else "—"}
            checked += 5

        for sec, full in (("full", True), ("attempted", False)):
            for c, a in res[sec]["overall"].items():
                agg(a, D[sec]["overall"][c])
            for ax, groups in res[sec]["by_axis"].items():
                for g, cell in groups.items():
                    for c, a in cell["configs"].items():
                        agg(a, D[sec]["by_axis"][ax][g]["configs"][c])
            for pk, p in res[sec]["pairs"].items():
                pair(p["overall"], D[sec]["pairs"][pk]["overall"], full)
                for ax, groups in p["by_axis"].items():
                    for g, o in groups.items():
                        pair(o, D[sec]["pairs"][pk]["by_axis"][ax][g], full)
        four = res["attempted"]["all_four"]
        for c, a in four["overall"]["configs"].items():
            agg(a, D["attempted"]["all_four"]["overall"]["configs"][c])
        for ax, groups in four["by_axis"].items():
            for g, cell in groups.items():
                for c, a in cell["configs"].items():
                    agg(a, D["attempted"]["all_four"]["by_axis"][ax][g]["configs"][c])
        for pk, t in res["full"]["transitions"].items():
            for ob, row in t["matrix"].items():
                for oa in outs:
                    assert D["full"]["transitions"][pk]["matrix"][ob][oa] == tool.pct(row[oa], sum(row.values())); checked += 1
        n = res["cohorts"]["full"]["n"]
        for k, v in res["full"]["attempt_patterns"].items():
            assert D["full"]["attempt_patterns"][k] == tool.pct(v, n); checked += 1
        F = res["faithfulness"]
        for c, v in F["own"].items():
            fmean(v, D["faithfulness"]["own"][c])
        for ax, groups in F["by_axis"].items():
            for g, cell in groups.items():
                for c, v in cell["configs"].items():
                    fmean(v, D["faithfulness"]["by_axis"][ax][g]["configs"][c])
        for c, v in F["all_three"]["overall"]["configs"].items():
            fmean(v, D["faithfulness"]["all_three"]["overall"]["configs"][c])
        for ax, groups in F["all_three"]["by_axis"].items():
            for g, cell in groups.items():
                for c, v in cell["configs"].items():
                    fmean(v, D["faithfulness"]["all_three"]["by_axis"][ax][g]["configs"][c])
        for pk, p in F["pairs"].items():
            if not p.get("available", True):
                assert D["faithfulness"]["pairs"][pk] is None
                continue
            fpair(p["overall"], D["faithfulness"]["pairs"][pk]["overall"])
            for ax, groups in p["by_axis"].items():
                for g, o in groups.items():
                    fpair(o, D["faithfulness"]["pairs"][pk]["by_axis"][ax][g])
        for c, o in F["by_outcome"].items():
            ub = o.get("unpaired_bootstrap_correct_minus_hallucination")
            gc, gh = o["grounded"]["correct"], o["grounded"]["hallucination"]
            assert D["faithfulness"]["by_outcome"][c] == {
                "correct_mean": tool.fs(o["correct"]["mean"]), "hallucination_mean": tool.fs(o["hallucination"]["mean"]),
                "delta": tool.dfs(ub["delta"]) if ub else "—", "ci": tool.ci_f(ub) if ub else "—",
                "p": tool.fmt_p(ub["p_value"]) if ub else "—",
                "ungrounded_correct_pct": tool.pct(gc["ungrounded"], gc["grounded"] + gc["ungrounded"]),
                "ungrounded_hallucination_pct": tool.pct(gh["ungrounded"], gh["grounded"] + gh["ungrounded"])}
            checked += 7
        assert checked > 200, checked                       # the fixture exercises every view

        # and nothing else exists in D: the walk above visited every leaf
        def count_leaves(x):
            return sum(count_leaves(v) for v in x.values()) if isinstance(x, dict) else 1
        assert count_leaves(D) >= checked

    def test_audit_cases_follow_the_python_rule(self, tool):
        # Values the retired JavaScript printed as 5.3 / 31.0 / 9.5 / +2.3 in the 2026-09-12 audit
        assert tool.pct(21, 400) == "5.2"
        assert tool.pct(1238, 4000) == "30.9"
        assert tool.pct(378, 4000) == "9.4"
        assert tool.pp(0.0225) == "+2.2"       # TEST-4000 full C3 − C2, intersection stratum
        assert tool.share(0.3095) == "30.9"                 # all-four common attempts, 1238/4000
