"""
Unit tests for tools/emit_judge_validation.py — the D7 judge-validation set.

Pins the three properties the tool exists for: the draw is stratified across
configs AND judge-score bands (the previous validation set failed with 11 of
12 cases carrying one label), hand annotations survive a re-emit verbatim,
and agreement is computed by answer with half-annotated or all-invalid cases
excluded rather than silently scored.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_emit_judge_validation.py -v
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _load():
    spec = importlib.util.spec_from_file_location(
        "emit_judge_validation",
        os.path.join(_ROOT, "tools", "emit_judge_validation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ejv = _load()


def _rec(i, config, score, n_claims=2, outcome="correct"):
    return {
        "id": f"q{i:03d}", "config": config, "outcome": outcome,
        "question": f"question {i}", "answer": f"answer {i}",
        "claims": [f"claim {i}.{j}" for j in range(n_claims)],
        "score": score, "status": "ok",
    }


def _pool():
    """Every config × band populated, plus records the draw must skip."""
    out, i = [], 0
    for cfg in ("rag", "graph_rag", "rerank"):
        for score in (0.0, 0.5, 1.0):
            for _ in range(6):
                out.append(_rec(i, cfg, score)); i += 1
    out.append(_rec(900, "rag", None))                 # unscored — ineligible
    out.append({**_rec(901, "rag", 1.0), "claims": []})  # claimless — ineligible
    return out


class TestDraw:
    def test_stratified_across_configs_and_bands(self):
        picked = ejv.draw_sample(_pool(), 25, seed=1)
        assert len(picked) == 25
        by_cfg = {c: sum(1 for r in picked if r["config"] == c)
                  for c in ("rag", "graph_rag", "rerank")}
        assert all(8 <= v <= 9 for v in by_cfg.values()), by_cfg
        # ⚠️ The property the old validation set lacked: every judge-score
        # band is represented, so agreement is measured where the judge can
        # actually be wrong.
        bands = {ejv.score_band(r["score"]) for r in picked}
        assert bands == {"full", "partial", "zero"}

    def test_deterministic_and_order_independent(self):
        pool = _pool()
        a = ejv.draw_sample(pool, 25, seed=7)
        b = ejv.draw_sample(list(reversed(pool)), 25, seed=7)
        assert [(r["config"], r["id"]) for r in a] == \
               [(r["config"], r["id"]) for r in b]

    def test_unscored_and_claimless_records_never_qualify(self):
        picked = ejv.draw_sample(_pool(), 100, seed=1)
        ids = {r["id"] for r in picked}
        assert "q900" not in ids and "q901" not in ids


class TestMerge:
    def test_hand_annotations_survive_a_re_emit_verbatim(self):
        done = ejv.blank_case(_rec(1, "rag", 0.5))
        done["claims"][0]["supported"] = "yes"
        done["claims"][0]["claim_valid"] = "yes"
        done["decomposition_complete"] = "no"

        fresh_same = ejv.blank_case(_rec(1, "rag", 0.5))   # same key, blank
        fresh_new = ejv.blank_case(_rec(2, "graph_rag", 1.0))
        merged = ejv.merge([done], [fresh_same, fresh_new])

        assert len(merged) == 2
        assert merged[0] is done                            # kept, not rebuilt
        assert merged[0]["claims"][0]["supported"] == "yes"


class TestAgreement:
    def test_human_score_is_none_until_fully_annotated(self):
        case = ejv.blank_case(_rec(1, "rag", 0.5))
        case["claims"][0].update(supported="yes", claim_valid="yes")
        assert ejv.human_score(case) is None                # claim 2 blank

    def test_invalid_claims_leave_the_denominator(self):
        case = ejv.blank_case(_rec(1, "rag", 0.5, n_claims=3))
        case["claims"][0].update(supported="yes", claim_valid="yes")
        case["claims"][1].update(supported="no", claim_valid="yes")
        case["claims"][2].update(supported="no", claim_valid="no")  # excluded
        assert ejv.human_score(case) == 0.5

    def test_all_claims_invalid_is_no_measurement(self):
        case = ejv.blank_case(_rec(1, "rag", 0.5))
        for cl in case["claims"]:
            cl.update(supported="no", claim_valid="no")
        assert ejv.human_score(case) is None

    def test_blind_emission_carries_no_judge_fields(self):
        case = ejv.blank_case(_rec(1, "rag", 0.5))
        assert "score" not in case and "verdicts" not in case
        assert all(set(cl) == {"text", "supported", "claim_valid"}
                   for cl in case["claims"])


class TestFrozenJudgePassGuard:
    """The sidecar pins the sha256 of the faithfulness pass it was drawn from."""

    def test_read_sidecar_accepts_both_shapes(self, tmp_path):
        import json
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps([{"id": "q1", "config": "rag", "claims": []}]))
        sha, cases = ejv.read_sidecar(legacy)
        assert sha is None and len(cases) == 1

        current = tmp_path / "current.json"
        current.write_text(json.dumps(
            {"faithfulness_sha256": "abc123", "cases": [{"id": "q2"}]}))
        sha, cases = ejv.read_sidecar(current)
        assert sha == "abc123" and cases == [{"id": "q2"}]

    def test_has_annotations_detects_any_hand_filled_field(self):
        blank = ejv.blank_case(_rec(1, "rag", 0.5))
        assert not ejv.has_annotations([blank])

        sup = ejv.blank_case(_rec(1, "rag", 0.5))
        sup["claims"][0]["supported"] = "yes"
        assert ejv.has_annotations([sup])

        val = ejv.blank_case(_rec(1, "rag", 0.5))
        val["claims"][1]["claim_valid"] = "no"
        assert ejv.has_annotations([val])

        dec = ejv.blank_case(_rec(1, "rag", 0.5))
        dec["decomposition_complete"] = "yes"
        assert ejv.has_annotations([dec])

    def test_whitespace_is_not_an_annotation(self):
        case = ejv.blank_case(_rec(1, "rag", 0.5))
        case["claims"][0]["supported"] = "  "
        assert not ejv.has_annotations([case])

    def test_load_faithfulness_returns_the_hash_of_the_bytes_read(self, tmp_path, monkeypatch):
        import hashlib, json
        out_dir = tmp_path
        monkeypatch.setattr(ejv, "OUT_DIR", out_dir)
        run = tmp_path / "myrun_raw.json"
        faith = out_dir / "faithfulness_myrun_raw.json"
        payload = [{"id": "q1", "config": "rag", "score": 1.0, "claims": ["c"]}]
        faith.write_text(json.dumps(payload), encoding="utf-8")

        records, sha = ejv.load_faithfulness(run)
        assert records == payload
        assert sha == hashlib.sha256(faith.read_bytes()).hexdigest()
