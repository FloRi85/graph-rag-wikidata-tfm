"""
The grounded×outcome decomposition (2×2) and the type-plausibility cut.

Declared 2026-08-19 in the original private protocol's execution log;
provenance is summarised in docs/reproduction.md. Grounded := faithfulness
score >= 0.5 (at least HALF of
parsed claims supported — a score of exactly 0.5 counts as grounded; the
pre-registration's "majority" gloss was imprecise, corrected by dated append
2026-08-21; the threshold itself is unchanged); records with score None
(`unparsed_verdicts`) are
EXCLUDED, never counted as 0.0, and reported separately. The two declared
rates: (a) the confabulation-analogue rate — the share of HALLUCINATION
outcomes that are ungrounded, over scored hallucinations per config; (b) each
2×2 cell over scored attempted answers per config. Undefined for C1 by
construction, mirroring faithfulness itself. §5 error-analysis decomposition,
analogy not diagnosis — NOT a headline metric.

The type-plausibility cut (second table) covers HALLUCINATION outcomes on
numerical/date/boolean questions only — entity/string are vacuously
type-plausible — and asks whether the wrong answer contains at least one
parseable value of the gold answer type, using the SCORER'S OWN parsers.

⚠️ OFFLINE BY CONSTRUCTION: reads a stored run raw file plus its stored
faithfulness JSON (`data/analysis/faithfulness_{run_stem}.json`). Zero API
calls. Outcomes are re-derived through the canonical recipe and a mismatch
with the outcome stored in a faithfulness record is FATAL — the two files
must describe the same run under the same scorer.

Usage (from repo root):
    venv/Scripts/python tools/grounding_outcome_matrix.py              # reference-v8
    venv/Scripts/python tools/grounding_outcome_matrix.py RUN_raw.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import (_CONFIG_LABELS, per_question_scores,     # noqa: E402
                              CORRECT, HALLUCINATION,
                              _extract_numbers, _extract_years,
                              _extract_full_dates, _parse_boolean)
from src.eval.gold_source import resolve_gold_source                   # noqa: E402

ANALYSIS_DIR = ROOT / "data" / "analysis"
DEFAULT_RUN = (ROOT / "data" / "results" / "dev_runs"
               / "20260818_1113_nvidia-llama-3-3-nemotron-super-49b-v1_reference-v8"
               / "20260818_1113_reference-v8_raw.json")

# Same tuple as faithfulness.RETRIEVAL_CONFIGS, re-declared like
# emit_judge_validation does: this tool is offline, and importing the
# faithfulness module pulls in the LLM client stack for three strings.
RETRIEVAL_CONFIGS = ("rag", "graph_rag", "rerank")

# The pre-registered grounding threshold. Do not make this a CLI flag: the
# definition is fixed in the protocol (see docs/reproduction.md); a knob invites
# picking the threshold by its result.
GROUNDED_THRESHOLD = 0.5

# Answer types whose "contains a value of the right type" question is
# non-vacuous; entity/string are excluded as declared.
_TYPED = ("numerical", "date", "boolean")


def _sha_read(path: Path) -> tuple[object, str]:
    blob = path.read_bytes()
    return json.loads(blob.decode("utf-8")), hashlib.sha256(blob).hexdigest()


def _type_plausible(answer: str, answer_type: str) -> bool:
    if answer_type == "numerical":
        return bool(_extract_numbers(answer))
    if answer_type == "date":
        return bool(_extract_years(answer)) or bool(_extract_full_dates(answer))
    if answer_type == "boolean":
        return _parse_boolean(answer) is not None
    raise ValueError(f"type-plausibility is not defined for {answer_type!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN,
                    help="run *_raw.json (default: the reference-v8 run of record)")
    ap.add_argument("--questions", default=None,
                    help="explicit raw Mintaka file for gold (else gold_source resolves)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: data/analysis/grounding_outcome_matrix_{stem}.json)")
    args = ap.parse_args()

    run_path = args.run
    if not run_path.exists():
        sys.exit(f"run file not found: {run_path}")
    faith_path = ANALYSIS_DIR / f"faithfulness_{run_path.stem}.json"
    if not faith_path.exists():
        sys.exit(f"no faithfulness scores for this run: {faith_path} does not exist\n"
                 f"(produce them with tools/score_faithfulness.py first)")

    rows, run_sha = _sha_read(run_path)
    if not isinstance(rows, list):
        sys.exit(f"{run_path.name}: not a run_eval raw file (expected a list of rows)")
    faith, faith_sha = _sha_read(faith_path)

    golds, qbi, _ = resolve_gold_source({r.get("id", "") for r in rows},
                                        explicit=args.questions, verbose=False)
    # Default all-or-nothing exclusion: a within-run table needs one shared
    # denominator across configs (contrast compare_runs.py, which pairs one
    # config across two runs and uses per-config exclusion).
    scored = per_question_scores(rows, qbi, golds=golds)
    answers = {r["id"]: r.get("answers", {}) for r in rows}

    # ---- join faithfulness records onto re-derived outcomes -----------------
    per_config: dict[str, dict] = {}
    for config in RETRIEVAL_CONFIGS:
        per_config[config] = {
            "cells": {CORRECT: {"grounded": 0, "ungrounded": 0},
                      HALLUCINATION: {"grounded": 0, "ungrounded": 0}},
            "unscored": 0,             # score is None — excluded, never 0.0
            "excluded_by_run": 0,      # question dropped by all-or-nothing exclusion
            "confab_ids": [],          # the confabulation-analogue cell, by id
        }

    for rec in faith:
        config, qid = rec["config"], rec["id"]
        if config not in per_config:
            sys.exit(f"faithfulness record for unknown config {config!r} (id {qid})")
        bucket = per_config[config]
        if qid not in scored.get(config, {}):
            bucket["excluded_by_run"] += 1
            continue
        outcome = scored[config][qid]["outcome"]
        if outcome != rec["outcome"]:
            sys.exit(f"OUTCOME MISMATCH on ({config}, {qid}): run scores "
                     f"{outcome!r} but the faithfulness record says "
                     f"{rec['outcome']!r}. The faithfulness file does not "
                     f"describe this run under this scorer — refusing to "
                     f"produce numbers from a broken join.")
        if rec["score"] is None:
            bucket["unscored"] += 1
            continue
        grounded = "grounded" if rec["score"] >= GROUNDED_THRESHOLD else "ungrounded"
        bucket["cells"][outcome][grounded] += 1
        if outcome == HALLUCINATION and grounded == "ungrounded":
            bucket["confab_ids"].append(qid)

    # ---- the declared rates -------------------------------------------------
    for config, b in per_config.items():
        cells = b["cells"]
        n_scored = sum(cells[o][g] for o in cells for g in cells[o])
        n_hall = sum(cells[HALLUCINATION].values())
        b["n_scored_attempted"] = n_scored
        b["n_scored_hallucinations"] = n_hall
        b["confabulation_rate"] = (
            cells[HALLUCINATION]["ungrounded"] / n_hall if n_hall else None)
        b["cell_shares"] = {
            f"{o}_{g}": (cells[o][g] / n_scored if n_scored else None)
            for o in cells for g in cells[o]}

    # ---- type-plausibility of hallucinations (from the run alone) -----------
    type_plaus: dict[str, dict] = {}
    for config in RETRIEVAL_CONFIGS + ("base_llm_abstain",):
        counts = {t: {"plausible": 0, "total": 0} for t in _TYPED}
        for qid, rec in scored.get(config, {}).items():
            if rec["outcome"] != HALLUCINATION or rec["answer_type"] not in _TYPED:
                continue
            counts[rec["answer_type"]]["total"] += 1
            if _type_plausible(answers[qid].get(config, ""), rec["answer_type"]):
                counts[rec["answer_type"]]["plausible"] += 1
        type_plaus[config] = counts

    # ---- print --------------------------------------------------------------
    print("=" * 74)
    print(f"GROUNDED×OUTCOME MATRIX — {run_path.stem}")
    print(f"grounded := faithfulness score >= {GROUNDED_THRESHOLD} "
          f"(pre-registered 2026-08-19; score-None excluded)")
    print("=" * 74)
    for config in RETRIEVAL_CONFIGS:
        b = per_config[config]
        c = b["cells"]
        print(f"\n{_CONFIG_LABELS.get(config, config)}  "
              f"(scored attempted {b['n_scored_attempted']}, "
              f"unscored {b['unscored']}, run-excluded {b['excluded_by_run']})")
        print(f"{'':>16}{'grounded':>10}{'ungrounded':>12}")
        for outcome in (CORRECT, HALLUCINATION):
            print(f"{outcome:>16}{c[outcome]['grounded']:>10}"
                  f"{c[outcome]['ungrounded']:>12}")
        rate = b["confabulation_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "n/a (no hallucinations)"
        print(f"  confabulation-analogue rate "
              f"(ungrounded share of {b['n_scored_hallucinations']} "
              f"hallucinations): {rate_s}")

    print("\n" + "=" * 74)
    print("TYPE-PLAUSIBILITY of hallucinations (numerical/date/boolean only;")
    print("entity/string excluded as vacuously type-plausible — declared)")
    print("=" * 74)
    header = f"{'config':<22}" + "".join(f"{t:>18}" for t in _TYPED)
    print(header)
    for config, counts in type_plaus.items():
        cells = []
        for t in _TYPED:
            n, tot = counts[t]["plausible"], counts[t]["total"]
            cells.append(f"{n}/{tot}" if tot else "—")
        print(f"{config:<22}" + "".join(f"{c:>18}" for c in cells))

    # Pooled across the three typed answer types, per config — the range the
    # thesis quotes. Without this the pooled figure was hand-added from the
    # per-type cells, which is exactly what a sanctioned source exists to end.
    print(f"\n{'pooled (num+date+bool)':<26}{'plausible/total':>18}{'rate':>9}")
    type_plaus_pooled: dict[str, dict] = {}
    for config, counts in type_plaus.items():
        n = sum(counts[t]["plausible"] for t in _TYPED)
        tot = sum(counts[t]["total"] for t in _TYPED)
        rate = n / tot if tot else None
        type_plaus_pooled[config] = {"plausible": n, "total": tot, "rate": rate}
        print(f"{config:<26}{f'{n}/{tot}' if tot else '—':>18}"
              + (f"{rate:>9.1%}" if rate is not None else f"{'—':>9}"))

    # ---- persist ------------------------------------------------------------
    out_path = args.out or (ANALYSIS_DIR
                            / f"grounding_outcome_matrix_{run_path.stem}.json")
    payload = {
        "metric": ("2x2 outcome (correct/hallucination) x grounding "
                   "(faithfulness >= threshold) per retrieval config, plus "
                   "type-plausibility of hallucinations"),
        "pre_registered": "docs/reference_run_plan.md, execution log 2026-08-19",
        "grounded_threshold": GROUNDED_THRESHOLD,
        "run": {"file": run_path.name, "sha256": run_sha},
        "faithfulness": {"file": faith_path.name, "sha256": faith_sha},
        "configs": per_config,
        "type_plausibility": type_plaus,
        "type_plausibility_pooled": type_plaus_pooled,
    }
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
