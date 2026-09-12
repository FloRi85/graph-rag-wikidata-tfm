"""
Measure full stored context sizes and compare two runs by question ID.

Counts the exact context string supplied to the answering model, including
headers and bullet markers. No live retrieval or model call is needed.
The two inputs must contain identical question-ID sets and a stored context
for each retrieval configuration. See docs/reproduction.md.

By default both inputs are the sealed TEST-4000 run, reproducing its context
medians and C2-to-C3/C4 size ratios. Restore the sealed raw file first.
To compare different runs, pass their directories with --before and --after.

Run from repo root:
    venv/Scripts/python tools/compare_context_sizes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.context_format import context_words

BEFORE = (ROOT / "data" / "results" / "test_runs"
          / "20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw")
AFTER = BEFORE
OUT = ROOT / "output" / "context_sizes.json"

CONFIGS = ("rag", "graph_rag", "rerank")
LABELS = {"rag": "C2 RAG", "graph_rag": "C3 Graph-RAG", "rerank": "C4 Condense"}


def load_run(run_dir: Path) -> tuple[list[dict], str, str]:
    """Rows, the raw file's name and its sha256."""
    candidates = sorted(run_dir.glob("*_raw.json"))
    if not candidates:
        sys.exit(f"no *_raw.json in {run_dir}")
    raw_path = candidates[0]
    data = raw_path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    rows = parsed["results"] if isinstance(parsed, dict) and "results" in parsed else parsed
    return rows, raw_path.name, hashlib.sha256(data).hexdigest()


def measure(rows: list[dict], config: str) -> dict[str, int]:
    """{question_id: context words} for one config.

    Counts the FULL stored context -- header and bullets included -- because
    that is the string the answering LLM received. `context_words` is the same
    statistic `metrics.context_words` records, so a live measurement and an
    archived one remain comparable.
    """
    out: dict[str, int] = {}
    for row in rows:
        qid = row.get("id")
        capture = (row.get("contexts") or {}).get(config) or {}
        ctx = capture.get("context") if isinstance(capture, dict) else capture
        if qid and isinstance(ctx, str):
            out[qid] = context_words(ctx)
    return out


def _summary(values: list[int]) -> dict:
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "median": statistics.median(values), "min": min(values),
            "max": max(values)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, default=BEFORE)
    ap.add_argument("--after", type=Path, default=AFTER)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    before_rows, before_file, before_sha = load_run(args.before)
    after_rows, after_file, after_sha = load_run(args.after)

    # 🔴 EXACT COHORT EQUALITY, NOT "no questions were skipped". Two medians over
    # different question sets are not a before/after comparison, however close
    # the counts look -- the superseded measurement differed by one question and
    # nobody noticed. Duplicates are fatal for the same reason: a repeated id
    # silently reweights the median.
    before_ids = [r.get("id") for r in before_rows]
    after_ids = [r.get("id") for r in after_rows]
    for name, ids in (("before", before_ids), ("after", after_ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            sys.exit(f"{name}: duplicate question ids {sorted(dupes)[:5]}")
    if set(before_ids) != set(after_ids):
        only_b = sorted(set(before_ids) - set(after_ids))[:5]
        only_a = sorted(set(after_ids) - set(before_ids))[:5]
        sys.exit(f"cohorts differ — before-only {only_b}, after-only {only_a}")
    ids = sorted(set(before_ids))

    configs: dict[str, dict] = {}
    for config in CONFIGS:
        b, a = measure(before_rows, config), measure(after_rows, config)
        missing = [q for q in ids if q not in b or q not in a]
        if missing:
            sys.exit(f"{config}: {len(missing)} questions lack a stored context "
                     f"in one of the runs (first: {missing[:5]})")
        bv = [b[q] for q in ids]
        av = [a[q] for q in ids]
        configs[config] = {
            "label": LABELS[config],
            "before": _summary(bv),
            "after": _summary(av),
            "median_ratio": round(statistics.median(av) / statistics.median(bv), 3),
            # Per-id deltas, so a reader can check the shape rather than trusting
            # two medians. Paired by id, never by file order.
            "per_question_delta": {q: a[q] - b[q] for q in ids},
        }

    c2m = configs["rag"]["after"]["median"]
    gaps = {c: {"before": round(configs["rag"]["before"]["median"]
                               / configs[c]["before"]["median"], 2),
                "after": round(c2m / configs[c]["after"]["median"], 2)}
            for c in ("graph_rag", "rerank")}

    report = {
        "metric": "context words = len(context.split()) over the FULL stored "
                  "context string, header and bullet markers included — the "
                  "exact text the answering LLM received",
        "source": "the two immutable raw run files; no live retrieval",
        "paired_by": "question id",
        "n_questions": len(ids),
        "runs": {
            "before": {"dir": args.before.name, "raw_file": before_file,
                       "raw_sha256": before_sha},
            "after": {"dir": args.after.name, "raw_file": after_file,
                      "raw_sha256": after_sha},
        },
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "per_question_delta"}
                    for k, v in configs.items()},
        "c2_gap_multiples": gaps,
        "per_question_delta": {k: v["per_question_delta"] for k, v in configs.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("=" * 74)
    print(f"CONTEXT SIZE — {args.before.name}")
    print(f"            -> {args.after.name}")
    print("=" * 74)
    print(f"paired on {len(ids)} question ids; full stored contexts\n")
    print(f"{'Config':<16}{'before':>10}{'after':>10}{'ratio':>9}"
          f"{'gap to C2 before':>19}{'after':>8}")
    for c in CONFIGS:
        v = configs[c]
        g = gaps.get(c)
        print(f"{v['label']:<16}{v['before']['median']:>10.0f}"
              f"{v['after']['median']:>10.0f}{v['median_ratio']:>9.2f}"
              + (f"{g['before']:>18.1f}x{g['after']:>7.1f}x" if g else " " * 26))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print("\nContext sizes include headers and bullet markers, matching the stored")
    print("strings supplied to the answering model.")


if __name__ == "__main__":
    main()
