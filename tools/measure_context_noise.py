"""
Quantify the two context-quality observations from the 2026-08-19 faithfulness
annotation session: subject==object lines in C3 contexts, and how much they
correlate with C3's behaviour. THE SANCTIONED SOURCE for the numbers quoted in
the SS5/SSLimitations context-noise paragraph.

Reads STORED contexts from an immutable raw run file -- no live retrieval, no
API calls -- so the numbers cannot drift with the cache (same discipline as
`tools/compare_context_sizes.py`).

What it measures, per run file:

1. **Tautology lines** -- rendered C3 lines whose [subject] label equals the
   value label case-insensitively (qualifier parenthetical stripped). On
   reference-v8: 135/5,918 top-k lines (2.3%), across 70/200 questions,
   0 exact duplicate lines.

   +/- A LABEL-BASED DROP FILTER WOULD BE UNSAFE, which is why these numbers
   exist instead of a fix: several such lines are REAL facts between two
   DISTINCT entities sharing a label ("[Gone With The Wind] based on: Gone
   with the Wind" -- film -> novel; likewise `derivative work`, `supplement
   to`). Distinguishing self-description (`title`) from cross-entity facts
   needs a QID comparison, not a label one.

2. **Impact proxies** on the affected questions: C3 abstention (exact
   sentinel) affected vs unaffected, whether the gold shares the ambiguous
   label (score cannot be hurt there), and whether the source entity's own
   ranked `description:` line made the top-k (subject-side disambiguation).
   On reference-v8: affected questions abstain LESS (45.7% vs 53.8%), gold
   shares the label on 14/70, description line present on 48/70.

The C2 half of the observation needs no measurement: mid-sentence chunk
boundaries are BY CONSTRUCTION (`wikipedia._windows`, 300-word windows /
50-word overlap, no sentence alignment), and the overlap deliberately
duplicates 50 words so a boundary-spanning fact is not lost.

Decision recorded 2026-08-19: no retrieval change before TEST-4,000.
See docs/reproduction.md for the retained protocol and run provenance.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_RAW = (
    Path(__file__).resolve().parents[1]
    / "data" / "results" / "dev_runs"
    / "20260818_1113_nvidia-llama-3-3-nemotron-super-49b-v1_reference-v8"
    / "20260818_1113_reference-v8_raw.json"
)

SENTINEL = "The answer is not in the context."

# One rendered statement line: bullet, [subject], property, value. The
# qualifier parenthetical is stripped before comparing labels, so
# "[X] award received: X (point in time: 1994)" still counts as a collision.
_LINE = re.compile(r"^\s*•?\s*\[(?P<subj>[^\]]+)\]\s+(?P<prop>[^:]+):\s+(?P<val>.*)$")


def _context(row: dict, config: str) -> str:
    ctx = (row.get("contexts") or {}).get(config) or ""
    if isinstance(ctx, dict):
        ctx = ctx.get("context") or ctx.get("text") or ""
    return ctx


def _strip_qualifiers(val: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", val).strip()


def measure(rows: list[dict], config: str = "graph_rag") -> dict:
    total_lines = taut_lines = dup_lines = 0
    affected: list[dict] = []
    unaffected: list[dict] = []
    gold_shares_label = 0
    desc_present_affected = 0
    examples: list[str] = []

    for row in rows:
        ctx = _context(row, config)
        if not ctx:
            continue
        lines = [l for l in ctx.splitlines() if _LINE.match(l)]
        total_lines += len(lines)
        dup_lines += sum(n - 1 for n in Counter(l.strip() for l in lines).values() if n > 1)

        taut_labels: set[str] = set()
        has_desc = False
        for l in lines:
            m = _LINE.match(l)
            if m.group("prop").strip() == "description":
                has_desc = True
            val = _strip_qualifiers(m.group("val"))
            if m.group("subj").strip().lower() == val.lower():
                taut_labels.add(val.lower())
                taut_lines += 1
                if len(examples) < 8:
                    examples.append(l.strip())

        ans = ((row.get("answers") or {}).get(config) or "").strip()
        rec = {"id": row.get("id"), "abstained": ans.startswith(SENTINEL)}
        if taut_labels:
            affected.append(rec)
            if has_desc:
                desc_present_affected += 1
            golds = [str(g).lower() for g in (row.get("answer_forms") or [])]
            golds.append(str(row.get("expected", "")).lower())
            if any(t in g or g in t for t in taut_labels for g in golds if g):
                gold_shares_label += 1
        else:
            unaffected.append(rec)

    def abst(rs: list[dict]) -> float:
        return 100 * sum(r["abstained"] for r in rs) / max(len(rs), 1)

    return {
        "config": config,
        "total_lines": total_lines,
        "tautology_lines": taut_lines,
        "tautology_pct": round(100 * taut_lines / max(total_lines, 1), 1),
        "duplicate_lines": dup_lines,
        "questions_affected": len(affected),
        "questions_unaffected": len(unaffected),
        "abstention_affected_pct": round(abst(affected), 1),
        "abstention_unaffected_pct": round(abst(unaffected), 1),
        "gold_shares_ambiguous_label": gold_shares_label,
        "description_line_in_topk_on_affected": desc_present_affected,
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("raw", nargs="?", type=Path, default=DEFAULT_RAW,
                    help="run raw file (default: the reference-v8 run of record)")
    ap.add_argument("--config", default="graph_rag",
                    help="context key to measure (default: graph_rag)")
    args = ap.parse_args()

    data = json.loads(args.raw.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    out = measure(rows, args.config)

    print(f"run: {args.raw.name}  ({len(rows)} rows)")
    print(f"config: {out['config']}")
    print(f"parsed statement lines: {out['total_lines']}")
    print(f"subject==object lines: {out['tautology_lines']} ({out['tautology_pct']}%)"
          f" in {out['questions_affected']} questions")
    print(f"exact duplicate lines: {out['duplicate_lines']}")
    print(f"abstention affected vs unaffected: {out['abstention_affected_pct']}%"
          f" vs {out['abstention_unaffected_pct']}%")
    print(f"gold shares an ambiguous label: {out['gold_shares_ambiguous_label']}"
          f"/{out['questions_affected']}")
    print(f"own description line in top-k on affected: "
          f"{out['description_line_in_topk_on_affected']}/{out['questions_affected']}")
    print("examples:")
    for e in out["examples"]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
