"""
Emit the claim-level validation set for the faithfulness judge (D7), and score it.

The validation checks the judge against human annotations of the same
claim-level support construct. The protocol (D7; see docs/reproduction.md)
specifies:

  - attempted answers stratified across C2/C3/C4 and judge score bands,
  - hand labels at CLAIM level: is each decomposed claim supported by the
    context, is the claim actually asserted by the answer (`claim_valid`), and
    does the claim set cover the answer (`decomposition_complete`) — so an
    omitted or distorted claim is detectable, not just a mis-verdict,
  - agreement computed BY ANSWER, never by claim: claims cluster within
    answers (~2.3 each), so claim-level counts overstate precision.

⚠️ THE ANNOTATION IS BLIND. The markdown shows the claims but NOT the judge's
verdicts or score, so the human cannot anchor on the thing being validated.
The judge's numbers stay in the faithfulness JSON and meet the annotations
only in `--agreement`.

⚠️ THE SAMPLE IS STRATIFIED BY JUDGE SCORE BAND (1.0 / partial / 0.0) within
each config, round-robin, seeded. That is what prevents a rerun of the
"11 of one label" failure: every band the judge actually uses is represented,
so agreement is measured where the judge can be wrong, not only where it is
confident.

⚠️ ANNOTATIONS MERGE, NEVER OVERWRITE:
hand work is the one thing in this repo that cannot be regenerated. Re-running
the emitter keeps every filled field of an existing sidecar and only adds
cases that are new.

No API calls. Reads the stored faithfulness JSON and the run's raw file.

Usage (from repo root):
    venv\\Scripts\\python tools/emit_judge_validation.py <run>_raw.json
    venv\\Scripts\\python tools/emit_judge_validation.py <run>_raw.json --n 25 --seed 7
    venv\\Scripts\\python tools/emit_judge_validation.py <run>_raw.json --agreement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import _CONFIG_LABELS

RETRIEVAL_CONFIGS = ("rag", "graph_rag", "rerank")
OUT_DIR = ROOT / "data" / "analysis"

# Judge score bands. Stratifying on them is the point — see module docstring.
BAND_FULL, BAND_PARTIAL, BAND_ZERO = "full", "partial", "zero"


def score_band(score: float) -> str:
    if score >= 1.0:
        return BAND_FULL
    if score <= 0.0:
        return BAND_ZERO
    return BAND_PARTIAL


def load_faithfulness(run_path: Path) -> tuple[list[dict], str]:
    """(records, sha256 of the file they came from).

    The hash is the frozen-judge-pass guard: the annotation labels THIS pass's
    decomposition, and `score_faithfulness` overwrites its output on a re-run
    (`temperature=0` is not deterministic here, so a re-run is a different
    decomposition). The sha256 is stamped into the sidecar at emission and
    checked on every later invocation, so a regenerated judge pass fails
    loudly instead of being compared against silently.
    """
    p = OUT_DIR / f"faithfulness_{run_path.stem}.json"
    if not p.exists():
        sys.exit(f"{p.name} not found — run tools/score_faithfulness.py on this "
                 f"run first; the validation set is drawn from its claims.")
    blob = p.read_bytes()
    return json.loads(blob.decode("utf-8")), hashlib.sha256(blob).hexdigest()


def read_sidecar(path: Path) -> tuple[str | None, list[dict]]:
    """(recorded faithfulness sha256 or None, cases). Accepts the pre-guard
    bare-list shape so the file emitted on 2026-08-18 still reads."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return None, blob
    return blob.get("faithfulness_sha256"), blob.get("cases", [])


def has_annotations(cases: list[dict]) -> bool:
    """True once any hand-filled field exists — the point of no return."""
    for c in cases:
        if (c.get("decomposition_complete") or "").strip():
            return True
        for cl in c.get("claims", []):
            if (cl.get("supported") or "").strip() or \
               (cl.get("claim_valid") or "").strip():
                return True
    return False


def load_contexts(run_path: Path) -> dict[tuple[str, str], str]:
    """(config, id) -> the context string handed to the answering LLM."""
    rows = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        sys.exit(f"{run_path.name}: expected a run_eval result list.")
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        for cfg in RETRIEVAL_CONFIGS:
            capture = (row.get("contexts") or {}).get(cfg) or {}
            ctx = capture if isinstance(capture, str) else capture.get("context")
            if isinstance(ctx, str) and ctx:
                out[(cfg, row.get("id", ""))] = ctx
    return out


def draw_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """
    ~n answers, stratified config × judge-score-band, round-robin, seeded.

    Only records the judge actually scored qualify (`score is not None` and at
    least one claim): an unscored answer validates nothing about the judge.
    """
    eligible = [r for r in records
                if r.get("config") in RETRIEVAL_CONFIGS
                and r.get("score") is not None
                and r.get("claims")]
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in eligible:
        buckets.setdefault((r["config"], score_band(r["score"])), []).append(r)
    # One rng per bucket, seeded by (seed, key): the draw is then independent
    # of both input order and bucket-creation order.
    for key, b in buckets.items():
        b.sort(key=lambda r: r["id"])
        random.Random(f"{seed}:{key}").shuffle(b)

    # Round-robin over configs, and over bands within a config, so no config
    # and no band is exhausted first by construction.
    order = [(cfg, band) for band in (BAND_ZERO, BAND_PARTIAL, BAND_FULL)
             for cfg in RETRIEVAL_CONFIGS]
    picked: list[dict] = []
    while len(picked) < n and any(buckets.get(k) for k in order):
        for key in order:
            if len(picked) >= n:
                break
            if buckets.get(key):
                picked.append(buckets[key].pop())
    return sorted(picked, key=lambda r: (r["config"], r["id"]))


def blank_case(rec: dict) -> dict:
    """The sidecar entry a human fills in. No judge verdicts — blind."""
    return {
        "id": rec["id"],
        "config": rec["config"],
        "outcome": rec["outcome"],
        "question": rec["question"],
        "answer": rec["answer"],
        "claims": [{"text": c, "supported": "", "claim_valid": ""}
                   for c in rec["claims"]],
        "decomposition_complete": "",
        "notes": "",
    }


def merge(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Keep every existing case verbatim (hand work); append only new keys."""
    have = {(c["config"], c["id"]) for c in existing}
    return existing + [c for c in fresh if (c["config"], c["id"]) not in have]


def render_markdown(cases: list[dict], contexts: dict, run_name: str) -> str:
    L = [f"# Faithfulness-judge validation — {run_name}", "",
         "Hand-label each claim against the CONTEXT (not against the truth):",
         "`supported:` yes|no — does the context entail the claim?",
         "`claim_valid:` yes|no — does the ANSWER actually assert this claim?",
         "`decomposition_complete:` yes|no — do the claims cover everything the answer asserts?",
         "",
         "⚠️ Blind on purpose: the judge's verdicts are not shown here.",
         "Fill the JSON sidecar (same fields) or this file — the sidecar is what",
         "`--agreement` reads.", ""]
    for i, c in enumerate(cases, 1):
        label = _CONFIG_LABELS.get(c["config"], c["config"])
        L += [f"---", "",
              f"## case {i} — {label} · `{c['id']}` · outcome {c['outcome']}", "",
              f"**Question:** {c['question']}",
              f"**Answer:** {c['answer']}", ""]
        for j, cl in enumerate(c["claims"], 1):
            L += [f"- claim {j}: {cl['text']}",
                  f"  - supported: {cl['supported']}",
                  f"  - claim_valid: {cl['claim_valid']}"]
        L += ["", f"decomposition_complete: {c['decomposition_complete']}", ""]
        ctx = contexts.get((c["config"], c["id"]), "")
        L += ["<details><summary>context</summary>", "", "```",
              ctx or "(no stored context)", "```", "</details>", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def human_score(case: dict) -> float | None:
    """
    Supported fraction over the claims the human accepted as valid.

    None until every claim in the case carries both labels — a half-annotated
    case must not enter the comparison looking finished. If the human rejects
    EVERY claim (`claim_valid: no` throughout), the decomposition measured
    nothing and the case is excluded the same way.
    """
    vals = []
    for cl in case["claims"]:
        sup = (cl.get("supported") or "").strip().lower()
        val = (cl.get("claim_valid") or "").strip().lower()
        if sup not in ("yes", "no") or val not in ("yes", "no"):
            return None
        if val == "yes":
            vals.append(1.0 if sup == "yes" else 0.0)
    return (sum(vals) / len(vals)) if vals else None


def report_agreement(cases: list[dict], records: list[dict]) -> None:
    by_key = {(r["config"], r["id"]): r for r in records}
    pairs, pending, degenerate = [], 0, 0
    invalid_claims = incomplete_decomp = 0
    for c in cases:
        for cl in c["claims"]:
            if (cl.get("claim_valid") or "").strip().lower() == "no":
                invalid_claims += 1
        if (c.get("decomposition_complete") or "").strip().lower() == "no":
            incomplete_decomp += 1
        h = human_score(c)
        if h is None:
            has_any = any((cl.get("supported") or "").strip() for cl in c["claims"])
            all_invalid = all((cl.get("claim_valid") or "").strip().lower() == "no"
                              for cl in c["claims"])
            if has_any and all_invalid:
                degenerate += 1
            else:
                pending += 1
            continue
        rec = by_key.get((c["config"], c["id"]))
        if rec and rec.get("score") is not None:
            pairs.append((c, h, rec["score"]))

    print(f"\n  annotated: {len(pairs)}  pending: {pending}  "
          f"all-claims-invalid: {degenerate}")
    print(f"  claims rejected as invalid: {invalid_claims}   "
          f"answers with incomplete decomposition: {incomplete_decomp}")
    if not pairs:
        print("\n  agreement: nothing fully annotated yet — the judge is "
              "UNVALIDATED and its numbers must not be quoted as validated.")
        return

    # BY ANSWER, per D7. Exact = same score to 2dp; the mean |diff| is the
    # magnitude view for the write-up.
    exact = sum(1 for _, h, j in pairs if abs(h - j) < 0.005)
    mad = sum(abs(h - j) for _, h, j in pairs) / len(pairs)
    print(f"\n  AGREEMENT BY ANSWER (n={len(pairs)}):")
    print(f"    exact:      {exact}/{len(pairs)}  ({100 * exact / len(pairs):.0f}%)")
    print(f"    mean |Δ|:   {mad:.3f}")
    for c, h, j in pairs:
        if abs(h - j) >= 0.005:
            print(f"    disagrees  {c['config']:10} {c['id']}  human {h:.2f}  judge {j:.2f}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run's *_raw.json")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260818,
                    help="declared before drawing; changing it re-draws the sample")
    ap.add_argument("--agreement", action="store_true",
                    help="score the annotated sidecar instead of emitting")
    args = ap.parse_args()

    records, faith_sha = load_faithfulness(args.run)
    sidecar = OUT_DIR / f"judge_validation_{args.run.stem}.json"

    if args.agreement:
        if not sidecar.exists():
            sys.exit(f"{sidecar.name} not found — emit it first (no --agreement).")
        recorded_sha, cases = read_sidecar(sidecar)
        if recorded_sha != faith_sha:
            sys.exit(
                f"🔴 FROZEN-JUDGE-PASS GUARD: the faithfulness JSON is not the one "
                f"this validation set was drawn from.\n"
                f"   sidecar records: {recorded_sha or '(none — pre-guard sidecar; re-emit to stamp it)'}\n"
                f"   file on disk:    {faith_sha}\n"
                f"   The annotations label a specific decomposition; comparing them "
                f"against a regenerated judge pass would be silently invalid.\n"
                f"   Restore the committed faithfulness JSON (git) rather than "
                f"re-scoring the run.")
        report_agreement(cases, records)
        return

    existing_sha, existing = (read_sidecar(sidecar) if sidecar.exists()
                              else (None, []))
    if existing and existing_sha != faith_sha:
        if has_annotations(existing):
            sys.exit(
                f"🔴 FROZEN-JUDGE-PASS GUARD: {sidecar.name} carries hand "
                f"annotations drawn from a DIFFERENT faithfulness pass "
                f"(recorded {existing_sha or '(none)'}, on disk {faith_sha}).\n"
                f"   Merging across passes would mix decompositions. Restore the "
                f"committed faithfulness JSON instead of re-scoring the run.")
        # Blank sidecar from another pass: its cases reference claims that no
        # longer exist, and nothing hand-made is lost by rebuilding.
        print(f"  note: discarding blank sidecar from a different judge pass "
              f"({existing_sha or 'no recorded hash'}).")
        existing = []

    fresh = [blank_case(r) for r in draw_sample(records, args.n, args.seed)]
    cases = merge(existing, fresh)
    kept = len(cases) - len([c for c in fresh
                             if (c["config"], c["id"])
                             not in {(e["config"], e["id"]) for e in existing}])

    contexts = load_contexts(args.run)
    md = OUT_DIR / f"judge_validation_{args.run.stem}.md"
    sidecar.write_text(json.dumps(
        {"faithfulness_sha256": faith_sha, "cases": cases},
        indent=1, ensure_ascii=False), encoding="utf-8")
    md.write_text(render_markdown(cases, contexts, args.run.stem),
                  encoding="utf-8")

    n_claims = sum(len(c["claims"]) for c in cases)
    by_cfg = {cfg: sum(1 for c in cases if c["config"] == cfg)
              for cfg in RETRIEVAL_CONFIGS}
    print(f"  cases: {len(cases)} ({kept} pre-existing kept verbatim)  "
          f"claims: {n_claims}")
    print(f"  per config: " + "  ".join(f"{k} {v}" for k, v in by_cfg.items()))
    print(f"  written: {sidecar.relative_to(ROOT)}")
    print(f"           {md.relative_to(ROOT)}")
    print(f"\n  annotate, then: venv/Scripts/python tools/emit_judge_validation.py "
          f"{args.run.name} --agreement")


if __name__ == "__main__":
    main()
