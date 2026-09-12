"""
Annotate the faithfulness-judge validation set one case at a time, per keypress.

`tools/emit_judge_validation.py` emits ~50 blind cases with three blank fields
each (`supported` / `claim_valid` per claim, `decomposition_complete` per
answer). Filling them by hand in the markdown works but is friction — 124
claims of yes/no lines to hunt through an editor — and the JSON sidecar is
worse (the same fields inside nested claim objects, syntax errors waiting).
This interactive workflow makes those annotations easier while preserving
the validation rules:

  - BLIND: the judge's verdicts and scores are never shown — they live in the
    faithfulness JSON and meet these annotations only in `--agreement`.
  - saves after EVERY answer, atomically, so a closed terminal at case 30
    costs nothing; re-running resumes at the first unfinished case.
  - WHY A HUMAN AND NOT A MODEL: these labels validate the LLM judge. Filling
    them in with a model would be the model validating itself.

The frozen-judge-pass guard is honoured: the sidecar's recorded sha256 of the
faithfulness JSON is re-checked at startup, so annotating against a silently
regenerated judge pass fails loudly (same rule as the emitter and --agreement).

Contexts are read from the run's raw file (the sidecar deliberately does not
duplicate them). C3/C4 contexts are short and print whole; C2's ~7,500-word
articles print as a head-excerpt with 'c' to page the full text.

Usage (from repo root, in a REAL terminal — it reads from stdin):
    venv\\Scripts\\python tools/review_judge_validation.py
    venv\\Scripts\\python tools/review_judge_validation.py --sidecar data/analysis/judge_validation_..._raw.json
    venv\\Scripts\\python tools/review_judge_validation.py --redo --config graph_rag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.metrics import _CONFIG_LABELS  # noqa: E402

ANALYSIS_DIR = ROOT / "data" / "analysis"

# Contexts at or under this many characters print in full. C3 (~280 words) and
# C4 (~37 words) always fit; C2's articles do not, and skimming 7,500 words
# defeats the judgement — the head plus targeted paging is the honest view.
FULL_CONTEXT_CHARS = 4000
HEAD_CHARS = 1500
WRAP = 96


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sidecar(path: Path) -> tuple[dict, list[dict]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, list):  # pre-guard shape, kept readable
        return {"cases": blob}, blob
    return blob, blob.get("cases", [])


def check_frozen_judge(sidecar_blob: dict, sidecar_path: Path) -> None:
    """Refuse to annotate against a regenerated judge pass (D7 guard)."""
    recorded = sidecar_blob.get("faithfulness_sha256")
    if not recorded:
        return  # pre-guard sidecar; nothing to check against
    run_stem = sidecar_path.stem.removeprefix("judge_validation_")
    p = ANALYSIS_DIR / f"faithfulness_{run_stem}.json"
    if not p.exists():
        sys.exit(f"{p.name} not found — the faithfulness pass these annotations "
                 f"label is missing. Restore it from git; do not re-score.")
    live = hashlib.sha256(p.read_bytes()).hexdigest()
    if live != recorded:
        sys.exit(f"{p.name} has CHANGED since this validation set was emitted "
                 f"(sha256 mismatch). The annotation labels one specific "
                 f"decomposition — restore the committed JSON from git rather "
                 f"than re-scoring.")


def find_run_file(sidecar_path: Path) -> Path:
    """The run raw file the sidecar's name points at, under dev_runs/."""
    run_stem = sidecar_path.stem.removeprefix("judge_validation_")
    hits = list((ROOT / "data" / "results" / "dev_runs").glob(f"*/{run_stem}.json"))
    if len(hits) != 1:
        sys.exit(f"expected exactly one dev_runs/*/{run_stem}.json, found "
                 f"{len(hits)} — pass --run explicitly.")
    return hits[0]


def load_contexts(run_path: Path) -> dict[tuple[str, str], str]:
    """(config, id) -> the context string handed to the answering LLM.
    Same reading as emit_judge_validation.load_contexts."""
    rows = json.loads(run_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        for cfg, capture in (row.get("contexts") or {}).items():
            ctx = capture if isinstance(capture, str) else (capture or {}).get("context")
            if isinstance(ctx, str) and ctx:
                out[(cfg, row.get("id", ""))] = ctx
    return out


# ---------------------------------------------------------------------------
# Persistence — atomic, after every answer so interrupted reviews can resume
# ---------------------------------------------------------------------------

def save(blob: dict, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def case_done(case: dict) -> bool:
    if (case.get("decomposition_complete") or "").strip().lower() not in ("yes", "no"):
        return False
    return all((cl.get("supported") or "").strip().lower() in ("yes", "no")
               and (cl.get("claim_valid") or "").strip().lower() in ("yes", "no")
               for cl in case.get("claims", []))


def progress(cases: list[dict]) -> tuple[int, int]:
    return sum(1 for c in cases if case_done(c)), len(cases)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(line, WRAP, initial_indent=indent,
                                   subsequent_indent=indent)
                     if len(line) > WRAP else indent + line
                     for line in text.splitlines())


def show_context(context: str, full: bool = False) -> None:
    if not context:
        print("    (no stored context)")
        return
    if full or len(context) <= FULL_CONTEXT_CHARS:
        print(_wrap(context))
    else:
        print(_wrap(context[:HEAD_CHARS]))
        print(f"    … [{len(context):,} chars total — 'c' shows everything]")


def render(case: dict, n: int, total: int, context: str) -> None:
    label = _CONFIG_LABELS.get(case["config"], case["config"])
    print("\n" + "=" * 78)
    print(f"  case {n}/{total}   ·   {label}   ·   id {case['id']}   ·   "
          f"outcome {case.get('outcome', '?')}")
    print("=" * 78)
    print(f"\n  QUESTION   {case['question']}")
    print(f"  ANSWER     {case['answer']!r}")
    print(f"\n  ┌─ CONTEXT the model was given "
          "(judge against THIS, not against the truth) ─┐")
    show_context(context)
    print()


# ---------------------------------------------------------------------------
# Annotation loop
# ---------------------------------------------------------------------------

YN = {"y": "yes", "n": "no"}
HELP = """
  For each claim, two calls:
    supported     does the CONTEXT entail the claim?   (not: is it true)
    claim_valid   does the ANSWER actually assert it?  (no = decomposer invented/distorted it)
  Then once per case:
    decomposition_complete   do the claims cover everything the answer asserts?

  Keys:  y = yes   n = no
         b = go BACK one prompt (re-answer; works across claims within the case)
         c = show the full context     ? = reprint this help
         t = add a note                s = skip this case for now
         q = save and quit
"""

BACK = object()  # sentinel: the reviewer wants the previous prompt


def ask(prompt: str, case: dict, blob: dict, path: Path, context: str):
    """One y/n with the shared escape keys. Returns 'yes'/'no', BACK to
    revisit the previous prompt, or None to skip the case; quits the process
    on q (after saving)."""
    while True:
        choice = input(prompt).strip().lower()
        if choice in YN:
            return YN[choice]
        if choice == "b":
            return BACK
        if choice == "c":
            print()
            show_context(context, full=True)
            print()
            continue
        if choice == "?":
            print(HELP)
            continue
        if choice == "t":
            case["notes"] = (case.get("notes", "") + " " +
                             input("  note > ").strip()).strip()
            save(blob, path)
            continue
        if choice == "s":
            return None
        if choice == "q":
            save(blob, path)
            done, total = progress(blob.get("cases", []))
            print(f"\nSaved. {done}/{total} cases fully annotated. "
                  f"Re-run to pick up where you stopped.")
            sys.exit(0)
        print("  ? use y, n, b, c, t, s, ? or q")


def annotate(blob: dict, cases: list[dict], path: Path,
             contexts: dict, redo: bool, only_config: str | None) -> None:
    todo = [c for c in cases
            if (redo or not case_done(c))
            and (not only_config or c["config"] == only_config)]
    if not todo:
        done, total = progress(cases)
        print(f"Nothing to annotate — {done}/{total} cases complete. "
              f"Use --redo to revisit.")
        return

    print(f"\n{len(todo)} case(s) to annotate. BLIND: the judge's verdicts "
          f"are not shown.")
    print(HELP)

    for i, case in enumerate(todo, 1):
        ctx = contexts.get((case["config"], case["id"]), "")
        render(case, i, len(todo), ctx)

        # The case as a flat list of prompts, so 'b' can walk backwards across
        # claim boundaries. Each step is (target dict, field key).
        steps: list[tuple[dict, str]] = []
        for cl in case["claims"]:
            steps.append((cl, "supported"))
            steps.append((cl, "claim_valid"))
        steps.append((case, "decomposition_complete"))

        # Resume at the first unanswered prompt (a case abandoned mid-way with
        # 's' keeps its partial answers; re-asking them all would punish the
        # skip). --redo re-walks from the top so every field is re-askable.
        j = 0
        if not redo:
            while j < len(steps) and \
                    (steps[j][0].get(steps[j][1]) or "").strip().lower() in ("yes", "no"):
                j = j + 1
            j = min(j, len(steps) - 1) if j else 0

        last_claim_shown = None
        while j < len(steps):
            target, field = steps[j]
            if field == "decomposition_complete":
                prompt = "  decomposition_complete? [y/n] > "
            else:
                k = case["claims"].index(target) + 1
                if last_claim_shown != k:
                    print(f"  claim {k}/{len(case['claims'])}: {target['text']}")
                    last_claim_shown = k
                prompt = (f"    supported?   [y/n] > " if field == "supported"
                          else f"    claim_valid? [y/n] > ")
            v = ask(prompt, case, blob, path, ctx)
            if v is None:               # skip case; partial answers persist
                break
            if v is BACK:
                if j == 0:
                    print("  (already at the first prompt of this case)")
                else:
                    j -= 1
                    prev_target, prev_field = steps[j]
                    if prev_field != "decomposition_complete":
                        last_claim_shown = None  # re-show the claim text
                    print(f"  (back — {prev_field} was "
                          f"{prev_target.get(prev_field)!r}, answer again)")
                continue
            target[field] = v
            save(blob, path)
            j += 1

    done, total = progress(cases)
    print(f"\nDone. {done}/{total} cases fully annotated in {path.name}")
    if done == total:
        run_stem = path.stem.removeprefix("judge_validation_")
        print("\nNext:")
        print(f"  venv\\Scripts\\python tools/emit_judge_validation.py "
              f"{run_stem}.json --agreement")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="judge_validation_*.json (default: the single one "
                         "under data/analysis/)")
    ap.add_argument("--run", type=Path, default=None,
                    help="the run's *_raw.json (default: located from the "
                         "sidecar's name under dev_runs/)")
    ap.add_argument("--config", default=None,
                    help="annotate one config only (rag / graph_rag / rerank)")
    ap.add_argument("--redo", action="store_true",
                    help="revisit cases already fully annotated")
    args = ap.parse_args()

    sidecar = args.sidecar
    if sidecar is None:
        hits = sorted(ANALYSIS_DIR.glob("judge_validation_*.json"))
        if len(hits) != 1:
            sys.exit(f"found {len(hits)} judge_validation_*.json under "
                     f"data/analysis/ — pass --sidecar explicitly.")
        sidecar = hits[0]
    if not sidecar.exists():
        sys.exit(f"not found: {sidecar}")

    blob, cases = load_sidecar(sidecar)
    check_frozen_judge(blob, sidecar)
    contexts = load_contexts(args.run or find_run_file(sidecar))

    done, total = progress(cases)
    print(f"{sidecar.name}: {total} cases, {done} fully annotated")
    annotate(blob, cases, sidecar, contexts, args.redo, args.config)


if __name__ == "__main__":
    main()
