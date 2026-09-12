"""
Build a stratified sample of Mintaka questions, balanced across complexity types.

The raw split files are sorted by category, so a naive head-slice is
~92% movies. This script draws an even number of questions from each of the
nine complexity types (generic, count, comparative, multihop, ...) using a
fixed seed for reproducibility.

⚠️ Defaults to the DEV split (all tuning happens on dev; test is spent once).
Sampling from the sealed TEST split requires naming it via --src, and the
output filename is then forced to contain "test".

Use --exclude to draw a sample DISJOINT from questions already used for tuning
(reporting numbers on questions that were read during tuning would be a
best-of-many-configurations score on fitted data):

    ... --n 300 --exclude data/questions/<earlier_sample>.json \\
                --out data/questions/<new_sample>.json

Usage (from repo root):
    venv\\Scripts\\python tools/sample_mintaka.py --n 200
    venv\\Scripts\\python tools/sample_mintaka.py --n 200 --out data/questions/mintaka_sample_dev_200.json

The DEV-200 sample shipped with the repository (`mintaka_sample_dev_200.json`)
is the output of this script with the default seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.parse_questions import load_question_ids

# ⚠️ Default source is the DEV split (since 2026-08-18). The protocol is
# "all tuning happens on dev; test is spent once"; a sampler that defaulted to
# TEST once wrote a test-drawn sample under a neutral name, which the old
# filename guard in run_eval treated as a dev file. Sampling from test is still
# possible via --src, but the output name is then FORCED to carry "test"
# (run_eval also checks question ids, so the name is belt to those braces).
SRC = ROOT / "data" / "questions" / "mintaka_dev_raw.json"
TEST_RAW = ROOT / "data" / "questions" / "mintaka_test_raw.json"
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Total questions to sample")
    parser.add_argument("--src", default=str(SRC), help="Source questions JSON")
    parser.add_argument("--out", default=None, help="Output path (default derives from --n)")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="FILE",
                        help="Question file(s) whose ids must NOT appear in the sample "
                             "(e.g. an earlier tuning sample drawn from the same split)")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing output file")
    args = parser.parse_args()

    # A sample drawn from the SEALED split must say so in its filename. The
    # check is by source identity here and by question id in run_eval — a
    # neutral name is the only thing this guard can still lose to, so it is
    # refused outright rather than warned about.
    src_is_test = TEST_RAW.exists() and Path(args.src).resolve() == TEST_RAW.resolve()
    default_name = f"mintaka_sample_{'test_' if src_is_test else ''}{args.n}.json"
    out_path = Path(args.out) if args.out else ROOT / "data" / "questions" / default_name
    if src_is_test and "test" not in out_path.stem.lower():
        sys.exit(f"Refusing to write a TEST-split sample to {out_path.name}: the "
                 f"filename must contain 'test' so nothing downstream can "
                 f"mistake it for a dev file.")

    # Silently overwriting a sample would destroy the record of which questions a
    # past run was tuned on — and that record is what makes disjointness checkable.
    if out_path.exists() and not args.force:
        sys.exit(f"Refusing to overwrite existing {out_path} (pass --force to replace it)")

    # Raw records, verbatim, in and out. `load_question_ids` is the only parser
    # entry point used here, and only to read an --exclude file's ids -- it
    # accepts a bare id list or any record shape carrying an `id`, so an old
    # flattened sample can still say WHICH questions it selected.
    questions = json.loads(Path(args.src).read_text(encoding="utf-8"))

    excluded_ids: set[str] = set()
    for path in args.exclude:
        excluded_ids |= load_question_ids(path)
    if excluded_ids:
        before = len(questions)
        questions = [q for q in questions if q.get("id") not in excluded_ids]
        print(f"Excluded {before - len(questions)} of {before} source questions "
              f"({len(excluded_ids)} ids from {len(args.exclude)} file(s))")

    # Bucket by complexity. RAW key (`complexityType`), because these are raw
    # records -- the flattened `complexity` no longer exists here.
    buckets: dict[str, list] = defaultdict(list)
    for q in questions:
        buckets[q.get("complexityType", "unknown")].append(q)

    complexities = sorted(buckets)
    base = args.n // len(complexities)
    remainder = args.n - base * len(complexities)

    rng = random.Random(SEED)
    sample: list = []
    # Distribute the remainder to the largest buckets for stability
    order = sorted(complexities, key=lambda c: len(buckets[c]), reverse=True)
    extra = set(order[:remainder])

    for complexity in complexities:
        take = base + (1 if complexity in extra else 0)
        pool = buckets[complexity]
        take = min(take, len(pool))
        sample.extend(rng.sample(pool, take))

    rng.shuffle(sample)

    out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    # Report
    from collections import Counter
    print(f"Wrote {len(sample)} questions to {out_path}")
    if len(sample) < args.n:
        print(f"  ! asked for {args.n}; a complexity bucket ran out after exclusions")
    print("complexity:", dict(Counter(q.get("complexityType") for q in sample)))
    print("category:  ", dict(Counter(q.get("category") for q in sample)))

    # Verify the guarantee rather than trusting the filter above — a silent
    # overlap here is exactly the leakage this flag exists to prevent.
    if excluded_ids:
        leaked = {q["id"] for q in sample} & excluded_ids
        print(f"disjointness: {'OK - 0 overlap' if not leaked else f'FAILED - {len(leaked)} leaked'}")
        if leaked:
            sys.exit(1)


if __name__ == "__main__":
    main()
