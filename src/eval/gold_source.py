"""
Which raw Mintaka file supplies the gold for a stored run.

Shared by result-reporting and faithfulness tools so every analysis resolves
the gold source using the same rule. Existence alone is insufficient: the file
must cover every question being scored.

⭐ COVERAGE IS CHECKED, NOT ASSUMED. A candidate is accepted only if it holds
every question id being scored. `metrics.per_question_scores` raises on a
missing id rather than falling back to the row, so a wrong file now fails loudly
at the boundary instead of quietly producing a table.
"""

from __future__ import annotations

from pathlib import Path

from src.eval.parse_questions import NotRawMintakaError, load_questions
from src.eval.parse_answers import load_answers

ROOT = Path(__file__).resolve().parent.parent.parent
QDIR = ROOT / "data" / "questions"

# Tried in order when nothing is named explicitly. FULL SPLITS, not samples: a
# sample can be deleted (`mintaka_sample_100.json` was) or renamed, while the
# splits are the downloaded corpus and do not move.
FALLBACK_SPLITS = ("mintaka_dev_raw.json", "mintaka_test_raw.json")


def resolve_gold_source(row_ids: set[str], *, explicit: str | None = None,
                        recorded: str | None = None, verbose: bool = True):
    """
    Return `(golds, entities_by_id, path)` for the run covering `row_ids`.

    `golds`
        question id -> `parse_answers.GoldAnswer`. Drives scoring.

    `entities_by_id`
        question id -> {entity_names, entity_labels}. The either-or guard in
        `metrics.entity_match` needs the question's entity surface forms, and
        result rows do not store them. `entity_names` holds the MENTIONS and
        `entity_labels` the canonical labels, matching the schema the flattened
        loader used before it was retired.

    Resolution order: `explicit` (a --questions flag), then `recorded` (a sweep
    file's own `sample` field), then the dev and test splits.

    Raises SystemExit naming everything it tried, because the alternative --
    scoring against whatever was found -- is the failure this exists to prevent.
    """
    candidates: list[Path] = []
    for cand in (explicit, recorded):
        if cand:
            p = Path(cand)
            candidates.append(p if p.exists() else QDIR / Path(cand).name)
    candidates += [QDIR / name for name in FALLBACK_SPLITS]

    tried: list[str] = []
    for p in candidates:
        if not p.exists():
            tried.append(f"{p.name} (not found)")
            continue
        try:
            golds = load_answers(p)
        except NotRawMintakaError:
            # A flattened file from before the raw-record migration. Named
            # rather than skipped silently: it is usually the file the caller
            # meant, and "not raw Mintaka" is the actionable message.
            tried.append(f"{p.name} (not raw Mintaka)")
            continue
        missing = row_ids - set(golds)
        if missing:
            tried.append(f"{p.name} (missing {len(missing)} of {len(row_ids)} ids)")
            continue
        entities = {
            q.id: {"entity_names": q.entity_mentions, "entity_labels": q.entity_names}
            for q in load_questions(p)
        }
        if verbose:
            print(f"gold answers: {p.name}")
        return golds, entities, p

    raise SystemExit(
        "\n🔴 NO QUESTIONS FILE COVERS THIS RUN.\n"
        f"   tried: {', '.join(tried) or '(nothing found)'}\n\n"
        "   Scoring needs the raw Mintaka split the run was drawn from.\n"
        "   Pass it explicitly:  --questions data/questions/mintaka_dev_raw.json\n")
