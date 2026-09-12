# TEST-4,000 metric reports — reading guide

A reproducible supplement documenting the sealed TEST run of the frozen configuration across four metrics, two populations and the dataset's three native attributes. Everything is computed offline from the sealed answers, the benchmark annotations and the stored judge output by `tools/build_metric_reports.py`; nothing is re-run, re-judged or re-retrieved. The thesis quotes only what its arguments need.

## Files

| file | what it is |
|---|---|
| `results.json` | every number, count, delta, interval and cohort membership hash, unrounded; timestamp-free — the record |
| `manifest.json` | generation time, input/code/output SHA-256s, command line — the only volatile file |
| `explorer.html` | self-contained interactive view of `results.json` (select population × metric × attribute × pair; colour scales); no network; no scoring or statistical inference is performed there, and every printed number string is pre-rendered in Python by the formatters of these reports |
| `full_4000/hallucination_abstention.md` | hallucination and abstention over all questions; six pairs on identical id sets; outcome-transition matrices; attempt patterns |
| `full_4000/f1_correctness.md` | correct % and F1 over all questions; six pairs; threshold sensitivity |
| `full_4000/faithfulness.md` | conditional faithfulness with the scoreable share of the benchmark; coverage accounting |
| `attempted/hallucination_abstention.md` | hallucination among own attempts; six pairs on common attempts; all-four common set |
| `attempted/f1_correctness.md` | F1 and correct % among own attempts; pairs on common attempts; all-four common set |
| `attempted/faithfulness.md` | scoreability among attempts; pairs on common scoreable attempts; faithfulness by outcome |

## Three independent choices behind every table

1. **Metric.** Benchmark hallucination (confident disagreement with the 2021 Mintaka gold — a factuality proxy), operational correctness (the scorer's CORRECT outcome at threshold 0.5), token/set F1 (0–100), or answer-to-supplied-context faithfulness (0–1, judge score). Abstention, Other and coverage are companions, never a fifth report.
2. **Population.** `full_4000`: every question, pairs on the identical id set. `attempted`: each configuration's own attempts (CORRECT or HALLUCINATION; Other excluded) for overviews, and the INTERSECTION of two attempt sets for pairs, with both means recomputed there. Faithfulness: scoreable attempts (attempted and judge-scored), pairs on the common scoreable set; C1 is N/A.
3. **Dataset attribute.** Answer type (`answer.answerType`, 5 groups), question complexity (`complexityType`, 9), topic category (`category`, 8). Three separate marginal partitions of the same questions — never crossed in the reports, never pooled as independent observations. Group order is the parser vocabulary order.

## Units and rounding

- Count-based percentages use the shared divide-first rule (`tools/document_run._pct1`: `round(count / n * 100, 1)`), so a printed digit here equals the digit in the run's `report.md`. F1 prints as mean × 100, one decimal; faithfulness means print with three decimals. Deltas are A − B on the unrounded values.
- `†` marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero); `—` marks a value that does not exist (C1 faithfulness, an empty denominator).
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted) and exact McNemar on discordant pairs for binary outcomes; unpaired bootstrap only for the disjoint correct-vs-hallucination faithfulness contrast. Bootstrap p below the Monte Carlo resolution prints as `<.001`. Only OVERALL pair rows carry tests; every per-group pair delta is descriptive. Where a persisted sanctioned file already analysed the same population, the number here reproduces it (`results.json` → `precedent_checks`).

## The faithfulness exception (both folders)

There is no four-configuration, denominator-4,000 faithfulness mean. C1 has no context (N/A, not zero). C2–C4 means are conditional on a scoreable attempt; abstentions, Other and judge-unscored attempts are excluded, not zeroed, and the reasons are accounted for separately. The own-scoreable means are identical in the two folders because the conditional population is the same; the folders differ in what the scoreable SHARE is taken over (benchmark vs attempts). Pair means need both configurations attempted and scored on the same question; each score still judges its answer against its own supplied context.

## Dataset composition (counts only)

Answer type × complexity over the 4,000 test questions — shown so the near-deterministic crossings (`count` → numerical, `yesno` → boolean) and the one split class (`comparative`: entity vs boolean) are visible. Not a stratification axis.

| complexity \ answer type | entity | numerical | boolean | date | string | n |
|---|---:|---:|---:|---:|---:|---:|
| generic | 582 | 100 | 1 | 113 | 4 | 800 |
| intersection | 399 | 0 | 0 | 1 | 0 | 400 |
| count | 0 | 400 | 0 | 0 | 0 | 400 |
| comparative | 225 | 1 | 171 | 0 | 3 | 400 |
| yesno | 0 | 0 | 400 | 0 | 0 | 400 |
| ordinal | 311 | 3 | 0 | 86 | 0 | 400 |
| multihop | 229 | 111 | 1 | 58 | 1 | 400 |
| difference | 358 | 36 | 0 | 6 | 0 | 400 |
| superlative | 395 | 4 | 0 | 1 | 0 | 400 |
| **total** | 2499 | 655 | 573 | 265 | 8 | 4000 |

Topic categories are exactly balanced across complexity classes in Mintaka (every category has the same complexity proportions), so that crossing carries no information and is not shown.

## Cohorts

Every population is a named cohort. Membership hashes (SHA-256 of the sorted question IDs) are retained in `results.json`; `tools/build_metric_reports.py --dump-ids PATH` writes the ID lists.

| cohort | n |
|---|---:|
| full | 4000 |
| own_attempted:C1 | 3484 |
| own_attempted:C2 | 2227 |
| own_attempted:C3 | 1707 |
| own_attempted:C4 | 1962 |
| common_attempted:C2-C1 | 2166 |
| common_attempted:C3-C1 | 1651 |
| common_attempted:C4-C1 | 1864 |
| common_attempted:C3-C2 | 1371 |
| common_attempted:C4-C2 | 1441 |
| common_attempted:C4-C3 | 1552 |
| common_attempted:all_four | 1238 |
| own_scoreable:C2 | 2187 |
| own_scoreable:C3 | 1674 |
| own_scoreable:C4 | 1928 |
| common_scoreable:C3-C2 | 1324 |
| common_scoreable:C4-C2 | 1394 |
| common_scoreable:C4-C3 | 1498 |
| common_scoreable:all_three | 1194 |

## Regeneration

```
venv/Scripts/python tools/build_metric_reports.py
venv/Scripts/python tools/build_metric_reports.py --check   # regenerate in memory and compare; exit 1 on any difference
```

Inputs (hashes in `manifest.json`): the sealed answer file (untracked, 215 MB; hash pinned in `data/results/test_runs/README.md`), `data/questions/mintaka_test_raw.json`, and `data/analysis/faithfulness_20260819_2017_mintaka_test_raw_raw.json`. Tests: `tests/unit/test_build_metric_reports.py`.
