# Faithfulness — full benchmark view (conditional on a scoreable attempt)

## 1. Definition and reading guide

- **Faithfulness** = the stored answer-level judge score (0–1, RAGAS-style: share of the answer's claims entailed by the context that configuration was supplied). Means are macro-averages of answer scores, never total supported claims over total claims.
- **There is no four-configuration, denominator-4,000 faithfulness mean in this experiment.** C1 receives no context: N/A, not zero, in every table and every pair involving it. C2–C4 means are conditional on a **scoreable attempt**: the answer was attempted (CORRECT or HALLUCINATION) and the judge returned a usable score.
- Abstentions, Other and judge-unscored attempts are EXCLUDED from every mean, never assigned zero; the reasons are separated in the coverage accounting. `partial_score` is never used as a substitute.
- This folder reports scoreable share relative to the FULL benchmark; the means themselves are identical to `../attempted/faithfulness.md` because the conditional population is the same — the two folders differ only in what the share is taken over.
- Pair rows use the COMMON scoreable set (both attempted, both scored, same question). Each score still evaluates its answer against ITS OWN supplied context, not a shared one.
- Judge contract-audit flags (degenerate verdict lines) are retained under the existing availability policy; the sensitivity means without them are recorded, not reported.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview

| Config | Benchmark n | Attempted n | Scoreable n | Unscored attempts | Scoreable % of benchmark | Mean faithfulness (conditional) |
|---|---:|---:|---:|---:|---:|---:|
| C1 Base LLM | 4000 | 3484 | N/A | N/A | N/A | N/A — no context |
| C2 RAG | 4000 | 2227 | 2187 | 40 | 54.7 | 0.723 |
| C3 Graph-RAG | 4000 | 1707 | 1674 | 33 | 41.9 | 0.616 |
| C4 Graph-RAG + condensing | 4000 | 1962 | 1928 | 34 | 48.2 | 0.859 |

Contract-audit flags (faithfulness_contract_20260819_2017_mintaka_test_raw_raw.json): 25 scoreable records flagged (C2 24, C3 0, C4 1), retained. Means excluding them: C2 0.724, C3 0.616, C4 0.859.

## 3. Dataset stratification

### 3.1 By answer type

Cells: scoreable n (share of the group's benchmark n) / mean. C1: N/A.

| Group | n | C1 | C2 scoreable (%) / mean | C3 scoreable (%) / mean | C4 scoreable (%) / mean |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | N/A | 1231 (49.3%) / 0.777 | 895 (35.8%) / 0.552 | 1032 (41.3%) / 0.838 |
| numerical | 655 | N/A | 322 (49.2%) / 0.558 | 254 (38.8%) / 0.468 | 324 (49.5%) / 0.850 |
| boolean | 573 | N/A | 447 (78.0%) / 0.639 | 376 (65.6%) / 0.780 | 406 (70.9%) / 0.894 |
| date | 265 | N/A | 180 (67.9%) / 0.850 | 141 (53.2%) / 0.845 | 158 (59.6%) / 0.926 |
| string | 8† | N/A | 7† (87.5%) / 1.000 | 8† (100.0%) / 0.750 | 8† (100.0%) / 0.821 |

### 3.2 By question complexity

Cells: scoreable n (share of the group's benchmark n) / mean. C1: N/A.

| Group | n | C1 | C2 scoreable (%) / mean | C3 scoreable (%) / mean | C4 scoreable (%) / mean |
|---|---:|---:|---:|---:|---:|
| generic | 800 | N/A | 560 (70.0%) / 0.839 | 454 (56.8%) / 0.713 | 474 (59.2%) / 0.887 |
| intersection | 400 | N/A | 219 (54.8%) / 0.806 | 162 (40.5%) / 0.463 | 177 (44.2%) / 0.824 |
| count | 400 | N/A | 227 (56.8%) / 0.522 | 187 (46.8%) / 0.430 | 221 (55.2%) / 0.850 |
| comparative | 400 | N/A | 324 (81.0%) / 0.647 | 230 (57.5%) / 0.713 | 236 (59.0%) / 0.825 |
| yesno | 400 | N/A | 307 (76.8%) / 0.657 | 257 (64.2%) / 0.776 | 284 (71.0%) / 0.904 |
| ordinal | 400 | N/A | 214 (53.5%) / 0.793 | 147 (36.8%) / 0.622 | 173 (43.2%) / 0.920 |
| multihop | 400 | N/A | 106 (26.5%) / 0.790 | 63 (15.8%) / 0.267 | 132 (33.0%) / 0.836 |
| difference | 400 | N/A | 125 (31.2%) / 0.707 | 114 (28.5%) / 0.477 | 153 (38.2%) / 0.771 |
| superlative | 400 | N/A | 105 (26.2%) / 0.599 | 60 (15.0%) / 0.431 | 78 (19.5%) / 0.803 |

### 3.3 By topic category

Cells: scoreable n (share of the group's benchmark n) / mean. C1: N/A.

| Group | n | C1 | C2 scoreable (%) / mean | C3 scoreable (%) / mean | C4 scoreable (%) / mean |
|---|---:|---:|---:|---:|---:|
| history | 500 | N/A | 267 (53.4%) / 0.732 | 272 (54.4%) / 0.577 | 277 (55.4%) / 0.871 |
| movies | 500 | N/A | 301 (60.2%) / 0.756 | 222 (44.4%) / 0.666 | 270 (54.0%) / 0.871 |
| music | 500 | N/A | 281 (56.2%) / 0.771 | 161 (32.2%) / 0.661 | 195 (39.0%) / 0.888 |
| videogames | 500 | N/A | 297 (59.4%) / 0.780 | 215 (43.0%) / 0.606 | 254 (50.8%) / 0.865 |
| sports | 500 | N/A | 266 (53.2%) / 0.677 | 165 (33.0%) / 0.668 | 206 (41.2%) / 0.874 |
| books | 500 | N/A | 293 (58.6%) / 0.784 | 228 (45.6%) / 0.593 | 250 (50.0%) / 0.844 |
| geography | 500 | N/A | 242 (48.4%) / 0.620 | 208 (41.6%) / 0.645 | 228 (45.6%) / 0.820 |
| politics | 500 | N/A | 240 (48.0%) / 0.626 | 203 (40.6%) / 0.544 | 248 (49.6%) / 0.840 |

## 4. Direct paired comparisons (common scoreable attempts)

### 4.1 Overall, six pairs

| Pair (A − B) | Common scoreable n | % of 4,000 | Mean A | Mean B | Δ | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C3 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C4 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C3 − C2 | 1324 | 33.1 | 0.643 | 0.744 | -0.101 | [-0.131, -0.072] | <.001 |
| C4 − C2 | 1394 | 34.8 | 0.879 | 0.741 | +0.138 | [+0.112, +0.163] | <.001 |
| C4 − C3 | 1498 | 37.5 | 0.888 | 0.660 | +0.228 | [+0.206, +0.251] | <.001 |

### 4.2 Per-pair attribute breakdowns

Rendered once, in `../attempted/faithfulness.md` §4 — identical by construction (same common scoreable sets).

## 5. Supplement: coverage accounting

Why a question contributes no faithfulness score. The four reasons partition the benchmark for each configuration.

| Config | Abstention | Other | Attempted, judge unscored | Scoreable | Total | = 4,000 |
|---|---:|---:|---:|---:|---:|---:|
| C2 RAG | 1759 | 14 | 40 | 2187 | 4000 | yes |
| C3 Graph-RAG | 2289 | 4 | 33 | 1674 | 4000 | yes |
| C4 Graph-RAG + condensing | 2037 | 1 | 34 | 1928 | 4000 | yes |

Unscored-attempt reasons (judge status): C2: unparsed_verdicts 40; C3: error 11, unparsed_verdicts 22; C4: error 7, unparsed_verdicts 27.

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- Conditional mean faithfulness, highest to lowest: C4 0.859 > C2 0.723 > C3 0.616 — on different scoreable sets (C2 n=2187, C3 n=1674, C4 n=1928); not testable against each other.
- Scoreable share of the benchmark: C2 54.7%, C3 41.9%, C4 48.2%; the rest is abstention, Other or an unscored attempt (section 5).
- C3 − C2 on 1324 common scoreable attempts: 0.643 vs 0.744, Δ -0.101 [-0.131, -0.072], p <.001 — CI excludes zero.
- C4 − C2 on 1394 common scoreable attempts: 0.879 vs 0.741, Δ +0.138 [+0.112, +0.163], p <.001 — CI excludes zero.
- C4 − C3 on 1498 common scoreable attempts: 0.888 vs 0.660, Δ +0.228 [+0.206, +0.251], p <.001 — CI excludes zero.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Faithfulness is undefined for C1; the headline C1-vs-retrieval comparison is a factuality comparison by construction.
- A faithful-but-wrong answer scores 1.0; the metric rewards small contexts and terse answers and does not measure benchmark correctness. C3's ~30×-smaller context cannot support as many claims as C2's.
- The judge was validated on DEV with moderate answer-level agreement: quote configuration-level comparisons only.

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
