# Faithfulness — attempted answers (scoreable share among attempts; pairs on common scoreable attempts)

## 1. Definition and reading guide

- **Faithfulness** = the stored answer-level judge score (0–1, RAGAS-style: share of the answer's claims entailed by the context that configuration was supplied). Means are macro-averages of answer scores, never total supported claims over total claims.
- **There is no four-configuration, denominator-4,000 faithfulness mean in this experiment.** C1 receives no context: N/A, not zero, in every table and every pair involving it. C2–C4 means are conditional on a **scoreable attempt**: the answer was attempted (CORRECT or HALLUCINATION) and the judge returned a usable score.
- Abstentions, Other and judge-unscored attempts are EXCLUDED from every mean, never assigned zero; the reasons are separated in the coverage accounting. `partial_score` is never used as a substitute.
- This folder reports scoreability AMONG ATTEMPTS; the own-scoreable means are identical to `../full_4000/faithfulness.md` by construction (same conditional population).
- Pair rows use the COMMON scoreable set (both attempted, both scored, same question). Each score still evaluates its answer against ITS OWN supplied context, not a shared one.
- Judge contract-audit flags (degenerate verdict lines) are retained under the existing availability policy; the sensitivity means without them are recorded, not reported.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview

| Config | Attempted n | Scoreable n | Unscored attempts | Scoreable % of attempts | Mean faithfulness (conditional) |
|---|---:|---:|---:|---:|---:|
| C1 Base LLM | 3484 | N/A | N/A | N/A | N/A — no context |
| C2 RAG | 2227 | 2187 | 40 | 98.2 | 0.723 |
| C3 Graph-RAG | 1707 | 1674 | 33 | 98.1 | 0.616 |
| C4 Graph-RAG + condensing | 1962 | 1928 | 34 | 98.3 | 0.859 |

## 3. Dataset stratification

### 3.1 By answer type

Cells: attempted n / scoreable n / mean. C1: N/A.

| Group | n | C1 | C2 att. / scor. / mean | C3 att. / scor. / mean | C4 att. / scor. / mean |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | N/A | 1245 / 1231 / 0.777 | 913 / 895 / 0.552 | 1051 / 1032 / 0.838 |
| numerical | 655 | N/A | 332 / 322 / 0.558 | 258 / 254 / 0.468 | 331 / 324 / 0.850 |
| boolean | 573 | N/A | 460 / 447 / 0.639 | 386 / 376 / 0.780 | 412 / 406 / 0.894 |
| date | 265 | N/A | 183 / 180 / 0.850 | 142 / 141 / 0.845 | 160 / 158 / 0.926 |
| string | 8† | N/A | 7 / 7† / 1.000 | 8 / 8† / 0.750 | 8 / 8† / 0.821 |

### 3.2 By question complexity

Cells: attempted n / scoreable n / mean. C1: N/A.

| Group | n | C1 | C2 att. / scor. / mean | C3 att. / scor. / mean | C4 att. / scor. / mean |
|---|---:|---:|---:|---:|---:|
| generic | 800 | N/A | 564 / 560 / 0.839 | 456 / 454 / 0.713 | 475 / 474 / 0.887 |
| intersection | 400 | N/A | 221 / 219 / 0.806 | 167 / 162 / 0.463 | 178 / 177 / 0.824 |
| count | 400 | N/A | 233 / 227 / 0.522 | 190 / 187 / 0.430 | 224 / 221 / 0.850 |
| comparative | 400 | N/A | 333 / 324 / 0.647 | 242 / 230 / 0.713 | 243 / 236 / 0.825 |
| yesno | 400 | N/A | 313 / 307 / 0.657 | 260 / 257 / 0.776 | 288 / 284 / 0.904 |
| ordinal | 400 | N/A | 216 / 214 / 0.793 | 149 / 147 / 0.622 | 175 / 173 / 0.920 |
| multihop | 400 | N/A | 112 / 106 / 0.790 | 66 / 63 / 0.267 | 141 / 132 / 0.836 |
| difference | 400 | N/A | 129 / 125 / 0.707 | 117 / 114 / 0.477 | 159 / 153 / 0.771 |
| superlative | 400 | N/A | 106 / 105 / 0.599 | 60 / 60 / 0.431 | 79 / 78 / 0.803 |

### 3.3 By topic category

Cells: attempted n / scoreable n / mean. C1: N/A.

| Group | n | C1 | C2 att. / scor. / mean | C3 att. / scor. / mean | C4 att. / scor. / mean |
|---|---:|---:|---:|---:|---:|
| history | 500 | N/A | 271 / 267 / 0.732 | 276 / 272 / 0.577 | 281 / 277 / 0.871 |
| movies | 500 | N/A | 306 / 301 / 0.756 | 233 / 222 / 0.666 | 279 / 270 / 0.871 |
| music | 500 | N/A | 283 / 281 / 0.771 | 164 / 161 / 0.661 | 199 / 195 / 0.888 |
| videogames | 500 | N/A | 304 / 297 / 0.780 | 219 / 215 / 0.606 | 258 / 254 / 0.865 |
| sports | 500 | N/A | 275 / 266 / 0.677 | 167 / 165 / 0.668 | 209 / 206 / 0.874 |
| books | 500 | N/A | 299 / 293 / 0.784 | 231 / 228 / 0.593 | 255 / 250 / 0.844 |
| geography | 500 | N/A | 245 / 242 / 0.620 | 211 / 208 / 0.645 | 232 / 228 / 0.820 |
| politics | 500 | N/A | 244 / 240 / 0.626 | 206 / 203 / 0.544 | 249 / 248 / 0.840 |

## 4. Direct paired comparisons (common scoreable attempts)

### 4.1 Overall, six pairs

| Pair (A − B) | Common scoreable n | % of A's / B's scoreable | Mean A | Mean B | Δ | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C3 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C4 − C1 | — | — | — | — | — | — | N/A: involves C1, which has no context: faithfulness N/A |
| C3 − C2 | 1324 | 79.1 / 60.5 | 0.643 | 0.744 | -0.101 | [-0.131, -0.072] | <.001 |
| C4 − C2 | 1394 | 72.3 / 63.7 | 0.879 | 0.741 | +0.138 | [+0.112, +0.163] | <.001 |
| C4 − C3 | 1498 | 77.7 / 89.5 | 0.888 | 0.660 | +0.228 | [+0.206, +0.251] | <.001 |

### 4.2 C2 − C1 by attribute

N/A — involves C1, which has no context: faithfulness N/A.

### 4.3 C3 − C1 by attribute

N/A — involves C1, which has no context: faithfulness N/A.

### 4.4 C4 − C1 by attribute

N/A — involves C1, which has no context: faithfulness N/A.

### 4.5 C3 − C2 by attribute

Descriptive per group (no tests).

| Group | Common scoreable n | Mean C3 | Mean C2 | Δ |
|---|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |
| entity | 666 | 0.577 | 0.812 | -0.235 |
| numerical | 193 | 0.485 | 0.577 | -0.092 |
| boolean | 340 | 0.790 | 0.662 | +0.128 |
| date | 118 | 0.842 | 0.849 | -0.007 |
| string | 7† | 0.714 | 1.000 | -0.286 |
| **Question complexity** |  |  |  |  |
| generic | 399 | 0.727 | 0.856 | -0.129 |
| intersection | 122 | 0.454 | 0.806 | -0.352 |
| count | 135 | 0.445 | 0.552 | -0.107 |
| comparative | 215 | 0.725 | 0.706 | +0.019 |
| yesno | 228 | 0.791 | 0.671 | +0.121 |
| ordinal | 103 | 0.640 | 0.762 | -0.122 |
| multihop | 38† | 0.260 | 0.842 | -0.582 |
| difference | 57 | 0.411 | 0.683 | -0.272 |
| superlative | 27† | 0.364 | 0.580 | -0.216 |
| **Topic category** |  |  |  |  |
| history | 192 | 0.617 | 0.757 | -0.140 |
| movies | 190 | 0.685 | 0.790 | -0.105 |
| music | 133 | 0.649 | 0.829 | -0.180 |
| videogames | 175 | 0.625 | 0.787 | -0.162 |
| sports | 132 | 0.658 | 0.655 | +0.003 |
| books | 189 | 0.604 | 0.782 | -0.177 |
| geography | 166 | 0.686 | 0.682 | +0.005 |
| politics | 147 | 0.623 | 0.637 | -0.014 |

### 4.6 C4 − C2 by attribute

Descriptive per group (no tests).

| Group | Common scoreable n | Mean C4 | Mean C2 | Δ |
|---|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |
| entity | 718 | 0.858 | 0.812 | +0.046 |
| numerical | 213 | 0.871 | 0.578 | +0.293 |
| boolean | 335 | 0.904 | 0.648 | +0.256 |
| date | 121 | 0.946 | 0.848 | +0.097 |
| string | 7† | 0.939 | 1.000 | -0.061 |
| **Question complexity** |  |  |  |  |
| generic | 406 | 0.912 | 0.846 | +0.066 |
| intersection | 128 | 0.852 | 0.851 | +0.001 |
| count | 148 | 0.861 | 0.560 | +0.302 |
| comparative | 211 | 0.825 | 0.679 | +0.146 |
| yesno | 227 | 0.921 | 0.656 | +0.265 |
| ordinal | 122 | 0.916 | 0.783 | +0.133 |
| multihop | 50 | 0.812 | 0.833 | -0.022 |
| difference | 70 | 0.802 | 0.719 | +0.083 |
| superlative | 32† | 0.838 | 0.568 | +0.270 |
| **Topic category** |  |  |  |  |
| history | 193 | 0.874 | 0.753 | +0.121 |
| movies | 205 | 0.891 | 0.782 | +0.109 |
| music | 140 | 0.912 | 0.818 | +0.094 |
| videogames | 193 | 0.888 | 0.811 | +0.077 |
| sports | 149 | 0.897 | 0.632 | +0.264 |
| books | 192 | 0.866 | 0.796 | +0.070 |
| geography | 165 | 0.865 | 0.670 | +0.194 |
| politics | 157 | 0.845 | 0.629 | +0.215 |

### 4.7 C4 − C3 by attribute

Descriptive per group (no tests).

| Group | Common scoreable n | Mean C4 | Mean C3 | Δ |
|---|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |
| entity | 787 | 0.876 | 0.598 | +0.278 |
| numerical | 230 | 0.879 | 0.499 | +0.380 |
| boolean | 340 | 0.904 | 0.830 | +0.074 |
| date | 133 | 0.942 | 0.866 | +0.076 |
| string | 8† | 0.821 | 0.750 | +0.071 |
| **Question complexity** |  |  |  |  |
| generic | 425 | 0.904 | 0.745 | +0.159 |
| intersection | 137 | 0.880 | 0.493 | +0.387 |
| count | 168 | 0.864 | 0.461 | +0.403 |
| comparative | 200 | 0.846 | 0.785 | +0.060 |
| yesno | 234 | 0.914 | 0.822 | +0.092 |
| ordinal | 135 | 0.962 | 0.653 | +0.309 |
| multihop | 48† | 0.883 | 0.319 | +0.564 |
| difference | 99 | 0.825 | 0.496 | +0.329 |
| superlative | 52 | 0.844 | 0.488 | +0.356 |
| **Topic category** |  |  |  |  |
| history | 242 | 0.880 | 0.620 | +0.260 |
| movies | 195 | 0.911 | 0.715 | +0.196 |
| music | 145 | 0.896 | 0.702 | +0.194 |
| videogames | 195 | 0.894 | 0.628 | +0.266 |
| sports | 151 | 0.906 | 0.704 | +0.201 |
| books | 204 | 0.879 | 0.646 | +0.233 |
| geography | 188 | 0.864 | 0.686 | +0.178 |
| politics | 178 | 0.884 | 0.607 | +0.277 |

## 5. Supplements

### 5.1 Faithfulness by outcome within each configuration

Correct vs hallucinated answers are DISJOINT groups within one configuration, so the contrast is an unpaired two-sample bootstrap. Grounded := score ≥ 0.5 (the pre-registered grounded×outcome cut); shares are within the outcome.

| Config | Correct n | mean | Hallucination n | mean | Δ corr − hall | 95% CI | p | Ungrounded among correct | Ungrounded among hallucinations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 RAG | 1840 | 0.758 | 347 | 0.538 | +0.220 | [+0.170, +0.270] | <.001 | 383 (20.8%) | 139 (40.1%) |
| C3 Graph-RAG | 1343 | 0.650 | 331 | 0.477 | +0.173 | [+0.122, +0.226] | <.001 | 425 (31.6%) | 159 (48.0%) |
| C4 Graph-RAG + condensing | 1513 | 0.889 | 415 | 0.750 | +0.139 | [+0.101, +0.178] | <.001 | 133 (8.8%) | 98 (23.6%) |

The last column is the confabulation-ANALOGUE rate (ungrounded share of hallucinations): an entailment verdict, not proof of source — judge error and derivable-but-unstated answers land in the same bucket.

### 5.2 Questions scoreable for all three retrieval configurations

n = 1194. Cells: mean faithfulness on that common set.

| Group | Common n | C2 mean | C3 mean | C4 mean |
|---|---:|---:|---:|---:|
| overall | 1194 | 0.746 | 0.685 | 0.904 |
| **Answer type** |  |  |  |  |
| entity | 592 | 0.817 | 0.620 | 0.891 |
| numerical | 177 | 0.569 | 0.511 | 0.892 |
| boolean | 306 | 0.666 | 0.842 | 0.915 |
| date | 112 | 0.854 | 0.869 | 0.959 |
| string | 7† | 1.000 | 0.714 | 0.939 |
| **Question complexity** |  |  |  |  |
| generic | 376 | 0.855 | 0.756 | 0.925 |
| intersection | 104 | 0.838 | 0.476 | 0.893 |
| count | 123 | 0.548 | 0.472 | 0.872 |
| comparative | 187 | 0.705 | 0.799 | 0.843 |
| yesno | 207 | 0.672 | 0.838 | 0.931 |
| ordinal | 97 | 0.758 | 0.659 | 0.950 |
| multihop | 29† | 0.859 | 0.332 | 0.890 |
| difference | 49† | 0.670 | 0.411 | 0.860 |
| superlative | 22† | 0.576 | 0.424 | 0.938 |
| **Topic category** |  |  |  |  |
| history | 177 | 0.756 | 0.647 | 0.887 |
| movies | 167 | 0.794 | 0.738 | 0.920 |
| music | 122 | 0.837 | 0.677 | 0.919 |
| videogames | 157 | 0.796 | 0.654 | 0.915 |
| sports | 121 | 0.642 | 0.698 | 0.907 |
| books | 169 | 0.788 | 0.656 | 0.902 |
| geography | 153 | 0.685 | 0.714 | 0.889 |
| politics | 128 | 0.638 | 0.701 | 0.894 |

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- Scoreable share among attempts: C2 98.2%, C3 98.1%, C4 98.3% (unscored: 40, 33, 34).
- C3 − C2 on 1324 common scoreable attempts: Δ -0.101 [-0.131, -0.072], p <.001 — CI excludes zero.
- C4 − C2 on 1394 common scoreable attempts: Δ +0.138 [+0.112, +0.163], p <.001 — CI excludes zero.
- C4 − C3 on 1498 common scoreable attempts: Δ +0.228 [+0.206, +0.251], p <.001 — CI excludes zero.
- C2: correct answers mean 0.758 (n=1840) vs hallucinations 0.538 (n=347); 139 of 347 hallucinations (40.1%) are ungrounded.
- C3: correct answers mean 0.650 (n=1343) vs hallucinations 0.477 (n=331); 159 of 331 hallucinations (48.0%) are ungrounded.
- C4: correct answers mean 0.889 (n=1513) vs hallucinations 0.750 (n=415); 98 of 415 hallucinations (23.6%) are ungrounded.
- All-three common set (1194): C2 0.746, C3 0.685, C4 0.904.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Faithfulness is computed only over attempted answers with an available judge score. It is undefined for C1 and conditional on attempting, which is itself a treatment outcome.
- Higher faithfulness does not imply benchmark correctness. For C4, support is judged against condensed prose; this does not test whether the condenser preserved the retrieved statements faithfully.

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
