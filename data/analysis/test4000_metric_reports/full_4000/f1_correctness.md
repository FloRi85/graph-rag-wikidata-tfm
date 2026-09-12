# F1 and correctness — full benchmark (n = 4,000)

## 1. Definition and reading guide

- **Correct %** = 100 × questions with the scorer's CORRECT outcome / n: thresholded operational correctness (score ≥ 0.5 under the type-aware matcher), not exact match and not verified world truth.
- **F1** = 100 × mean per-question token/set F1 from the shared scorer, on a 0–100 scale. Abstention and Other carry F1 = 0 under the current contract, so full-benchmark F1 = coverage × F1@attempted exactly.
- **F1@attempted** is each configuration's OWN attempted set (different denominators; see `../attempted/f1_correctness.md`). The threshold never replaces the outcome classifier.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview

| Config | n | Correct n (%) | F1 | Coverage % | F1@attempted (own) | coverage × F1@att. | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 Base LLM | 4000 | 2493 (62.3) | 58.7 | 87.1 | 67.4 | 58.7 | 50.8 |
| C2 RAG | 4000 | 1867 (46.7) | 43.0 | 55.7 | 77.2 | 43.0 | 37.7 |
| C3 Graph-RAG | 4000 | 1369 (34.2) | 33.0 | 42.7 | 77.4 | 33.0 | 30.6 |
| C4 Graph-RAG + condensing | 4000 | 1537 (38.4) | 37.5 | 49.0 | 76.4 | 37.5 | 34.4 |

EM (exact match, 0–100) is shown as a companion only; the thesis reports F1.

## 3. Dataset stratification

### 3.1 By answer type

Cells: correct % / F1 over all questions in the group.

| Group | n | C1 Corr / F1 | C2 Corr / F1 | C3 Corr / F1 | C4 Corr / F1 |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | 63.2 / 57.4 | 43.3 / 37.5 | 30.5 / 28.6 | 34.9 / 33.3 |
| numerical | 655 | 43.5 / 43.5 | 32.2 / 32.2 | 22.1 / 22.1 | 26.1 / 26.1 |
| boolean | 573 | 79.6 / 79.6 | 71.4 / 71.4 | 60.6 / 60.6 | 64.6 / 64.6 |
| date | 265 | 63.4 / 63.5 | 59.6 / 59.6 | 40.8 / 40.9 | 45.3 / 45.4 |
| string | 8† | 50.0 / 55.0 | 87.5 / 77.1 | 87.5 / 85.0 | 62.5 / 66.7 |

### 3.2 By question complexity

Cells: correct % / F1 over all questions in the group.

| Group | n | C1 Corr / F1 | C2 Corr / F1 | C3 Corr / F1 | C4 Corr / F1 |
|---|---:|---:|---:|---:|---:|
| generic | 800 | 68.1 / 65.2 | 58.1 / 54.1 | 46.1 / 44.0 | 47.8 / 45.9 |
| intersection | 400 | 71.2 / 68.5 | 52.2 / 49.9 | 36.5 / 35.5 | 39.5 / 38.6 |
| count | 400 | 51.7 / 51.7 | 40.0 / 40.0 | 28.0 / 28.0 | 29.8 / 29.8 |
| comparative | 400 | 76.5 / 68.5 | 70.0 / 54.7 | 53.2 / 49.4 | 54.5 / 54.2 |
| yesno | 400 | 82.8 / 82.8 | 69.8 / 69.8 | 57.8 / 57.8 | 63.0 / 63.0 |
| ordinal | 400 | 59.8 / 55.1 | 47.5 / 44.7 | 30.0 / 29.4 | 34.5 / 33.5 |
| multihop | 400 | 44.8 / 39.6 | 21.8 / 20.3 | 13.8 / 13.3 | 25.5 / 24.7 |
| difference | 400 | 46.8 / 40.4 | 28.0 / 23.7 | 20.0 / 18.1 | 28.2 / 25.5 |
| superlative | 400 | 53.5 / 50.1 | 21.2 / 18.8 | 10.8 / 10.7 | 13.8 / 13.5 |

### 3.3 By topic category

Cells: correct % / F1 over all questions in the group.

| Group | n | C1 Corr / F1 | C2 Corr / F1 | C3 Corr / F1 | C4 Corr / F1 |
|---|---:|---:|---:|---:|---:|
| history | 500 | 72.2 / 66.4 | 48.2 / 41.2 | 43.8 / 40.6 | 45.6 / 44.8 |
| movies | 500 | 59.8 / 57.3 | 50.8 / 45.8 | 38.2 / 37.4 | 41.8 / 40.7 |
| music | 500 | 50.4 / 45.5 | 45.4 / 42.2 | 24.4 / 23.9 | 28.4 / 27.7 |
| videogames | 500 | 65.6 / 62.1 | 54.4 / 51.4 | 35.6 / 34.1 | 41.2 / 40.1 |
| sports | 500 | 60.4 / 58.6 | 43.6 / 41.4 | 28.2 / 27.8 | 32.2 / 31.8 |
| books | 500 | 57.6 / 54.7 | 49.8 / 46.7 | 36.6 / 35.5 | 41.2 / 40.3 |
| geography | 500 | 64.4 / 60.5 | 40.8 / 38.0 | 32.6 / 31.8 | 36.4 / 35.1 |
| politics | 500 | 68.2 / 64.7 | 40.4 / 37.3 | 34.4 / 33.1 | 40.6 / 39.1 |

## 4. Direct paired comparisons (identical id sets)

### 4.1 Overall, six pairs

ΔCorr = correct A − B (pp) with bootstrap CI and exact McNemar; ΔF1 = F1 A − B (points) with bootstrap CI.

| Pair (A − B) | n | Corr% A | Corr% B | ΔCorr pp | 95% CI | p | McNemar p | F1 A | F1 B | ΔF1 | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | 4000 | 46.7 | 62.3 | -15.6 | [-17.3, -14.0] | <.001 | <.001 | 43.0 | 58.7 | -15.7 | [-17.2, -14.2] | <.001 |
| C3 − C1 | 4000 | 34.2 | 62.3 | -28.1 | [-29.8, -26.4] | <.001 | <.001 | 33.0 | 58.7 | -25.7 | [-27.3, -24.1] | <.001 |
| C4 − C1 | 4000 | 38.4 | 62.3 | -23.9 | [-25.7, -22.2] | <.001 | <.001 | 37.5 | 58.7 | -21.3 | [-22.9, -19.6] | <.001 |
| C3 − C2 | 4000 | 34.2 | 46.7 | -12.4 | [-14.0, -10.9] | <.001 | <.001 | 33.0 | 43.0 | -10.0 | [-11.5, -8.5] | <.001 |
| C4 − C2 | 4000 | 38.4 | 46.7 | -8.3 | [-10.0, -6.7] | <.001 | <.001 | 37.5 | 43.0 | -5.5 | [-7.2, -4.0] | <.001 |
| C4 − C3 | 4000 | 38.4 | 34.2 | +4.2 | [+3.2, +5.2] | <.001 | <.001 | 37.5 | 33.0 | +4.4 | [+3.5, +5.4] | <.001 |

### 4.2 C2 − C1 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C2 | Corr% C1 | ΔCorr pp | F1 C2 | F1 C1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 43.3 | 63.2 | -19.9 | 37.5 | 57.4 | -20.0 |
| numerical | 655 | 32.2 | 43.5 | -11.3 | 32.2 | 43.5 | -11.3 |
| boolean | 573 | 71.4 | 79.6 | -8.2 | 71.4 | 79.6 | -8.2 |
| date | 265 | 59.6 | 63.4 | -3.8 | 59.6 | 63.5 | -3.9 |
| string | 8† | 87.5 | 50.0 | +37.5 | 77.1 | 55.0 | +22.1 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 58.1 | 68.1 | -10.0 | 54.1 | 65.2 | -11.1 |
| intersection | 400 | 52.2 | 71.2 | -19.0 | 49.9 | 68.5 | -18.6 |
| count | 400 | 40.0 | 51.7 | -11.7 | 40.0 | 51.7 | -11.7 |
| comparative | 400 | 70.0 | 76.5 | -6.5 | 54.7 | 68.5 | -13.8 |
| yesno | 400 | 69.8 | 82.8 | -13.0 | 69.8 | 82.8 | -13.0 |
| ordinal | 400 | 47.5 | 59.8 | -12.3 | 44.7 | 55.1 | -10.4 |
| multihop | 400 | 21.8 | 44.8 | -23.0 | 20.3 | 39.6 | -19.3 |
| difference | 400 | 28.0 | 46.8 | -18.8 | 23.7 | 40.4 | -16.7 |
| superlative | 400 | 21.2 | 53.5 | -32.2 | 18.8 | 50.1 | -31.3 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 48.2 | 72.2 | -24.0 | 41.2 | 66.4 | -25.3 |
| movies | 500 | 50.8 | 59.8 | -9.0 | 45.8 | 57.3 | -11.4 |
| music | 500 | 45.4 | 50.4 | -5.0 | 42.2 | 45.5 | -3.2 |
| videogames | 500 | 54.4 | 65.6 | -11.2 | 51.4 | 62.1 | -10.7 |
| sports | 500 | 43.6 | 60.4 | -16.8 | 41.4 | 58.6 | -17.2 |
| books | 500 | 49.8 | 57.6 | -7.8 | 46.7 | 54.7 | -8.0 |
| geography | 500 | 40.8 | 64.4 | -23.6 | 38.0 | 60.5 | -22.5 |
| politics | 500 | 40.4 | 68.2 | -27.8 | 37.3 | 64.7 | -27.4 |

### 4.3 C3 − C1 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C3 | Corr% C1 | ΔCorr pp | F1 C3 | F1 C1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 30.5 | 63.2 | -32.7 | 28.6 | 57.4 | -28.9 |
| numerical | 655 | 22.1 | 43.5 | -21.4 | 22.1 | 43.5 | -21.4 |
| boolean | 573 | 60.6 | 79.6 | -19.0 | 60.6 | 79.6 | -19.0 |
| date | 265 | 40.8 | 63.4 | -22.6 | 40.9 | 63.5 | -22.6 |
| string | 8† | 87.5 | 50.0 | +37.5 | 85.0 | 55.0 | +30.0 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 46.1 | 68.1 | -22.0 | 44.0 | 65.2 | -21.2 |
| intersection | 400 | 36.5 | 71.2 | -34.8 | 35.5 | 68.5 | -33.0 |
| count | 400 | 28.0 | 51.7 | -23.7 | 28.0 | 51.7 | -23.7 |
| comparative | 400 | 53.2 | 76.5 | -23.3 | 49.4 | 68.5 | -19.1 |
| yesno | 400 | 57.8 | 82.8 | -25.0 | 57.8 | 82.8 | -25.0 |
| ordinal | 400 | 30.0 | 59.8 | -29.8 | 29.4 | 55.1 | -25.6 |
| multihop | 400 | 13.8 | 44.8 | -31.0 | 13.3 | 39.6 | -26.4 |
| difference | 400 | 20.0 | 46.8 | -26.8 | 18.1 | 40.4 | -22.4 |
| superlative | 400 | 10.8 | 53.5 | -42.8 | 10.7 | 50.1 | -39.4 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 43.8 | 72.2 | -28.4 | 40.6 | 66.4 | -25.9 |
| movies | 500 | 38.2 | 59.8 | -21.6 | 37.4 | 57.3 | -19.9 |
| music | 500 | 24.4 | 50.4 | -26.0 | 23.9 | 45.5 | -21.5 |
| videogames | 500 | 35.6 | 65.6 | -30.0 | 34.1 | 62.1 | -28.0 |
| sports | 500 | 28.2 | 60.4 | -32.2 | 27.8 | 58.6 | -30.8 |
| books | 500 | 36.6 | 57.6 | -21.0 | 35.5 | 54.7 | -19.3 |
| geography | 500 | 32.6 | 64.4 | -31.8 | 31.8 | 60.5 | -28.7 |
| politics | 500 | 34.4 | 68.2 | -33.8 | 33.1 | 64.7 | -31.5 |

### 4.4 C4 − C1 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C4 | Corr% C1 | ΔCorr pp | F1 C4 | F1 C1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 34.9 | 63.2 | -28.4 | 33.3 | 57.4 | -24.1 |
| numerical | 655 | 26.1 | 43.5 | -17.4 | 26.1 | 43.5 | -17.4 |
| boolean | 573 | 64.6 | 79.6 | -15.0 | 64.6 | 79.6 | -15.0 |
| date | 265 | 45.3 | 63.4 | -18.1 | 45.4 | 63.5 | -18.1 |
| string | 8† | 62.5 | 50.0 | +12.5 | 66.7 | 55.0 | +11.7 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 47.8 | 68.1 | -20.4 | 45.9 | 65.2 | -19.3 |
| intersection | 400 | 39.5 | 71.2 | -31.8 | 38.6 | 68.5 | -29.9 |
| count | 400 | 29.8 | 51.7 | -22.0 | 29.8 | 51.7 | -22.0 |
| comparative | 400 | 54.5 | 76.5 | -22.0 | 54.2 | 68.5 | -14.3 |
| yesno | 400 | 63.0 | 82.8 | -19.8 | 63.0 | 82.8 | -19.8 |
| ordinal | 400 | 34.5 | 59.8 | -25.3 | 33.5 | 55.1 | -21.6 |
| multihop | 400 | 25.5 | 44.8 | -19.2 | 24.7 | 39.6 | -14.9 |
| difference | 400 | 28.2 | 46.8 | -18.5 | 25.5 | 40.4 | -15.0 |
| superlative | 400 | 13.8 | 53.5 | -39.8 | 13.5 | 50.1 | -36.6 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 45.6 | 72.2 | -26.6 | 44.8 | 66.4 | -21.6 |
| movies | 500 | 41.8 | 59.8 | -18.0 | 40.7 | 57.3 | -16.5 |
| music | 500 | 28.4 | 50.4 | -22.0 | 27.7 | 45.5 | -17.7 |
| videogames | 500 | 41.2 | 65.6 | -24.4 | 40.1 | 62.1 | -22.0 |
| sports | 500 | 32.2 | 60.4 | -28.2 | 31.8 | 58.6 | -26.8 |
| books | 500 | 41.2 | 57.6 | -16.4 | 40.3 | 54.7 | -14.5 |
| geography | 500 | 36.4 | 64.4 | -28.0 | 35.1 | 60.5 | -25.3 |
| politics | 500 | 40.6 | 68.2 | -27.6 | 39.1 | 64.7 | -25.5 |

### 4.5 C3 − C2 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C3 | Corr% C2 | ΔCorr pp | F1 C3 | F1 C2 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 30.5 | 43.3 | -12.8 | 28.6 | 37.5 | -8.9 |
| numerical | 655 | 22.1 | 32.2 | -10.1 | 22.1 | 32.2 | -10.1 |
| boolean | 573 | 60.6 | 71.4 | -10.8 | 60.6 | 71.4 | -10.8 |
| date | 265 | 40.8 | 59.6 | -18.9 | 40.9 | 59.6 | -18.7 |
| string | 8† | 87.5 | 87.5 | +0.0 | 85.0 | 77.1 | +7.9 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 46.1 | 58.1 | -12.0 | 44.0 | 54.1 | -10.0 |
| intersection | 400 | 36.5 | 52.2 | -15.7 | 35.5 | 49.9 | -14.4 |
| count | 400 | 28.0 | 40.0 | -12.0 | 28.0 | 40.0 | -12.0 |
| comparative | 400 | 53.2 | 70.0 | -16.8 | 49.4 | 54.7 | -5.3 |
| yesno | 400 | 57.8 | 69.8 | -12.0 | 57.8 | 69.8 | -12.0 |
| ordinal | 400 | 30.0 | 47.5 | -17.5 | 29.4 | 44.7 | -15.3 |
| multihop | 400 | 13.8 | 21.8 | -8.0 | 13.3 | 20.3 | -7.0 |
| difference | 400 | 20.0 | 28.0 | -8.0 | 18.1 | 23.7 | -5.7 |
| superlative | 400 | 10.8 | 21.2 | -10.5 | 10.7 | 18.8 | -8.0 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 43.8 | 48.2 | -4.4 | 40.6 | 41.2 | -0.6 |
| movies | 500 | 38.2 | 50.8 | -12.6 | 37.4 | 45.8 | -8.4 |
| music | 500 | 24.4 | 45.4 | -21.0 | 23.9 | 42.2 | -18.3 |
| videogames | 500 | 35.6 | 54.4 | -18.8 | 34.1 | 51.4 | -17.3 |
| sports | 500 | 28.2 | 43.6 | -15.4 | 27.8 | 41.4 | -13.6 |
| books | 500 | 36.6 | 49.8 | -13.2 | 35.5 | 46.7 | -11.3 |
| geography | 500 | 32.6 | 40.8 | -8.2 | 31.8 | 38.0 | -6.2 |
| politics | 500 | 34.4 | 40.4 | -6.0 | 33.1 | 37.3 | -4.1 |

### 4.6 C4 − C2 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C4 | Corr% C2 | ΔCorr pp | F1 C4 | F1 C2 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 34.9 | 43.3 | -8.4 | 33.3 | 37.5 | -4.2 |
| numerical | 655 | 26.1 | 32.2 | -6.1 | 26.1 | 32.2 | -6.1 |
| boolean | 573 | 64.6 | 71.4 | -6.8 | 64.6 | 71.4 | -6.8 |
| date | 265 | 45.3 | 59.6 | -14.3 | 45.4 | 59.6 | -14.2 |
| string | 8† | 62.5 | 87.5 | -25.0 | 66.7 | 77.1 | -10.4 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 47.8 | 58.1 | -10.4 | 45.9 | 54.1 | -8.2 |
| intersection | 400 | 39.5 | 52.2 | -12.7 | 38.6 | 49.9 | -11.2 |
| count | 400 | 29.8 | 40.0 | -10.3 | 29.8 | 40.0 | -10.3 |
| comparative | 400 | 54.5 | 70.0 | -15.5 | 54.2 | 54.7 | -0.5 |
| yesno | 400 | 63.0 | 69.8 | -6.8 | 63.0 | 69.8 | -6.8 |
| ordinal | 400 | 34.5 | 47.5 | -13.0 | 33.5 | 44.7 | -11.2 |
| multihop | 400 | 25.5 | 21.8 | +3.8 | 24.7 | 20.3 | +4.4 |
| difference | 400 | 28.2 | 28.0 | +0.2 | 25.5 | 23.7 | +1.7 |
| superlative | 400 | 13.8 | 21.2 | -7.5 | 13.5 | 18.8 | -5.2 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 45.6 | 48.2 | -2.6 | 44.8 | 41.2 | +3.6 |
| movies | 500 | 41.8 | 50.8 | -9.0 | 40.7 | 45.8 | -5.1 |
| music | 500 | 28.4 | 45.4 | -17.0 | 27.7 | 42.2 | -14.5 |
| videogames | 500 | 41.2 | 54.4 | -13.2 | 40.1 | 51.4 | -11.2 |
| sports | 500 | 32.2 | 43.6 | -11.4 | 31.8 | 41.4 | -9.6 |
| books | 500 | 41.2 | 49.8 | -8.6 | 40.3 | 46.7 | -6.5 |
| geography | 500 | 36.4 | 40.8 | -4.4 | 35.1 | 38.0 | -2.9 |
| politics | 500 | 40.6 | 40.4 | +0.2 | 39.1 | 37.3 | +1.8 |

### 4.7 C4 − C3 by attribute

Descriptive per group (no tests).

| Group | n | Corr% C4 | Corr% C3 | ΔCorr pp | F1 C4 | F1 C3 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 34.9 | 30.5 | +4.4 | 33.3 | 28.6 | +4.7 |
| numerical | 655 | 26.1 | 22.1 | +4.0 | 26.1 | 22.1 | +4.0 |
| boolean | 573 | 64.6 | 60.6 | +4.0 | 64.6 | 60.6 | +4.0 |
| date | 265 | 45.3 | 40.8 | +4.5 | 45.4 | 40.9 | +4.5 |
| string | 8† | 62.5 | 87.5 | -25.0 | 66.7 | 85.0 | -18.3 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 47.8 | 46.1 | +1.6 | 45.9 | 44.0 | +1.9 |
| intersection | 400 | 39.5 | 36.5 | +3.0 | 38.6 | 35.5 | +3.1 |
| count | 400 | 29.8 | 28.0 | +1.7 | 29.8 | 28.0 | +1.7 |
| comparative | 400 | 54.5 | 53.2 | +1.3 | 54.2 | 49.4 | +4.8 |
| yesno | 400 | 63.0 | 57.8 | +5.2 | 63.0 | 57.8 | +5.2 |
| ordinal | 400 | 34.5 | 30.0 | +4.5 | 33.5 | 29.4 | +4.1 |
| multihop | 400 | 25.5 | 13.8 | +11.8 | 24.7 | 13.3 | +11.5 |
| difference | 400 | 28.2 | 20.0 | +8.2 | 25.5 | 18.1 | +7.4 |
| superlative | 400 | 13.8 | 10.8 | +3.0 | 13.5 | 10.7 | +2.8 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 45.6 | 43.8 | +1.8 | 44.8 | 40.6 | +4.2 |
| movies | 500 | 41.8 | 38.2 | +3.6 | 40.7 | 37.4 | +3.4 |
| music | 500 | 28.4 | 24.4 | +4.0 | 27.7 | 23.9 | +3.8 |
| videogames | 500 | 41.2 | 35.6 | +5.6 | 40.1 | 34.1 | +6.0 |
| sports | 500 | 32.2 | 28.2 | +4.0 | 31.8 | 27.8 | +4.0 |
| books | 500 | 41.2 | 36.6 | +4.6 | 40.3 | 35.5 | +4.8 |
| geography | 500 | 36.4 | 32.6 | +3.8 | 35.1 | 31.8 | +3.4 |
| politics | 500 | 40.6 | 34.4 | +6.2 | 39.1 | 33.1 | +6.0 |

## 5. Supplement: threshold sensitivity

Outcomes re-scored with the wrong-answer cutoff at 0.3 / 0.5 / 0.7 (0.5 is the reported contract). Cells: correct % / hallucination %. The question is whether the ORDERING depends on the cutoff, not the level.

| Config | t = 0.3: Corr / H | t = 0.5: Corr / H | t = 0.7: Corr / H |
|---|---:|---:|---:|
| C1 Base LLM | 63.7 / 23.4 | 62.3 / 24.8 | 58.8 / 28.3 |
| C2 RAG | 47.0 / 8.6 | 46.7 / 9.0 | 45.1 / 10.5 |
| C3 Graph-RAG | 34.5 / 8.2 | 34.2 / 8.5 | 33.1 / 9.6 |
| C4 Graph-RAG + condensing | 38.8 / 10.2 | 38.4 / 10.6 | 37.1 / 11.9 |

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- Full-benchmark correct %, highest to lowest: C1 62.3% > C2 46.7% > C4 38.4% > C3 34.2%.
- Full-benchmark F1, highest to lowest: C1 58.7 > C2 43.0 > C4 37.5 > C3 33.0.
- F1 on own attempts, highest to lowest: C3 77.4 > C2 77.2 > C4 76.4 > C1 67.4 — different denominators, not testable against each other.
- C2 − C1: ΔCorr -15.6 pp [-17.3, -14.0], p <.001; ΔF1 -15.7 [-17.2, -14.2], p <.001.
- C3 − C1: ΔCorr -28.1 pp [-29.8, -26.4], p <.001; ΔF1 -25.7 [-27.3, -24.1], p <.001.
- C4 − C1: ΔCorr -23.9 pp [-25.7, -22.2], p <.001; ΔF1 -21.3 [-22.9, -19.6], p <.001.
- C3 − C2: ΔCorr -12.4 pp [-14.0, -10.9], p <.001; ΔF1 -10.0 [-11.5, -8.5], p <.001.
- C4 − C2: ΔCorr -8.3 pp [-10.0, -6.7], p <.001; ΔF1 -5.5 [-7.2, -4.0], p <.001.
- C4 − C3: ΔCorr +4.2 pp [+3.2, +5.2], p <.001; ΔF1 +4.4 [+3.5, +5.4], p <.001.
- Hallucination ordering at t = 0.3: C3 8.2% < C2 8.6% < C4 10.2% < C1 23.4%.
- Hallucination ordering at t = 0.5: C3 8.5% < C2 9.0% < C4 10.6% < C1 24.8%.
- Hallucination ordering at t = 0.7: C3 9.6% < C2 10.5% < C4 11.9% < C1 28.3%.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Hallucination here is confident disagreement with the 2021 Mintaka gold (a factuality proxy), scored by the shared type-aware scorer at threshold 0.5; correctness is that scorer's CORRECT outcome, not exact match or verified world truth.
- C1 never receives the question-entity block that C2–C4 receive, so a C1-vs-retrieval delta measures retrieval plus that annotation. C4's prompt differs from C3's in more than the condensing step.
- Run-to-run noise (byte-identical repeats): hallucination ≤ 1.0 pt, correct/F1 2–4 pts, abstention 2–5 pts. A p<.05 on correctness, F1 or abstention is not by itself evidence of an effect.
- Full-benchmark F1 and correct % reward answering: a configuration that abstains often scores low here even when its attempted answers are as accurate — read them beside coverage, never alone.

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
