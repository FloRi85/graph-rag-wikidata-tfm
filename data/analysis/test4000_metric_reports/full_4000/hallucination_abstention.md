# Hallucination and abstention — full benchmark (n = 4,000)

## 1. Definition and reading guide

- **Population:** every question of the sealed TEST split, for every configuration. Pair rows compare the identical id set; nothing is filtered to common attempts in this folder (see `../attempted/` for that).
- **Hallucination %** = 100 × questions with the scorer's HALLUCINATION outcome / n. **Abstention %** = 100 × ABSTENTION / n. The four outcomes (correct, hallucination, abstention, other) partition every group exactly; **coverage** = attempted / n = (correct + hallucination) / n.
- Other = empty, truncated or refusal-shaped-but-non-compliant replies: not evidence about accuracy, counted but never graded.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview

| Config | n | Hallucination n (%) | Abstention n (%) | Correct n (%) | Other n (%) | Coverage % | Hall. % of attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 Base LLM | 4000 | 991 (24.8) | 489 (12.2) | 2493 (62.3) | 27 (0.7) | 87.1 | 28.4 |
| C2 RAG | 4000 | 360 (9.0) | 1759 (44.0) | 1867 (46.7) | 14 (0.4) | 55.7 | 16.2 |
| C3 Graph-RAG | 4000 | 338 (8.5) | 2289 (57.2) | 1369 (34.2) | 4 (0.1) | 42.7 | 19.8 |
| C4 Graph-RAG + condensing | 4000 | 425 (10.6) | 2037 (50.9) | 1537 (38.4) | 1 (0.0) | 49.0 | 21.7 |

The last column is each configuration's OWN attempted set (denominators differ between rows; see `../attempted/hallucination_abstention.md`).

## 3. Dataset stratification

### 3.1 By answer type

Cells: hallucination % / abstention % over all questions in the group.

| Group | n | C1 H / A | C2 H / A | C3 H / A | C4 H / A |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | 21.4 / 14.6 | 6.5 / 49.8 | 6.0 / 63.4 | 7.2 / 57.9 |
| numerical | 655 | 42.1 / 13.9 | 18.5 / 48.9 | 17.3 / 60.3 | 24.4 / 49.5 |
| boolean | 573 | 16.9 / 3.1 | 8.9 / 19.4 | 6.8 / 32.6 | 7.3 / 28.1 |
| date | 265 | 30.2 / 5.7 | 9.4 / 30.9 | 12.8 / 46.4 | 15.1 / 39.2 |
| string | 8† | 50.0 / 0.0 | 0.0 / 12.5 | 12.5 / 0.0 | 37.5 / 0.0 |

Other counts (excluded from H and A): entity: C1 20, C2 9, C3 2; numerical: C1 3, C2 3, C3 2; boolean: C1 2, C2 2; date: C1 2, C4 1.

### 3.2 By question complexity

Cells: hallucination % / abstention % over all questions in the group.

| Group | n | C1 H / A | C2 H / A | C3 H / A | C4 H / A |
|---|---:|---:|---:|---:|---:|
| generic | 800 | 25.5 / 6.2 | 12.4 / 29.1 | 10.9 / 42.8 | 11.6 / 40.6 |
| intersection | 400 | 16.0 / 11.8 | 3.0 / 44.8 | 5.2 / 58.2 | 5.0 / 55.5 |
| count | 400 | 42.0 / 6.0 | 18.2 / 41.2 | 19.5 / 52.5 | 26.2 / 44.0 |
| comparative | 400 | 19.0 / 3.8 | 13.2 / 16.0 | 7.2 / 39.2 | 6.2 / 39.2 |
| yesno | 400 | 14.0 / 3.2 | 8.5 / 21.5 | 7.2 / 35.0 | 9.0 / 28.0 |
| ordinal | 400 | 28.5 / 11.2 | 6.5 / 45.8 | 7.2 / 62.7 | 9.2 / 56.2 |
| multihop | 400 | 30.0 / 24.2 | 6.2 / 71.8 | 2.8 / 83.5 | 9.8 / 64.5 |
| difference | 400 | 20.2 / 31.0 | 4.2 / 67.0 | 9.2 / 70.5 | 11.5 / 60.2 |
| superlative | 400 | 27.0 / 18.5 | 5.2 / 73.5 | 4.2 / 85.0 | 6.0 / 80.2 |

Other counts (excluded from H and A): generic: C1 1, C2 3, C3 2; intersection: C1 4; count: C1 1, C2 2; comparative: C1 3, C2 3, C3 1; yesno: C2 1; ordinal: C1 2, C2 1; multihop: C1 4, C2 1, C4 1; difference: C1 8, C2 3, C3 1; superlative: C1 4.

### 3.3 By topic category

Cells: hallucination % / abstention % over all questions in the group.

| Group | n | C1 H / A | C2 H / A | C3 H / A | C4 H / A |
|---|---:|---:|---:|---:|---:|
| history | 500 | 22.2 / 5.0 | 6.0 / 45.0 | 11.4 / 44.6 | 10.6 / 43.8 |
| movies | 500 | 28.4 / 11.4 | 10.4 / 38.4 | 8.4 / 53.2 | 14.0 / 44.2 |
| music | 500 | 33.6 / 14.4 | 11.2 / 43.2 | 8.4 / 67.0 | 11.4 / 60.0 |
| videogames | 500 | 24.0 / 10.0 | 6.4 / 39.2 | 8.2 / 56.2 | 10.4 / 48.4 |
| sports | 500 | 23.4 / 15.8 | 11.4 / 45.0 | 5.2 / 66.6 | 9.6 / 58.2 |
| books | 500 | 23.0 / 18.0 | 10.0 / 39.4 | 9.6 / 53.8 | 9.8 / 49.0 |
| geography | 500 | 23.2 / 12.2 | 8.2 / 50.6 | 9.6 / 57.8 | 10.0 / 53.6 |
| politics | 500 | 20.4 / 11.0 | 8.4 / 51.0 | 6.8 / 58.6 | 9.2 / 50.2 |

Other counts (excluded from H and A): history: C1 3, C2 4, C3 1; movies: C1 2, C2 2, C3 1; music: C1 8, C2 1, C3 1, C4 1; videogames: C1 2; sports: C1 2; books: C1 7, C2 4; geography: C1 1, C2 2; politics: C1 2, C2 1, C3 1.

## 4. Direct paired comparisons (identical id sets)

### 4.1 Overall, six pairs

ΔH = hallucination A − B; ΔA = abstention A − B (secondary: abstention has a 2–5 pt noise floor). McNemar: exact binomial on the discordant pairs (A-only / B-only hallucinations).

| Pair (A − B) | n | H% A | H% B | ΔH pp | 95% CI | p | A-only / B-only | McNemar p | ΔA pp | 95% CI (ΔA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | 4000 | 9.0 | 24.8 | -15.8 | [-17.2, -14.4] | <.001 | 144 / 775 | <.001 | +31.8 | [+30.2, +33.3] |
| C3 − C1 | 4000 | 8.5 | 24.8 | -16.3 | [-17.8, -14.8] | <.001 | 166 / 819 | <.001 | +45.0 | [+43.4, +46.6] |
| C4 − C1 | 4000 | 10.6 | 24.8 | -14.2 | [-15.7, -12.7] | <.001 | 212 / 778 | <.001 | +38.7 | [+37.1, +40.4] |
| C3 − C2 | 4000 | 8.5 | 9.0 | -0.5 | [-1.6, +0.5] | 0.323 | 238 / 260 | 0.347 | +13.3 | [+11.6, +14.9] |
| C4 − C2 | 4000 | 10.6 | 9.0 | +1.6 | [+0.5, +2.8] | 0.005 | 318 / 253 | 0.007 | +7.0 | [+5.2, +8.7] |
| C4 − C3 | 4000 | 10.6 | 8.5 | +2.2 | [+1.4, +3.0] | <.001 | 186 / 99 | <.001 | -6.3 | [-7.4, -5.1] |

### 4.2 C2 − C1 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C2 | H% C1 | ΔH pp | A% C2 | A% C1 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 6.5 | 21.4 | -14.8 | 49.8 | 14.6 | +35.2 |
| numerical | 655 | 18.5 | 42.1 | -23.7 | 48.9 | 13.9 | +35.0 |
| boolean | 573 | 8.9 | 16.9 | -8.0 | 19.4 | 3.1 | +16.2 |
| date | 265 | 9.4 | 30.2 | -20.8 | 30.9 | 5.7 | +25.3 |
| string | 8† | 0.0 | 50.0 | -50.0 | 12.5 | 0.0 | +12.5 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 12.4 | 25.5 | -13.1 | 29.1 | 6.2 | +22.9 |
| intersection | 400 | 3.0 | 16.0 | -13.0 | 44.8 | 11.8 | +33.0 |
| count | 400 | 18.2 | 42.0 | -23.8 | 41.2 | 6.0 | +35.2 |
| comparative | 400 | 13.2 | 19.0 | -5.8 | 16.0 | 3.8 | +12.2 |
| yesno | 400 | 8.5 | 14.0 | -5.5 | 21.5 | 3.2 | +18.2 |
| ordinal | 400 | 6.5 | 28.5 | -22.0 | 45.8 | 11.2 | +34.5 |
| multihop | 400 | 6.2 | 30.0 | -23.8 | 71.8 | 24.2 | +47.5 |
| difference | 400 | 4.2 | 20.2 | -16.0 | 67.0 | 31.0 | +36.0 |
| superlative | 400 | 5.2 | 27.0 | -21.8 | 73.5 | 18.5 | +55.0 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 6.0 | 22.2 | -16.2 | 45.0 | 5.0 | +40.0 |
| movies | 500 | 10.4 | 28.4 | -18.0 | 38.4 | 11.4 | +27.0 |
| music | 500 | 11.2 | 33.6 | -22.4 | 43.2 | 14.4 | +28.8 |
| videogames | 500 | 6.4 | 24.0 | -17.6 | 39.2 | 10.0 | +29.2 |
| sports | 500 | 11.4 | 23.4 | -12.0 | 45.0 | 15.8 | +29.2 |
| books | 500 | 10.0 | 23.0 | -13.0 | 39.4 | 18.0 | +21.4 |
| geography | 500 | 8.2 | 23.2 | -15.0 | 50.6 | 12.2 | +38.4 |
| politics | 500 | 8.4 | 20.4 | -12.0 | 51.0 | 11.0 | +40.0 |

### 4.3 C3 − C1 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C3 | H% C1 | ΔH pp | A% C3 | A% C1 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 6.0 | 21.4 | -15.3 | 63.4 | 14.6 | +48.8 |
| numerical | 655 | 17.3 | 42.1 | -24.9 | 60.3 | 13.9 | +46.4 |
| boolean | 573 | 6.8 | 16.9 | -10.1 | 32.6 | 3.1 | +29.5 |
| date | 265 | 12.8 | 30.2 | -17.4 | 46.4 | 5.7 | +40.8 |
| string | 8† | 12.5 | 50.0 | -37.5 | 0.0 | 0.0 | +0.0 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 10.9 | 25.5 | -14.6 | 42.8 | 6.2 | +36.5 |
| intersection | 400 | 5.2 | 16.0 | -10.8 | 58.2 | 11.8 | +46.5 |
| count | 400 | 19.5 | 42.0 | -22.5 | 52.5 | 6.0 | +46.5 |
| comparative | 400 | 7.2 | 19.0 | -11.8 | 39.2 | 3.8 | +35.5 |
| yesno | 400 | 7.2 | 14.0 | -6.8 | 35.0 | 3.2 | +31.8 |
| ordinal | 400 | 7.2 | 28.5 | -21.2 | 62.7 | 11.2 | +51.5 |
| multihop | 400 | 2.8 | 30.0 | -27.2 | 83.5 | 24.2 | +59.2 |
| difference | 400 | 9.2 | 20.2 | -11.0 | 70.5 | 31.0 | +39.5 |
| superlative | 400 | 4.2 | 27.0 | -22.8 | 85.0 | 18.5 | +66.5 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 11.4 | 22.2 | -10.8 | 44.6 | 5.0 | +39.6 |
| movies | 500 | 8.4 | 28.4 | -20.0 | 53.2 | 11.4 | +41.8 |
| music | 500 | 8.4 | 33.6 | -25.2 | 67.0 | 14.4 | +52.6 |
| videogames | 500 | 8.2 | 24.0 | -15.8 | 56.2 | 10.0 | +46.2 |
| sports | 500 | 5.2 | 23.4 | -18.2 | 66.6 | 15.8 | +50.8 |
| books | 500 | 9.6 | 23.0 | -13.4 | 53.8 | 18.0 | +35.8 |
| geography | 500 | 9.6 | 23.2 | -13.6 | 57.8 | 12.2 | +45.6 |
| politics | 500 | 6.8 | 20.4 | -13.6 | 58.6 | 11.0 | +47.6 |

### 4.4 C4 − C1 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C4 | H% C1 | ΔH pp | A% C4 | A% C1 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 7.2 | 21.4 | -14.2 | 57.9 | 14.6 | +43.3 |
| numerical | 655 | 24.4 | 42.1 | -17.7 | 49.5 | 13.9 | +35.6 |
| boolean | 573 | 7.3 | 16.9 | -9.6 | 28.1 | 3.1 | +25.0 |
| date | 265 | 15.1 | 30.2 | -15.1 | 39.2 | 5.7 | +33.6 |
| string | 8† | 37.5 | 50.0 | -12.5 | 0.0 | 0.0 | +0.0 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 11.6 | 25.5 | -13.9 | 40.6 | 6.2 | +34.4 |
| intersection | 400 | 5.0 | 16.0 | -11.0 | 55.5 | 11.8 | +43.8 |
| count | 400 | 26.2 | 42.0 | -15.7 | 44.0 | 6.0 | +38.0 |
| comparative | 400 | 6.2 | 19.0 | -12.8 | 39.2 | 3.8 | +35.5 |
| yesno | 400 | 9.0 | 14.0 | -5.0 | 28.0 | 3.2 | +24.8 |
| ordinal | 400 | 9.2 | 28.5 | -19.2 | 56.2 | 11.2 | +45.0 |
| multihop | 400 | 9.8 | 30.0 | -20.2 | 64.5 | 24.2 | +40.2 |
| difference | 400 | 11.5 | 20.2 | -8.8 | 60.2 | 31.0 | +29.3 |
| superlative | 400 | 6.0 | 27.0 | -21.0 | 80.2 | 18.5 | +61.7 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 10.6 | 22.2 | -11.6 | 43.8 | 5.0 | +38.8 |
| movies | 500 | 14.0 | 28.4 | -14.4 | 44.2 | 11.4 | +32.8 |
| music | 500 | 11.4 | 33.6 | -22.2 | 60.0 | 14.4 | +45.6 |
| videogames | 500 | 10.4 | 24.0 | -13.6 | 48.4 | 10.0 | +38.4 |
| sports | 500 | 9.6 | 23.4 | -13.8 | 58.2 | 15.8 | +42.4 |
| books | 500 | 9.8 | 23.0 | -13.2 | 49.0 | 18.0 | +31.0 |
| geography | 500 | 10.0 | 23.2 | -13.2 | 53.6 | 12.2 | +41.4 |
| politics | 500 | 9.2 | 20.4 | -11.2 | 50.2 | 11.0 | +39.2 |

### 4.5 C3 − C2 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C3 | H% C2 | ΔH pp | A% C3 | A% C2 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 6.0 | 6.5 | -0.5 | 63.4 | 49.8 | +13.6 |
| numerical | 655 | 17.3 | 18.5 | -1.2 | 60.3 | 48.9 | +11.5 |
| boolean | 573 | 6.8 | 8.9 | -2.1 | 32.6 | 19.4 | +13.3 |
| date | 265 | 12.8 | 9.4 | +3.4 | 46.4 | 30.9 | +15.5 |
| string | 8† | 12.5 | 0.0 | +12.5 | 0.0 | 12.5 | -12.5 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 10.9 | 12.4 | -1.5 | 42.8 | 29.1 | +13.6 |
| intersection | 400 | 5.2 | 3.0 | +2.2 | 58.2 | 44.8 | +13.5 |
| count | 400 | 19.5 | 18.2 | +1.3 | 52.5 | 41.2 | +11.3 |
| comparative | 400 | 7.2 | 13.2 | -6.0 | 39.2 | 16.0 | +23.2 |
| yesno | 400 | 7.2 | 8.5 | -1.3 | 35.0 | 21.5 | +13.5 |
| ordinal | 400 | 7.2 | 6.5 | +0.7 | 62.7 | 45.8 | +17.0 |
| multihop | 400 | 2.8 | 6.2 | -3.5 | 83.5 | 71.8 | +11.7 |
| difference | 400 | 9.2 | 4.2 | +5.0 | 70.5 | 67.0 | +3.5 |
| superlative | 400 | 4.2 | 5.2 | -1.0 | 85.0 | 73.5 | +11.5 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 11.4 | 6.0 | +5.4 | 44.6 | 45.0 | -0.4 |
| movies | 500 | 8.4 | 10.4 | -2.0 | 53.2 | 38.4 | +14.8 |
| music | 500 | 8.4 | 11.2 | -2.8 | 67.0 | 43.2 | +23.8 |
| videogames | 500 | 8.2 | 6.4 | +1.8 | 56.2 | 39.2 | +17.0 |
| sports | 500 | 5.2 | 11.4 | -6.2 | 66.6 | 45.0 | +21.6 |
| books | 500 | 9.6 | 10.0 | -0.4 | 53.8 | 39.4 | +14.4 |
| geography | 500 | 9.6 | 8.2 | +1.4 | 57.8 | 50.6 | +7.2 |
| politics | 500 | 6.8 | 8.4 | -1.6 | 58.6 | 51.0 | +7.6 |

### 4.6 C4 − C2 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C4 | H% C2 | ΔH pp | A% C4 | A% C2 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 7.2 | 6.5 | +0.7 | 57.9 | 49.8 | +8.1 |
| numerical | 655 | 24.4 | 18.5 | +6.0 | 49.5 | 48.9 | +0.6 |
| boolean | 573 | 7.3 | 8.9 | -1.6 | 28.1 | 19.4 | +8.7 |
| date | 265 | 15.1 | 9.4 | +5.7 | 39.2 | 30.9 | +8.3 |
| string | 8† | 37.5 | 0.0 | +37.5 | 0.0 | 12.5 | -12.5 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 11.6 | 12.4 | -0.7 | 40.6 | 29.1 | +11.5 |
| intersection | 400 | 5.0 | 3.0 | +2.0 | 55.5 | 44.8 | +10.8 |
| count | 400 | 26.2 | 18.2 | +8.0 | 44.0 | 41.2 | +2.8 |
| comparative | 400 | 6.2 | 13.2 | -7.0 | 39.2 | 16.0 | +23.2 |
| yesno | 400 | 9.0 | 8.5 | +0.5 | 28.0 | 21.5 | +6.5 |
| ordinal | 400 | 9.2 | 6.5 | +2.7 | 56.2 | 45.8 | +10.5 |
| multihop | 400 | 9.8 | 6.2 | +3.5 | 64.5 | 71.8 | -7.3 |
| difference | 400 | 11.5 | 4.2 | +7.3 | 60.2 | 67.0 | -6.8 |
| superlative | 400 | 6.0 | 5.2 | +0.8 | 80.2 | 73.5 | +6.8 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 10.6 | 6.0 | +4.6 | 43.8 | 45.0 | -1.2 |
| movies | 500 | 14.0 | 10.4 | +3.6 | 44.2 | 38.4 | +5.8 |
| music | 500 | 11.4 | 11.2 | +0.2 | 60.0 | 43.2 | +16.8 |
| videogames | 500 | 10.4 | 6.4 | +4.0 | 48.4 | 39.2 | +9.2 |
| sports | 500 | 9.6 | 11.4 | -1.8 | 58.2 | 45.0 | +13.2 |
| books | 500 | 9.8 | 10.0 | -0.2 | 49.0 | 39.4 | +9.6 |
| geography | 500 | 10.0 | 8.2 | +1.8 | 53.6 | 50.6 | +3.0 |
| politics | 500 | 9.2 | 8.4 | +0.8 | 50.2 | 51.0 | -0.8 |

### 4.7 C4 − C3 by attribute

Descriptive per group (no tests). Columns: hallucination A, B, Δ; abstention A, B, Δ.

| Group | n | H% C4 | H% C3 | ΔH pp | A% C4 | A% C3 | ΔA pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 2499 | 7.2 | 6.0 | +1.2 | 57.9 | 63.4 | -5.4 |
| numerical | 655 | 24.4 | 17.3 | +7.2 | 49.5 | 60.3 | -10.8 |
| boolean | 573 | 7.3 | 6.8 | +0.5 | 28.1 | 32.6 | -4.5 |
| date | 265 | 15.1 | 12.8 | +2.3 | 39.2 | 46.4 | -7.2 |
| string | 8† | 37.5 | 12.5 | +25.0 | 0.0 | 0.0 | +0.0 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 800 | 11.6 | 10.9 | +0.8 | 40.6 | 42.8 | -2.1 |
| intersection | 400 | 5.0 | 5.2 | -0.2 | 55.5 | 58.2 | -2.7 |
| count | 400 | 26.2 | 19.5 | +6.8 | 44.0 | 52.5 | -8.5 |
| comparative | 400 | 6.2 | 7.2 | -1.0 | 39.2 | 39.2 | +0.0 |
| yesno | 400 | 9.0 | 7.2 | +1.8 | 28.0 | 35.0 | -7.0 |
| ordinal | 400 | 9.2 | 7.2 | +2.0 | 56.2 | 62.7 | -6.5 |
| multihop | 400 | 9.8 | 2.8 | +7.0 | 64.5 | 83.5 | -19.0 |
| difference | 400 | 11.5 | 9.2 | +2.3 | 60.2 | 70.5 | -10.2 |
| superlative | 400 | 6.0 | 4.2 | +1.7 | 80.2 | 85.0 | -4.7 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 500 | 10.6 | 11.4 | -0.8 | 43.8 | 44.6 | -0.8 |
| movies | 500 | 14.0 | 8.4 | +5.6 | 44.2 | 53.2 | -9.0 |
| music | 500 | 11.4 | 8.4 | +3.0 | 60.0 | 67.0 | -7.0 |
| videogames | 500 | 10.4 | 8.2 | +2.2 | 48.4 | 56.2 | -7.8 |
| sports | 500 | 9.6 | 5.2 | +4.4 | 58.2 | 66.6 | -8.4 |
| books | 500 | 9.8 | 9.6 | +0.2 | 49.0 | 53.8 | -4.8 |
| geography | 500 | 10.0 | 9.6 | +0.4 | 53.6 | 57.8 | -4.2 |
| politics | 500 | 9.2 | 6.8 | +2.4 | 50.2 | 58.6 | -8.4 |

## 5. Supplements: conditional outcomes

### 5.1 Outcome-transition matrices

For each pair, rows are the outcome of the reference configuration B and columns the outcome of A on the same questions; cells are counts with the row share in brackets. Read a row as *"where B did X, A did …"*. These are descriptive comparisons of independent outputs on a selected subset, not causal decompositions.

**C2 − C1** — rows: C1 outcome · columns: C2 outcome

|  | row n | C2 correct | C2 hallucination | C2 abstention | C2 other |
|---|---:|---:|---:|---:|---:|
| C1 correct | 2493 | 1589 (63.7) | 125 (5.0) | 774 (31.0) | 5 (0.2) |
| C1 hallucination | 991 | 236 (23.8) | 216 (21.8) | 536 (54.1) | 3 (0.3) |
| C1 abstention | 489 | 37 (7.6) | 18 (3.7) | 431 (88.1) | 3 (0.6) |
| C1 other | 27 | 5 (18.5) | 1 (3.7) | 18 (66.7) | 3 (11.1) |

**C3 − C1** — rows: C1 outcome · columns: C3 outcome

|  | row n | C3 correct | C3 hallucination | C3 abstention | C3 other |
|---|---:|---:|---:|---:|---:|
| C1 correct | 2493 | 1192 (47.8) | 134 (5.4) | 1166 (46.8) | 1 (0.0) |
| C1 hallucination | 991 | 153 (15.4) | 172 (17.4) | 663 (66.9) | 3 (0.3) |
| C1 abstention | 489 | 22 (4.5) | 29 (5.9) | 438 (89.6) | 0 (0.0) |
| C1 other | 27 | 2 (7.4) | 3 (11.1) | 22 (81.5) | 0 (0.0) |

**C4 − C1** — rows: C1 outcome · columns: C4 outcome

|  | row n | C4 correct | C4 hallucination | C4 abstention | C4 other |
|---|---:|---:|---:|---:|---:|
| C1 correct | 2493 | 1297 (52.0) | 156 (6.3) | 1040 (41.7) | 0 (0.0) |
| C1 hallucination | 991 | 198 (20.0) | 213 (21.5) | 579 (58.4) | 1 (0.1) |
| C1 abstention | 489 | 42 (8.6) | 51 (10.4) | 396 (81.0) | 0 (0.0) |
| C1 other | 27 | 0 (0.0) | 5 (18.5) | 22 (81.5) | 0 (0.0) |

**C3 − C2** — rows: C2 outcome · columns: C3 outcome

|  | row n | C3 correct | C3 hallucination | C3 abstention | C3 other |
|---|---:|---:|---:|---:|---:|
| C2 correct | 1867 | 1094 (58.6) | 108 (5.8) | 665 (35.6) | 0 (0.0) |
| C2 hallucination | 360 | 69 (19.2) | 100 (27.8) | 189 (52.5) | 2 (0.6) |
| C2 abstention | 1759 | 200 (11.4) | 129 (7.3) | 1429 (81.2) | 1 (0.1) |
| C2 other | 14 | 6 (42.9) | 1 (7.1) | 6 (42.9) | 1 (7.1) |

**C4 − C2** — rows: C2 outcome · columns: C4 outcome

|  | row n | C4 correct | C4 hallucination | C4 abstention | C4 other |
|---|---:|---:|---:|---:|---:|
| C2 correct | 1867 | 1127 (60.4) | 126 (6.7) | 614 (32.9) | 0 (0.0) |
| C2 hallucination | 360 | 81 (22.5) | 107 (29.7) | 171 (47.5) | 1 (0.3) |
| C2 abstention | 1759 | 323 (18.4) | 190 (10.8) | 1246 (70.8) | 0 (0.0) |
| C2 other | 14 | 6 (42.9) | 2 (14.3) | 6 (42.9) | 0 (0.0) |

**C4 − C3** — rows: C3 outcome · columns: C4 outcome

|  | row n | C4 correct | C4 hallucination | C4 abstention | C4 other |
|---|---:|---:|---:|---:|---:|
| C3 correct | 1369 | 1229 (89.8) | 33 (2.4) | 107 (7.8) | 0 (0.0) |
| C3 hallucination | 338 | 51 (15.1) | 239 (70.7) | 48 (14.2) | 0 (0.0) |
| C3 abstention | 2289 | 257 (11.2) | 150 (6.6) | 1881 (82.2) | 1 (0.0) |
| C3 other | 4 | 0 (0.0) | 3 (75.0) | 1 (25.0) | 0 (0.0) |

### 5.2 Exact attempt patterns

Which configurations attempted each question. Unlike the common-attempt intersections in `../attempted/`, these mutually exclusive patterns partition the 4,000 questions. They describe coverage overlap, not accuracy.

| Configurations that attempted | n | % of 4,000 |
|---|---:|---:|
| C1 | 833 | 20.8 |
| C1+C2 | 639 | 16.0 |
| C1+C2+C3 | 112 | 2.8 |
| C1+C2+C3+C4 | 1238 | 30.9 |
| C1+C2+C4 | 177 | 4.4 |
| C1+C3 | 36 | 0.9 |
| C1+C3+C4 | 265 | 6.6 |
| C1+C4 | 184 | 4.6 |
| C2 | 33 | 0.8 |
| C2+C3 | 2 | 0.1 |
| C2+C3+C4 | 19 | 0.5 |
| C2+C4 | 7 | 0.2 |
| C3 | 5 | 0.1 |
| C3+C4 | 30 | 0.8 |
| C4 | 42 | 1.1 |
| none | 378 | 9.4 |

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- Full-benchmark hallucination, lowest to highest: C3 8.5% < C2 9.0% < C4 10.6% < C1 24.8%.
- Abstention, lowest to highest: C1 12.2% < C2 44.0% < C4 50.9% < C3 57.2%.
- By answer type (groups with n ≥ 50): C1 highest on numerical (42.1%), lowest on boolean (16.9%); C2 highest on numerical (18.5%), lowest on entity (6.5%); C3 highest on numerical (17.3%), lowest on entity (6.0%); C4 highest on numerical (24.4%), lowest on entity (7.2%).
- By question complexity (groups with n ≥ 50): C1 highest on count (42.0%), lowest on yesno (14.0%); C2 highest on count (18.2%), lowest on intersection (3.0%); C3 highest on count (19.5%), lowest on multihop (2.8%); C4 highest on count (26.2%), lowest on intersection (5.0%).
- By topic category (groups with n ≥ 50): C1 highest on music (33.6%), lowest on politics (20.4%); C2 highest on sports (11.4%), lowest on history (6.0%); C3 highest on history (11.4%), lowest on sports (5.2%); C4 highest on movies (14.0%), lowest on politics (9.2%).
- C2 − C1: ΔH -15.8 pp [-17.2, -14.4], bootstrap p <.001, McNemar p <.001 — CI excludes zero.
- C3 − C1: ΔH -16.3 pp [-17.8, -14.8], bootstrap p <.001, McNemar p <.001 — CI excludes zero.
- C4 − C1: ΔH -14.2 pp [-15.7, -12.7], bootstrap p <.001, McNemar p <.001 — CI excludes zero.
- C3 − C2: ΔH -0.5 pp [-1.6, +0.5], bootstrap p 0.323, McNemar p 0.347 — CI includes zero.
- C4 − C2: ΔH +1.6 pp [+0.5, +2.8], bootstrap p 0.005, McNemar p 0.007 — CI excludes zero.
- C4 − C3: ΔH +2.2 pp [+1.4, +3.0], bootstrap p <.001, McNemar p <.001 — CI excludes zero.
- Where C1's outcome is hallucination (991 questions), C2: correct 236 (23.8%), hallucination 216 (21.8%), abstention 536 (54.1%), other 3.
- Where C1's outcome is abstention (489 questions), C2: correct 37 (7.6%), hallucination 18 (3.7%), abstention 431 (88.1%), other 3.
- Where C1's outcome is hallucination (991 questions), C3: correct 153 (15.4%), hallucination 172 (17.4%), abstention 663 (66.9%), other 3.
- Where C1's outcome is abstention (489 questions), C3: correct 22 (4.5%), hallucination 29 (5.9%), abstention 438 (89.6%), other 0.
- Where C1's outcome is hallucination (991 questions), C4: correct 198 (20.0%), hallucination 213 (21.5%), abstention 579 (58.4%), other 1.
- Where C1's outcome is abstention (489 questions), C4: correct 42 (8.6%), hallucination 51 (10.4%), abstention 396 (81.0%), other 0.
- Where C2's outcome is hallucination (360 questions), C3: correct 69 (19.2%), hallucination 100 (27.8%), abstention 189 (52.5%), other 2.
- Where C2's outcome is abstention (1759 questions), C3: correct 200 (11.4%), hallucination 129 (7.3%), abstention 1429 (81.2%), other 1.
- Where C2's outcome is hallucination (360 questions), C4: correct 81 (22.5%), hallucination 107 (29.7%), abstention 171 (47.5%), other 1.
- Where C2's outcome is abstention (1759 questions), C4: correct 323 (18.4%), hallucination 190 (10.8%), abstention 1246 (70.8%), other 0.
- Where C3's outcome is hallucination (338 questions), C4: correct 51 (15.1%), hallucination 239 (70.7%), abstention 48 (14.2%), other 0.
- Where C3's outcome is abstention (2289 questions), C4: correct 257 (11.2%), hallucination 150 (6.6%), abstention 1881 (82.2%), other 1.
- Tiny groups (†): string (n=8) — no ranking should rest on them.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Hallucination here is confident disagreement with the 2021 Mintaka gold (a factuality proxy), scored by the shared type-aware scorer at threshold 0.5; correctness is that scorer's CORRECT outcome, not exact match or verified world truth.
- C1 never receives the question-entity block that C2–C4 receive, so a C1-vs-retrieval delta measures retrieval plus that annotation. C4's prompt differs from C3's in more than the condensing step.
- Run-to-run noise (byte-identical repeats): hallucination ≤ 1.0 pt, correct/F1 2–4 pts, abstention 2–5 pts. A p<.05 on correctness, F1 or abstention is not by itself evidence of an effect.
- Lower hallucination does not imply higher coverage or F1: the configurations with the fewest confident errors are also the ones that abstain most (see section 2 and `f1_correctness.md`).

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
