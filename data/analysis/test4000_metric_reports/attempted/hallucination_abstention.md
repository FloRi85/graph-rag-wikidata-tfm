# Hallucination — attempted answers (each configuration's own attempts; pairs on common attempts)

## 1. Definition and reading guide

- **Attempted** = the scorer's `is_attempted`: outcome CORRECT or HALLUCINATION. Abstention AND Other are excluded — attempted does not mean "not abstention".
- **Hallucination % of attempts** = 100 × HALLUCINATION / attempted. Within attempts, correct % is its complement (they sum to 100), so a test on one IS the test on the other — `f1_correctness.md` does not repeat it.
- Overview and attribute tables use each configuration's OWN attempted ids (denominators differ per config). Pair rows use the INTERSECTION of the two attempt sets and recompute both rates there; they never subtract two own-attempted rates.
- These are answer-selected, post-treatment populations: a common id set removes question-set mismatch within a comparison but not selection bias, and identifies no causal context effect. An inclusive C1/C2 intersection also contains questions C3/C4 attempted.
- **Retained share** = attempted / benchmark n for the same group.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview (own attempts)

| Config | Benchmark n | Attempted n | Retained % | Excl. abstention | Excl. other | Hallucination n (% of attempts) | Correct % of attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 Base LLM | 4000 | 3484 | 87.1 | 489 | 27 | 991 (28.4) | 71.6 |
| C2 RAG | 4000 | 2227 | 55.7 | 1759 | 14 | 360 (16.2) | 83.8 |
| C3 Graph-RAG | 4000 | 1707 | 42.7 | 2289 | 4 | 338 (19.8) | 80.2 |
| C4 Graph-RAG + condensing | 4000 | 1962 | 49.0 | 2037 | 1 | 425 (21.7) | 78.3 |

## 3. Dataset stratification (own attempts)

### 3.1 By answer type

Cells: attempted n / hallucination % of those attempts. Group n is the benchmark count.

| Group | n | C1 att. n / H% | C2 att. n / H% | C3 att. n / H% | C4 att. n / H% |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | 2114 / 25.3 | 1245 / 13.1 | 913 / 16.5 | 1051 / 17.1 |
| numerical | 655 | 561 / 49.2 | 332 / 36.4 | 258 / 43.8 | 331 / 48.3 |
| boolean | 573 | 553 / 17.5 | 460 / 11.1 | 386 / 10.1 | 412 / 10.2 |
| date | 265 | 248 / 32.3 | 183 / 13.7 | 142 / 23.9 | 160 / 25.0 |
| string | 8† | 8† / 50.0 | 7† / 0.0 | 8† / 12.5 | 8† / 37.5 |

### 3.2 By question complexity

Cells: attempted n / hallucination % of those attempts. Group n is the benchmark count.

| Group | n | C1 att. n / H% | C2 att. n / H% | C3 att. n / H% | C4 att. n / H% |
|---|---:|---:|---:|---:|---:|
| generic | 800 | 749 / 27.2 | 564 / 17.6 | 456 / 19.1 | 475 / 19.6 |
| intersection | 400 | 349 / 18.3 | 221 / 5.4 | 167 / 12.6 | 178 / 11.2 |
| count | 400 | 375 / 44.8 | 233 / 31.3 | 190 / 41.1 | 224 / 46.9 |
| comparative | 400 | 382 / 19.9 | 333 / 15.9 | 242 / 12.0 | 243 / 10.3 |
| yesno | 400 | 387 / 14.5 | 313 / 10.9 | 260 / 11.2 | 288 / 12.5 |
| ordinal | 400 | 353 / 32.3 | 216 / 12.0 | 149 / 19.5 | 175 / 21.1 |
| multihop | 400 | 299 / 40.1 | 112 / 22.3 | 66 / 16.7 | 141 / 27.7 |
| difference | 400 | 268 / 30.2 | 129 / 13.2 | 117 / 31.6 | 159 / 28.9 |
| superlative | 400 | 322 / 33.5 | 106 / 19.8 | 60 / 28.3 | 79 / 30.4 |

### 3.3 By topic category

Cells: attempted n / hallucination % of those attempts. Group n is the benchmark count.

| Group | n | C1 att. n / H% | C2 att. n / H% | C3 att. n / H% | C4 att. n / H% |
|---|---:|---:|---:|---:|---:|
| history | 500 | 472 / 23.5 | 271 / 11.1 | 276 / 20.7 | 281 / 18.9 |
| movies | 500 | 441 / 32.2 | 306 / 17.0 | 233 / 18.0 | 279 / 25.1 |
| music | 500 | 420 / 40.0 | 283 / 19.8 | 164 / 25.6 | 199 / 28.6 |
| videogames | 500 | 448 / 26.8 | 304 / 10.5 | 219 / 18.7 | 258 / 20.2 |
| sports | 500 | 419 / 27.9 | 275 / 20.7 | 167 / 15.6 | 209 / 23.0 |
| books | 500 | 403 / 28.5 | 299 / 16.7 | 231 / 20.8 | 255 / 19.2 |
| geography | 500 | 438 / 26.5 | 245 / 16.7 | 211 / 22.7 | 232 / 21.6 |
| politics | 500 | 443 / 23.0 | 244 / 17.2 | 206 / 16.5 | 249 / 18.5 |

## 4. Direct paired comparisons (common attempts)

### 4.1 Overall, six pairs

Both rates recomputed on the questions BOTH configurations attempted. Discordant cells: A-only = A hallucinated and B was correct; B-only the reverse.

| Pair (A − B) | Common n | Retained % of 4,000 | H% A | H% B | ΔH pp | 95% CI | p | both / A-only / B-only / neither | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | 2166 | 54.1 | 15.7 | 20.9 | -5.1 | [-6.8, -3.4] | <.001 | 216 / 125 / 236 / 1589 | <.001 |
| C3 − C1 | 1651 | 41.3 | 18.5 | 19.7 | -1.2 | [-3.1, +0.9] | 0.278 | 172 / 134 / 153 / 1192 | 0.288 |
| C4 − C1 | 1864 | 46.6 | 19.8 | 22.0 | -2.3 | [-4.2, -0.2] | 0.028 | 213 / 156 / 198 / 1297 | 0.029 |
| C3 − C2 | 1371 | 34.3 | 15.2 | 12.3 | +2.8 | [+0.9, +4.8] | 0.004 | 100 / 108 / 69 / 1094 | 0.004 |
| C4 − C2 | 1441 | 36.0 | 16.2 | 13.0 | +3.1 | [+1.2, +5.1] | 0.001 | 107 / 126 / 81 / 1127 | 0.002 |
| C4 − C3 | 1552 | 38.8 | 17.5 | 18.7 | -1.2 | [-2.3, +0.0] | 0.052 | 239 / 33 / 51 / 1229 | 0.063 |

### 4.2 C2 − C1 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C2 | H% C1 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 1198 | 47.9 | 12.4 | 16.4 | -3.9 |
| numerical | 655 | 323 | 49.3 | 36.2 | 39.3 | -3.1 |
| boolean | 573 | 456 | 79.6 | 11.2 | 16.7 | -5.5 |
| date | 265 | 182 | 68.7 | 13.2 | 26.9 | -13.7 |
| string | 8† | 7† | 87.5 | 0.0 | 57.1 | -57.1 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 552 | 69.0 | 17.0 | 21.2 | -4.2 |
| intersection | 400 | 209 | 52.2 | 4.3 | 12.9 | -8.6 |
| count | 400 | 231 | 57.8 | 31.6 | 35.1 | -3.5 |
| comparative | 400 | 329 | 82.2 | 15.8 | 19.5 | -3.6 |
| yesno | 400 | 312 | 78.0 | 10.9 | 13.5 | -2.6 |
| ordinal | 400 | 208 | 52.0 | 10.6 | 23.6 | -13.0 |
| multihop | 400 | 110 | 27.5 | 22.7 | 26.4 | -3.6 |
| difference | 400 | 112 | 28.0 | 11.6 | 18.8 | -7.1 |
| superlative | 400 | 103 | 25.8 | 18.4 | 21.4 | -2.9 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 270 | 54.0 | 11.1 | 13.3 | -2.2 |
| movies | 500 | 295 | 59.0 | 16.6 | 25.8 | -9.2 |
| music | 500 | 276 | 55.2 | 20.3 | 29.7 | -9.4 |
| videogames | 500 | 298 | 59.6 | 10.7 | 23.2 | -12.4 |
| sports | 500 | 266 | 53.2 | 19.2 | 21.4 | -2.3 |
| books | 500 | 281 | 56.2 | 15.7 | 23.1 | -7.5 |
| geography | 500 | 241 | 48.2 | 15.8 | 12.9 | +2.9 |
| politics | 500 | 239 | 47.8 | 17.2 | 15.1 | +2.1 |

### 4.3 C3 − C1 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C3 | H% C1 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 869 | 34.8 | 14.5 | 14.3 | +0.2 |
| numerical | 655 | 253 | 38.6 | 43.5 | 38.3 | +5.1 |
| boolean | 573 | 382 | 66.7 | 9.7 | 16.2 | -6.5 |
| date | 265 | 139 | 52.5 | 23.0 | 27.3 | -4.3 |
| string | 8† | 8† | 100.0 | 12.5 | 50.0 | -37.5 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 441 | 55.1 | 17.5 | 18.6 | -1.1 |
| intersection | 400 | 162 | 40.5 | 11.7 | 10.5 | +1.2 |
| count | 400 | 188 | 47.0 | 41.0 | 36.2 | +4.8 |
| comparative | 400 | 239 | 59.8 | 11.3 | 21.3 | -10.0 |
| yesno | 400 | 258 | 64.5 | 10.9 | 12.0 | -1.2 |
| ordinal | 400 | 144 | 36.0 | 18.1 | 24.3 | -6.2 |
| multihop | 400 | 66 | 16.5 | 16.7 | 13.6 | +3.0 |
| difference | 400 | 96 | 24.0 | 27.1 | 26.0 | +1.0 |
| superlative | 400 | 57 | 14.2 | 26.3 | 12.3 | +14.0 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 270 | 54.0 | 19.6 | 16.3 | +3.3 |
| movies | 500 | 226 | 45.2 | 16.4 | 22.6 | -6.2 |
| music | 500 | 157 | 31.4 | 24.2 | 28.0 | -3.8 |
| videogames | 500 | 214 | 42.8 | 17.8 | 18.7 | -0.9 |
| sports | 500 | 160 | 32.0 | 15.0 | 19.4 | -4.4 |
| books | 500 | 218 | 43.6 | 18.3 | 20.6 | -2.3 |
| geography | 500 | 205 | 41.0 | 22.0 | 18.5 | +3.4 |
| politics | 500 | 201 | 40.2 | 15.4 | 15.9 | -0.5 |

### 4.4 C4 − C1 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C4 | H% C1 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 980 | 39.2 | 14.6 | 16.3 | -1.7 |
| numerical | 655 | 314 | 47.9 | 45.9 | 40.8 | +5.1 |
| boolean | 573 | 405 | 70.7 | 9.9 | 17.0 | -7.2 |
| date | 265 | 157 | 59.2 | 24.8 | 31.8 | -7.0 |
| string | 8† | 8† | 100.0 | 37.5 | 50.0 | -12.5 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 459 | 57.4 | 17.9 | 19.2 | -1.3 |
| intersection | 400 | 171 | 42.8 | 9.4 | 9.4 | +0.0 |
| count | 400 | 217 | 54.2 | 45.2 | 37.3 | +7.8 |
| comparative | 400 | 238 | 59.5 | 9.2 | 21.4 | -12.2 |
| yesno | 400 | 283 | 70.8 | 12.0 | 12.7 | -0.7 |
| ordinal | 400 | 167 | 41.8 | 19.8 | 27.5 | -7.8 |
| multihop | 400 | 130 | 32.5 | 26.9 | 33.8 | -6.9 |
| difference | 400 | 125 | 31.2 | 23.2 | 28.8 | -5.6 |
| superlative | 400 | 74 | 18.5 | 27.0 | 17.6 | +9.5 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 275 | 55.0 | 17.8 | 18.2 | -0.4 |
| movies | 500 | 263 | 52.6 | 24.0 | 27.8 | -3.8 |
| music | 500 | 190 | 38.0 | 26.8 | 32.6 | -5.8 |
| videogames | 500 | 249 | 49.8 | 18.9 | 21.3 | -2.4 |
| sports | 500 | 194 | 38.8 | 20.6 | 19.6 | +1.0 |
| books | 500 | 233 | 46.6 | 16.3 | 20.2 | -3.9 |
| geography | 500 | 223 | 44.6 | 19.3 | 19.7 | -0.4 |
| politics | 500 | 237 | 47.4 | 16.0 | 18.6 | -2.5 |

### 4.5 C3 − C2 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C3 | H% C2 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 689 | 27.6 | 11.0 | 8.6 | +2.5 |
| numerical | 655 | 203 | 31.0 | 37.9 | 30.5 | +7.4 |
| boolean | 573 | 353 | 61.6 | 8.5 | 9.1 | -0.6 |
| date | 265 | 119 | 44.9 | 20.2 | 13.4 | +6.7 |
| string | 8† | 7† | 87.5 | 14.3 | 0.0 | +14.3 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 403 | 50.4 | 17.1 | 14.1 | +3.0 |
| intersection | 400 | 128 | 32.0 | 7.0 | 5.5 | +1.6 |
| count | 400 | 143 | 35.8 | 34.3 | 25.9 | +8.4 |
| comparative | 400 | 229 | 57.2 | 10.9 | 13.5 | -2.6 |
| yesno | 400 | 233 | 58.2 | 9.0 | 8.2 | +0.9 |
| ordinal | 400 | 105 | 26.2 | 14.3 | 5.7 | +8.6 |
| multihop | 400 | 41† | 10.2 | 12.2 | 7.3 | +4.9 |
| difference | 400 | 62 | 15.5 | 17.7 | 11.3 | +6.5 |
| superlative | 400 | 27† | 6.8 | 14.8 | 7.4 | +7.4 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 199 | 39.8 | 11.6 | 9.5 | +2.0 |
| movies | 500 | 201 | 40.2 | 14.9 | 11.4 | +3.5 |
| music | 500 | 136 | 27.2 | 19.1 | 12.5 | +6.6 |
| videogames | 500 | 182 | 36.4 | 14.8 | 6.0 | +8.8 |
| sports | 500 | 139 | 27.8 | 15.8 | 15.8 | +0.0 |
| books | 500 | 193 | 38.6 | 16.6 | 14.0 | +2.6 |
| geography | 500 | 170 | 34.0 | 17.6 | 14.7 | +2.9 |
| politics | 500 | 151 | 30.2 | 11.9 | 16.6 | -4.6 |

### 4.6 C4 − C2 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C4 | H% C2 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 737 | 29.5 | 11.7 | 9.8 | +1.9 |
| numerical | 655 | 226 | 34.5 | 42.0 | 31.0 | +11.1 |
| boolean | 573 | 347 | 60.6 | 8.1 | 8.6 | -0.6 |
| date | 265 | 124 | 46.8 | 17.7 | 12.9 | +4.8 |
| string | 8† | 7† | 87.5 | 28.6 | 0.0 | +28.6 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 410 | 51.2 | 17.6 | 14.9 | +2.7 |
| intersection | 400 | 131 | 32.8 | 6.9 | 5.3 | +1.5 |
| count | 400 | 156 | 39.0 | 39.7 | 24.4 | +15.4 |
| comparative | 400 | 222 | 55.5 | 9.0 | 13.5 | -4.5 |
| yesno | 400 | 234 | 58.5 | 9.4 | 8.1 | +1.3 |
| ordinal | 400 | 123 | 30.8 | 14.6 | 8.1 | +6.5 |
| multihop | 400 | 58 | 14.5 | 19.0 | 19.0 | +0.0 |
| difference | 400 | 75 | 18.8 | 18.7 | 9.3 | +9.3 |
| superlative | 400 | 32† | 8.0 | 15.6 | 15.6 | +0.0 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 198 | 39.6 | 13.6 | 10.1 | +3.5 |
| movies | 500 | 216 | 43.2 | 20.4 | 14.8 | +5.6 |
| music | 500 | 144 | 28.8 | 18.8 | 13.2 | +5.6 |
| videogames | 500 | 200 | 40.0 | 15.0 | 7.0 | +8.0 |
| sports | 500 | 157 | 31.4 | 21.7 | 17.2 | +4.5 |
| books | 500 | 199 | 39.8 | 16.1 | 14.1 | +2.0 |
| geography | 500 | 167 | 33.4 | 13.2 | 14.4 | -1.2 |
| politics | 500 | 160 | 32.0 | 10.6 | 15.0 | -4.4 |

### 4.7 C4 − C3 by attribute

Descriptive per group (no tests). Common n = questions both attempted within the group; retained % is relative to the group's benchmark n.

| Group | n | Common n | Retained % | H% C4 | H% C3 | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |
| entity | 2499 | 815 | 32.6 | 13.7 | 15.5 | -1.7 |
| numerical | 655 | 238 | 36.3 | 41.2 | 42.9 | -1.7 |
| boolean | 573 | 355 | 62.0 | 8.2 | 8.5 | -0.3 |
| date | 265 | 136 | 51.3 | 22.1 | 22.8 | -0.7 |
| string | 8† | 8† | 100.0 | 37.5 | 12.5 | +25.0 |
| **Question complexity** |  |  |  |  |  |  |
| generic | 800 | 428 | 53.5 | 18.5 | 18.5 | +0.0 |
| intersection | 400 | 141 | 35.2 | 7.8 | 8.5 | -0.7 |
| count | 400 | 174 | 43.5 | 38.5 | 39.7 | -1.1 |
| comparative | 400 | 216 | 54.0 | 8.3 | 11.6 | -3.2 |
| yesno | 400 | 241 | 60.2 | 9.5 | 8.7 | +0.8 |
| ordinal | 400 | 139 | 34.8 | 17.3 | 18.7 | -1.4 |
| multihop | 400 | 55 | 13.8 | 14.5 | 16.4 | -1.8 |
| difference | 400 | 106 | 26.5 | 25.5 | 31.1 | -5.7 |
| superlative | 400 | 52 | 13.0 | 28.8 | 30.8 | -1.9 |
| **Topic category** |  |  |  |  |  |  |
| history | 500 | 250 | 50.0 | 18.8 | 19.6 | -0.8 |
| movies | 500 | 211 | 42.2 | 17.5 | 18.5 | -0.9 |
| music | 500 | 152 | 30.4 | 21.1 | 22.4 | -1.3 |
| videogames | 500 | 202 | 40.4 | 16.3 | 17.3 | -1.0 |
| sports | 500 | 155 | 31.0 | 15.5 | 15.5 | +0.0 |
| books | 500 | 210 | 42.0 | 17.1 | 18.1 | -1.0 |
| geography | 500 | 192 | 38.4 | 19.3 | 21.4 | -2.1 |
| politics | 500 | 180 | 36.0 | 14.4 | 16.7 | -2.2 |

## 5. Supplement: questions attempted by all four configurations

One common set for all four (n = 1238, 30.9% of the benchmark). The most selected population in this supplement: every configuration chose to answer.

| Group | Common n | Benchmark n | Retained % | C1 H% | C2 H% | C3 H% | C4 H% |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 1238 | 4000 | 30.9 | 16.6 | 11.6 | 13.9 | 13.7 |
| **Answer type** |  |  |  |  |  |  |  |
| entity | 609 | 2499 | 24.4 | 10.5 | 8.0 | 9.7 | 9.0 |
| numerical | 186 | 655 | 28.4 | 32.3 | 29.0 | 37.1 | 38.2 |
| boolean | 321 | 573 | 56.0 | 15.0 | 8.1 | 6.5 | 6.2 |
| date | 115 | 265 | 43.4 | 25.2 | 13.0 | 19.1 | 18.3 |
| string | 7† | 8 | 87.5 | 57.1 | 0.0 | 14.3 | 28.6 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 376 | 800 | 47.0 | 17.0 | 13.0 | 15.7 | 15.7 |
| intersection | 107 | 400 | 26.8 | 5.6 | 3.7 | 3.7 | 4.7 |
| count | 132 | 400 | 33.0 | 28.8 | 25.0 | 33.3 | 34.8 |
| comparative | 203 | 400 | 50.7 | 21.7 | 13.3 | 10.3 | 7.9 |
| yesno | 215 | 400 | 53.8 | 9.3 | 7.0 | 6.5 | 6.5 |
| ordinal | 99 | 400 | 24.8 | 18.2 | 6.1 | 14.1 | 12.1 |
| multihop | 34† | 400 | 8.5 | 11.8 | 8.8 | 11.8 | 11.8 |
| difference | 51 | 400 | 12.8 | 15.7 | 11.8 | 17.6 | 19.6 |
| superlative | 21† | 400 | 5.2 | 14.3 | 4.8 | 14.3 | 14.3 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 185 | 500 | 37.0 | 14.1 | 9.7 | 11.9 | 13.0 |
| movies | 178 | 500 | 35.6 | 22.5 | 11.8 | 14.0 | 15.7 |
| music | 127 | 500 | 25.4 | 21.3 | 11.8 | 16.5 | 16.5 |
| videogames | 166 | 500 | 33.2 | 14.5 | 5.4 | 13.3 | 12.0 |
| sports | 127 | 500 | 25.4 | 18.9 | 15.0 | 15.7 | 15.7 |
| books | 171 | 500 | 34.2 | 17.0 | 12.3 | 14.0 | 14.0 |
| geography | 155 | 500 | 31.0 | 11.6 | 12.9 | 14.8 | 11.6 |
| politics | 129 | 500 | 25.8 | 13.2 | 16.3 | 11.6 | 10.9 |

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- Hallucination among own attempts, lowest to highest: C2 16.2% < C3 19.8% < C4 21.7% < C1 28.4%.
- Retained share (coverage), highest to lowest: C1 87.1% > C2 55.7% > C4 49.0% > C3 42.7%.
- C2 − C1 on 2166 common attempts: 15.7% vs 20.9%, Δ -5.1 pp [-6.8, -3.4], p <.001, McNemar p <.001 — CI excludes zero.
- C3 − C1 on 1651 common attempts: 18.5% vs 19.7%, Δ -1.2 pp [-3.1, +0.9], p 0.278, McNemar p 0.288 — CI includes zero.
- C4 − C1 on 1864 common attempts: 19.8% vs 22.0%, Δ -2.3 pp [-4.2, -0.2], p 0.028, McNemar p 0.029 — CI excludes zero.
- C3 − C2 on 1371 common attempts: 15.2% vs 12.3%, Δ +2.8 pp [+0.9, +4.8], p 0.004, McNemar p 0.004 — CI excludes zero.
- C4 − C2 on 1441 common attempts: 16.2% vs 13.0%, Δ +3.1 pp [+1.2, +5.1], p 0.001, McNemar p 0.002 — CI excludes zero.
- C4 − C3 on 1552 common attempts: 17.5% vs 18.7%, Δ -1.2 pp [-2.3, +0.0], p 0.052, McNemar p 0.063 — CI includes zero.
- On the all-four common set (1238): C1 16.6%, C2 11.6%, C3 13.9%, C4 13.7%.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Hallucination here is confident disagreement with the 2021 Mintaka gold (a factuality proxy), scored by the shared type-aware scorer at threshold 0.5; correctness is that scorer's CORRECT outcome, not exact match or verified world truth.
- C1 never receives the question-entity block that C2–C4 receive, so a C1-vs-retrieval delta measures retrieval plus that annotation. C4's prompt differs from C3's in more than the condensing step.
- Run-to-run noise (byte-identical repeats): hallucination ≤ 1.0 pt, correct/F1 2–4 pts, abstention 2–5 pts. A p<.05 on correctness, F1 or abstention is not by itself evidence of an effect.
- Own-attempt rates sit on different question sets per configuration and are not testable against each other; only the common-attempt rows are.
- A common-attempt comparison conditions on both configurations choosing to answer, which is itself a treatment outcome; it complements the full-benchmark comparison and does not replace it.

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
