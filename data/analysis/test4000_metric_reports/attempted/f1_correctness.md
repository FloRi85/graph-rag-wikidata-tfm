# F1 and correctness — attempted answers (own attempts; pairs on common attempts)

## 1. Definition and reading guide

- **Attempted** = outcome CORRECT or HALLUCINATION (Other excluded). **Correct % of attempts** is the complement of hallucination % of attempts (they sum to 100): its paired test is the one in `hallucination_abstention.md` §4.1 and is NOT repeated here as independent evidence.
- **F1@attempted** = 100 × mean per-question F1 over the attempted set (0–100). A partial-credit measure: it separates from correct % where answers are partially right (set answers, extra tokens).
- Overview and attribute tables use each configuration's OWN attempts; pair rows use the common attempts of the two configurations and recompute both means there.
- † marks a cell with fewer than 50 questions; `n/a` marks an empty group (unavailable, not zero). Percentages are divide-first, one decimal; deltas are A − B in percentage points (pp) and are computed from the unrounded rates, so they can differ from the difference of the printed values by 0.1.

## 2. Overview (own attempts)

| Config | Attempted n | Retained % | Correct % of attempts | F1@attempted | EM@attempted |
|---|---:|---:|---:|---:|---:|
| C1 Base LLM | 3484 | 87.1 | 71.6 | 67.4 | 58.4 |
| C2 RAG | 2227 | 55.7 | 83.8 | 77.2 | 67.8 |
| C3 Graph-RAG | 1707 | 42.7 | 80.2 | 77.4 | 71.7 |
| C4 Graph-RAG + condensing | 1962 | 49.0 | 78.3 | 76.4 | 70.1 |

## 3. Dataset stratification (own attempts)

### 3.1 By answer type

Cells: attempted n / correct % / F1 on those attempts.

| Group | n | C1 n / Corr / F1 | C2 n / Corr / F1 | C3 n / Corr / F1 | C4 n / Corr / F1 |
|---|---:|---:|---:|---:|---:|
| entity | 2499 | 2114 / 74.7 / 67.9 | 1245 / 86.9 / 75.2 | 913 / 83.5 / 78.2 | 1051 / 82.9 / 79.2 |
| numerical | 655 | 561 / 50.8 / 50.8 | 332 / 63.6 / 63.6 | 258 / 56.2 / 56.2 | 331 / 51.7 / 51.7 |
| boolean | 573 | 553 / 82.5 / 82.5 | 460 / 88.9 / 88.9 | 386 / 89.9 / 89.9 | 412 / 89.8 / 89.8 |
| date | 265 | 248 / 67.7 / 67.9 | 183 / 86.3 / 86.3 | 142 / 76.1 / 76.3 | 160 / 75.0 / 75.2 |
| string | 8† | 8† / 50.0 / 55.0 | 7† / 100.0 / 88.1 | 8† / 87.5 / 85.0 | 8† / 62.5 / 66.7 |

### 3.2 By question complexity

Cells: attempted n / correct % / F1 on those attempts.

| Group | n | C1 n / Corr / F1 | C2 n / Corr / F1 | C3 n / Corr / F1 | C4 n / Corr / F1 |
|---|---:|---:|---:|---:|---:|
| generic | 800 | 749 / 72.8 / 69.7 | 564 / 82.4 / 76.7 | 456 / 80.9 / 77.3 | 475 / 80.4 / 77.4 |
| intersection | 400 | 349 / 81.7 / 78.5 | 221 / 94.6 / 90.3 | 167 / 87.4 / 85.0 | 178 / 88.8 / 86.8 |
| count | 400 | 375 / 55.2 / 55.2 | 233 / 68.7 / 68.7 | 190 / 58.9 / 58.9 | 224 / 53.1 / 53.1 |
| comparative | 400 | 382 / 80.1 / 71.7 | 333 / 84.1 / 65.7 | 242 / 88.0 / 81.6 | 243 / 89.7 / 89.2 |
| yesno | 400 | 387 / 85.5 / 85.5 | 313 / 89.1 / 89.1 | 260 / 88.8 / 88.8 | 288 / 87.5 / 87.5 |
| ordinal | 400 | 353 / 67.7 / 62.4 | 216 / 88.0 / 82.8 | 149 / 80.5 / 79.1 | 175 / 78.9 / 76.6 |
| multihop | 400 | 299 / 59.9 / 53.0 | 112 / 77.7 / 72.5 | 66 / 83.3 / 80.4 | 141 / 72.3 / 70.1 |
| difference | 400 | 268 / 69.8 / 60.3 | 129 / 86.8 / 73.6 | 117 / 68.4 / 61.8 | 159 / 71.1 / 64.1 |
| superlative | 400 | 322 / 66.5 / 62.2 | 106 / 80.2 / 70.8 | 60 / 71.7 / 71.5 | 79 / 69.6 / 68.5 |

### 3.3 By topic category

Cells: attempted n / correct % / F1 on those attempts.

| Group | n | C1 n / Corr / F1 | C2 n / Corr / F1 | C3 n / Corr / F1 | C4 n / Corr / F1 |
|---|---:|---:|---:|---:|---:|
| history | 500 | 472 / 76.5 / 70.4 | 271 / 88.9 / 75.9 | 276 / 79.3 / 73.5 | 281 / 81.1 / 79.7 |
| movies | 500 | 441 / 67.8 / 64.9 | 306 / 83.0 / 74.9 | 233 / 82.0 / 80.2 | 279 / 74.9 / 73.0 |
| music | 500 | 420 / 60.0 / 54.1 | 283 / 80.2 / 74.6 | 164 / 74.4 / 73.0 | 199 / 71.4 / 69.7 |
| videogames | 500 | 448 / 73.2 / 69.3 | 304 / 89.5 / 84.5 | 219 / 81.3 / 77.9 | 258 / 79.8 / 77.8 |
| sports | 500 | 419 / 72.1 / 70.0 | 275 / 79.3 / 75.3 | 167 / 84.4 / 83.3 | 209 / 77.0 / 76.1 |
| books | 500 | 403 / 71.5 / 67.9 | 299 / 83.3 / 78.1 | 231 / 79.2 / 76.8 | 255 / 80.8 / 78.9 |
| geography | 500 | 438 / 73.5 / 69.0 | 245 / 83.3 / 77.6 | 211 / 77.3 / 75.3 | 232 / 78.4 / 75.7 |
| politics | 500 | 443 / 77.0 / 73.0 | 244 / 82.8 / 76.4 | 206 / 83.5 / 80.5 | 249 / 81.5 / 78.6 |

## 4. Direct paired comparisons (common attempts)

### 4.1 Overall, six pairs

ΔF1 with bootstrap CI (the F1 test). ΔCorr is descriptive here — its test is the hallucination test.

| Pair (A − B) | Common n | F1 A | F1 B | ΔF1 | 95% CI | p | Corr% A | Corr% B | ΔCorr pp (see H file) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 − C1 | 2166 | 77.6 | 75.5 | +2.1 | [+0.3, +3.8] | 0.017 | 84.3 | 79.1 | +5.1 |
| C3 − C1 | 1651 | 78.6 | 77.2 | +1.4 | [-0.6, +3.3] | 0.171 | 81.5 | 80.3 | +1.2 |
| C4 − C1 | 1864 | 78.2 | 74.6 | +3.6 | [+1.6, +5.6] | <.001 | 80.2 | 78.0 | +2.3 |
| C3 − C2 | 1371 | 81.8 | 81.2 | +0.6 | [-1.4, +2.5] | 0.590 | 84.8 | 87.7 | -2.8 |
| C4 − C2 | 1441 | 82.1 | 81.1 | +1.0 | [-1.0, +3.0] | 0.365 | 83.8 | 87.0 | -3.1 |
| C4 − C3 | 1552 | 80.5 | 78.6 | +1.9 | [+0.8, +3.1] | 0.002 | 82.5 | 81.3 | +1.2 |

### 4.2 C2 − C1 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C2 | F1 C1 | ΔF1 | Corr% C2 | Corr% C1 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 1198 | 75.6 | 77.1 | -1.5 | 87.6 | 83.6 | +3.9 |
| numerical | 323 | 63.8 | 60.7 | +3.1 | 63.8 | 60.7 | +3.1 |
| boolean | 456 | 88.8 | 83.3 | +5.5 | 88.8 | 83.3 | +5.5 |
| date | 182 | 86.8 | 73.1 | +13.7 | 86.8 | 73.1 | +13.7 |
| string | 7† | 88.1 | 48.6 | +39.5 | 100.0 | 42.9 | +57.1 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 552 | 77.4 | 76.1 | +1.3 | 83.0 | 78.8 | +4.2 |
| intersection | 209 | 91.1 | 85.0 | +6.1 | 95.7 | 87.1 | +8.6 |
| count | 231 | 68.4 | 64.9 | +3.5 | 68.4 | 64.9 | +3.5 |
| comparative | 329 | 65.6 | 72.5 | -6.9 | 84.2 | 80.5 | +3.6 |
| yesno | 312 | 89.1 | 86.5 | +2.6 | 89.1 | 86.5 | +2.6 |
| ordinal | 208 | 84.4 | 73.2 | +11.2 | 89.4 | 76.4 | +13.0 |
| multihop | 110 | 72.0 | 66.3 | +5.7 | 77.3 | 73.6 | +3.6 |
| difference | 112 | 73.4 | 69.3 | +4.1 | 88.4 | 81.2 | +7.1 |
| superlative | 103 | 71.9 | 74.4 | -2.5 | 81.6 | 78.6 | +2.9 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 270 | 76.0 | 79.9 | -3.9 | 88.9 | 86.7 | +2.2 |
| movies | 295 | 75.0 | 72.0 | +3.0 | 83.4 | 74.2 | +9.2 |
| music | 276 | 73.9 | 65.2 | +8.7 | 79.7 | 70.3 | +9.4 |
| videogames | 298 | 84.4 | 74.2 | +10.2 | 89.3 | 76.8 | +12.4 |
| sports | 266 | 76.6 | 76.8 | -0.1 | 80.8 | 78.6 | +2.3 |
| books | 281 | 79.1 | 73.6 | +5.5 | 84.3 | 76.9 | +7.5 |
| geography | 241 | 78.8 | 83.3 | -4.5 | 84.2 | 87.1 | -2.9 |
| politics | 239 | 76.3 | 81.6 | -5.3 | 82.8 | 84.9 | -2.1 |

### 4.3 C3 − C1 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C3 | F1 C1 | ΔF1 | Corr% C3 | Corr% C1 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 869 | 80.1 | 79.8 | +0.3 | 85.5 | 85.7 | -0.2 |
| numerical | 253 | 56.5 | 61.7 | -5.1 | 56.5 | 61.7 | -5.1 |
| boolean | 382 | 90.3 | 83.8 | +6.5 | 90.3 | 83.8 | +6.5 |
| date | 139 | 77.2 | 72.9 | +4.3 | 77.0 | 72.7 | +4.3 |
| string | 8† | 85.0 | 55.0 | +30.0 | 87.5 | 50.0 | +37.5 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 441 | 78.8 | 78.7 | +0.1 | 82.5 | 81.4 | +1.1 |
| intersection | 162 | 85.8 | 87.7 | -1.9 | 88.3 | 89.5 | -1.2 |
| count | 188 | 59.0 | 63.8 | -4.8 | 59.0 | 63.8 | -4.8 |
| comparative | 239 | 82.3 | 71.1 | +11.2 | 88.7 | 78.7 | +10.0 |
| yesno | 258 | 89.1 | 88.0 | +1.2 | 89.1 | 88.0 | +1.2 |
| ordinal | 144 | 80.3 | 71.4 | +8.9 | 81.9 | 75.7 | +6.2 |
| multihop | 66 | 80.4 | 82.3 | -1.9 | 83.3 | 86.4 | -3.0 |
| difference | 96 | 65.8 | 67.9 | -2.1 | 72.9 | 74.0 | -1.0 |
| superlative | 57 | 73.8 | 82.1 | -8.3 | 73.7 | 87.7 | -14.0 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 270 | 74.5 | 77.2 | -2.7 | 80.4 | 83.7 | -3.3 |
| movies | 226 | 81.8 | 75.8 | +6.0 | 83.6 | 77.4 | +6.2 |
| music | 157 | 73.8 | 67.2 | +6.5 | 75.8 | 72.0 | +3.8 |
| videogames | 214 | 78.7 | 78.8 | -0.1 | 82.2 | 81.3 | +0.9 |
| sports | 160 | 84.6 | 80.4 | +4.2 | 85.0 | 80.6 | +4.4 |
| books | 218 | 79.3 | 77.3 | +2.0 | 81.7 | 79.4 | +2.3 |
| geography | 205 | 76.1 | 78.3 | -2.2 | 78.0 | 81.5 | -3.4 |
| politics | 201 | 81.5 | 81.5 | -0.0 | 84.6 | 84.1 | +0.5 |

### 4.4 C4 − C1 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C4 | F1 C1 | ΔF1 | Corr% C4 | Corr% C1 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 980 | 81.6 | 77.3 | +4.3 | 85.4 | 83.7 | +1.7 |
| numerical | 314 | 54.1 | 59.2 | -5.1 | 54.1 | 59.2 | -5.1 |
| boolean | 405 | 90.1 | 83.0 | +7.2 | 90.1 | 83.0 | +7.2 |
| date | 157 | 75.4 | 68.4 | +7.0 | 75.2 | 68.2 | +7.0 |
| string | 8† | 66.7 | 55.0 | +11.7 | 62.5 | 50.0 | +12.5 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 459 | 79.0 | 78.0 | +1.0 | 82.1 | 80.8 | +1.3 |
| intersection | 171 | 88.6 | 89.3 | -0.7 | 90.6 | 90.6 | +0.0 |
| count | 217 | 54.8 | 62.7 | -7.8 | 54.8 | 62.7 | -7.8 |
| comparative | 238 | 90.3 | 70.8 | +19.5 | 90.8 | 78.6 | +12.2 |
| yesno | 283 | 88.0 | 87.3 | +0.7 | 88.0 | 87.3 | +0.7 |
| ordinal | 167 | 77.7 | 67.7 | +9.9 | 80.2 | 72.5 | +7.8 |
| multihop | 130 | 70.7 | 58.9 | +11.8 | 73.1 | 66.2 | +6.9 |
| difference | 125 | 69.1 | 65.2 | +4.0 | 76.8 | 71.2 | +5.6 |
| superlative | 74 | 71.9 | 77.8 | -5.9 | 73.0 | 82.4 | -9.5 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 275 | 80.8 | 76.0 | +4.8 | 82.2 | 81.8 | +0.4 |
| movies | 263 | 74.2 | 69.5 | +4.8 | 76.0 | 72.2 | +3.8 |
| music | 190 | 71.4 | 63.1 | +8.3 | 73.2 | 67.4 | +5.8 |
| videogames | 249 | 78.8 | 76.0 | +2.9 | 81.1 | 78.7 | +2.4 |
| sports | 194 | 78.9 | 80.0 | -1.1 | 79.4 | 80.4 | -1.0 |
| books | 233 | 81.7 | 76.5 | +5.2 | 83.7 | 79.8 | +3.9 |
| geography | 223 | 78.0 | 76.9 | +1.0 | 80.7 | 80.3 | +0.4 |
| politics | 237 | 80.9 | 78.0 | +2.8 | 84.0 | 81.4 | +2.5 |

### 4.5 C3 − C2 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C3 | F1 C2 | ΔF1 | Corr% C3 | Corr% C2 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 689 | 82.9 | 78.7 | +4.2 | 89.0 | 91.4 | -2.5 |
| numerical | 203 | 62.1 | 69.5 | -7.4 | 62.1 | 69.5 | -7.4 |
| boolean | 353 | 91.5 | 90.9 | +0.6 | 91.5 | 90.9 | +0.6 |
| date | 119 | 80.1 | 86.8 | -6.7 | 79.8 | 86.6 | -6.7 |
| string | 7† | 82.9 | 88.1 | -5.2 | 85.7 | 100.0 | -14.3 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 403 | 79.1 | 79.5 | -0.4 | 82.9 | 85.9 | -3.0 |
| intersection | 128 | 91.3 | 91.2 | +0.0 | 93.0 | 94.5 | -1.6 |
| count | 143 | 65.7 | 74.1 | -8.4 | 65.7 | 74.1 | -8.4 |
| comparative | 229 | 82.4 | 67.2 | +15.1 | 89.1 | 86.5 | +2.6 |
| yesno | 233 | 91.0 | 91.8 | -0.9 | 91.0 | 91.8 | -0.9 |
| ordinal | 105 | 84.3 | 89.4 | -5.1 | 85.7 | 94.3 | -8.6 |
| multihop | 41† | 85.2 | 90.4 | -5.2 | 87.8 | 92.7 | -4.9 |
| difference | 62 | 72.5 | 77.7 | -5.2 | 82.3 | 88.7 | -6.5 |
| superlative | 27† | 84.0 | 86.8 | -2.9 | 85.2 | 92.6 | -7.4 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 199 | 80.8 | 76.0 | +4.8 | 88.4 | 90.5 | -2.0 |
| movies | 201 | 83.1 | 81.1 | +2.0 | 85.1 | 88.6 | -3.5 |
| music | 136 | 78.7 | 83.0 | -4.3 | 80.9 | 87.5 | -6.6 |
| videogames | 182 | 81.7 | 89.1 | -7.4 | 85.2 | 94.0 | -8.8 |
| sports | 139 | 84.1 | 82.5 | +1.6 | 84.2 | 84.2 | +0.0 |
| books | 193 | 80.8 | 79.8 | +1.0 | 83.4 | 86.0 | -2.6 |
| geography | 170 | 80.7 | 80.0 | +0.7 | 82.4 | 85.3 | -2.9 |
| politics | 151 | 84.5 | 79.3 | +5.2 | 88.1 | 83.4 | +4.6 |

### 4.6 C4 − C2 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C4 | F1 C2 | ΔF1 | Corr% C4 | Corr% C2 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 737 | 84.8 | 78.9 | +6.0 | 88.3 | 90.2 | -1.9 |
| numerical | 226 | 58.0 | 69.0 | -11.1 | 58.0 | 69.0 | -11.1 |
| boolean | 347 | 91.9 | 91.4 | +0.6 | 91.9 | 91.4 | +0.6 |
| date | 124 | 82.5 | 87.4 | -4.8 | 82.3 | 87.1 | -4.8 |
| string | 7† | 76.2 | 88.1 | -11.9 | 71.4 | 100.0 | -28.6 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 410 | 79.3 | 79.2 | +0.1 | 82.4 | 85.1 | -2.7 |
| intersection | 131 | 92.3 | 92.5 | -0.2 | 93.1 | 94.7 | -1.5 |
| count | 156 | 60.3 | 75.6 | -15.4 | 60.3 | 75.6 | -15.4 |
| comparative | 222 | 90.5 | 68.6 | +21.9 | 91.0 | 86.5 | +4.5 |
| yesno | 234 | 90.6 | 91.9 | -1.3 | 90.6 | 91.9 | -1.3 |
| ordinal | 123 | 82.6 | 86.5 | -4.0 | 85.4 | 91.9 | -6.5 |
| multihop | 58 | 79.0 | 75.8 | +3.2 | 81.0 | 81.0 | +0.0 |
| difference | 75 | 74.3 | 83.2 | -8.9 | 81.3 | 90.7 | -9.3 |
| superlative | 32† | 83.3 | 77.0 | +6.3 | 84.4 | 84.4 | +0.0 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 198 | 85.5 | 76.7 | +8.8 | 86.4 | 89.9 | -3.5 |
| movies | 216 | 78.3 | 78.8 | -0.6 | 79.6 | 85.2 | -5.6 |
| music | 144 | 79.5 | 83.1 | -3.7 | 81.2 | 86.8 | -5.6 |
| videogames | 200 | 82.4 | 88.3 | -5.8 | 85.0 | 93.0 | -8.0 |
| sports | 157 | 78.1 | 81.0 | -2.9 | 78.3 | 82.8 | -4.5 |
| books | 199 | 82.0 | 80.8 | +1.1 | 83.9 | 85.9 | -2.0 |
| geography | 167 | 84.8 | 79.9 | +4.9 | 86.8 | 85.6 | +1.2 |
| politics | 160 | 86.1 | 80.5 | +5.6 | 89.4 | 85.0 | +4.4 |

### 4.7 C4 − C3 by attribute

Descriptive per group (no tests).

| Group | Common n | F1 C4 | F1 C3 | ΔF1 | Corr% C4 | Corr% C3 | ΔCorr pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Answer type** |  |  |  |  |  |  |  |
| entity | 815 | 82.5 | 79.4 | +3.1 | 86.3 | 84.5 | +1.7 |
| numerical | 238 | 58.8 | 57.1 | +1.7 | 58.8 | 57.1 | +1.7 |
| boolean | 355 | 91.8 | 91.5 | +0.3 | 91.8 | 91.5 | +0.3 |
| date | 136 | 78.2 | 77.5 | +0.7 | 77.9 | 77.2 | +0.7 |
| string | 8† | 66.7 | 85.0 | -18.3 | 62.5 | 87.5 | -25.0 |
| **Question complexity** |  |  |  |  |  |  |  |
| generic | 428 | 78.3 | 77.8 | +0.5 | 81.5 | 81.5 | +0.0 |
| intersection | 141 | 89.8 | 89.0 | +0.8 | 92.2 | 91.5 | +0.7 |
| count | 174 | 61.5 | 60.3 | +1.1 | 61.5 | 60.3 | +1.1 |
| comparative | 216 | 91.4 | 81.7 | +9.7 | 91.7 | 88.4 | +3.2 |
| yesno | 241 | 90.5 | 91.3 | -0.8 | 90.5 | 91.3 | -0.8 |
| ordinal | 139 | 81.0 | 80.2 | +0.8 | 82.7 | 81.3 | +1.4 |
| multihop | 55 | 84.0 | 81.9 | +2.2 | 85.5 | 83.6 | +1.8 |
| difference | 106 | 66.8 | 63.8 | +3.0 | 74.5 | 68.9 | +5.7 |
| superlative | 52 | 70.0 | 69.1 | +1.0 | 71.2 | 69.2 | +1.9 |
| **Topic category** |  |  |  |  |  |  |  |
| history | 250 | 80.1 | 74.8 | +5.3 | 81.2 | 80.4 | +0.8 |
| movies | 211 | 80.7 | 80.4 | +0.4 | 82.5 | 81.5 | +0.9 |
| music | 152 | 77.3 | 75.8 | +1.5 | 78.9 | 77.6 | +1.3 |
| videogames | 202 | 81.7 | 79.6 | +2.1 | 83.7 | 82.7 | +1.0 |
| sports | 155 | 83.3 | 83.3 | +0.0 | 84.5 | 84.5 | +0.0 |
| books | 210 | 80.3 | 79.2 | +1.0 | 82.9 | 81.9 | +1.0 |
| geography | 192 | 78.2 | 76.5 | +1.7 | 80.7 | 78.6 | +2.1 |
| politics | 180 | 82.9 | 80.8 | +2.2 | 85.6 | 83.3 | +2.2 |

## 5. Supplement: questions attempted by all four configurations

n = 1238 (30.9% of the benchmark). Cells: correct % / F1.

| Group | Common n | C1 Corr / F1 | C2 Corr / F1 | C3 Corr / F1 | C4 Corr / F1 |
|---|---:|---:|---:|---:|---:|
| overall | 1238 | 83.4 / 80.6 | 88.4 / 82.4 | 86.1 / 83.3 | 86.3 / 84.9 |
| **Answer type** |  |  |  |  |  |
| entity | 609 | 89.5 / 83.5 | 92.0 / 79.9 | 90.3 / 84.5 | 91.0 / 87.9 |
| numerical | 186 | 67.7 / 67.7 | 71.0 / 71.0 | 62.9 / 62.9 | 61.8 / 61.8 |
| boolean | 321 | 85.0 / 85.0 | 91.9 / 91.9 | 93.5 / 93.5 | 93.8 / 93.8 |
| date | 115 | 74.8 / 75.1 | 87.0 / 87.2 | 80.9 / 81.2 | 81.7 / 82.0 |
| string | 7† | 42.9 / 48.6 | 100.0 / 88.1 | 85.7 / 82.9 | 71.4 / 76.2 |
| **Question complexity** |  |  |  |  |  |
| generic | 376 | 83.0 / 80.0 | 87.0 / 80.9 | 84.3 / 80.4 | 84.3 / 81.0 |
| intersection | 107 | 94.4 / 93.0 | 96.3 / 93.7 | 96.3 / 94.7 | 95.3 / 94.4 |
| count | 132 | 71.2 / 71.2 | 75.0 / 75.0 | 66.7 / 66.7 | 65.2 / 65.2 |
| comparative | 203 | 78.3 / 70.7 | 86.7 / 68.3 | 89.7 / 82.5 | 92.1 / 91.8 |
| yesno | 215 | 90.7 / 90.7 | 93.0 / 93.0 | 93.5 / 93.5 | 93.5 / 93.5 |
| ordinal | 99 | 81.8 / 77.6 | 93.9 / 88.9 | 85.9 / 85.0 | 87.9 / 86.1 |
| multihop | 34† | 88.2 / 87.4 | 91.2 / 88.4 | 88.2 / 87.9 | 88.2 / 88.5 |
| difference | 51 | 84.3 / 79.0 | 88.2 / 81.2 | 82.4 / 76.7 | 80.4 / 76.4 |
| superlative | 21† | 85.7 / 84.1 | 95.2 / 87.8 | 85.7 / 84.1 | 85.7 / 84.1 |
| **Topic category** |  |  |  |  |  |
| history | 185 | 85.9 / 79.6 | 90.3 / 76.8 | 88.1 / 80.9 | 87.0 / 86.6 |
| movies | 178 | 77.5 / 75.7 | 88.2 / 82.3 | 86.0 / 84.7 | 84.3 / 83.2 |
| music | 127 | 78.7 / 75.0 | 88.2 / 83.8 | 83.5 / 81.1 | 83.5 / 81.9 |
| videogames | 166 | 85.5 / 83.6 | 94.6 / 89.6 | 86.7 / 83.7 | 88.0 / 85.7 |
| sports | 127 | 81.1 / 80.8 | 85.0 / 83.1 | 84.3 / 84.5 | 84.3 / 84.3 |
| books | 171 | 83.0 / 80.5 | 87.7 / 82.4 | 86.0 / 83.0 | 86.0 / 83.7 |
| geography | 155 | 88.4 / 85.3 | 87.1 / 81.8 | 85.2 / 83.3 | 88.4 / 86.8 |
| politics | 129 | 86.8 / 84.5 | 83.7 / 79.9 | 88.4 / 85.5 | 89.1 / 86.5 |

## 6. Findings and limitations

Generated from the tables above; no interpretation beyond them.

- F1 on own attempts, highest to lowest: C3 77.4 > C2 77.2 > C4 76.4 > C1 67.4 (different denominators).
- Correct % of own attempts, highest to lowest: C2 83.8% > C3 80.2% > C4 78.3% > C1 71.6%.
- C2 − C1 on 2166 common attempts: F1 77.6 vs 75.5, Δ +2.1 [+0.3, +3.8], p 0.017 — CI excludes zero.
- C3 − C1 on 1651 common attempts: F1 78.6 vs 77.2, Δ +1.4 [-0.6, +3.3], p 0.171 — CI includes zero.
- C4 − C1 on 1864 common attempts: F1 78.2 vs 74.6, Δ +3.6 [+1.6, +5.6], p <.001 — CI excludes zero.
- C3 − C2 on 1371 common attempts: F1 81.8 vs 81.2, Δ +0.6 [-1.4, +2.5], p 0.590 — CI includes zero.
- C4 − C2 on 1441 common attempts: F1 82.1 vs 81.1, Δ +1.0 [-1.0, +3.0], p 0.365 — CI includes zero.
- C4 − C3 on 1552 common attempts: F1 80.5 vs 78.6, Δ +1.9 [+0.8, +3.1], p 0.002 — CI excludes zero.
- All-four common set (1238), F1: C1 80.6, C2 82.4, C3 83.3, C4 84.9.

Limitations:

- The three attribute axes are separate marginal partitions of the same questions; a finding that repeats across axes is the same questions seen three times, not three findings.
- Per-group pair deltas are descriptive. A within-group difference is not evidence of a subgroup interaction, and one group being significant while another is not says nothing about their contrast.
- Hallucination here is confident disagreement with the 2021 Mintaka gold (a factuality proxy), scored by the shared type-aware scorer at threshold 0.5; correctness is that scorer's CORRECT outcome, not exact match or verified world truth.
- C1 never receives the question-entity block that C2–C4 receive, so a C1-vs-retrieval delta measures retrieval plus that annotation. C4's prompt differs from C3's in more than the condensing step.
- Run-to-run noise (byte-identical repeats): hallucination ≤ 1.0 pt, correct/F1 2–4 pts, abstention 2–5 pts. A p<.05 on correctness, F1 or abstention is not by itself evidence of an effect.
- Correct % and F1 on attempts can move by 2–4 pts between byte-identical runs; a common-attempt F1 gap inside that band is not a finding.

## 7. Provenance and reproduction

- Input files and checksums are recorded in [`manifest.json`](../manifest.json).
- Every number here is copied from [`results.json`](../results.json); the reading guide is [`README.md`](../README.md); browse interactively in [`explorer.html`](../explorer.html).
- Regenerate: `venv/Scripts/python tools/build_metric_reports.py` — verify without overwriting: `venv/Scripts/python tools/build_metric_reports.py --check`.
- Statistics: paired bootstrap (10,000 resamples, seed 0, nominal 95% percentile CIs, unadjusted), exact McNemar on discordant pairs for binary outcomes. Bootstrap p below the Monte Carlo resolution prints as `<.001`, never as zero.
- Precedent checks against the persisted sanctioned files: 77 matched, 0 mismatched (list in `results.json` → `precedent_checks`).
