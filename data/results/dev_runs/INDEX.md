# Development evidence

These runs use the fixed 200-question development sample. The final
development reference is **18 August, `reference-v8`**. The earlier records
are retained only where they support a reported design decision or repeated-run
comparison; their retrieval scores do not replace the final reference or TEST
results. Each directory contains its original raw answers, `meta.json` and
`report.md`.

The [reproduction guide](../../../docs/reproduction.md) explains the metrics,
cache versions and provenance. Original run-purpose and report text may refer
to private historical documents or withdrawn interpretations. Use the current
guide and final thesis for interpretation, and the immutable raw files for
re-scoring.

## Final reference and structural sensitivity

| Exact run directory | Role |
|---|---|
| [20260818_1113_nvidia-llama-3-3-nemotron-super-49b-v1_reference-v8](20260818_1113_nvidia-llama-3-3-nemotron-super-49b-v1_reference-v8/) | Final DEV reference: k=30, global allocation, full entity block, fixed condenser, cache v8, two workers. |
| [20260818_1211_nvidia-llama-3-3-nemotron-super-49b-v1_ref-alloc-floor-v8](20260818_1211_nvidia-llama-3-3-nemotron-super-49b-v1_ref-alloc-floor-v8/) | Per-entity floor allocation, same total item budget; compare with reference-v8. |
| [20260818_1315_nvidia-llama-3-3-nemotron-super-49b-v1_ref-entity-block-v8](20260818_1315_nvidia-llama-3-3-nemotron-super-49b-v1_ref-entity-block-v8/) | Entity-only annotation block, omitting question-literal annotations; compare with reference-v8. |

For each sensitivity arm, four configurations and four metrics give 16
comparisons: hallucination, correctness, abstention and F1. The two checks
therefore contain 32 comparisons, not 32 distinct experiments.

## Retrieval depth, condensing and repeat-generation checks

| Exact run directory | Role |
|---|---|
| [20260817_0245_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k10](20260817_0245_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k10/) | Depth k=10. |
| [20260817_0324_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k30](20260817_0324_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k30/) | Depth k=30. |
| [20260817_0402_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k50](20260817_0402_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k50/) | Depth k=50. |
| [20260817_1009_nvidia-llama-3-3-nemotron-super-49b-v1_preref-check](20260817_1009_nvidia-llama-3-3-nemotron-super-49b-v1_preref-check/) | Repeat of sweep-k30 settings, before the condensing-prompt change. |
| [20260817_1127_nvidia-llama-3-3-nemotron-super-49b-v1_condense-fix-check](20260817_1127_nvidia-llama-3-3-nemotron-super-49b-v1_condense-fix-check/) | Tightened condensing prompt; compare with preref-check, noting that workers also changed from eight to two. |
| [20260817_2040_nvidia-llama-3-3-nemotron-super-49b-v1_final-preref-check](20260817_2040_nvidia-llama-3-3-nemotron-super-49b-v1_final-preref-check/) | Repeat of condense-fix-check settings, with two workers. |
| [20260817_2226_nvidia-llama-3-3-nemotron-super-49b-v1_reference](20260817_2226_nvidia-llama-3-3-nemotron-super-49b-v1_reference/) | Earlier cache-v7 reference, retained for the v8 transition check; superseded by reference-v8. |

The depth decision compares k30 minus k10 and k50 minus k30. Its evidence
rests on C3's k30 F1 increase and C2's k50 F1 decrease, not an assumed C4 gain.
The sweep predates the fixed condensing prompt and cache v8; it is not a
second evaluation of the final configuration.

The unchanged-setting repeat pairs are `sweep-k30` versus `preref-check`, and
`condense-fix-check` versus `final-preref-check`. The older pair contains
pipeline failures, so the analysis reports the matched denominator rather
than assuming 200 usable pairs. Observed variation in these small numbers of
repeats is descriptive, not a general noise bound for future runs.

Example offline comparison, from the repository root:

```powershell
$devRoot = 'data/results/dev_runs'
$k10 = Join-Path $devRoot '20260817_0245_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k10/20260817_0245_sweep-k10_raw.json'
$k30 = Join-Path $devRoot '20260817_0324_nvidia-llama-3-3-nemotron-super-49b-v1_sweep-k30/20260817_0324_sweep-k30_raw.json'
.\venv\Scripts\python.exe tools/compare_runs.py $k30 $k10 --metric f1
```

`compare_runs.py` reports the first file minus the second, paired by question
ID within each configuration. For the structural and prompt checks, pass an
arm's raw file first and the `reference-v8` raw file second. Repeat with
`--metric hallucination`, `--metric correct` and `--metric abstention` as
needed. Each raw filename is recorded in its run's metadata and ends in
`_raw.json`.

## Ten answer-prompt variants

Each arm compares with reference-v8 and keeps the condensing prompt fixed.
Eight alter answer-prompt wording for the applicable configurations; the
last two change only C1. Exact strings are preserved in metadata and in
`src/prompts.py`'s variant definitions.

| Exact run directory | Changed surface |
|---|---|
| [20260818_2158_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c-prime](20260818_2158_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c-prime/) | Refusal condition: answer not in context. |
| [20260818_2241_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-verify](20260818_2241_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-verify/) | Refusal condition: cannot verify from context. |
| [20260818_2330_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-no-guess](20260818_2330_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-no-guess/) | Explicit instruction not to guess. |
| [20260819_0017_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-keep-short](20260819_0017_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-keep-short/) | Short-answer instruction in place of fact-only wording. |
| [20260819_0106_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-refusal-ok](20260819_0106_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-refusal-ok/) | Explicitly acceptable refusal. |
| [20260819_0149_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-decline-over-guess](20260819_0149_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-decline-over-guess/) | Persona preferring refusal to guessing. |
| [20260819_0237_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-no-persona](20260819_0237_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-no-persona/) | Persona removed. |
| [20260819_0320_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-only-below](20260819_0320_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-only-below/) | C2–C4 instructed to use only the supplied information. |
| [20260819_0406_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c1-dont-know](20260819_0406_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c1-dont-know/) | C1 refusal conditioned on not knowing. |
| [20260819_0453_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c1-no-clause](20260819_0453_nvidia-llama-3-3-nemotron-super-49b-v1_prompt-c1-no-clause/) | C1 refusal clause removed. |

## Historical representation and refusal comparisons

| Exact run directory | Retained purpose and limitation |
|---|---|
| [archive_truthy_retrieval/20260806_2315_nvidia-llama-3-3-nemotron-super-49b-v1_reference-c-prime](archive_truthy_retrieval/20260806_2315_nvidia-llama-3-3-nemotron-super-49b-v1_reference-c-prime/) | Earlier truthy-retrieval C-prime refusal condition. Paired with variant F below to document the original prompt-dependent graph/text difference. |
| [archive_truthy_retrieval/20260807_1855_nvidia-llama-3-3-nemotron-super-49b-v1_reference-variant-f](archive_truthy_retrieval/20260807_1855_nvidia-llama-3-3-nemotron-super-49b-v1_reference-variant-f/) | Earlier truthy-retrieval variant F, before the statement representation. Not final retrieval performance. |
| [20260812_1738_nvidia-llama-3-3-nemotron-super-49b-v1_statement-model-v1](20260812_1738_nvidia-llama-3-3-nemotron-super-49b-v1_statement-model-v1/) | First statement-model run. Six hub entities lacked incoming facts, affecting 24 sample questions. It is retained for the documented transition, not evidence of a retrieval-quality gain. |

Historical caches and executable versions are not included. The stored
contexts and outputs allow offline re-scoring; a live rerun with current
retrieval code would not recreate those earlier conditions.
