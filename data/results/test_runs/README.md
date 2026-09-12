# Sealed test run

The final evaluation generated answers for all 4,000 English Mintaka test
questions on 19–20 August 2026. Its settings were selected on development data
before this run. The four configurations produced 16,000 graded responses,
with no final pipeline errors or excluded questions.

The run is stored in
[`20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw/`](20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw/).

| File | Purpose |
|---|---|
| `20260819_2017_mintaka_test_raw_raw.json` | Immutable answers and supplied contexts; downloaded separately. |
| `meta.json` | Run-start settings, raw-result hash, and appended scorer provenance. |
| `report.md` | Recorded settings and outcome summary. |

## Restore the raw answers

The raw file is approximately 215 MB and exceeds GitHub's per-file limit. It
is distributed as an uncompressed asset attached to
[the submission release](https://github.com/FloRi85/graph-rag-wikidata-tfm/releases/tag/v1.0-submission),
not through Git LFS. Download `20260819_2017_mintaka_test_raw_raw.json` and put it
beside the run's `meta.json`.

From the repository root, verify it in PowerShell:

```powershell
$testDir = 'data/results/test_runs/20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw'
$testRaw = Join-Path $testDir '20260819_2017_mintaka_test_raw_raw.json'
Get-FileHash -Algorithm SHA256 -LiteralPath $testRaw
```

The expected SHA-256 is:

```text
01b8b82c81c410aedc61aa4f3f58d4a762822007fcdec240c5ecd5a23328f627
```

This hash identifies the **answer file**, not the Mintaka question file.
Keep the bytes unchanged; converting line endings would change the hash.

## Reproduce the results

After installing the project dependencies and restoring the raw file:

```powershell
.\venv\Scripts\python.exe tools/report.py $testRaw
.\venv\Scripts\python.exe tools/report.py $testRaw --baseline rag
.\venv\Scripts\python.exe tools/build_metric_reports.py --check
```

These commands use stored answers and make no model calls. The first two
regenerate statistical JSON files; the last verifies the metric supplement in
memory without writing files. The [reproduction guide](../../../docs/reproduction.md)
describes the remaining analyses, denominators and provenance.

Re-scoring is distinct from generating new answers. The evaluator's
`--test-run` guard checks question IDs, requires the complete test split and
enforces frozen settings. New generations are not replacements for this
record. Run documentation preserves the start settings, checks the raw hash
and appends scorer records instead of relabelling how the answers were made.
Historical Git identifiers in metadata belong to the private development
history and need not resolve in this single-commit public snapshot.

## Evaluation dependence

During early development, outputs for 100 of the 4,000 test questions were
inspected to refine the scoring implementation. No model was trained on these
examples, but their inclusion in the final evaluation introduces some
dependence between scorer development and evaluation. The full 4,000-question
result is reported. The 2.5% sample share is not a measured bound on that
dependence.
