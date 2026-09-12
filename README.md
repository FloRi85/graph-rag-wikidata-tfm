# Graph-RAG on Wikidata

**Improving the Factual Reliability of LLMs through Knowledge-Graph-Based Retrieval-Augmented Generation**

Master's Thesis (TFM), Universidad de Málaga — II Máster de Formación Permanente
en Big Data, IA e Ingeniería de Datos. Author: Florian Ringsgwandl.
Tutor: María del Mar Roldán García.

This repository contains the implementation, frozen research inputs, retained experimental
results, tests, and analysis instructions. The thesis PDF and signed declaration are submitted
separately through Campus Virtual. Thesis sources and private development records are omitted.

- [Reproduction guide and experiment map](docs/reproduction.md)
- [Detailed metric reports](data/analysis/test4000_metric_reports/README.md)
- [Demo video (YouTube)](https://youtu.be/4Ap8gAPAJCY)
- [Interactive results explorer](data/analysis/test4000_metric_reports/explorer.html)
- [Submission release: sealed answers and demo video](https://github.com/FloRi85/graph-rag-wikidata-tfm/releases/tag/v1.0-submission)

Download `explorer.html` and open it in a browser. No Python installation or API key is required.

## Research question and configurations

Can Wikidata-based retrieval reduce hallucination in question answering compared with
a base LLM and Wikipedia-text retrieval? The same Mintaka questions are evaluated under
four configurations:

| Configuration | Context supplied to the answering model |
|---|---|
| C1 — Base LLM | No retrieved context |
| C2 — RAG | Wikipedia article chunks selected by embedding similarity |
| C3 — Graph-RAG | Wikidata statements selected by embedding similarity |
| C4 — Graph-RAG + condensing | The C3 statements rewritten into prose by an additional LLM call |

C2–C4 share the encoder (`multi-qa-MiniLM-L6-cos-v1`), cosine similarity,
within-entity z-normalisation and global selection of up to 30 items. Wikidata statements
retain qualifiers, ranks, units and time precision. Evaluation uses Mintaka's annotated
question entities, not an automatic entity linker. All configurations may abstain;
C2–C4 additionally receive a question-entity block that C1 does not receive. Thus the
C1 comparison does not isolate retrieval from this annotation difference.

## Results on all 4,000 test questions

The recorded model is `nvidia/llama-3.3-nemotron-super-49b-v1`, served through NVIDIA Build.
Generation used temperature 0 and an answer limit of 128 tokens; C4's condensing limit
was 384 tokens. There were no pipeline failures or excluded questions in the final run.

| Configuration | Abstention % | Correct % | Hallucination % | F1 | F1 on attempted answers | Mean context words |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 12.2 | 62.3 | 24.8 | 58.7 | 67.4 | — |
| C2 | 44.0 | 46.7 | 9.0 | 43.0 | 77.2 | 7,684 |
| C3 | 57.2 | 34.2 | 8.5 | 33.0 | 77.4 | 293 |
| C4 | 50.9 | 38.4 | 10.6 | 37.5 | 76.4 | 38 |

Values come from the saved final-run report produced by `tools/report.py`.
Hallucination denotes an attempted answer scored below the correctness threshold against
Mintaka's reference answers, not an independent verification of present-day world facts.
Attempted answers are those classified as correct or hallucinated; abstentions and Other
responses are excluded from attempted-only metrics. Each configuration's attempted set differs.

Retrieval configurations reduce the full-test hallucination rate by 14.1–16.3 percentage
points relative to C1 (all paired-bootstrap p < .001), alongside substantially greater
abstention. C3's 0.5-point reduction relative to C2 is not statistically significant
(p = .323). C4 increases hallucination relative to both C2 and C3. The detailed reports
separate full-test comparisons from comparisons on jointly attempted questions.

Context faithfulness is a separate, RAGAS-style claim-support measure. Its means are
0.723, 0.616 and 0.859 for C2, C3 and C4, respectively, over 2,187, 1,674 and 1,928
judge-scored attempted answers. It is undefined for C1. These marginal means use different
question sets; paired comparisons use only common scorable answers.

## Install and run

Use Python 3.11.9, the version recorded for the saved metric reports. Install the pinned dependencies
in a virtual environment. Installation and the first embedding-model download need internet
access; unit tests and analyses of stored answers do not need an API key.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pytest tests/unit/ -q
```

On Linux/macOS, use `venv/bin/python` in place of `.\venv\Scripts\python.exe`.
The first retrieval invocation downloads the embedding model from Hugging Face;
subsequent invocations can use its local cache.

For byte-for-byte report checks, use the recorded Python version and pinned packages.
Newer Python versions can produce last-decimal floating-point differences even when
all displayed results agree.

### Demo

Watch the [YouTube setup guide and demonstration](https://youtu.be/4Ap8gAPAJCY) for installation instructions and a walkthrough of the demo interface.

Create a local `.env` file (never commit a real key):

```dotenv
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=your-key
LLM_RPM_LIMIT=33
```

Then start the demo with an explicitly selected model:

```powershell
.\venv\Scripts\python.exe tools/demo_server.py --model nvidia/nemotron-3-super-120b-a12b
```

Open `http://127.0.0.1:8765/`. Preset mode uses DEV-200 questions and their annotated
entities. Manual mode lets the user select Wikidata search candidates and is not
evaluation-equivalent. The interface shows each answer and the context supplied to it;
that display does not establish which evidence the model actually used.

The original model endpoint returned HTTP 410 when checked after retirement in August
2026. The documented demo substitute is therefore visibly marked **non-reference**.
Its answers do not reproduce the sealed experiment. Other OpenAI-compatible endpoints
can be configured, but a matching model name alone does not guarantee comparable outputs.

Demo cache writes go to `data/cache/demo/`, leaving the frozen corpus untouched.
The server binds locally by default. Do not run the demo concurrently with an evaluation
against the same endpoint: request pacing is process-local.

## Data and offline analysis

The repository includes the original Mintaka development and test question splits and
the fixed DEV-200 sample. The unused training split is omitted; `tools/download_mintaka.py`
can download the original splits. Active Wikidata statement and Wikipedia article caches
are retained because they are the frozen retrieval inputs, not disposable build files.

The final raw answer file is distributed separately as a Release asset because it exceeds
GitHub's per-file limit. Before reproducing test analyses:

1. Download `20260819_2017_mintaka_test_raw_raw.json` from the submission release.
2. Place it beside `meta.json` in
   `data/results/test_runs/20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw/`.
3. Verify SHA-256:
   `01b8b82c81c410aedc61aa4f3f58d4a762822007fcdec240c5ecd5a23328f627`.

For example, after restoring that file:

```powershell
$testRaw = 'data/results/test_runs/20260819_2017_nvidia-llama-3-3-nemotron-super-49b-v1_mintaka_test_raw/20260819_2017_mintaka_test_raw_raw.json'
Get-FileHash -Algorithm SHA256 $testRaw
.\venv\Scripts\python.exe tools/report.py $testRaw
.\venv\Scripts\python.exe tools/report.py $testRaw --baseline rag
.\venv\Scripts\python.exe tools/report.py $testRaw --baseline graph_rag
.\venv\Scripts\python.exe tools/build_metric_reports.py --check
```

The report commands regenerate their statistics JSONs. `--check` compares regenerated
metric reports with the saved supplement. These commands analyse stored answers; they
do not call an LLM or repeat the sealed evaluation. Additional commands and the exact
development comparisons are listed in the [reproduction guide](docs/reproduction.md).

### Evaluation dependence

During early development, outputs for 100 of the 4,000 test questions were inspected to
refine the scoring implementation. No model was trained on these examples, but their
inclusion in the final evaluation introduces some dependence between scorer development
and evaluation. The full 4,000 questions are reported. Separately, possible exposure to
Mintaka during model pretraining cannot be established from the undisclosed training corpus.

## Repository layout

| Path | Purpose |
|---|---|
| `src/pipelines/` | Four answering configurations |
| `src/retrieval/` | Statement/article retrieval, caching, ranking and context formatting |
| `src/eval/` | Benchmark parsing, scoring, statistics, faithfulness and evaluation runner |
| `src/llm_config.py`, `src/prompts.py` | Shared model interface and prompt templates |
| `tools/` | Demo, data preparation and retained reproduction tools |
| `tests/unit/` | Tests with model and network calls mocked |
| `data/questions/`, `data/cache/` | Benchmark questions and frozen retrieval corpus |
| `data/results/` | Retained raw runs, metadata and reports |
| `data/analysis/` | Saved analyses, judge validation and metric-report supplement |
| `docs/reproduction.md` | Method summary, run map and reproduction commands |

This is a single-commit submission snapshot, not the private development history.
Historical commit hashes in immutable run metadata identify the original execution and
scoring states; they do not resolve in this repository. Older tools and research records
that are unnecessary for the retained analyses are not included.

## Licences and AI use

Project code is provided under the [MIT licence](LICENSE). Third-party data retain their
licences: Mintaka © Amazon.com, Inc. or its affiliates (CC BY 4.0); Wikipedia article text
© Wikipedia contributors (CC BY-SA 4.0); Wikidata statements (CC0 1.0). Cached article
metadata records source titles and available revision identifiers.

Generative AI was used both as the object of study (answering, condensing and faithfulness
judging) and as a development and writing assistant: Claude Code for code, tests,
experiment documentation and analysis support; OpenAI Codex for an independent review
and preparation of the draft report. The author defined the research question and
experimental decisions, reviewed the material, checked reported results against project
data, and takes responsibility for the submission.
