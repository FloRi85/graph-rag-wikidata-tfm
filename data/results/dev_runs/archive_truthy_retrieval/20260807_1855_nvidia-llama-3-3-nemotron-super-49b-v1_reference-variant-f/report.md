# reference-variant-f

**REFERENCE DEV RUN v2 (variant F clause).**

- Run: `20260807_1855`  ·  documented 2026-08-09T16:08:17
- Sample: `data/questions/mintaka_sample_dev_200.json` (n=200)
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1` @ `https://integrate.api.nvidia.com/v1`  ·  temperature 0  ·  max_tokens 128
- Recorded code state: uncommitted changes.

## Outcomes

| Arm | n | Abst% | Correct% | Halluc% | Other% | Truncated | Errors | Median words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_llm_abstain | 200 | 11.0 | 67.0 | 20.5 | 1.5 | 2 | 0 | 2 |
| rag | 200 | 39.0 | 55.0 | 5.5 | 0.5 | 1 | 0 | 4 |
| graph_rag | 200 | 58.0 | 37.0 | 5.0 | 0.0 | 0 | 0 | 7 |
| rerank | 200 | 42.0 | 46.5 | 8.0 | 3.5 | 1 | 0 | 6 |

> `hallucination` = a clean attempt scored < 0.50. `abstention` = the exact
> instructed refusal string. `other` = empty, truncated, or a refusal in the
> model's own words — reported rather than guessed at.
> Abst + Correct + Halluc + Other = 100. Scores are regenerable from the raw file;
> re-run this tool after any scorer change rather than editing this file.

## Notes

C3 -15.5, C2 -15.0, C4 -12.5 vs C1, all p<.001. C3-C2 collapses to -0.5 p=.914 (was -4.5 p=.048 under C-prime).

PROVENANCE CORRECTION (2026-08-10): this run's `prompt_framing` was rebuilt. tools/document_run.py had called prompt_style(role) positionally against a (model, role) signature, so the role was looked up as a model name and the default OpenAI framing was recorded instead of the Nemotron framing actually used -- meta.json claimed the system prompt was the factual-assistant persona when the calls sent 'detailed thinking off' with the persona and brevity line in the user preamble. Answers, scores, git SHA and every other spec are untouched; only the framing record was wrong.

PROVENANCE CORRECTION (2026-08-10), second field: `retrieval.noise_filter_mode` read null in this file. tools/document_run.py asked for `wikidata.FILTER_MODE` through a getattr default, but the live constant is NOISE_FILTER_MODE, so the wrong name was recorded as a null instead of raising. ⚠️ The value below is RECONSTRUCTED, not recovered: NOISE_FILTER_MODE resolves from the environment at import and nothing archived captured it. 'property_type' is the module default, appears in no .env, is set by no archived runner, and is assigned only by tools/sweep_dev_filter_topk.py -- which none of these runs used. Flagged `reconstructed: true` so it is not mistaken for a measurement.

## Prompt

⚠️ Prompt wording is an experimental variable in this project, not boilerplate.

**Framing — `answer` role**
```
system:   'detailed thinking off'
preamble: 'You are a factual question-answering assistant. Answer questions concisely and accurately. Answer with the fact only - no explanation, no restatement of the question.\n\n'
```

**Framing — `condense` role**
```
system:   'detailed thinking off'
preamble: 'You are a factual question-answering assistant. Answer questions concisely and accurately.\n\n'
```

- `ABSTAIN_SENTINEL` = `The answer is not in the context.`
- `ABSTAIN_CLAUSE` = `Refuse if you don't find the answer and reply exactly: "The answer is not in the context."`
- `ANSWER_INSTRUCTION` = `Use the following information to answer the question as concisely as possible.`
- `ANSWER_INSTRUCTION_NO_CONTEXT` = `Answer the following question as concisely as possible.`

**C1_base_llm**
```
Answer the following question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

Question: {question}
Answer:
```

**C2_rag**
```
Use the following information to answer the question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

Context:
{context}

Question: {question}
Answer:
```

**C3_graph_rag**
```
Use the following information to answer the question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

Context:
{context}

Question: {question}
Answer:
```

**C4_answer**
```
Use the following information to answer the question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

Context:
{context}

Question: {question}
Answer:
```

**C4_condense**
```
Given the following Wikidata facts and the question, select only the relevant facts and summarise them in one or two sentences.

Facts:
{top_k_triples}

Question: {question}
Summary:
```

## Retrieval settings

- top_k: C2 30 · C3 30 · C4 30
- embedding model: `multi-qa-MiniLM-L6-cos-v1`
- noise filter: `property_type` · reverse limit 200 · per-property cap 30

## Operational

- workers: 3
- calls: 1001
- errors: 0
- retries_429: 27

## Files

- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**
- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.
- `report.md` — this file, regenerated by `tools/document_run.py`.
