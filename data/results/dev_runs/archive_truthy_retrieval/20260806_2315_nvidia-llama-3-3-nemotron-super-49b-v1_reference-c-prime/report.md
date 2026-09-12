# reference-c-prime

**REFERENCE DEV RUN: all 4 configs on Nemotron under the shipped C-prime prompt. Establishes whether the gpt-4o-mini headline replicates once the prompt confound is removed.**

- Run: `20260806_2315`  ·  documented 2026-08-07T17:32:55
- Sample: `data/questions/mintaka_sample_dev_200.json` (n=200)
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1` @ `https://integrate.api.nvidia.com/v1`  ·  temperature 0  ·  max_tokens 128
- Recorded code state: uncommitted changes.

## Outcomes

| Arm | n | Abst% | Correct% | Halluc% | Other% | Truncated | Errors | Median words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_llm_abstain | 200 | 8.0 | 68.0 | 24.0 | 0.0 | 0 | 0 | 2 |
| rag | 200 | 32.0 | 59.5 | 8.0 | 0.5 | 6 | 0 | 3 |
| graph_rag | 200 | 60.5 | 35.0 | 4.5 | 0.0 | 0 | 0 | 7 |
| rerank | 200 | 45.5 | 42.5 | 7.5 | 4.5 | 1 | 0 | 7 |

> `hallucination` = a clean attempt scored < 0.50. `abstention` = the exact
> instructed refusal string. `other` = empty, truncated, or a refusal in the
> model's own words — reported rather than guessed at.
> Abst + Correct + Halluc + Other = 100. Scores are regenerable from the raw file;
> re-run this tool after any scorer change rather than editing this file.

## Notes

REPLICATES. Paired vs C1: C3 -19.5 [-26.0,-13.5], C4 -16.5 [-22.5,-10.5], C2 -16.0 [-23.0,-9.5], all p<.001. Scored under the FOUR-WAY classifier. Stronger than gpt-4o-mini in three ways: the selection effect is gone (C3 +3.8 p=.284 on its attempted subset vs -9.3 p=.036 there), threshold ordering is identical at all three cutoffs, and C4 now beats C2. Report beside it: C3-C2 = -4.5 p=.048 borderline; C3 buys 4.5% hallucination with 60.5% abstention; C4 is the only config with a non-zero Other% - it refuses in its own words rather than the instructed sentinel.

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
- `ABSTAIN_CLAUSE` = `If the answer is not in the context, reply exactly: "The answer is not in the context."`
- `ANSWER_INSTRUCTION` = `Use the following information to answer the question as concisely as possible.`
- `ANSWER_INSTRUCTION_NO_CONTEXT` = `Answer the following question as concisely as possible.`

**C1_base_llm**
```
Answer the following question as concisely as possible.
If the answer is not in the context, reply exactly: "The answer is not in the context."

Question: {question}
Answer:
```

**C2_rag**
```
Use the following information to answer the question as concisely as possible.
If the answer is not in the context, reply exactly: "The answer is not in the context."

Context:
{context}

Question: {question}
Answer:
```

**C3_graph_rag**
```
Use the following information to answer the question as concisely as possible.
If the answer is not in the context, reply exactly: "The answer is not in the context."

Context:
{context}

Question: {question}
Answer:
```

**C4_answer**
```
Use the following information to answer the question as concisely as possible.
If the answer is not in the context, reply exactly: "The answer is not in the context."

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
- rpm: unpaced (limiter shipped after this run)
- calls: 1001
- errors: 0
- retries_429: 47

## Files

- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**
- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.
- `report.md` — this file, regenerated by `tools/document_run.py`.
