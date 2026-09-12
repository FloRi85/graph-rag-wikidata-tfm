# sweep-k10

**k-sweep arm 1 of 3 (k=10). Declared in advance in the development protocol: criterion is overall F1 per config with paired bootstrap CIs, ties to the smaller k. Primary allocation (global z-norm), ENTITY_BLOCK_MODE=all. NOTE: the question-entity block reaches C2/C3/C4 but not C1, so the C1-vs-retrieval delta measures retrieval PLUS an entity annotation.**

- Run: `20260817_0245`  ·  documented 2026-08-17T16:46:47
- Sample: `data/questions/mintaka_sample_dev_200.json` (n=200)
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1` @ `https://integrate.api.nvidia.com/v1`  ·  temperature 0  ·  max_tokens 128

## Outcomes

**Scoring** — every metric `metrics.score_results` computes.

| Arm | n scored | Attempted | Abst% | Correct% | Halluc% | Correct%@att | Halluc%@att | Other% | Coverage% | EM | F1 | EM@att | F1@att | Score | CtxW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_llm_abstain | 195 | 172 | 11.3 | 68.2 | 20.0 | 77.3 | 22.7 | 0.5 | 88.2 | 56.9 | 65.0 | 64.5 | 73.7 | 67.6 | — |
| rag | 195 | 125 | 35.4 | 57.9 | 6.2 | 90.4 | 9.6 | 0.5 | 64.1 | 47.2 | 53.4 | 73.6 | 83.2 | 56.6 | 2803 |
| graph_rag | 195 | 80 | 59.0 | 34.4 | 6.7 | 83.8 | 16.2 | 0.0 | 41.0 | 30.3 | 33.3 | 73.8 | 81.1 | 34.0 | 102 |
| rerank | 195 | 103 | 45.6 | 43.6 | 9.2 | 82.5 | 17.5 | 1.5 | 52.8 | 38.5 | 42.5 | 72.8 | 80.4 | 43.6 | 186 |

**Generation** — how the run behaved, not how it scored.

| Arm | Truncated | Errors | Excluded | Median words |
|---|---:|---:|---:|---:|
| base_llm_abstain | 1 | 0 | 5 | 2 |
| rag | 0 | 0 | 5 | 3 |
| graph_rag | 0 | 0 | 5 | 7 |
| rerank | 0 | 5 | 0 | 4 |

> `hallucination` = a clean attempt scored < 0.50. `abstention` = the exact
> instructed refusal string. `other` = empty, truncated, or a refusal in the
> model's own words — reported rather than guessed at.
> Abst + Correct + Halluc + Other = 100. Coverage = 1 - Abst.
> EM/F1 are over ALL questions (a non-attempt scores 0); the @att pair is
> over the attempted subset only, which is why it is far higher.
> Scores are regenerable from the raw file; re-run this tool after any
> scorer change rather than editing this file.

**F1 by answer type**

| answer type | n | base_llm_abstain | graph_rag | rag | rerank |
|---|---:|---:|---:|---:|---:|
| boolean | 33 | 93.9 | 60.6 | 78.8 | 66.7 |
| date | 14 | 71.4 | 35.7 | 50.0 | 50.0 |
| entity | 122 | 57.2 | 28.6 | 46.8 | 38.3 |
| numerical | 26 | 61.5 | 19.2 | 53.8 | 26.9 |

**F1 by complexity**

| complexity | n | base_llm_abstain | graph_rag | rag | rerank |
|---|---:|---:|---:|---:|---:|
| comparative | 23 | 74.9 | 45.3 | 55.4 | 69.6 |
| count | 21 | 71.4 | 23.8 | 61.9 | 33.3 |
| difference | 21 | 41.9 | 19.8 | 39.4 | 17.2 |
| generic | 22 | 71.5 | 49.1 | 62.2 | 49.1 |
| intersection | 22 | 75.3 | 31.2 | 74.5 | 48.4 |
| multihop | 21 | 36.5 | 4.8 | 24.3 | 21.8 |
| ordinal | 22 | 69.5 | 48.5 | 63.6 | 50.0 |
| superlative | 21 | 50.1 | 9.5 | 13.6 | 24.5 |
| yesno | 22 | 90.9 | 63.6 | 81.8 | 63.6 |

**rerank — condensing step (Configuration 4 only)**

- condensations: 195
- truncated at CONDENSE_MAX_TOKENS: 44 (22.6%)  ⚠️ **those answering calls received a context cut off mid-sentence**
- condensation length: median 201 words, max 306

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
- `ENTITY_BLOCK_MODE` = `all`

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

{entities}Context:
{context}

Question: {question}
Answer:
```

**C3_graph_rag**
```
Use the following information to answer the question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

{entities}Context:
{context}

Question: {question}
Answer:
```

**C4_answer**
```
Use the following information to answer the question as concisely as possible.
Refuse if you don't find the answer and reply exactly: "The answer is not in the context."

{entities}Context:
{context}

Question: {question}
Answer:
```

**C4_condense**
```
Given the following Wikidata facts and the question, select only the relevant facts and summarise them in one or two sentences.

{entities}Facts:
{top_k_facts}

Question: {question}
Summary:
```

## Retrieval settings

- top_k: C2 10 · C3 10 · C4 10
- embedding model: `multi-qa-MiniLM-L6-cos-v1` · top-k allocation: `global`
- noise filter: `property_type` · per-property cap 30
- statements: cache v7 · row limit 4000 · rank policy ['preferred', 'normal'] · labels `en,mul`
- strict retrieval: True · follows redirects: True
- articles: cache v1 · 300-word windows, 50 overlap · strict retrieval: True · 3 fetch attempts
- dormant: {'hops': 1} · legacy: None

## Generation

- answer max_tokens 128 · C4 condense max_tokens 384 · temperature 0

## Retrieval preflight

- gold entities checked: 238
- truncated (served an arbitrary slice): 154 of 238
    - reverse 154 — a property hit REVERSE_PER_PROP_CAP: Q102427, Q103916, Q1046088, Q1066, Q10853588, Q1125021, Q11424, Q11571, Q1163715, Q11649, Q11679, Q11696, Q11701, Q11930, Q1196645, Q12003, Q1203, Q1215884, Q1215892, Q12391356, Q12393, Q12557, Q12560, Q125904, Q1261, Q131976, Q134969, Q1370, Q1384, Q1393, Q1397, Q1407, Q142, Q1439, Q145, Q15228, Q153056, Q154958, Q155, Q155223, Q1558, Q15615, Q1603, Q17, Q170263, Q172678, Q173, Q173496, Q1744, Q1747150, Q178750, Q181484, Q183492, Q184839, Q18642757, Q18756, Q190155, Q19020, Q192812, Q19610114, Q208408, Q2117272, Q211872, Q2133344, Q213417, Q213812, Q216930, Q220, Q222047, Q2263, Q23548, Q23666, Q23780734, Q241163, Q265538, Q268181, Q27593, Q277551, Q27950674, Q289, Q29468, Q29552, Q30, Q32096, Q3308007, Q33240, Q33999, Q35332, Q35657, Q36153, Q36159, Q362, Q364295, Q37175, Q3772, Q38022, Q38111, Q396, Q40096, Q405, Q41254, Q41396, Q41421, Q420292, Q42493, Q43919, Q441214, Q4439148, Q44703, Q458346, Q45875, Q46, Q462, Q4630676, Q4692, Q48, Q49, Q49201, Q49740, Q50868, Q5107, Q515, Q5389, Q545007, Q5451, Q546692, Q5484, Q5705738, Q57147, Q600344, Q615, Q62, Q6256, Q65, Q671069, Q7322, Q739499, Q745979, Q782, Q7889, Q79, Q7926203, Q797, Q812, Q81931, Q8337, Q844, Q864, Q8740, Q8877, Q889821, Q904528, Q95074, Q9960
    - forward 0 — STATEMENT_ROW_LIMIT bound (never bound)

- articles checked: 238
- no English article: 7 — contributing no chunks to Configuration 2's pool: Q14174302, Q14175641, Q2133344, Q29559108, Q4439148, Q5705738, Q94820254

## Operational

- LLM calls (logical): 990
    795 answers + 195 C4 condensations
- API attempts: 1222
    1033 succeeded · 189 failed
    232 extra from retries (a retry re-runs C4's condense as well as its answer)
- workers: 8
- responses_with_usage: 1033
- includes_sdk_internal_retries: False
- truncated: 1
- started: 20260817_024521
- documented_at: 2026-08-17T16:46:47
- raw_file: 20260817_0245_sweep-k10_raw.json

## Provenance

File checksums and available scoring history are retained in [meta.json](meta.json). Historical scorer hashes before and after 17 August 2026 are not directly comparable.

## Files

- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**
- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.
- `report.md` — this file, regenerated by `tools/document_run.py`.
