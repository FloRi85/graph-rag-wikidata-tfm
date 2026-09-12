# sweep-k30

**k-sweep arm 2 of 3 (k=30). Declared in advance in the development protocol: criterion is overall F1 per config with paired bootstrap CIs, ties to the smaller k. Primary allocation (global z-norm), ENTITY_BLOCK_MODE=all. NOTE: the question-entity block reaches C2/C3/C4 but not C1, so the C1-vs-retrieval delta measures retrieval PLUS an entity annotation.**

- Run: `20260817_0324`  ·  documented 2026-08-17T16:46:48
- Sample: `data/questions/mintaka_sample_dev_200.json` (n=200)
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1` @ `https://integrate.api.nvidia.com/v1`  ·  temperature 0  ·  max_tokens 128
- Recorded code state: uncommitted changes.

## Outcomes

**Scoring** — every metric `metrics.score_results` computes.

| Arm | n scored | Attempted | Abst% | Correct% | Halluc% | Correct%@att | Halluc%@att | Other% | Coverage% | EM | F1 | EM@att | F1@att | Score | CtxW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_llm_abstain | 197 | 174 | 11.7 | 65.5 | 22.8 | 74.1 | 25.9 | 0.0 | 88.3 | 55.8 | 62.9 | 63.2 | 71.2 | 65.1 | — |
| rag | 197 | 116 | 41.1 | 54.3 | 4.6 | 92.2 | 7.8 | 0.0 | 58.9 | 43.1 | 50.1 | 73.3 | 85.1 | 53.5 | 7509 |
| graph_rag | 197 | 96 | 51.3 | 44.2 | 4.6 | 90.6 | 9.4 | 0.0 | 48.7 | 39.1 | 42.3 | 80.2 | 86.9 | 43.6 | 300 |
| rerank | 197 | 112 | 43.1 | 49.7 | 7.1 | 87.5 | 12.5 | 0.0 | 56.9 | 44.7 | 48.3 | 78.6 | 85.0 | 49.0 | 162 |

**Generation** — how the run behaved, not how it scored.

| Arm | Truncated | Errors | Excluded | Median words |
|---|---:|---:|---:|---:|
| base_llm_abstain | 0 | 0 | 3 | 2 |
| rag | 0 | 0 | 3 | 4 |
| graph_rag | 0 | 0 | 3 | 7 |
| rerank | 0 | 3 | 0 | 4 |

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
| boolean | 32 | 93.8 | 62.5 | 71.9 | 68.8 |
| date | 14 | 64.3 | 50.0 | 64.3 | 57.1 |
| entity | 124 | 55.5 | 37.4 | 42.5 | 43.7 |
| numerical | 27 | 59.3 | 37.0 | 51.9 | 40.7 |

**F1 by complexity**

| complexity | n | base_llm_abstain | graph_rag | rag | rerank |
|---|---:|---:|---:|---:|---:|
| comparative | 23 | 77.0 | 54.1 | 48.8 | 69.6 |
| count | 21 | 66.7 | 42.9 | 57.1 | 42.9 |
| difference | 22 | 37.4 | 12.1 | 25.0 | 14.4 |
| generic | 23 | 63.3 | 57.1 | 63.1 | 61.2 |
| intersection | 22 | 75.3 | 45.5 | 74.8 | 50.0 |
| multihop | 22 | 39.9 | 18.2 | 27.7 | 31.8 |
| ordinal | 22 | 69.5 | 69.7 | 59.2 | 71.2 |
| superlative | 21 | 46.3 | 13.3 | 13.6 | 25.4 |
| yesno | 21 | 90.5 | 66.7 | 81.0 | 66.7 |

**rerank — condensing step (Configuration 4 only)**

- condensations: 197
- truncated at CONDENSE_MAX_TOKENS: 35 (17.8%)  ⚠️ **those answering calls received a context cut off mid-sentence**
- condensation length: median 166 words, max 297

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

- top_k: C2 30 · C3 30 · C4 30
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

- LLM calls (logical): 994
    797 answers + 197 C4 condensations
- API attempts: 1190
    1028 succeeded · 162 failed
    196 extra from retries (a retry re-runs C4's condense as well as its answer)
- workers: 8
- responses_with_usage: 1028
- includes_sdk_internal_retries: False
- truncated: 0
- started: 20260817_032438
- documented_at: 2026-08-17T16:46:48
- raw_file: 20260817_0324_sweep-k30_raw.json

## Provenance

File checksums and available scoring history are retained in [meta.json](meta.json). Historical scorer hashes before and after 17 August 2026 are not directly comparable.

## Files

- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**
- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.
- `report.md` — this file, regenerated by `tools/document_run.py`.
