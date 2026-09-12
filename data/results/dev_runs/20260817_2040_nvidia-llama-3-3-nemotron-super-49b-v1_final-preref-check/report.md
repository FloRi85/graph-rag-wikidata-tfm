# final-preref-check

**Final pre-reference validation at frozen settings (k=30, global, entity block all, fixed condense prompt): end-to-end check that parsing and metrics are pulled correctly before the reference run. Byte-identical settings to condense-fix-check.**

- Run: `20260817_2040`  ·  documented 2026-08-17T21:26:59
- Sample: `data/questions/mintaka_sample_dev_200.json` (n=200)
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1` @ `https://integrate.api.nvidia.com/v1`  ·  temperature 0  ·  max_tokens 128

## Outcomes

**Scoring** — every metric `metrics.score_results` computes.

| Arm | n scored | Attempted | Abst% | Correct% | Halluc% | Correct%@att | Halluc%@att | Other% | Coverage% | EM | F1 | EM@att | F1@att | Score | CtxW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_llm_abstain | 200 | 176 | 12.0 | 65.0 | 23.0 | 73.9 | 26.1 | 0.0 | 88.0 | 56.0 | 62.7 | 63.6 | 71.3 | 64.5 | — |
| rag | 200 | 121 | 39.0 | 54.5 | 6.0 | 90.1 | 9.9 | 0.5 | 60.5 | 42.0 | 49.2 | 69.4 | 81.3 | 53.7 | 7487 |
| graph_rag | 200 | 99 | 50.5 | 44.0 | 5.5 | 88.9 | 11.1 | 0.0 | 49.5 | 39.0 | 42.2 | 78.8 | 85.2 | 43.4 | 300 |
| rerank | 200 | 104 | 48.0 | 44.5 | 7.5 | 85.6 | 14.4 | 0.0 | 52.0 | 40.5 | 43.2 | 77.9 | 83.0 | 43.8 | 37 |

**Generation** — how the run behaved, not how it scored.

| Arm | Truncated | Errors | Excluded | Median words |
|---|---:|---:|---:|---:|
| base_llm_abstain | 0 | 0 | 0 | 2 |
| rag | 2 | 0 | 0 | 4 |
| graph_rag | 0 | 0 | 0 | 7 |
| rerank | 0 | 0 | 0 | 7 |

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
| boolean | 33 | 90.9 | 60.6 | 69.7 | 66.7 |
| date | 14 | 64.3 | 50.0 | 57.1 | 50.0 |
| entity | 125 | 55.5 | 37.9 | 41.1 | 37.1 |
| numerical | 28 | 60.7 | 35.7 | 57.1 | 39.3 |

**F1 by complexity**

| complexity | n | base_llm_abstain | graph_rag | rag | rerank |
|---|---:|---:|---:|---:|---:|
| comparative | 23 | 74.9 | 50.3 | 42.4 | 56.5 |
| count | 22 | 68.2 | 40.9 | 63.6 | 45.5 |
| difference | 22 | 44.3 | 14.4 | 25.0 | 20.7 |
| generic | 23 | 63.3 | 58.6 | 63.0 | 56.9 |
| intersection | 22 | 72.7 | 45.5 | 73.9 | 45.5 |
| multihop | 22 | 36.8 | 18.2 | 25.0 | 18.2 |
| ordinal | 22 | 69.5 | 69.7 | 63.7 | 62.1 |
| superlative | 22 | 47.8 | 12.7 | 13.0 | 13.6 |
| yesno | 22 | 86.4 | 68.2 | 72.7 | 68.2 |

**rerank — condensing step (Configuration 4 only)**

- condensations: 200
- truncated at CONDENSE_MAX_TOKENS: 1 (0.5%)  ⚠️ **those answering calls received a context cut off mid-sentence**
- condensation length: median 33 words, max 297

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
Given the following Wikidata facts and the question, select only the relevant facts and summarise them in at most two sentences.
Write only the facts themselves. Do not explain your reasoning, do not comment on what the facts fail to cover, and do not use knowledge beyond them.

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

- LLM calls (logical): 1000
    800 answers + 200 C4 condensations
- API attempts: 1008
    1002 succeeded · 6 failed
    8 extra from retries (a retry re-runs C4's condense as well as its answer)
- workers: 2
- responses_with_usage: 1002
- includes_sdk_internal_retries: False
- truncated: 2
- started: 20260817_204000
- documented_at: 2026-08-17T21:26:59
- raw_file: 20260817_2040_final-preref-check_raw.json

## Provenance

File checksums and available scoring history are retained in [meta.json](meta.json). Historical scorer hashes before and after 17 August 2026 are not directly comparable.

## Files

- `<run_id>_<slug>_raw.json` — every answer, verbatim and untruncated. **Immutable.**
- [meta.json](meta.json) — recorded settings, file checksums and available scoring history.
- `report.md` — this file, regenerated by `tools/document_run.py`.
