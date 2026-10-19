# Environment — frozen protocol

## Answer model / semantic judge

- Server: local Ollama at `http://127.0.0.1:11434` (override with `ANSWER_BASE_URL`)
- Answer model: `qwen2.5:7b` (override with `ANSWER_MODEL`), temperature 0 (`ANSWER_TEMPERATURE`)
- Semantic judge: same local model by default (`SEMANTIC_JUDGE_MODEL`), temperature 0
- **Pin the model digest**, not just the tag, for any published run:
  record `ollama show qwen2.5:7b --modelfile` digest in the run summary.

## Embeddings (Phase 2 decision)

- Embedding model: **`bge-m3:latest`** via Ollama `/api/embeddings`
  (override with `OLLAMA_EMBEDDING_MODEL`).
- **Strict failure policy (C8 fix, 2026-10-18)**: a failing embedding request is
  retried 3× (60s timeout, 0.5s backoff; env `EMBEDDING_RETRIES`,
  `EMBEDDING_TIMEOUT`, `EMBEDDING_RETRY_BACKOFF`) and then raises
  `EmbeddingUnavailableError`. The historical silent hash fallback contaminated
  study records (see docs/AUDIT-2026-10-18.md C8) and is opt-in only:
  `USE_OLLAMA_EMBEDDING=0` (offline) or `EMBEDDING_ALLOW_FALLBACK=1`.
- Every run warms the embedding backend up front and aborts on failure.
- Verify a finished run with
  `python experiments/audit_embedding_dims.py --expect 1024`
  (scans `data/*.db`; every stored vector must be 1024-dim bge-m3).
- Repairing individual failed records (do not re-run the whole study):
  `python -m experiments.run_oracle_experiments --dataset <id> --limit <n> \
   --out results/<study> --append --ids <qid1>,<qid2>`
- The memory analyzer (fact/relation/event extraction) is deterministic and
  rule-based; it makes no LLM calls. Only answer generation, semantic judging
  and embeddings hit the model server.

## Hardware

Experiments were developed on RTX 5060 Laptop (8 GB VRAM). Measured warm
generation speed: qwen2.5:7b ≈ 2 tok/s, gemma3:12b ≈ 0.65 tok/s. Budget
accordingly: a qwen run of the 500-record LongMemEval oracle set costs
roughly 3–5 min/record (3 answer generations + 4 judge calls + embeddings).

## Python

- Python 3.11+ (developed on 3.13)
- `pip install -r requirements.txt`
- Run everything from the repository root.

## Phase 2 run matrix

| Study | Dataset id | n | Answer model | Judge model | Purpose |
|---|---|---|---|---|---|
| 2A | longmemeval_s_sample10 (full haystack) | 10 | qwen2.5:7b | qwen2.5:7b | protocol replication of the pre-fix n=10 run |
| 2B | longmemeval_oracle_n50 (filtered haystack, 6 question types) | 50 | qwen2.5:7b | qwen2.5:7b | primary n=50 scale-up |
| 2C | locomo_n50 (question-level subset; conversations joined at load) | 50 | qwen2.5:7b | qwen2.5:7b | cross-benchmark replication |
| 2D | longmemeval_oracle_n50 | 50 | gemma3:12b | qwen2.5:7b | model robustness (cross-design) |
| 2E | 2B subsample | 20 | — | gemma3:12b (re-judge) | inter-judge agreement (audit I2) |

Commands (run from the repository root; results go to per-study directories):

    # 2B (primary) and 2A (pilot) — same command shape, different dataset/limit
    OLLAMA_EMBEDDING_MODEL=bge-m3:latest \
    OLLAMA_MODEL=qwen2.5:7b OLLAMA_JUDGE_MODEL=qwen2.5:7b \
      python -m experiments.run_oracle_experiments \
      --dataset longmemeval_oracle_n50 --limit 50 \
      --out results/study2b_longmemeval_oracle_n50

    # 2C — LoCoMo cross-benchmark
    OLLAMA_EMBEDDING_MODEL=bge-m3:latest \
    OLLAMA_MODEL=qwen2.5:7b OLLAMA_JUDGE_MODEL=qwen2.5:7b \
      python -m experiments.run_oracle_experiments \
      --dataset locomo_n50 --limit 50 --out results/study2c_locomo_n50

    # 2D — gemma3:12b answers, qwen judge (cross-model design)
    OLLAMA_EMBEDDING_MODEL=bge-m3:latest \
    OLLAMA_MODEL=gemma3:12b OLLAMA_JUDGE_MODEL=qwen2.5:7b \
      python -m experiments.run_oracle_experiments \
      --dataset longmemeval_oracle_n50 --limit 50 \
      --out results/study2d_longmemeval_gemma

    # 2E — re-judge a seeded 20-record subsample of 2B with gemma3:12b, then kappa
    SEMANTIC_JUDGE_MODEL=gemma3:12b \
      python -m experiments.rejudge_answers \
      --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
      --output results/study2e_judge_gemma/comparison.jsonl \
      --limit 20 --seed 20261018
    python experiments/paired_stats.py \
      --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
      --judge-input results/study2e_judge_gemma/comparison.jsonl \
      --dataset longmemeval_oracle_n50

After every study: `python experiments/audit_embedding_dims.py` must report
all-1024-dim, then `python experiments/paired_stats.py --input
results/<study>/comparison.jsonl --dataset <dataset id>` for the condition
comparison.

Study sets are rebuilt with `experiments/build_study_set.py` at fixed seeds
(20261018); see README data section for provenance and license notes.

Cross-model judging in 2D/2E avoids answer-model self-preference bias in the
semantic evaluator. Study 2B uses a single judge for internal consistency;
2E quantifies how much that choice matters. Judge agreement is measured with
`experiments/rejudge_answers.py` (re-judges the *same stored answers* with a
second judge model, isolating judge variance from answer variance) and
`experiments/paired_stats.py --judge-input` (Cohen's kappa + raw agreement).
Condition comparisons use the same script's exact McNemar test and paired
bootstrap CIs.

## Reproducibility rules

1. The protocol is frozen before any n>=50 run (see docs/PROTOCOL.md).
2. Every run must store: dataset id, limit, model + digest, seeds, code commit hash.
3. `results/` is never committed; publish run summaries + per-question traces
   as release artifacts or in the paper appendix.

## Model selection env vars (verified 2026-10-18)

The pipeline reads exactly two model-selection variables:

- `ANSWER_MODEL` — answer generation (default `qwen2.5:7b`), read at import
  time by `evaluation/answer_generation.py`.
- `SEMANTIC_JUDGE_MODEL` — semantic judge (defaults to `ANSWER_MODEL`), read
  at import time by `evaluation/answer_semantic_eval.py`.

`OLLAMA_MODEL` / `OLLAMA_JUDGE_MODEL` are NOT consumed anywhere; studies 2A/2B
were unaffected only because their values equaled the defaults. Studies 2D/2E
must use the variables above.

## Study 2D (cross-model answer: gemma3:12b)

```bash
OLLAMA_EMBEDDING_MODEL=bge-m3:latest ANSWER_MODEL=gemma3:12b \
  SEMANTIC_JUDGE_MODEL=qwen2.5:7b \
  python -m experiments.run_oracle_experiments \
  --dataset longmemeval_oracle_n50 --limit 50 --out results/study2d_longmemeval_gemma
```

## Study 2E (judge reliability: gemma3:12b re-judging 2B's stored answers)

```bash
OLLAMA_EMBEDDING_MODEL=bge-m3:latest SEMANTIC_JUDGE_MODEL=gemma3:12b \
  python -m experiments.rejudge_answers \
  --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
  --output results/study2e_judge_gemma/comparison.jsonl \
  --limit 20 --seed 20261018
# then: paired_stats.py --judge-input results/study2e_judge_gemma/comparison.jsonl ...
# (same seed -> same 20 records as the 2B subsample; kappa computed on the pair)
```

## Study 2F (extractor ablation: LLM-driven formation, n=50)

```bash
EXTRACTOR_BACKEND=llm OLLAMA_EMBEDDING_MODEL=bge-m3:latest \
  ANSWER_MODEL=qwen2.5:7b SEMANTIC_JUDGE_MODEL=qwen2.5:7b \
  python -m experiments.run_oracle_experiments \
  --dataset longmemeval_oracle_n50 --limit 50 \
  --out results/study2f_longmemeval_llm_extractor
```

`EXTRACTOR_BACKEND=llm` swaps the formation extractor for
`memory_engine/llm_analyzer.LLMMemoryAnalyzer` (qwen2.5:7b, temperature 0,
4 messages per call, JSON-only). Everything downstream of formation is
identical to the rule arm; the runner records per-record extraction
diagnostics (`llm_calls`, `parse_failures`, `fallback_messages`,
`skipped_messages`, `facts`) in `comparison.jsonl` and aggregates them in
`summary.json`. `EXTRACTOR_MODEL` overrides the extractor model;
`EXTRACTOR_TIMEOUT`/`EXTRACTOR_RETRIES` tune the client.

### Study 2F results (completed 2026-10-19, 50/50, 0 errors, 1h34m)

Same 50 LongMemEval-s oracle questions as Study 2B (verified identical
question_id set and order; gold evidence counts identical, 100 total).
Extractor is the only varying component.

| metric | rule arm (2B) | LLM arm (2F) |
|---|---|---|
| formation stage recall | 0.236 [0.147, 0.331] | **0.838 [0.747, 0.918]** |
| evolution / context recall | 0.236 / 0.695 | 0.802 / 0.847 |
| first loss = F1 | 45/50 | 13/50 (none: 5/50 → 29/50) |
| semantic accuracy real / oracle-M / oracle-C | 66.0 / 83.3 / 75.0% | 70.8 / 71.4 / 70.8% |
| real vs oracle-M (McNemar) | +0.170, b=10 c=2, p=0.0386 | **+0.000, b=4 c=4, p=1.0000** |
| attributable failures | F1×9, E0×7, EVAL×3 | E0×6, F6×4, F1×3, EVAL×2, F5×1 |

Extraction diagnostics (LLM arm): 354 calls, 23 parse failures (rule
fallback, 37 messages, zero silent drops), 1451 facts; formation observes
2,262 memory items vs 335 under the rule arm (6.8×, median per-question
ratio). Conclusion: the oracle-memory advantage is a property of the
extractor, not of the pipeline family — replacing the extractor collapses
it from +17.3 points to zero and relocates the residual bottleneck to
benchmark evidence sufficiency (E0) and reasoning (F6).
