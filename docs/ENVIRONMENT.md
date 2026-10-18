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

    OLLAMA_EMBEDDING_MODEL=bge-m3:latest \
    OLLAMA_MODEL=qwen2.5:7b OLLAMA_JUDGE_MODEL=qwen2.5:7b \
      python -m experiments.run_oracle_experiments \
      --dataset longmemeval_oracle_n50 --limit 50 \
      --out results/study2b_longmemeval_oracle_n50

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
