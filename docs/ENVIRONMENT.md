# Environment — frozen protocol

## Answer model / semantic judge

- Server: local Ollama at `http://127.0.0.1:11434` (override with `ANSWER_BASE_URL`)
- Answer model: `qwen2.5:7b` (override with `ANSWER_MODEL`), temperature 0 (`ANSWER_TEMPERATURE`)
- Semantic judge: same local model by default (`SEMANTIC_JUDGE_MODEL`), temperature 0
- **Pin the model digest**, not just the tag, for any published run:
  record `ollama show qwen2.5:7b --modelfile` digest in the run summary.

## Hardware

Experiments were developed on RTX 5060 Laptop (8 GB VRAM). Expected runtime
scales roughly linearly with n; the n=10 LongMemEval sanity run took ~22 min.

## Python

- Python 3.11+ (developed on 3.13)
- `pip install -r requirements.txt`
- Run everything from the repository root.

## Reproducibility rules

1. The protocol is frozen before any n>=50 run (see docs/PROTOCOL.md).
2. Every run must store: dataset id, limit, model + digest, seeds, code commit hash.
3. `results/` is never committed; publish run summaries + per-question traces
   as release artifacts or in the paper appendix.
