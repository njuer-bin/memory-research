# Study 2C paused (13/50 completed)

Resume with:

```bash
cd memory-research
OLLAMA_EMBEDDING_MODEL=bge-m3:latest OLLAMA_MODEL=qwen2.5:7b OLLAMA_JUDGE_MODEL=qwen2.5:7b \
  "C:/Users/13742/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \
  -m experiments.run_oracle_experiments \
  --dataset locomo_n50 --limit 50 --out results/study2c_locomo_n50 \
  --append --ids "$(cat results/study2c_locomo_n50/remaining_ids.txt)"
```

Completed rows are preserved in comparison.jsonl; the runner merges by
question_id and rewrites the deduplicated file at the end.
Error ids at pause: []
