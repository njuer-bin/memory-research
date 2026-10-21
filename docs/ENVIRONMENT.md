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

### Third-family blind judge (revision round 3, 2026-10-19)

Independence control for the author adjudication: a judge from a third
model family re-judges the same stored answers under the identical prompt
and deterministic rules, strictly blind (no access to the qwen/gemma
verdicts, the author adjudication, or the fact that the rows were
disagreement-selected). Any OpenAI/Anthropic-compatible endpoint works;
credentials come from the environment and are never written to artifacts.

```bash
BLIND_JUDGE_BASE_URL="https://opencode.ai/zen/go" \
  BLIND_JUDGE_API_KEY="<key>" \
  BLIND_JUDGE_MODEL="deepseek-v4.1-flash" \
  BLIND_JUDGE_SESSION="ses_2e_blindjudge_001" \
  python -m experiments.third_judge_blind \
  --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
  --out results/study2e_judge_gemma/third_judge_blind.json
# --retry-from <artifact> re-requests only failed/missing verdicts (thinking
# models need max_tokens>=8192; the gateway's WAF rejects the default
# Python-urllib user agent, so the script sets its own).
```

Results (60/60 verdicts, 0 errors): DeepSeek reproduces the author
adjudication on 13/16 disagreement cases (13/14 decidable), agrees with
qwen on only 9/16 and with gemma on only 4/16, and sides with the
adjudicated truth against the primary judge in 5 cases — the adjudication
is not a restatement of the primary judge. The blind judge is itself
strict (raw agreement 71.7%, kappa=0.409 vs qwen; 63.3%, kappa=0.198 vs
gemma), so it is used only as an independence check, never as an accuracy
source. Full table: `results/study2e_judge_gemma/third_judge_report.txt`.

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

## Phase 0 revision (2026-10-19) — reviewer-response hardening

Compiles to 12 pages total; the main text (Introduction–Conclusion) is
exactly **8 pages**, within the ARR limit. The appendix (pseudocode,
per-question tables, judge prompts, judge ledger, adjudication
guidelines, stratification recipes, statistical procedures, E0 ablation)
is excluded from the limit.

### G. Judge-study ledger (2E denominator hygiene)
All 2E denominators are tabulated in one place (Appendix "Judge-Study
Ledger"): 60 answers → 55 paired → 43 agreeing → 12 conflicting → 16
adjudicated → 14 decidable; plus the 9/60 model-independent deterministic
verdicts (identical for any judge, so not an inter-judge disagreement
source). The 2E body text no longer uses the misleading "9 more settled
by the deterministic channel" phrasing — the 9 is a subset count, stated
as such.

### E0. Anti-circular gate ablation (counterfactual)
Re-runs `attribute_failure` on stored 2B traces with the E0
evidence-sufficiency gate forced `sufficient`, to quantify what the gate
prevents being misattributed.

    python experiments/e0_gate_ablation.py \
      --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
      --summary results/study2b_longmemeval_oracle_n50/e0_gate_ablation.json

Result (real-memory arm): 7 E0 items exist; disabling the gate
reclassifies all 7 as lifecycle failures (6 → F1_FORMATION, 1 → F6),
inflating the F1-dominant count 9 → 15 (+67%) and the formation share
among attributed failures 47% → 79%. The gate is load-bearing, not
cosmetic (cited in §2B and Appendix "E0 Gate Ablation").

### B. Cluster-robust paired bootstrap
`paired_stats.py` now resamples by conversation (cluster), not by
question, correcting the correlation of within-session questions
(pseudoreplication). For LongMemEval (one question per session, 50/50)
the cluster-robust and naive CIs coincide; for LoCoMo (50 questions over
10 conversations) the cluster-robust CI is slightly wider but every
positive contrast survives. Enable with `--cluster-by-conversation` and
pass `--dataset` so the qid→conversation map is built.

    python experiments/paired_stats.py \
      --input results/study2c_locomo_n50/comparison.jsonl \
      --dataset locomo_n50 --cluster-by-conversation

Verified LoCoMo output: real vs Oracle Memory Δ+0.200, naive
[+0.040,+0.360] vs cluster [+0.038,+0.396]; real vs Oracle Context
Δ+0.220, naive [+0.040,+0.380] vs cluster [+0.062,+0.408].

### C. 2F closed-loop reframing
Study 2F is now presented as the protocol's prescriptive demonstration
(§2F "Prescriptive payoff"): the 2B diagnosis (formation = binding
constraint) motivates the LLM-extractor fix, and the boundary relocates
exactly as attribution predicts; lifecycle labels map to per-type
interventions (extractor/write-path for F1/storage, retriever/reranker
for F4/F5, question reform for E0, answer-model work for F6). This
answers the "diagnosis only, no fix" critique.

## Phase 1 revision (2026-10-19) — Study 2B scaled to n=200

The pilot n=50 Study 2B (qwen2.5:7b) is superseded by a scaled
n=200 run (`results/study2b_longmemeval_n200_deepseek/`), directly
answering reviewer P2 / reviewer-1 #2 (sample size, statistical power).
The run uses `deepseek-v4.1-flash` as BOTH answer and judge (temperature
0), source `longmemeval_oracle.json` (sha256_16 `5a6a52e1a98f85eb`, limit
200), git commit `84bedce2` (branch `fix/formation-diagnostics-evidence-based`).

**Regenerate / verify:**

    # paired stats (McNemar + paired bootstrap), naive == cluster (one
    # question per conversation, 200/200)
    python experiments/paired_stats.py \
      --input results/study2b_longmemeval_n200_deepseek/comparison.jsonl \
      --bootstrap-repeats 10000

    # E0 anti-circular gate ablation (pure counterfactual, no model calls)
    python experiments/e0_gate_ablation.py \
      --input results/study2b_longmemeval_n200_deepseek/comparison.jsonl \
      --out results/study2b_longmemeval_n200_deepseek/e0_gate_ablation.json

**Headline numbers (all traceable to summary.json / e0_gate_ablation.json):**
- Semantic accuracy: real 52.1% (85/163), Oracle Memory 69.1% (96/139),
  Oracle Context 82.1% (110/134).
- Real-vs-Oracle-Memory: McNemar b=20, c=5, p=0.0041, Δ+0.117,
  CI [+0.047,+0.195]. Real-vs-Oracle-Context: b=29, c=5, p<0.0001,
  Δ+0.205, CI [+0.111,+0.299]. (Both contrasts now decisive; at n=50 the
  second was marginal.)
- Stage recall (real): formation 0.427 [0.375,0.480], context 0.896
  [0.868,0.922]; storage/retrieval 1.000, rerank 0.990. First loss = F1
  in 155/200 (77.5%).
- E0 gate: 153 sufficient / 41 insufficient / 6 uncertain. Ablation:
  disabling the gate reclassifies all 10 E0 items as lifecycle failures
  (9 → F1), F1 count 61→72 (+18%), formation share 73%→86%.
- Subset draws on the three harder LongMemEval types only
  (knowledge-update 78, multi-session 62, temporal-reasoning 60); F1
  concentrates in multi-session (34).

**Paper consequences:** Study 2B is now the n=200 primary; 2D/2E/2F are
reframed as controlled ablations on the separate n=50 pilot subset (qwen
answer model). H2 is revised to a direction-agnostic, supported statement
(evidence availability dominates; reasoning on perfect evidence imperfect)
because the oracle-vs-oracle ordering is NOT robust — Oracle Context
exceeds Oracle Memory at n=200 and under gemma (2D) but not in the n=50
qwen pilot. The self-judging caveat (answer model = judge model in 2B) is
stated in §2B/§Limitations; the deterministic rule stack covers 67/163
real-memory verdicts. Main text compressed back to the ARR 8-page limit
(Conclusion ends p8, References start p9, 12 pages total).


---

## Phase 2 — External-system replication (Study 3A, Mem0 OSS)

Addresses the reviewers' #1 risk (external validity): run the frozen
lifecycle protocol unchanged on an external open-source memory system.

**Deployment (isolated venv, project-local stores):**
- `mem0ai` 2.2.1 in `$HOME/.workbuddy/binaries/python/envs/default`
  (`MEM0_TELEMETRY=false`); local Ollama `qwen2.5:7b` extraction,
  `bge-m3` (1024-dim) embeddings, persistent Chroma at
  `data/mem0_chroma`, isolated history DB at `data/mem0_history`.
- Ingestion streams conversations in 4-message chunks per `add()`
  (longer batches return 0 memories with local extractors — both qwen
  and gemma fail the JSON extraction schema at ≥6 messages).
- Adapter honesty: Mem0 artifacts carry no benchmark provenance, so
  evidence matching uses the exact/semantic tiers only (no provenance
  tier, no forged lineage); rerank = retrieval identity (Mem0 OSS has
  no reranker); storage holds extracted memories only (0 UPDATE/DELETE
  events observed on this data). Caveats recorded per-stage in details.

**Commands:**
    # full run (~75 min, 50 questions, qwen2.5:7b answer model — same
    # questions/judge as the Study 2B n=50 pilot subset)
    $HOME/.workbuddy/binaries/python/envs/default/Scripts/python.exe \
      experiments/run_mem0_external.py \
      --dataset longmemeval_oracle_n50 --limit 50 \
      --out results/study3a_mem0_n50

    # single-question repair (one transient Ollama error in the full run)
    ... run_mem0_external.py ... --ids gpt4_59149c77 --append

    # analysis: 3-condition accuracy, McNemar + bootstrap, stage recall,
    # failure labels, E0 distribution, per-type, side-by-side vs 2B n=50
    python experiments/analyze_mem0_external.py

    # E0 ablation + paired stats (same scripts as the in-house arms)
    python experiments/e0_gate_ablation.py \
      --input results/study3a_mem0_n50/comparison.jsonl \
      --out results/study3a_mem0_n50/e0_gate_ablation.json
    python experiments/paired_stats.py \
      --input results/study3a_mem0_n50/comparison.jsonl \
      --cluster-by-conversation

**Headline numbers (traceable to results/study3a_mem0_n50/summary.json):**
- Semantic accuracy: real 58.3% (28/48), Oracle Memory 79.6% (39/49),
  Oracle Context 77.1% (37/48).
- Real-vs-Oracle-Memory: b=11, c=1, p=0.0063, Δ+0.208, CI [+0.083,+0.333].
  Real-vs-Oracle-Context: b=11, c=2, p=0.0225, Δ+0.191, CI [+0.043,+0.340].
  Oracle-vs-oracle: p=1.0000, CI [-0.062,+0.104] — again null (evidence
  availability dominates regardless of which system forms the memory).
- Extraction: 1,101 messages → 283 LLM calls → 910 facts; 0 UPDATE,
  0 DELETE. Formation recall 0.359 (in-house rule arm on the same
  questions: 0.236).
- Failure labels (real): F1 9, E0 8, EVAL 2, F6 2, F4 1 — nearly
  identical to the in-house run on the same questions (9/7/3/0/0);
  F1+E0 = 17/22 attributable failures (77%).
- E0 gate: 40 sufficient / 8 insufficient / 2 uncertain. Ablation:
  6 of 8 E0 items would be mis-attributed to F1 (2 to F6), formation
  count 9→15 (+67%) — gate criticality replicates on an external system.

**Paper consequences:** Study 3A added to Results (n=50, same questions
as the 2B pilot), Analysis (taxonomy transfers), Conclusion,
Limitations (external validity now partially addressed, breadth remains),
contributions, and appendices (E0 ablation, statistics, environment).
Main text recompressed to the ARR 8-page limit (Conclusion ends p8,
References start p9, 13 pages total).

## Phase 3 (2026-10-20): 2B verdict repair, fourth-family re-judge, sensitivity

**2B exact-match repair (measurement fix).** The n=200 runner left 134
condition-answers unjudged whose text equals the gold answer string
(exact_match=1, semantic_correct=None; 28/47/59 per condition). These
are certainly correct; we count them True via a deterministic
exact-match channel (`experiments/sensitivity_analysis.py`, section 7).
Corrected headline: 59.2/76.9/87.6% (113/191, 143/186, 169/193);
McNemar Real-vs-OM b=40 c=7 p<1e-5, Real-vs-OC b=62 c=7 p<1e-5,
OM-vs-OC b=28 c=6 p=0.0002. Remaining exclusions 9/14/7 (API errors
6/9/2 + abstentions 3/5/5); counting them all wrong leaves every
contrast significant (56.5/71.5/84.5%; Real-vs-OM p=0.0001). The old
52.1/69.1/82.1% understated all conditions, oracle most.

**Cross-family blind re-judge (P0-1).** Seeded 30-record subsample
(seed 20261018) re-judged blind with qwen2.5:7b local Ollama
(`experiments/rejudge_2b_blind.py`, artifacts
`results/study2b_judge_qwen_n200/`): raw agreement 10/25, 15/23, 16/24;
kappa -0.214/+0.171/+0.143 vs corrected deepseek verdicts. Paired on
the subsample the primary contrast flips sign under qwen (+0.250 -> 
-0.080); disclosed in Results + Limitations with the confound (qwen is
the less truth-consistent family in 2E).

**LoCoMo n=200 external run (NOT merged).**
`G:/PythonProject/memory-failure-attribution/results/oracle_experiments_locomo_n200/`
(200/200, errors 0) is a valid *robustness signal* (first loss F1
140/200 even under a degraded backend) but is NOT protocol-consistent
with the paper: (1) embedding backend fell back to hash:384 (configured
qwen3-embedding:4b unavailable) vs frozen bge-m3; (2) E0 insufficient
86/200 (43%) vs 6% in 2C n=50; (3) code diverged (numeric_range judge
channel, commit 6f11aef0 dirty); (4) same deepseek self-judging caveat.
A merge requires a re-run from the paper repo with bge-m3 and the frozen
thresholds.

**Third API-family judge (standoff breaker, prepared 2026-10-20).** To break
the deepseek-vs-qwen 1-1 standoff on the subsample contrast direction,
`experiments/rejudge_2b_api_blind.py` re-judges the SAME seeded subsample
(seed 20261018, 30 records x 3 conditions) with an API-scale judge from a
family distinct from both deepseek and qwen (Claude/GPT class via an
Anthropic-compatible gateway). The prompt and all deterministic channels are
imported byte-identical from the primary judge (via
`experiments/third_judge_blind._blind_verdict`); only the backend changes, so
the local memory footprint is zero (no Ollama load). The report computes
per-condition kappa against both baselines, the discordant-row side-taking
(standoff resolution), paired McNemar under all three judges, and the
extrapolated full-sample movement. Run from repo root with the Anaconda
interpreter (pydantic lives there, not in the WorkBuddy-managed python):

```bash
BLIND_JUDGE_BASE_URL="https://<anthropic-compatible-gateway>" \
  BLIND_JUDGE_API_KEY="<key>" \
  BLIND_JUDGE_MODEL="<claude-or-gpt-family-model-id>" \
  BLIND_JUDGE_SESSION="ses_2b_apijudge_001" \
  /g/aconda/python.exe -m experiments.rejudge_2b_api_blind \
  --input results/study2b_longmemeval_n200_deepseek/comparison.jsonl \
  --qwen results/study2b_judge_qwen_n200/rejudge_blind.json \
  --outdir results/study2b_judge_api_n200 --limit 30 --seed 20261018
# --retry-from <artifact> re-requests only failed/missing verdicts
```

Validated without credentials: compile, full import chain, subsample
identity with the qwen re-judge (30/30 question_ids), retry reader
(decided-only), and statistic conventions (McNemar 28/6 -> p=0.0002).
Execution is blocked on `BLIND_JUDGE_API_KEY`; the paper currently discloses
the standoff in Results + Limitations (narrative refocused on the
judgment-free stage-recall chain as the headline).

## Phase 4 (2026-10-20): recompression to the ARR 8-page main-text limit

The review-response additions (EM-repair disclosure, re-judge paragraph,
uncertain-exclusion sensitivity, hard-subset reframe, Mem0 config) had
pushed the main text to 9 pages (Conclusion on p9). Recompressed back to
exactly 8 pages (Conclusion now ends on p8; Limitations and References
start p9; 15 pages total; 0 undefined refs/citations). Changes, all
redundancy removal with no number altered:

- Moved both per-type float tables (tab:2b-type, tab:2c-type) from the
  main text to Appendix B; §5 keeps one-sentence pointers
  (Appendix~\ref{app:traces}). Verified in the PDF: Table 2 (2B per-type)
  and Table 3 (2C per-category) render side by side at the top of p11.
- §6 Analysis: four paragraphs rewritten to drop numbers already in §5 /
  Table 1 (Where memory fails / oracle gap / E0 gate / taxonomy
  transfers); the "Reconciling" paragraph keeps the thesis sentence.
- §5: oracle-contrast paragraph tail, 2B opening (accuracy/EM-repair/E0
  sentences preserved verbatim), sensitivity paragraph, 2E, blind
  validation, 2F caveats and 3A paragraphs tightened; the 2F formation
  CIs ([0.147, 0.331] / [0.747, 0.918]) now live only in Appendix G.
- §1/§2/§3/§4: contribution 4 and post-contribution paragraph, the
  four-way divergence discussion, §3 evidence-tier example, metrics and
  sampling paragraphs trimmed.
- Verified post-edit: all headline numbers intact in the PDF
  (59.2/76.9/87.6%, 113/191/143/186/169/193, 155/200, 0.427/0.951,
  87=61+10+8+3+1+1+3 unlabeled, kappa -0.21/+0.17/+0.14, b=28 c=6
  p=0.0002, +17.3->0, 2F 2262/335/0.838/13-50/29-50, 3A 910/0.359).
