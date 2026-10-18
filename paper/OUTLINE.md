# Paper Outline — Memory Failure Attribution (working title)

**Working title**: *Where Does Long-Term Agent Memory Actually Fail? A Lifecycle-Boundary Attribution Protocol with Oracle Controls*

**Target venues** (CCF-B, main conference): EMNLP / NAACL / COLING.

**One-line claim**: when a long-term-memory LLM agent answers wrongly, the
failure can be attributed to a specific lifecycle boundary; we provide a
frozen, measurement-valid protocol (with three oracle control conditions)
that makes this attribution reliable, and we show on two benchmarks that the
dominant boundary is *not* the one commonly assumed.

---

## 1. Introduction

- Motivation: agent memory systems are evaluated end-to-end (QA accuracy);
  a wrong answer says nothing about *where* the pipeline broke.
- Gap: prior work reports failure taxonomies post-hoc from traces, but (a)
  lexical scorers conflate representation failures with reasoning failures,
  (b) evidence-lineage matches are not content-validated, (c) oracle controls
  are under-specified (gold injected into context vs. into memory are
  different interventions).
- Contributions:
  1. A frozen attribution protocol over a 9-stage memory lifecycle with an
     anti-circular evidence-sufficiency gate (E0) and a semantic-judge
     decision layer (lexical metrics never enter attribution).
  2. Three orthogonal control conditions: Real Memory / Oracle Memory /
     Oracle Context — separating retrieval-side from reasoning-side failures.
  3. Measurement-validity fixes with regression tests: provenance matches
     require content preservation (C3), semantic evidence matching requires
     faithful representation rather than topical overlap (C7), deterministic
     evaluator channels are narrowed to quote/date rules (C5).
  4. Empirical study on LongMemEval-s and LoCoMo-Refined with paired
     statistics (exact McNemar + paired bootstrap) and judge-reliability
     analysis.

## 2. Related Work

- Long-term memory benchmarks: LongMemEval, LoCoMo; memory architectures
  (RAG-style stores, reflective/consolidation memories).
- Agent failure analysis / error taxonomies; attribution debugging in ML
  systems; RAG failure analysis (retrieval vs. generation).
- Evaluation methodology: EM/F1 vs. semantic/LLM judging; judge reliability.
- Positioning: we contribute the *attribution instrument*, not a new memory
  system; our system is a concrete, open implementation the protocol runs on.

## 3. Protocol: Lifecycle-Boundary Attribution

- Lifecycle: Conversation → Formation → Storage → Evolution → Retrieval →
  Reranking → Context → Reasoning → Answer.
- Labels: F1..F6, E0, EVAL_SEMANTIC_UNCERTAIN; RC control label for oracle
  conditions.
- Decision order: semantic verdict → E0 gate (benchmark answer, never the
  model prediction) → earliest stage where gold evidence is not fully
  present → F6 candidate.
- Evidence matching tiers: exact → provenance (with content-preservation
  check, C3) → semantic (faithful-representation containment, C7).
- Unmeasured stages carry `measured: false` and are skipped, never read as
  perfect recall.
- All thresholds frozen in `docs/PROTOCOL.md` before any n≥50 run.

## 4. Experimental Design

- System under test: open rule-based analyzer + embedding retrieval +
  reranking + context assembly + local LLM answer model (qwen2.5:7b;
  gemma3:12b cross-check), bge-m3 embeddings, temperature 0.
- Conditions: Real / Oracle Memory / Oracle Context (definitions table).
- Datasets: LongMemEval-s n=10 full haystack (2A); LongMemEval oracle n=50
  stratified (2B); LoCoMo-Refined n=50 stratified, conversation-level (2C);
  gemma3:12b cross-model (2D); judge agreement n=20 (2E).
- Metrics: per-condition semantic accuracy; stage recall; failure-label
  distribution; exact McNemar + paired bootstrap CIs; Cohen's kappa between
  judges.
- Hypotheses:
  - H1: Real Memory loses a significant fraction of gold evidence before
    context (F1/F4/F5 dominate over F6).
  - H2: Oracle Memory ≫ Oracle Context accuracy implies retrieval/context
    assembly, not reasoning, is the bottleneck.
  - H3: Real-vs-Oracle inversions (oracle wrong, real right) are control
    failures, not evidence for reasoning strength.

## 5. Results (to be filled from Phase 2 runs)

- 5.1 Study 2A: protocol replication, n=10 full haystack; three-condition
  independence; calibration-period false negatives explained.
- 5.2 Study 2B: n=50 primary results; failure distribution; McNemar/bootstrap.
- 5.3 Study 2C: LoCoMo cross-benchmark replication; per-task-type breakdown.
- 5.4 Study 2D: gemma3:12b robustness; judge model effects.
- 5.5 Study 2E: judge agreement (kappa), uncertainty rate.

## 6. Analysis

- Where memory fails: boundary distribution and what it implies for
  architecture (formation coverage vs. retrieval vs. context budget).
- Why lexical scoring misleads: false-negative examples from the semantic
  layer; the C7 conflation mechanism behind inflated recall.
- Threats to validity: n=50 sample size, two benchmarks, local models,
  rule-based analyzer, judge model choice (quantified in 2E).

## 7. Conclusion

## Artifacts

- Code + frozen protocol + tests: this repository (public).
- Study subsets with seeds; run summaries with model digests and code
  commits; per-question traces in the appendix.
- Reproducibility anchor: Study 2A results committed under
  `results/study2a_longmemeval_s/`.

## Appendix plan

- A: full protocol pseudocode; B: per-question attribution table; C: judge
  prompts and normalization; D: stratification recipes; E: statistical
  procedures (exact McNemar, paired bootstrap); F: environment (models,
  digests, hardware, timings).
