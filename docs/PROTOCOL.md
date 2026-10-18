# Frozen Protocol: Memory Failure Attribution (v2)

This document is the **single normative reference** for how failures are
attributed in this repository. The protocol is *frozen*: once Phase 2 data
collection starts, no rule below may change without bumping the protocol
version and re-running affected experiments. Rationale for every rule is
traceable to `docs/AUDIT-2026-10-18.md`.

## 1. Lifecycle and failure labels

    Conversation → Formation → Storage → Evolution → Retrieval → Reranking → Context → Reasoning → Answer

| Label | Meaning |
|---|---|
| F1_FORMATION | evidence present in conversation, not extracted into usable memory |
| F2_STORAGE | extracted memory not persisted / not accessible |
| F3_EVOLUTION | formed memory superseded or retracted by governance before query time |
| F4_RETRIEVAL | evidence in memory but not retrieved / not reranked into the usable set |
| F5_CONTEXT | evidence lost while assembling the final answer context |
| F6_REASONING_CANDIDATE | all gold evidence reaches final context and the answer is still semantically wrong (*candidate* attribution) |
| E0_EVIDENCE_INSUFFICIENT | Gold Evidence itself cannot answer the question |
| EVAL_SEMANTIC_UNCERTAIN | evaluator cannot support a stable attribution |
| RC_REASONING_CONTROL_FAILURE | oracle-context control answered wrongly (reasoning-side control, never counted as F1–F6) |

## 2. Attribution decision procedure

`evaluation.attribution.attribute_failure(trace, semantic_correct, gate, gold_evidence)`
is the only implementation. Order matters:

1. Semantic evaluator says the answer is correct → **no failure**.
2. Semantic evaluation unavailable or uncertain (`semantic_correct is None`) →
   **EVAL_SEMANTIC_UNCERTAIN**.
3. E0 gate says the Gold Evidence cannot answer the question →
   **E0_EVIDENCE_INSUFFICIENT**.
4. Earliest lifecycle stage where gold evidence is not fully present →
   **F1..F5**.
5. All gold evidence reaches final context and the answer is still
   semantically incorrect → **F6_REASONING_CANDIDATE**.

Lexical metrics (EM / token-F1) are **never** an attribution input. They are
reported for reproducibility only: the lexical scorer produces false
negatives on concise correct answers (e.g. "The Glass Menagerie" against a
long reference sentence), which previously leaked into F6 candidates.

## 3. E0 evidence-sufficiency gate (anti-circularity)

`evaluation.gate.evaluate_evidence_gate(question, reference_answer, judge_gold)`
judges the Gold Evidence against the **benchmark reference answer** — never
against the model prediction being diagnosed. Using the model's failed answer
would make E0 depend on the very failure under attribution (audit finding C2).
Status is one of `sufficient` / `insufficient` / `uncertain` / `error`;
anything but `sufficient` blocks F1–F6 attribution.

## 4. Stage matching: provenance requires content preservation (C3)

`tracing.lifecycle.match_evidence` matches gold evidence in three tiers:
exact text, provenance, semantic. A **provenance match only counts as found
when the artifact content still recognizably preserves the gold text**:

- content-bearing vocabulary = words of length ≥ 3 plus every CJK character,
  minus a documented stopword list;
- at least `min(2, |gold content tokens|)` tokens are shared, and
- shared / min(|gold|, |artifact|) ≥ `PROVENANCE_CONTENT_THRESHOLD` (0.5).

Rejected matches are recorded in `provenance_only` with their shared-token
count and containment, so "lineage matched but content did not survive" is a
reported diagnostic instead of silent recall inflation.

**Semantic matching is faithful representation, not topical overlap (C7).**
The semantic tier uses the same content-bearing vocabulary and requires one
side's content tokens to be almost entirely contained in the other's
(`SEMANTIC_CONTENT_THRESHOLD` = 0.8, either direction), with the same absolute
floor of `min(2, |gold content tokens|)` shared tokens. Raw token overlap
(stopwords + character bigrams, symmetric containment at 0.55) used to score
two *distinct* same-topic evidence items at 0.56 — "…the park on 9 May 2023"
vs "…the park on 7 May 2023" — and conflate them, which is one mechanism
behind the legacy all-1.00 lifecycle recall. A distilled fact ("Caroline
attended the LGBTQ support group") still matches its longer gold turn at 1.0
by construction; a same-topic different-event sentence does not.

## 5. Evolution stage is measured, not assumed (C4)

- **Formation** = everything the extractor produced, including facts that
  governance later superseded.
- **Evolution** = the post-governance current belief state: facts with
  `status='active'` plus all relations / events / rules / profiles.

A fact that was formed but later superseded therefore surfaces as
**F3_EVOLUTION** instead of silently passing. Stages that a condition does
not measure carry `measured: false` and are skipped by the protocol, never
read as perfect recall.

## 6. Oracle controls are distinct conditions (C6)

| Condition | Memory content | Pipeline |
|---|---|---|
| Real Memory | learned from the conversation | full pipeline |
| Oracle Memory | gold evidence injected as already-formed active facts | **same** retrieval / rerank / context pipeline as Real Memory |
| Oracle Context | gold evidence injected directly into the answer context | pipeline bypassed |

Oracle Memory isolates retrieval/context-assembly failures; Oracle Context
isolates reasoning. Before the fix the two conditions shared one identical
answer context and were indistinguishable (audit finding C6).

## 7. Narrow deterministic rules in the semantic evaluator (C5)

`evaluation.answer_semantic_eval` may pass an answer **without** the LLM judge
only under two documented rules:

- **R1 (lexical)**: the longest shared span between candidate and evidence
  covers ≥ `DETERMINISTIC_COVERAGE` (0.8) of the candidate's content-bearing
  tokens — i.e. the candidate is essentially a verbatim quote of the
  evidence. Partial overlaps (e.g. "went to the park" vs "went to the LGBTQ
  support group") must be judged, never auto-passed.
- **R2 (temporal)**: the candidate is a pure date/relative expression that
  resolves to the same date as a relative expression in timestamped evidence
  (`score_answer` enforces full-string temporal forms).

The same R1 rule gates the post-judge repair of judge false negatives
(`lexical_anchor_override`).

## 8. Judge evidence normalization

The semantic judge reads the `content` key; lifecycle traces carry `text`.
Evidence passed to any judge is normalized (text → content) **with**
source/turn/timestamp metadata, because timestamps resolve relative temporal
expressions ("yesterday", "last week") in the judge prompt. Dropping them
makes the gate disagree with the runner on the same evidence.

## 9. Reproducibility

- Every result row carries `code_commit` (git HEAD at run time).
- Model endpoints, model names and temperatures are pinned in
  `docs/ENVIRONMENT.md` and overridable only via the documented environment
  variables.
- Runner v2 stores per-mode semantic verdicts and the E0 gate in
  `comparison.jsonl`; the post-hoc calibration layer reuses them unless
  `--force` is passed.
