"""Single source of truth for memory-lifecycle failure attribution.

Frozen protocol (see docs/PROTOCOL.md):

    1. Semantic evaluator judges the answer correct            -> no failure
    2. Semantic evaluation unavailable or uncertain            -> EVAL_SEMANTIC_UNCERTAIN
    3. Gold Evidence cannot answer the question (E0 gate,
       judged against the BENCHMARK REFERENCE ANSWER, never
       against the model prediction being diagnosed)           -> E0_EVIDENCE_INSUFFICIENT
    4. Earliest lifecycle stage where gold evidence is not
       fully present (content-validated)                       -> F1..F5
    5. All gold evidence reaches final context and the answer
       is still semantically incorrect                         -> F6_REASONING_CANDIDATE

Design notes
------------
* F6 is a *candidate* attribution: evidence sufficiency and answer correctness
  are both evaluated before a reasoning failure may be claimed.
* Lexical metrics (EM / token-F1) are reported separately and are NEVER an
  input to attribution. The lexical scorer produces false negatives on concise
  correct answers (e.g. "The Glass Menagerie" vs a long reference sentence),
  which previously leaked into F6 candidates.
* The E0 gate must be computed against the benchmark reference answer. Using
  the model's failed answer would make E0 depend on the very failure being
  attributed (circularity).
* Stages marked ``measured: false`` (e.g. an uninstrumented evolution stage)
  are skipped rather than reported as perfect recall.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class FailureLabel(str, Enum):
    F1_FORMATION = "F1_FORMATION"
    F2_STORAGE = "F2_STORAGE"
    F3_EVOLUTION = "F3_EVOLUTION"
    F4_RETRIEVAL = "F4_RETRIEVAL"
    F5_CONTEXT = "F5_CONTEXT"
    F6_REASONING_CANDIDATE = "F6_REASONING_CANDIDATE"
    E0_EVIDENCE_INSUFFICIENT = "E0_EVIDENCE_INSUFFICIENT"
    EVAL_SEMANTIC_UNCERTAIN = "EVAL_SEMANTIC_UNCERTAIN"


STAGE_TO_FAILURE = {
    "formation": FailureLabel.F1_FORMATION,
    "storage": FailureLabel.F2_STORAGE,
    "evolution": FailureLabel.F3_EVOLUTION,
    "retrieval": FailureLabel.F4_RETRIEVAL,
    "rerank": FailureLabel.F4_RETRIEVAL,
    "context": FailureLabel.F5_CONTEXT,
}

ATTRIBUTION_STAGES = ("formation", "storage", "evolution", "retrieval", "rerank", "context")


def _gold_ids(gold_evidence: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(x["evidence_id"]) for x in gold_evidence if x.get("evidence_id")}


def earliest_failed_stage(
    trace: Mapping[str, Any],
    gold_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> Optional[str]:
    """Earliest lifecycle stage where gold evidence is not fully present.

    Returns None when all gold evidence reaches final context (or when there
    is no gold evidence at all, which callers must treat as E0). Diagnostic
    runners that do not generate answers use this to report lifecycle recall
    without claiming anything about reasoning.
    """
    if gold_evidence is None:
        gold_evidence = trace.get("gold_evidence") or []
    gold = _gold_ids(gold_evidence)
    if not gold:
        return FailureLabel.E0_EVIDENCE_INSUFFICIENT.value

    for stage in ATTRIBUTION_STAGES:
        stage_data = trace.get(stage) or {}
        if stage_data.get("measured") is False:
            continue
        found = {str(x) for x in (stage_data.get("found") or [])}
        if not gold.issubset(found):
            return STAGE_TO_FAILURE[stage].value
    return None


def attribute_failure(
    trace: Mapping[str, Any],
    *,
    semantic_correct: Optional[bool],
    gate: Mapping[str, Any],
    gold_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> Optional[str]:
    """Attribute a failed answer to the earliest lifecycle boundary.

    Parameters
    ----------
    trace:
        Mapping with per-stage ``{"found": [...], "measured": bool}`` entries.
    semantic_correct:
        Verdict of the semantic evaluator on the *model answer*. ``None`` means
        the evaluator could not produce a stable verdict.
    gate:
        Precomputed E0 evidence-sufficiency gate (status in
        {"sufficient", "insufficient", "uncertain", "error"}).
    gold_evidence:
        Gold evidence records (used when the trace does not carry them).
    """
    if semantic_correct is True:
        return None
    if semantic_correct is None:
        return FailureLabel.EVAL_SEMANTIC_UNCERTAIN.value

    status = str(gate.get("status") or "")
    if status == "insufficient":
        return FailureLabel.E0_EVIDENCE_INSUFFICIENT.value
    if status != "sufficient":
        return FailureLabel.EVAL_SEMANTIC_UNCERTAIN.value

    return earliest_failed_stage(trace, gold_evidence) or (
        FailureLabel.F6_REASONING_CANDIDATE.value
    )


def control_failure_label(mode: str, *, semantically_wrong: bool) -> Optional[str]:
    """Label for oracle control conditions.

    Oracle Context supplies Gold Evidence directly to the answer model; a wrong
    answer there is a *reasoning control* failure, not a memory failure, and
    must never be counted as F1-F6 for the real memory system.
    """
    if not semantically_wrong:
        return None
    if mode == "oracle_context":
        return "RC_REASONING_CONTROL_FAILURE"
    return None
