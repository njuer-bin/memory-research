"""E0 evidence-sufficiency gate.

The gate answers one question: *can the Gold Evidence answer the exact
question, as judged against the BENCHMARK REFERENCE ANSWER?*

It must never be judged against the model prediction being diagnosed. Using
the model's failed answer would make E0 depend on the very failure under
attribution (circularity) -- see docs/AUDIT-2026-10-18.md finding C2.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from evaluation.answer_semantic_eval import evaluate_answer


def evaluate_evidence_gate(
    question: str,
    reference_answer: str,
    judge_gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a gate record with ``status`` in {sufficient, insufficient, uncertain, error}."""
    if not judge_gold:
        return {
            "status": "insufficient", "evidence_sufficient": False,
            "confidence": "high", "reason": "No benchmark gold evidence is available.",
            "adjudication": "no_gold_evidence",
        }
    try:
        verdict = evaluate_answer(question, str(reference_answer or ""), list(judge_gold))
    except Exception as exc:
        return {
            "status": "error", "evidence_sufficient": None, "confidence": "low",
            "reason": f"Evidence sufficiency judge failed: {type(exc).__name__}: {exc}",
            "adjudication": "semantic_judge_error",
        }
    sufficient = verdict.get("evidence_sufficient")
    confidence = str(verdict.get("confidence") or "medium").lower()
    if sufficient is False:
        status = "insufficient"
    elif sufficient is True and confidence != "low":
        status = "sufficient"
    else:
        status = "uncertain"
    return {
        "status": status,
        "evidence_sufficient": sufficient,
        "confidence": confidence,
        "reason": verdict.get("reason", ""),
        "adjudication": verdict.get("adjudication", "judge"),
        "adjudication_anchor": verdict.get("adjudication_anchor"),
        "model": verdict.get("model"),
    }
