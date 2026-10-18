"""Regression tests for the unified failure-attribution protocol.

These tests pin the frozen protocol from docs/PROTOCOL.md:

1. Semantic evaluator says correct                -> no failure
2. Semantic evaluation unavailable/uncertain      -> EVAL_SEMANTIC_UNCERTAIN
3. Gold Evidence cannot answer the question      -> E0_EVIDENCE_INSUFFICIENT
   (gate computed against the benchmark answer, never the model prediction)
4. Earliest lifecycle stage with content loss     -> F1..F5
5. All gold evidence reaches context, still wrong -> F6_REASONING_CANDIDATE

They also guard the two calibration-period false negatives that motivated the
semantic protocol in the first place (e47becba, 58bf7951).
"""
from evaluation.attribution import (
    FailureLabel,
    attribute_failure,
    control_failure_label,
)


def _trace(answer_correct=False):
    return {
        "question": "When did Caroline go to the LGBTQ support group?",
        "gold_evidence": [
            {
                "evidence_id": "e0",
                "text": "I went to a LGBTQ support group yesterday.",
                "source_id": "D1",
                "turn_id": "D1:turn_2",
            }
        ],
        "answer": {
            "answer": "Yesterday.",
            "correct": answer_correct,
        },
        "formation": {"found": ["e0"]},
        "storage": {"found": ["e0"]},
        "evolution": {"found": ["e0"]},
        "retrieval": {"found": ["e0"]},
        "rerank": {"found": ["e0"]},
        "context": {"found": ["e0"]},
    }


def _gate(status="sufficient"):
    return {"status": status}


# ---------------------------------------------------------------- protocol


def test_correct_answer_has_no_failure():
    trace = _trace(answer_correct=True)
    assert attribute_failure(trace, semantic_correct=True, gate=_gate()) is None


def test_missing_semantic_verdict_is_eval_uncertain():
    trace = _trace()
    assert (
        attribute_failure(trace, semantic_correct=None, gate=_gate())
        == FailureLabel.EVAL_SEMANTIC_UNCERTAIN.value
    )


def test_insufficient_evidence_is_e0():
    trace = _trace()
    assert (
        attribute_failure(
            trace, semantic_correct=False, gate=_gate("insufficient")
        )
        == FailureLabel.E0_EVIDENCE_INSUFFICIENT.value
    )


def test_uncertain_evidence_gate_is_eval_uncertain():
    trace = _trace()
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate("uncertain"))
        == FailureLabel.EVAL_SEMANTIC_UNCERTAIN.value
    )


def test_retrieval_failure_survives_sufficiency_gate():
    trace = _trace()
    trace["retrieval"] = {"found": []}
    trace["rerank"] = {"found": []}
    trace["context"] = {"found": []}
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate())
        == FailureLabel.F4_RETRIEVAL.value
    )


def test_formation_failure_reported_before_retrieval():
    trace = _trace()
    trace["formation"] = {"found": []}
    trace["storage"] = {"found": []}
    trace["retrieval"] = {"found": []}
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate())
        == FailureLabel.F1_FORMATION.value
    )


def test_f6_requires_sufficient_evidence_and_full_context():
    trace = _trace()
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate())
        == FailureLabel.F6_REASONING_CANDIDATE.value
    )


def test_unmeasured_stage_is_skipped_not_reported_perfect():
    """An uninstrumented evolution stage must not fake a full pass (C4)."""
    trace = _trace()
    trace["evolution"] = {"measured": False}
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate())
        == FailureLabel.F6_REASONING_CANDIDATE.value
    )


def test_gold_evidence_required_for_stage_scan():
    trace = _trace()
    trace["gold_evidence"] = []
    assert (
        attribute_failure(trace, semantic_correct=False, gate=_gate())
        == FailureLabel.E0_EVIDENCE_INSUFFICIENT.value
    )


# ------------------------------------------------------- oracle controls


def test_oracle_context_failure_is_reasoning_control():
    assert (
        control_failure_label("oracle_context", semantically_wrong=True)
        == "RC_REASONING_CONTROL_FAILURE"
    )


def test_oracle_context_correct_answer_has_no_failure():
    assert control_failure_label("oracle_context", semantically_wrong=False) is None


def test_oracle_memory_failure_is_not_control_failure():
    assert control_failure_label("oracle_memory", semantically_wrong=True) is None


# ------------------------------------------- calibration false negatives


def test_lexical_false_negative_e47becba_not_f6():
    """e47becba: judge false negative on a concise, evidence-backed answer.

    The semantic layer marks the answer correct, so attribution must return
    no failure even though the lexical scorer scored it wrong.
    """
    trace = _trace(answer_correct=False)
    assert attribute_failure(trace, semantic_correct=True, gate=_gate()) is None


def test_lexical_false_negative_58bf7951_not_f6():
    """58bf7951: relative-temporal answer resolved by evidence timestamps."""
    trace = _trace(answer_correct=False)
    assert attribute_failure(trace, semantic_correct=True, gate=_gate()) is None
