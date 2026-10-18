"""Integration test for run_oracle_experiments.run_record.

Regression context: the shared stage-measurement refactor (commit 9c9cd64)
renamed the runner's local ``_stage`` helper to the shared ``stage_dict`` but
left ``_oracle_context_trace`` calling the old name. Unit tests did not cover
the runner's per-record path, so the NameError surfaced only minutes into the
first full experiment run. This test exercises the whole per-record path with
engines and LLM calls mocked, so a regression in the three-condition trace
construction fails in the test suite instead of mid-experiment.
"""
from __future__ import annotations

from typing import Any

from experiments import run_oracle_experiments as runner


class _FakeStore:
    def __init__(self, facts: list[dict[str, Any]]):
        self._facts = facts

    def all_raw(self, user_id: str) -> list[dict[str, Any]]:
        return []

    def active_facts(self, user_id: str, include_history: bool = False):
        return list(self._facts)

    def events(self, user_id: str) -> list[dict[str, Any]]:
        return []


class _FakeEngine:
    def __init__(self, facts: list[dict[str, Any]], search_trace: dict[str, Any]):
        self.store = _FakeStore(facts)
        self.last_search_trace = search_trace

    def search(self, request: Any) -> list[Any]:
        return []


RECORD = {
    "question_id": "int-test-1",
    "question": "When did Caroline go to the park?",
    "answer": "7 May 2023",
    "gold_evidence": [
        {
            "evidence_id": "g1",
            "text": "Caroline went to the park on 7 May 2023.",
            "source_id": "s1",
            "turn_id": "t1",
            "timestamp": "2023-05-07T10:00:00Z",
        },
        {
            "evidence_id": "g2",
            "text": "She mentioned the park again on 9 May 2023.",
            "source_id": "s1",
            "turn_id": "t2",
            "timestamp": "2023-05-09T10:00:00Z",
        },
    ],
}


def _fact_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "f1",
            "content": "Caroline went to the park on 7 May 2023.",
            "status": "active",
            "source_session_id": "s1",
            "source_turn_id": "t1",
        },
        {
            "id": "f2",
            "content": "She mentioned the park again on 9 May 2023.",
            "status": "active",
            "source_session_id": "s1",
            "source_turn_id": "t2",
        },
    ]


def _search_trace(*, context_items: list[dict[str, Any]]) -> dict[str, Any]:
    both = [
        {"id": "f1", "content": "Caroline went to the park on 7 May 2023."},
        {"id": "f2", "content": "She mentioned the park again on 9 May 2023."},
    ]
    return {
        "context": context_items,
        "retrieval_candidates": both,
        "reranked": both,
    }


def _fake_verdict(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        "correct": True,
        "judge_correct": True,
        "evidence_sufficient": True,
        "confidence": "high",
        "reason": "matches gold evidence",
        "adjudication": "judge",
        "adjudication_anchor": None,
        "model": "fake-judge",
    }


def _patch_externals(monkeypatch) -> None:
    facts = _fact_rows()
    # The real engine's pipeline drops g2 at the context stage; the oracle
    # engine surfaces both gold facts. This mirrors the real asymmetry the
    # three conditions are designed to expose.
    real_trace = _search_trace(context_items=[facts[0]])
    oracle_trace = _search_trace(context_items=facts)
    monkeypatch.setattr(
        runner, "_real_engine",
        lambda record, idx: (_FakeEngine(facts, real_trace), "real-user"),
    )
    monkeypatch.setattr(
        runner, "_oracle_memory_engine",
        lambda gold, idx: (_FakeEngine(facts, oracle_trace), "oracle-user"),
    )
    monkeypatch.setattr(
        runner, "generate_and_score",
        lambda question, reference, context: {
            "answer": "7 May 2023",
            "correct": True,
            "exact_match": 1.0,
            "token_f1": 1.0,
            "threshold": 0.5,
            "temporal_equivalent": False,
            "temporal_anchor_date": None,
            "phrase_match": False,
            "explicit_date_mismatch": False,
            "model": "fake-answer",
            "base_url": "http://fake",
        },
    )
    monkeypatch.setattr(runner, "evaluate_answer", _fake_verdict)
    monkeypatch.setattr(
        runner, "evaluate_evidence_gate",
        lambda question, reference_answer, gold: {
            "status": "sufficient",
            "evidence_sufficient": True,
            "reason": "gold evidence supports the reference answer",
        },
    )


def test_run_record_measures_all_three_conditions(monkeypatch):
    _patch_externals(monkeypatch)

    row = runner.run_record(RECORD, 0)

    modes = row["modes"]
    assert set(modes) == {"real_memory", "oracle_memory", "oracle_context"}

    # Oracle Context: only context is measured; the other stages carry
    # measured=False so the attribution protocol skips them.
    oc = modes["oracle_context"]
    for stage in ("formation", "storage", "evolution", "retrieval", "rerank"):
        assert oc[stage].get("measured") is False, stage
    assert oc["context"].get("measured") is not False
    assert set(oc["context"]["found"]) == {"g1", "g2"}

    # Oracle Memory: full pipeline measured, gold injected as active facts.
    om = modes["oracle_memory"]
    assert set(om["formation"]["found"]) == {"g1", "g2"}
    assert set(om["context"]["found"]) == {"g1", "g2"}
    assert om["evolution"]["details"]["superseded_facts"] == 0

    # Real Memory: semantic verdict flows into attribution; a semantically
    # correct answer carries no failure label.
    rm = modes["real_memory"]
    assert rm["answer"]["semantic_correct"] is True
    assert rm["answer_failure_type"] is None
    assert rm["formation"]["details"]["semantic_memory_counts"]["total"] == 2


def test_run_record_oracle_context_control_failure(monkeypatch):
    """Oracle Context wrong answers are control failures, not F1-F6."""
    _patch_externals(monkeypatch)
    monkeypatch.setattr(
        runner, "_semantic_verdict",
        lambda question, answer_text, judge_gold: {
            "semantic_correct": False,
            "semantic_judge_correct": False,
            "semantic_evidence_sufficient": True,
            "semantic_confidence": "high",
            "semantic_reason": "wrong",
            "semantic_adjudication": "judge",
            "semantic_adjudication_anchor": None,
            "semantic_judge_model": "fake-judge",
        },
    )

    row = runner.run_record(RECORD, 0)

    oc = row["modes"]["oracle_context"]
    assert oc["answer"]["semantic_correct"] is False
    assert oc["answer_failure_type"] == "RC_REASONING_CONTROL_FAILURE"
    # Real Memory still runs the F1-F6 protocol: the context stage drops g2,
    # so the failure is attributed at the earliest failed stage (context).
    assert row["modes"]["real_memory"]["answer_failure_type"] == "F5_CONTEXT"
