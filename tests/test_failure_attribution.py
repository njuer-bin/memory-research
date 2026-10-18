import tracing.failure_attribution as failure_attribution


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


def test_insufficient_evidence_is_e0(monkeypatch):
    def fake_judge(question, candidate, evidence):
        return {
            "correct": False,
            "evidence_sufficient": False,
            "confidence": "high",
            "reason": "The evidence describes a different event.",
            "adjudication": "judge",
        }

    monkeypatch.setattr(failure_attribution, "evaluate_answer", fake_judge)
    trace = _trace()

    assert failure_attribution.attribute_failure(trace) == "E0_EVIDENCE_INSUFFICIENCY"
    assert trace["evidence_sufficiency"]["status"] == "insufficient"


def test_uncertain_evidence_gate_is_eval(monkeypatch):
    def fake_judge(question, candidate, evidence):
        return {
            "correct": False,
            "evidence_sufficient": True,
            "confidence": "low",
            "reason": "The evidence may be sufficient, but the case is ambiguous.",
            "adjudication": "judge",
        }

    monkeypatch.setattr(failure_attribution, "evaluate_answer", fake_judge)
    trace = _trace()

    assert failure_attribution.attribute_failure(trace) == "EVAL_EVIDENCE_SUFFICIENCY"
    assert trace["evidence_sufficiency"]["status"] == "uncertain"


def test_retrieval_failure_survives_sufficiency_gate(monkeypatch):
    def fake_judge(question, candidate, evidence):
        return {
            "correct": False,
            "evidence_sufficient": True,
            "confidence": "high",
            "reason": "The evidence directly answers the question.",
            "adjudication": "judge",
        }

    monkeypatch.setattr(failure_attribution, "evaluate_answer", fake_judge)
    trace = _trace()
    trace["retrieval"] = {"found": []}
    trace["rerank"] = {"found": []}
    trace["context"] = {"found": []}

    assert failure_attribution.attribute_failure(trace) == "F4_RETRIEVAL"


def test_f6_requires_sufficient_evidence_and_full_context(monkeypatch):
    def fake_judge(question, candidate, evidence):
        return {
            "correct": False,
            "evidence_sufficient": True,
            "confidence": "high",
            "reason": "The evidence directly answers the question.",
            "adjudication": "judge",
        }

    monkeypatch.setattr(failure_attribution, "evaluate_answer", fake_judge)
    trace = _trace()

    assert failure_attribution.attribute_failure(trace) == "F6_REASONING"
