from experiments import run_oracle_experiments as runner


def test_evidence_gate_uses_benchmark_answer_not_real_prediction(monkeypatch):
    seen = {}

    def fake_judge(question, candidate, evidence):
        seen["candidate"] = candidate
        return {
            "correct": True,
            "evidence_sufficient": True,
            "confidence": "high",
            "reason": "The gold evidence directly supports the benchmark answer.",
            "adjudication": "judge",
        }

    monkeypatch.setattr(runner, "evaluate_answer", fake_judge)

    gold = [{"evidence_id": "e0", "content": "Caroline went yesterday."}]
    result = runner._evidence_sufficiency(
        "When did Caroline go?",
        "7 May 2023",
        gold,
    )

    assert seen["candidate"] == "7 May 2023"
    assert result["status"] == "sufficient"
    assert result["evidence_sufficient"] is True
