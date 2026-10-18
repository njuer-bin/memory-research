from evaluation import gate as gate_module
from experiments import run_oracle_experiments as runner


def test_evidence_gate_uses_benchmark_answer_not_real_prediction(monkeypatch):
    """The E0 gate must be judged against the benchmark reference answer.

    Judging it against the Real Memory prediction would make E0 depend on the
    very system failure being attributed (circularity, audit finding C2).
    """
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

    monkeypatch.setattr(gate_module, "evaluate_answer", fake_judge)

    gold = [{"evidence_id": "e0", "content": "Caroline went yesterday."}]
    result = runner._evidence_sufficiency(
        "When did Caroline go?",
        "7 May 2023",
        gold,
    )

    assert seen["candidate"] == "7 May 2023"
    assert result["status"] == "sufficient"
    assert result["evidence_sufficient"] is True


def test_evidence_gate_without_gold_evidence_is_insufficient():
    gold = []
    result = runner._evidence_sufficiency(
        "When did Caroline go?",
        "7 May 2023",
        gold,
    )

    assert result["status"] == "insufficient"
