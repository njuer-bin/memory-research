from evaluation.answer_semantic_eval import _deterministic_support


def test_relative_temporal_evidence_is_deterministically_sufficient():
    evidence = [
        {
            "content": "I went to a LGBTQ support group yesterday and it was so powerful.",
            "timestamp": "1:56 pm on 8 May, 2023",
        }
    ]

    result = _deterministic_support("7 May 2023", evidence)

    assert result is not None
    assert result[0] == "deterministic_temporal"
    assert result[1] == "yesterday"


def test_explicit_answer_phrase_is_deterministically_sufficient():
    evidence = [
        {"content": "Caroline researched adoption agencies."},
    ]

    result = _deterministic_support(
        "Caroline researched adoption agencies.",
        evidence,
    )

    assert result is not None
    assert result[0] == "deterministic_lexical"
    assert result[1] == "caroline researched adoption agencies"


def test_unrelated_evidence_does_not_get_deterministic_override():
    evidence = [
        {"content": "Melanie visited the beach last week."},
    ]

    result = _deterministic_support("adoption agencies", evidence)

    assert result is None
