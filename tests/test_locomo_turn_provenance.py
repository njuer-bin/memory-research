"""Regression tests for LoCoMo-to-MemoryEngine turn provenance alignment."""
from datasets.adapters import _locomo_evidence, _locomo_turns


def test_locomo_turn_ids_follow_ingested_nonempty_message_index():
    conversation = {
        "conversation": [
            {
                "session_id": "session_1",
                "messages": [
                    {"dia_id": "D1", "text": "first turn"},
                    {"dia_id": "D2", "text": ""},
                    {"dia_id": "D3", "text": "third source item"},
                    {"dia_id": "D4", "text": "   "},
                    {"dia_id": "D5", "text": "last turn"},
                ],
            }
        ]
    }
    turns = _locomo_turns(conversation)
    assert [turn["turn_id"] for turn in turns] == [
        "D1:turn_0", "D1:turn_1", "D1:turn_2",
    ]
    assert [turn["source_id"] for turn in turns] == ["D1", "D1", "D1"]
    assert [turn["benchmark_turn_id"] for turn in turns] == ["D1", "D3", "D5"]
    assert [turn["text"] for turn in turns] == [
        "first turn", "third source item", "last turn",
    ]


def test_locomo_turn_ids_reset_for_each_session():
    conversation = {
        "conversation": [
            {"session_id": "session_1", "messages": [{"text": "one"}]},
            {"session_id": "session_2", "messages": [{"text": "two"}, {"text": "three"}]},
        ]
    }
    turns = _locomo_turns(conversation)
    assert [turn["turn_id"] for turn in turns] == [
        "session_1:turn_0", "session_2:turn_0", "session_2:turn_1",
    ]


def test_locomo_benchmark_dia_id_overrides_conflicting_container_session_id():
    conversation = {
        "sessions": [
            {
                "session_id": "D10",
                "messages": [
                    {"dia_id": "D1:1", "text": "first"},
                    {"dia_id": "D1:2", "text": "second"},
                    {"dia_id": "D1:3", "text": "target evidence"},
                ],
            }
        ]
    }
    turns = _locomo_turns(conversation)
    assert [turn["turn_id"] for turn in turns] == [
        "D1:turn_0", "D1:turn_1", "D1:turn_2",
    ]
    assert all(turn["source_id"] == "D1" for turn in turns)


def test_locomo_public_dia_reference_resolves_to_message_text():
    conversation = {
        "sample_id": "conv-26",
        "sessions": [
            {
                "session_index": 1,
                "date_time": "1:56 pm on 8 May, 2023",
                "messages": [
                    {"dia_id": "D1:1", "role": "user", "text": "first"},
                    {"dia_id": "D1:2", "role": "assistant", "text": "second"},
                    {"dia_id": "D1:3", "role": "user", "text": "I went to a LGBTQ support group yesterday and it was so powerful."},
                ],
            }
        ],
    }
    evidence = _locomo_evidence(
        "D1:3",
        example_id="locomo_refined_public:conv-26#q0000",
        conversation=conversation,
    )
    assert len(evidence) == 1
    assert evidence[0]["granularity"] == "turn"
    assert evidence[0]["source_id"] == "D1"
    assert evidence[0]["turn_id"] == "D1:turn_2"
    assert evidence[0]["text"].startswith("I went to a LGBTQ support group")
    assert evidence[0]["role"] == "user"


def test_locomo_public_conversation_is_normalized_to_runner_schema():
    from datasets.adapters import normalize_locomo_refined

    raw = {
        "sample_id": "conv-26",
        "question": "When did Caroline go to the LGBTQ support group?",
        "answer": "7 May 2023",
        "evidence": ["D1:3"],
    }
    conversation = {
        "sample_id": "conv-26",
        "sessions": [
            {
                "session_index": 1,
                "date_time": "1:56 pm on 8 May, 2023",
                "messages": [
                    {"dia_id": "D1:1", "speaker": "Caroline", "role": "user", "text": "hello"},
                    {"dia_id": "D1:2", "speaker": "Melanie", "role": "assistant", "text": "hi"},
                    {"dia_id": "D1:3", "speaker": "Caroline", "role": "user", "text": "I went to a LGBTQ support group yesterday and it was so powerful."},
                ],
            }
        ],
    }
    record = normalize_locomo_refined(raw, index=0, conversation=conversation)
    assert isinstance(record["conversation"], list)
    assert record["conversation"][0]["session_id"] == "D1"
    assert record["gold_evidence"][0]["granularity"] == "turn"
    assert record["gold_evidence"][0]["turn_id"] == "D1:turn_2"
    assert record["gold_evidence"][0]["text"].startswith("I went to a LGBTQ support group")
    assert record["answer"] == "7 May 2023"
