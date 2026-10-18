"""Tests for lifecycle evidence matching (audit findings C3 and C7).

Provenance matches (same source turn/session) must also validate that the
artifact content still recognizably preserves the gold text; otherwise a
stage could report perfect recall over corrupted or replaced content (C3).

Semantic matches must be faithful representations, not topical overlap: two
distinct evidence items about the same topic share stopwords and dates and
used to be conflated by raw token overlap, inflating recall (C7).
"""
from tracing.lifecycle import (
    PROVENANCE_CONTENT_THRESHOLD,
    SEMANTIC_CONTENT_THRESHOLD,
    match_evidence,
)


GOLD = [
    {
        "evidence_id": "e1",
        "text": "Caroline went to the LGBTQ support group yesterday.",
        "source_id": "D1",
        "turn_id": "D1:turn_2",
    }
]


def test_provenance_match_with_preserved_content_is_found():
    artifacts = [
        {
            "id": "m1",
            "content": "Caroline went to the LGBTQ support group on 7 May 2023.",
            "turn_id": "D1:turn_2",
        }
    ]

    result = match_evidence(GOLD, artifacts)

    assert result["found"] == ["e1"]
    assert result["match_types"]["e1"] == "provenance"
    assert result["provenance_only"] == []


def test_provenance_match_with_unrelated_content_is_not_found():
    """Same lineage, different content: lineage alone must not count."""
    artifacts = [
        {
            "id": "m1",
            "content": "The weather in Paris was mild and rainy.",
            "turn_id": "D1:turn_2",
        }
    ]

    result = match_evidence(GOLD, artifacts)

    assert result["found"] == []
    assert result["missing"] == ["e1"]
    assert len(result["provenance_only"]) == 1
    entry = result["provenance_only"][0]
    assert entry["evidence_id"] == "e1"
    assert entry["reason"] == "content_not_preserved"
    assert entry["shared_tokens"] == 0
    assert entry["containment"] < PROVENANCE_CONTENT_THRESHOLD


def test_provenance_match_with_empty_content_is_not_found():
    artifacts = [{"id": "m1", "content": "", "turn_id": "D1:turn_2"}]

    result = match_evidence(GOLD, artifacts)

    assert result["found"] == []
    assert result["provenance_only"][0]["reason"] == "empty_content"


def test_provenance_rejection_falls_through_to_exact_match():
    """Evidence preserved in another artifact still counts."""
    artifacts = [
        {"id": "m1", "content": "totally unrelated text", "turn_id": "D1:turn_2"},
        {"id": "m2", "content": "Caroline went to the LGBTQ support group yesterday."},
    ]

    result = match_evidence(GOLD, artifacts)

    assert result["found"] == ["e1"]
    assert result["match_types"]["e1"] == "exact"


def test_semantic_match_still_works_without_provenance():
    artifacts = [
        {"id": "m1", "content": "Caroline went to the LGBTQ support group yesterday."}
    ]

    result = match_evidence(GOLD, artifacts)

    assert result["found"] == ["e1"]


# --- Audit finding C7: semantic matching is faithful representation, not
# topical overlap. Raw token overlap (stopwords + bigrams) used to score two
# distinct same-topic evidence items at 0.56 and conflate them. ---

TOPIC_GOLD = [
    {
        "evidence_id": "t1",
        "text": "She mentioned the park again on 9 May 2023.",
    }
]


def test_topical_overlap_is_not_a_semantic_match():
    """Same topic, different event: must not be treated as the same evidence."""
    artifacts = [
        {"id": "a1", "content": "Caroline went to the park on 7 May 2023."}
    ]

    result = match_evidence(TOPIC_GOLD, artifacts)

    assert result["found"] == []
    assert result["missing"] == ["t1"]


def test_stopword_overlap_alone_is_not_a_match():
    gold = [{"evidence_id": "s1", "text": "I went to the store."}]
    artifacts = [{"id": "a1", "content": "She went to the bank."}]

    result = match_evidence(gold, artifacts)

    assert result["found"] == []


def test_distilled_fact_is_a_semantic_match():
    """A fact distilled from the gold turn is a faithful representation."""
    gold = [
        {
            "evidence_id": "d1",
            "text": "Caroline went to the LGBTQ support group yesterday and talked for an hour.",
        }
    ]
    artifacts = [{"id": "a1", "content": "Caroline attended the LGBTQ support group."}]

    result = match_evidence(gold, artifacts)

    assert result["found"] == ["d1"]
    assert result["match_types"]["d1"] == "semantic"


def test_gold_contained_in_longer_artifact_is_a_match():
    """The raw-storage direction: gold sentence inside a longer turn."""
    gold = [{"evidence_id": "r1", "text": "Caroline went to the park."}]
    artifacts = [
        {"id": "a1", "content": "In the morning Caroline went to the park, then she visited the library."}
    ]

    result = match_evidence(gold, artifacts)

    assert result["found"] == ["r1"]
    assert result["match_scores"]["r1"] >= SEMANTIC_CONTENT_THRESHOLD
