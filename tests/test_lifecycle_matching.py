"""Tests for lifecycle evidence matching (audit finding C3).

Provenance matches (same source turn/session) must also validate that the
artifact content still recognizably preserves the gold text; otherwise a
stage could report perfect recall over corrupted or replaced content.
"""
from tracing.lifecycle import PROVENANCE_CONTENT_THRESHOLD, match_evidence


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
