"""Vector index must refuse mixed-dimension vectors instead of zero-padding."""
from __future__ import annotations

import pytest

from memory_engine.vector_index import MemoryVectorIndex


class _Store:
    def __init__(self, rows):
        self._rows = rows

    def embeddings(self, user_id):
        return self._rows


def _vec(dim: int, first: float = 1.0):
    return [first] + [0.0] * (dim - 1)


def test_mixed_dimensions_in_store_raise():
    store = _Store([("a", _vec(1024)), ("b", _vec(384))])
    index = MemoryVectorIndex(store)
    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        index.search("u", _vec(1024))


def test_query_dimension_mismatch_raises():
    store = _Store([("a", _vec(1024))])
    index = MemoryVectorIndex(store)
    with pytest.raises(ValueError, match="does not match index dimension"):
        index.search("u", _vec(384))


def test_add_with_inconsistent_dimension_raises():
    index = MemoryVectorIndex(_Store([]))
    index.add("u", "a", _vec(1024))
    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        index.add("u", "b", _vec(384))


def test_add_then_search_same_dimension():
    index = MemoryVectorIndex(_Store([]))
    index.add("u", "a", _vec(1024))
    index.add("u", "b", _vec(1024, first=0.0))
    hits = index.search("u", _vec(1024), top_k=5)
    assert [mid for mid, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(1.0)


def test_empty_index_returns_no_hits():
    index = MemoryVectorIndex(_Store([]))
    assert index.search("u", _vec(1024)) == []
