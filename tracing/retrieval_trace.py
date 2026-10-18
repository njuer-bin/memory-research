"""Retrieval and reranking evidence attribution utilities."""
from typing import Any, Dict, Iterable

def _ids(items: Iterable[Dict[str, Any]], id_key: str):
    return [str(x[id_key]) for x in items if id_key in x]

def trace_retrieval(gold_evidence: Iterable[str], retrieved: Iterable[Dict[str, Any]],
                    id_key: str = "evidence_id") -> Dict[str, Any]:
    gold, ids = set(gold_evidence), set(_ids(retrieved, id_key))
    found, missing = sorted(gold & ids), sorted(gold - ids)
    return {"found": found, "missing": missing,
            "recall": len(found) / len(gold) if gold else 1.0,
            "retrieved_count": len(ids)}

def trace_rerank(gold_evidence: Iterable[str], before: Iterable[Dict[str, Any]],
                 after: Iterable[Dict[str, Any]], id_key: str = "evidence_id") -> Dict[str, Any]:
    gold, before_ids, after_ids = set(gold_evidence), set(_ids(before, id_key)), set(_ids(after, id_key))
    return {"found": sorted(gold & after_ids),
            "dropped": sorted((gold & before_ids) - after_ids),
            "missing": sorted(gold - after_ids),
            "before_count": len(before_ids), "after_count": len(after_ids)}
