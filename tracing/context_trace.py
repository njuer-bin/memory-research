"""Context-construction tracing: retrieved evidence -> final LLM context."""
from typing import Any, Dict, Iterable

def trace_context(gold_evidence: Iterable[str], context_items: Iterable[Dict[str, Any]],
                  id_key: str = "evidence_id", token_budget: int = 0) -> Dict[str, Any]:
    gold = set(gold_evidence)
    ids = {str(x[id_key]) for x in context_items if id_key in x}
    found, missing = sorted(gold & ids), sorted(gold - ids)
    return {"found": found, "missing": missing,
            "coverage": len(found) / len(gold) if gold else 1.0,
            "token_budget": token_budget, "context_item_count": len(ids)}
