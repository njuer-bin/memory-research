"""Formation-stage tracing: conversation evidence -> extracted memory."""
from typing import Any, Dict, Iterable

def trace_formation(gold_evidence: Iterable[str], extracted_memory: Iterable[Dict[str, Any]],
                    id_key: str = "evidence_id") -> Dict[str, Any]:
    gold = set(gold_evidence)
    extracted = {str(item[id_key]) for item in extracted_memory if id_key in item}
    found, missing = sorted(gold & extracted), sorted(gold - extracted)
    return {"found": found, "missing": missing,
            "recall": len(found) / len(gold) if gold else 1.0,
            "extracted_count": len(extracted)}
