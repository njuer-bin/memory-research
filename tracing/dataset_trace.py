"""Convert normalized benchmark records into trace-ready evidence objects."""
from __future__ import annotations

from typing import Any, Iterable

from tracing.memory_trace import EvidenceItem, MemoryTrace


def evidence_items(record: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for item in record.get("gold_evidence", []) or []:
        if not isinstance(item, dict) or not item.get("evidence_id"):
            continue
        items.append(
            EvidenceItem(
                evidence_id=str(item["evidence_id"]),
                text=str(item.get("text") or ""),
                source=item.get("source_id") or item.get("source"),
                turn_id=item.get("turn_id") or item.get("turn"),
                timestamp=item.get("timestamp"),
                metadata={
                    k: v
                    for k, v in item.items()
                    if k not in {"evidence_id", "text", "source_id", "source", "turn_id", "turn", "timestamp"}
                },
            )
        )
    return items


def record_to_trace(record: dict[str, Any]) -> MemoryTrace:
    """Create an empty lifecycle trace populated with benchmark gold evidence."""
    return MemoryTrace(
        question_id=str(record.get("question_id") or record.get("example_id") or ""),
        question=str(record.get("question") or ""),
        gold_evidence=evidence_items(record),
    )


def records_to_traces(records: Iterable[dict[str, Any]]) -> list[MemoryTrace]:
    return [record_to_trace(record) for record in records]
