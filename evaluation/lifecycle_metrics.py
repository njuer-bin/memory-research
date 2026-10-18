"""Metrics for stage-wise memory failure attribution."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


STAGES = ("formation", "storage", "evolution", "retrieval", "rerank", "context")


def _set(value: Any) -> set[str]:
    return {str(x) for x in (value or [])}


def stage_recall(trace: dict[str, Any], stage: str) -> float:
    gold = {str(x.get("evidence_id")) for x in trace.get("gold_evidence", []) if x.get("evidence_id")}
    found = _set((trace.get(stage) or {}).get("found"))
    return len(gold & found) / len(gold) if gold else 1.0


def first_loss_stage(trace: dict[str, Any]) -> str | None:
    gold = {str(x.get("evidence_id")) for x in trace.get("gold_evidence", []) if x.get("evidence_id")}
    if not gold:
        return None
    for stage in STAGES:
        if not gold.issubset(_set((trace.get(stage) or {}).get("found"))):
            return stage
    return None


def summarize_traces(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(traces)
    recalls = {stage: sum(stage_recall(t, stage) for t in rows) / len(rows) if rows else 0.0 for stage in STAGES}
    losses = Counter(first_loss_stage(t) or "none" for t in rows)
    return {
        "count": len(rows),
        "stage_recall": recalls,
        "first_loss_distribution": dict(losses),
    }
