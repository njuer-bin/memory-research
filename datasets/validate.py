"""Lightweight dataset inventory and schema validation."""
from __future__ import annotations
from collections import Counter
from typing import Any
from .loader import dataset_info, iter_normalized, load_manifest

def inventory(dataset_id:str, limit:int|None=None)->dict[str,Any]:
    count=0
    records_with_evidence=0
    evidence_count=0
    task_counts=Counter()
    missing_question=0
    missing_answer=0
    for row in iter_normalized(dataset_id,limit=limit):
        count+=1
        n_evidence=len(row.get("gold_evidence",[]))
        evidence_count+=n_evidence
        records_with_evidence += int(n_evidence > 0)
        task_counts.update(row.get("task_type",[]))
        if not row.get("question"):
            missing_question+=1
        if row.get("answer") in (None,"",[]):
            missing_answer+=1
    return {
        "dataset":dataset_id,
        "family":dataset_info(dataset_id)["family"],
        "records":count,
        "records_with_evidence":records_with_evidence,
        "gold_evidence_items":evidence_count,
        "task_type_counts":dict(task_counts),
        "missing_question":missing_question,
        "missing_answer":missing_answer,
    }

def validate_manifest()->list[str]:
    errors=[]
    for item in load_manifest()["datasets"]:
        if not item.get("id") or not item.get("source") or not item.get("format"):
            errors.append(f"invalid manifest entry: {item}")
    return errors
