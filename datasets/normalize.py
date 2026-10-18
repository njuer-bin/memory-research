"""Normalize heterogeneous memory benchmarks into one research record schema."""
from __future__ import annotations
from typing import Any, Iterable

def _first(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return default

def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)

def _evidence_items(raw: Any, example_id: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        return [{"evidence_id":f"{example_id}:e0","text":_as_text(raw)}]
    if isinstance(raw, dict):
        raw=[raw]
    result=[]
    for idx,item in enumerate(raw if isinstance(raw,list) else []):
        if isinstance(item,dict):
            evidence_id=_first(item,"evidence_id","id","memory_id","source_id",default=f"{example_id}:e{idx}")
            result.append({"evidence_id":str(evidence_id),"text":_first(item,"text","content","evidence","memory"),"source_id":_first(item,"source_id","doc_id","memory_id"),"turn_id":_first(item,"turn_id","turn","id"),"timestamp":_first(item,"timestamp","date","time")})
        else:
            result.append({"evidence_id":f"{example_id}:e{idx}","text":_as_text(item)})
    return result

def normalize_record(raw: dict[str, Any], *, dataset: str, family: str, index: int, task_type: Iterable[str] = ()) -> dict[str, Any]:
    question_id=_first(raw,"question_id","qid","id",default=str(index))
    conversation_id=_first(raw,"conversation_id","conversationId","dialogue_id","session_id")
    question=_first(raw,"question","query","input","prompt",default="")
    answer=_first(raw,"answer","gold_answer","reference_answer","target")
    evidence_raw=_first(raw,"gold_evidence","evidence","supporting_evidence","relevant_evidence","gold_memory","oracle_evidence")
    example_id=f"{dataset}:{question_id}"
    return {
        "dataset":dataset,"family":family,"split":_first(raw,"split","subset"),"example_id":example_id,
        "conversation_id":str(conversation_id) if conversation_id is not None else None,
        "question_id":str(question_id),"question":_as_text(question),
        "answer":_as_text(answer) if answer is not None else None,
        "gold_evidence":_evidence_items(evidence_raw,example_id),
        "conversation":_first(raw,"conversation","messages","dialogue",default=[]),
        "task_type":list(task_type) or list(_first(raw,"task_type","categories",default=[])),
        "metadata":{"raw":raw,"normalization":"generic_v1"}
    }
