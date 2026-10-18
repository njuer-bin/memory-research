"""Dataset registry and lazy loaders for the unified memory benchmark layer."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterator

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"datasets"/"manifest.json"

def load_manifest()->dict[str,Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def dataset_info(dataset_id:str)->dict[str,Any]:
    for item in load_manifest()["datasets"]:
        if item["id"]==dataset_id:
            return item
    raise KeyError(f"Unknown dataset: {dataset_id}")

def _iter_json(path:Path)->Iterator[dict[str,Any]]:
    with path.open("r",encoding="utf-8") as fh:
        obj=json.load(fh)
    if isinstance(obj,list):
        yield from (x for x in obj if isinstance(x,dict))
    elif isinstance(obj,dict):
        for key in ("data","questions","examples","records","conversations"):
            value=obj.get(key)
            if isinstance(value,list):
                yield from (x for x in value if isinstance(x,dict))
                return
        yield obj

def _iter_jsonl(path:Path)->Iterator[dict[str,Any]]:
    with path.open("r",encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if line:
                value=json.loads(line)
                if isinstance(value,dict):
                    yield value

def _iter_parquet(path:Path)->Iterator[dict[str,Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("BEAM parquet loading requires pyarrow. Install it with pip install pyarrow.") from exc
    for row in pq.read_table(path).to_pylist():
        if isinstance(row,dict):
            yield row

def _locomo_conversations()->dict[str,dict[str,Any]]:
    path=ROOT/"datasets/data/locomo_refined_dataset/public/conversations.jsonl"
    if not path.exists():
        return {}
    result={}
    for row in _iter_jsonl(path):
        cid=row.get("sample_id") or row.get("conversation_id") or row.get("id")
        if cid is not None:
            result[str(cid)]=row
    return result

def iter_raw(dataset_id:str,*,limit:int|None=None)->Iterator[dict[str,Any]]:
    info=dataset_info(dataset_id)
    source=ROOT/info["source"]
    if info["format"]=="json":
        iterator=_iter_json(source)
    elif info["format"]=="jsonl":
        iterator=_iter_jsonl(ROOT/"datasets/data/locomo_refined_dataset/public/questions.jsonl")
    elif info["format"]=="parquet":
        iterator=_iter_parquet(source)
    else:
        raise ValueError(f"Unsupported dataset format: {info['format']}")
    for idx,row in enumerate(iterator):
        if limit is not None and idx>=limit:
            break
        yield row

def iter_normalized(dataset_id:str,*,limit:int|None=None)->Iterator[dict[str,Any]]:
    from .adapters import normalize_locomo_refined, normalize_longmemeval
    from .normalize import normalize_record
    info=dataset_info(dataset_id)
    conversations=_locomo_conversations() if dataset_id=="locomo_refined_public" else {}
    for idx,raw in enumerate(iter_raw(dataset_id,limit=limit)):
        if dataset_id in {"longmemeval_s_sample10","longmemeval_oracle"}:
            yield normalize_longmemeval(raw,dataset=dataset_id,index=idx)
        elif dataset_id=="locomo_refined_public":
            cid = str(raw.get("sample_id") or raw.get("conversation_id") or "")
            conversation = conversations.get(cid)
            if conversation is None and raw.get("conversation_idx") is not None:
                try:
                    conversation = list(conversations.values())[int(raw["conversation_idx"])]
                except (ValueError, IndexError):
                    conversation = None
            record=normalize_locomo_refined(raw,index=idx,conversation=conversation)
            yield record
        else:
            yield normalize_record(raw,dataset=dataset_id,family=info["family"],index=idx,task_type=info.get("capabilities",[]))
