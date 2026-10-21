"""Study 3A: external-system deployment -- run the frozen protocol on Mem0.

External-validity arm (reviewer 1's top concern). The lifecycle-attribution
protocol (docs/PROTOCOL.md) is system-agnostic: it only needs, per lifecycle
stage, the list of artifact texts a memory system produced, and it matches
them against gold evidence itself (tracing/lifecycle.match_evidence, three
tiers: exact / provenance+content / semantic containment). This runner
therefore swaps the memory engine for Mem0 (OSS, v2.x) and reuses every
protocol component unchanged:

- same questions: longmemeval_oracle_n50 (the Study 2B protocol subset)
- same answer + judge model: qwen2.5:7b via local Ollama (env defaults)
- same three conditions: real memory (Mem0), oracle memory (gold injected
  as formed memories, then Mem0's own retrieval), oracle context (gold in
  the prompt)
- same E0 gate, same semantic judge, same attribute_failure labelling

Mem0 is adapted to the trace schema as follows (documented in
results/study3a_mem0_n50/summary.json and the paper):

- formation:  memories extracted by Mem0's LLM-driven ADD pipeline
- storage:    Mem0 OSS persists extracted memories only (no raw-message
              store); the stage is measured on the same set and flagged
              in details rather than faking a raw store
- evolution:  post-consolidation memory set; superseded counts come from
              Mem0's UPDATE/DELETE add-events
- retrieval:  Mem0 vector search top-k (threshold 0 = pure top-k, matching
              the oracle harness; Mem0's default 0.1 threshold is disabled
              so retrieval is measured, not policy-filtered)
- rerank:     identity of retrieval -- Mem0 OSS has no reranker; flagged
              in details ("rerank_is_identity": True)
- context:    the retrieved memories handed to the answer prompt

Provenance is deliberately NOT fabricated for Mem0 artifacts: Mem0
memories carry no benchmark turn/session lineage, so evidence matching
uses the exact/semantic tiers only. Faking lineage would let the
provenance tier inflate formation recall (audit finding C3).

Mem0 is configured fully locally: qwen2.5:7b (LLM extraction),
bge-m3 (1024-dim embeddings), Chroma persistent store per record.

Ingestion cadence (integration decision, recorded in every row): both
local models (qwen2.5:7b and gemma3:12b) collapse from Mem0's ADD
extraction schema when a single add() call carries a long message batch
(>=6 messages: qwen returns a conversational ``{"response": ...}``
object, gemma3 a conversation-analysis object -- both parse to zero
facts). Mem0's schema-following is reliable for small batches, so the
conversation is streamed to Mem0 in chunks of MEM0_ADD_CHUNK=4 messages
per add() call. Mem0's own dedup/update logic still operates across
calls (existing memories are retrieved and passed as context to every
extraction), so this is a cadence choice, not a semantic change. The
extraction model stays qwen2.5:7b (same as the answer/judge model in
the Study 2B protocol subset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("MEM0_TELEMETRY", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.loader import iter_normalized  # noqa: E402
from evaluation.answer_generation import generate_and_score  # noqa: E402
from evaluation.answer_semantic_eval import evaluate_answer  # noqa: E402
from evaluation.attribution import attribute_failure, control_failure_label  # noqa: E402
from evaluation.gate import evaluate_evidence_gate  # noqa: E402
from evaluation.lifecycle_metrics import summarize_traces  # noqa: E402
from tracing.dataset_trace import record_to_trace  # noqa: E402
from tracing.real_stages import stage_dict  # noqa: E402
from experiments.run_oracle_experiments import (  # noqa: E402
    _answer_summary,
    _commit_hash,
    _context_only_summary,
    _first_answer_failure_distribution,
    _gold_artifacts,
    _judge_evidence,
    _msg,
    _oracle_context_trace,
    _score_mode_answer,
    _semantic_verdict,
    _sessions,
)

OLLAMA_URL = os.getenv("MEM0_OLLAMA_BASE_URL", "http://localhost:11434")
MEM0_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", "qwen2.5:7b")
MEM0_EMBED_MODEL = os.getenv("MEM0_EMBED_MODEL", "bge-m3")
MEM0_EMBED_DIMS = int(os.getenv("MEM0_EMBED_DIMS", "1024"))
TOP_K = int(os.getenv("MEM0_TOP_K", "10"))
ADD_CHUNK = int(os.getenv("MEM0_ADD_CHUNK", "4"))

OUT = ROOT / "results" / "study3a_mem0_n50"
CHROMA_DIR = ROOT / "data" / "mem0_chroma"


def log(message: str) -> None:
    print(message, flush=True)


class Mem0System:
    """One isolated Mem0 instance (own Chroma collection + user id)."""

    def __init__(self, label: str) -> None:
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig

        self.label = label
        self.uid = f"{label}_{uuid.uuid4().hex}"
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "mem0_history").mkdir(parents=True, exist_ok=True)
        config = MemoryConfig(**{
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": MEM0_LLM_MODEL,
                    "ollama_base_url": OLLAMA_URL,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": MEM0_EMBED_MODEL,
                    "ollama_base_url": OLLAMA_URL,
                    "embedding_dims": MEM0_EMBED_DIMS,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"mem0_{self.uid}",
                    "path": str(CHROMA_DIR),
                },
            },
            "history_db_path": str(ROOT / "data" / "mem0_history" / f"history_{self.uid}.db"),
        })
        self.m = Memory(config)
        self.add_calls = 0
        self.messages_ingested = 0
        self.events: list[str] = []

    # -- ingestion ---------------------------------------------------------
    def ingest(self, sessions: list[dict[str, Any]]) -> None:
        """Stream the conversation to Mem0 in small add() chunks.

        Mem0 OSS add() accepts role/content only; timestamps are not part
        of its vanilla extraction path. See module docstring for why the
        cadence is chunked rather than session-per-call.
        """
        messages: list[dict[str, str]] = []
        for session in sessions:
            for raw in session["messages"]:
                role, text, _ts = _msg(raw)
                if text:
                    messages.append({"role": role, "content": text})
        self.messages_ingested = len(messages)
        for start in range(0, len(messages), ADD_CHUNK):
            chunk = messages[start:start + ADD_CHUNK]
            self.add_calls += 1
            results = self.m.add(chunk, user_id=self.uid, infer=True)
            events = results.get("results") if isinstance(results, dict) else results
            for event in events or []:
                if isinstance(event, dict):
                    self.events.append(str(event.get("event") or "UNKNOWN"))

    # -- reads -------------------------------------------------------------
    def stored(self) -> list[dict[str, Any]]:
        res = self.m.get_all(filters={"user_id": self.uid}, top_k=10000)
        return [self._artifact(r) for r in (res.get("results") or [])]

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        res = self.m.search(
            query, filters={"user_id": self.uid}, top_k=top_k, threshold=0.0
        )
        return [self._artifact(r) for r in (res.get("results") or [])]

    @staticmethod
    def _artifact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id") or ""),
            "content": str(row.get("memory") or ""),
            "memory_type": "mem0_memory",
            "score": row.get("score"),
        }

    # -- stats -------------------------------------------------------------
    def extraction_stats(self) -> dict[str, Any]:
        added = sum(1 for e in self.events if e == "ADD")
        updated = sum(1 for e in self.events if e == "UPDATE")
        deleted = sum(1 for e in self.events if e == "DELETE")
        return {
            "llm_calls": self.add_calls,
            "messages_ingested": self.messages_ingested,
            "add_chunk": ADD_CHUNK,
            "facts": added,
            "memories_updated": updated,
            "memories_deleted": deleted,
            "parse_failures": 0,
            "fallback_messages": 0,
            "extracted_messages": 0,
            "empty_messages": 0,
        }


def mem0_oracle_memory(gold: list[dict[str, Any]]) -> Mem0System:
    """Oracle Memory on Mem0: gold evidence inserted as already-formed
    memories (bypassing extraction), then Mem0's own retrieval runs --
    mirroring _oracle_memory_engine for the in-house pipeline."""
    system = Mem0System("mem0_oracle")
    for evidence in gold:
        text = str(evidence.get("text") or evidence.get("content") or "")
        if not text:
            continue
        vector = system.m.embedding_model.embed(text, "add")
        system.m.vector_store.insert(
            vectors=[vector],
            payloads=[{
                "data": text,
                "user_id": system.uid,
                "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
            }],
            ids=[str(uuid.uuid4())],
        )
    return system


def mem0_real_stages(
    gold: list[dict[str, Any]], system: Mem0System, search_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Six lifecycle stages for the Mem0 real-memory condition."""
    stored = system.stored()
    stats = system.extraction_stats()
    stages = {
        "formation": stage_dict(gold, stored, "mem0_formation"),
        "storage": stage_dict(gold, stored, "mem0_storage_extracted_only"),
        "evolution": stage_dict(gold, stored, "mem0_evolution_post_consolidation"),
        "retrieval": stage_dict(gold, search_results, "mem0_retrieval_topk"),
        "rerank": stage_dict(gold, search_results, "mem0_rerank_identity"),
        "context": stage_dict(gold, search_results, "mem0_context"),
    }
    stages["storage"]["details"]["note"] = (
        "mem0 OSS persists extracted memories only; no separate raw-message store"
    )
    stages["evolution"]["details"]["superseded_facts"] = (
        stats["memories_updated"] + stats["memories_deleted"]
    )
    stages["evolution"]["details"]["measured"] = True
    stages["rerank"]["details"]["rerank_is_identity"] = True
    stages["retrieval"]["details"]["top_k"] = TOP_K
    stages["retrieval"]["details"]["threshold"] = 0.0
    stages["formation"]["details"]["semantic_memory_counts"] = {
        "total": len(stored),
        "by_type": {"mem0_memory": len(stored)},
    }
    return stages


def mem0_oracle_stages(
    gold: list[dict[str, Any]],
    oracle_items: list[dict[str, Any]],
    system: Mem0System,
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Oracle Memory on Mem0: formation/storage/evolution are gold by
    construction (measured to validate the injection); retrieval/rerank/
    context go through Mem0's search."""
    gold_artifacts = [
        {"id": str(e.get("evidence_id") or ""), "content": str(e.get("text") or ""),
         "memory_type": "oracle", "score": None}
        for e in gold
        if (e.get("text") or e.get("content"))
    ]
    stages = {
        "formation": stage_dict(gold, gold_artifacts, "mem0_oracle_memory_formation"),
        "storage": stage_dict(gold, gold_artifacts, "mem0_oracle_memory_storage"),
        "evolution": stage_dict(gold, gold_artifacts, "mem0_oracle_memory_evolution"),
        "retrieval": stage_dict(gold, search_results, "mem0_oracle_memory_retrieval_topk"),
        "rerank": stage_dict(gold, search_results, "mem0_oracle_memory_rerank_identity"),
        "context": stage_dict(gold, search_results, "mem0_oracle_memory_context"),
    }
    stages["rerank"]["details"]["rerank_is_identity"] = True
    stages["retrieval"]["details"]["top_k"] = TOP_K
    stages["retrieval"]["details"]["threshold"] = 0.0
    return stages


def run_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    trace = record_to_trace(record)
    gold = [e.__dict__ for e in trace.gold_evidence]
    oracle_items = _gold_artifacts(record)
    judge_gold = _judge_evidence(gold)

    # --- real memory: Mem0 -------------------------------------------------
    log("    [mem0] ingest")
    t0 = time.perf_counter()
    system = Mem0System("mem0_real")
    system.ingest(_sessions(record))
    stored = system.stored()
    log(f"    [mem0] ingest done in {time.perf_counter() - t0:.2f}s; "
        f"messages={system.messages_ingested} add_calls={system.add_calls} "
        f"memories={len(stored)} events={sorted(set(system.events))}")
    log("    [mem0] retrieval")
    t0 = time.perf_counter()
    search_results = system.search(record["question"])
    log(f"    [mem0] retrieval done in {time.perf_counter() - t0:.2f}s; "
        f"results={len(search_results)}")

    real_stages = mem0_real_stages(gold, system, search_results)

    # --- oracle memory: gold injected as formed memories -------------------
    log("    [mem0-oracle] inject gold + retrieve")
    oracle_system = mem0_oracle_memory(gold)
    oracle_search = oracle_system.search(record["question"])
    oracle_stages = mem0_oracle_stages(gold, oracle_items, oracle_system, oracle_search)

    # --- oracle context: gold straight into the prompt ---------------------
    oracle_context_stages = _oracle_context_trace(gold, oracle_items)

    # --- answers ------------------------------------------------------------
    real_answer = _score_mode_answer(record["question"], record.get("answer", ""), search_results)
    oracle_memory_answer = _score_mode_answer(
        record["question"], record.get("answer", ""), oracle_search
    )
    oracle_context_answer = _score_mode_answer(
        record["question"], record.get("answer", ""), oracle_items
    )
    for ans in (real_answer, oracle_memory_answer, oracle_context_answer):
        ans.update(_semantic_verdict(record["question"], ans.get("answer") or "", judge_gold))

    gate = evaluate_evidence_gate(record["question"], record.get("answer", ""), judge_gold)

    real_mode = {**real_stages, "answer": {**real_answer, "mode": "real_memory"},
                 "evidence_sufficiency": gate}
    real_mode["answer_failure_type"] = attribute_failure(
        {"gold_evidence": gold, **real_mode},
        semantic_correct=real_answer.get("semantic_correct"),
        gate=gate,
        gold_evidence=gold,
    )

    oracle_memory = {**oracle_stages,
                     "answer": {**oracle_memory_answer, "mode": "oracle_memory"},
                     "evidence_sufficiency": gate}
    oracle_memory["answer_failure_type"] = attribute_failure(
        {"gold_evidence": gold, **oracle_memory},
        semantic_correct=oracle_memory_answer.get("semantic_correct"),
        gate=gate,
        gold_evidence=gold,
    )

    oracle_context = {**oracle_context_stages,
                      "answer": {**oracle_context_answer, "mode": "oracle_context"},
                      "evidence_sufficiency": gate}
    oracle_context["answer_failure_type"] = control_failure_label(
        "oracle_context", semantically_wrong=oracle_context_answer.get("semantic_correct") is False
    )

    return {
        "question_id": trace.question_id, "question": trace.question,
        "gold_answer": record.get("answer"),
        "modes": {"real_memory": real_mode, "oracle_memory": oracle_memory,
                  "oracle_context": oracle_context},
        "gold_evidence_count": len(gold), "gold_evidence": gold,
        "extraction": {"backend": "mem0", "stats": system.extraction_stats()},
        "memory_system": {
            "name": "mem0",
            "version": _mem0_version(),
            "llm": MEM0_LLM_MODEL,
            "embedder": f"{MEM0_EMBED_MODEL} ({MEM0_EMBED_DIMS}-dim)",
            "vector_store": "chroma (persistent)",
            "ingest_cadence": f"{ADD_CHUNK} messages per add() call (chunked streaming)",
        },
        "code_commit": _commit_hash(),
    }


def _mem0_version() -> str:
    try:
        import mem0
        return str(getattr(mem0, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _mode_trace(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return {"gold_evidence": row.get("gold_evidence", []), **row["modes"][mode]}


def _build_summary(dataset: str, limit: int, rows: list[dict[str, Any]], errors: int) -> dict[str, Any]:
    gate_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("modes", {}).get("real_memory", {}).get("evidence_sufficiency", {}).get("status") or "unknown")
        gate_counts[status] = gate_counts.get(status, 0) + 1
    extraction: dict[str, Any] = {"backend": "mem0", "totals": {}}
    keys = ("llm_calls", "messages_ingested", "facts", "memories_updated", "memories_deleted",
            "parse_failures", "fallback_messages", "extracted_messages", "empty_messages")
    for k in keys:
        extraction["totals"][k] = sum(int((r.get("extraction", {}).get("stats") or {}).get(k, 0) or 0) for r in rows)
    extraction["ingest_cadence"] = f"{ADD_CHUNK} messages per add() call (chunked streaming)"
    return {
        "dataset": dataset, "limit": limit, "records": len(rows), "errors": errors,
        "extraction": extraction,
        "memory_system": "mem0 (OSS) via local Ollama (qwen2.5:7b extraction, bge-m3 embeddings)",
        "real_memory": summarize_traces([_mode_trace(r, "real_memory") for r in rows]) if rows else {},
        "oracle_memory": summarize_traces([_mode_trace(r, "oracle_memory") for r in rows]) if rows else {},
        "oracle_context": _context_only_summary(rows),
        "evidence_sufficiency": {"status_distribution": gate_counts, "gate": "E0 before F1-F6; uncertain/error cases are EVAL"},
        "answer_scoring": {
            "method": "lexical (EM/token-F1, reported only) + semantic judge (attribution input)",
            "lexical_note": "Lexical correctness is NOT an attribution input.",
            "real_memory": _answer_summary(rows, "real_memory"),
            "oracle_memory": _answer_summary(rows, "oracle_memory"),
            "oracle_context": _answer_summary(rows, "oracle_context"),
            "first_answer_failure_distribution": _first_answer_failure_distribution(rows),
        },
        "models": {
            "answer_model": os.getenv("ANSWER_MODEL", "qwen2.5:7b"),
            "semantic_judge_model": os.getenv("SEMANTIC_JUDGE_MODEL", os.getenv("ANSWER_MODEL", "qwen2.5:7b")),
            "extractor_backend": "mem0 (external system)",
            "mem0_llm": MEM0_LLM_MODEL,
            "mem0_embedder": f"{MEM0_EMBED_MODEL} ({MEM0_EMBED_DIMS}-dim)",
            "code_commit": _commit_hash(),
        },
        "note": (
            "External-validity arm: the frozen lifecycle protocol run on Mem0. "
            "Same questions (longmemeval_oracle_n50) and same answer/judge model as the "
            "Study 2B protocol subset. Mem0 artifacts carry no benchmark provenance, so "
            "evidence matching uses the exact/semantic tiers only; rerank is the identity "
            "of retrieval (Mem0 OSS has no reranker) and storage holds extracted memories "
            "only. Both caveats are recorded per-stage in details."
        ),
    }


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prior_results(jsonl: Path, progress_path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    rows: dict[str, dict[str, Any]] = {}
    error_ids: set[str] = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("question_id"))] = row
    if progress_path.exists():
        try:
            error_ids = set(progress_path.read_text(encoding="utf-8") and json.loads(progress_path.read_text(encoding="utf-8")).get("error_ids", []))
        except Exception:
            error_ids = set()
    return rows, error_ids


def run(dataset: str, limit: int, out_dir: Path | None = None,
        ids: str | None = None, append: bool = False) -> dict[str, Any]:
    out = out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "comparison.jsonl"
    progress_path = out / "progress.json"
    summary_path = out / "summary.json"

    selected = {s.strip() for s in (ids or "").split(",") if s.strip()} or None
    if selected is not None and not append:
        raise ValueError("--ids requires --append (repair merges into an existing result directory)")

    if append:
        rows_by_id, error_ids = _load_prior_results(jsonl, progress_path)
    else:
        jsonl.write_text("", encoding="utf-8")
        rows_by_id, error_ids = {}, set()
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    started = time.perf_counter()
    log("=" * 72)
    mode = "repair" if selected else ("append" if append else "fresh")
    log(f"Mem0 external study started: dataset={dataset}, limit={limit}, mode={mode}"
        + (f", ids={sorted(selected)}" if selected else ""))
    log(f"Output: {out}")
    log(f"Mem0: llm={MEM0_LLM_MODEL} embedder={MEM0_EMBED_MODEL}({MEM0_EMBED_DIMS}-dim) "
        f"store=chroma@{CHROMA_DIR}")
    log("=" * 72)
    _write_progress(progress_path, {
        "status": "running", "dataset": dataset, "limit": limit,
        "completed": len(rows_by_id), "errors": len(error_ids),
        "error_ids": sorted(error_ids), "elapsed_seconds": 0.0,
    })

    for idx, record in enumerate(iter_normalized(dataset, limit=limit)):
        qid = str(record.get("question_id"))
        if selected is not None and qid not in selected:
            continue
        number = idx + 1
        log(f"\n[{number}/{limit}] START question_id={qid}")
        t0 = time.perf_counter()
        try:
            row = run_record(record, idx)
            rows_by_id[qid] = row
            error_ids.discard(qid)
            _write_jsonl(jsonl, row)
            failure = row["modes"]["real_memory"].get("answer_failure_type")
            log(f"[{number}/{limit}] DONE {time.perf_counter() - t0:.2f}s failure={failure}")
        except Exception as exc:
            error_ids.add(qid)
            log(f"[{number}/{limit}] ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
        _write_progress(progress_path, {
            "status": "running", "dataset": dataset, "limit": limit,
            "completed": len(rows_by_id), "errors": len(error_ids),
            "error_ids": sorted(error_ids),
            "elapsed_seconds": time.perf_counter() - started,
        })

    merged = list(rows_by_id.values())
    with jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _build_summary(dataset, limit, merged, len(error_ids))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, {
        "status": "completed", "dataset": dataset, "limit": limit,
        "completed": len(merged), "errors": len(error_ids),
        "error_ids": sorted(error_ids), "elapsed_seconds": time.perf_counter() - started,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="longmemeval_oracle_n50")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default=None,
                        help="Result directory (default: results/study3a_mem0_n50)")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated question_ids to (re)run; requires --append")
    parser.add_argument("--append", action="store_true",
                        help="Merge into an existing result directory instead of starting fresh")
    args = parser.parse_args()
    run(args.dataset, args.limit, Path(args.out) if args.out else None,
        ids=args.ids, append=args.append)


if __name__ == "__main__":
    main()
