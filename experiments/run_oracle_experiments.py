"""Run evidence-boundary control experiments with an explicit E0 gate."""
from __future__ import annotations

import argparse
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from datasets.loader import iter_normalized
from evaluation.answer_generation import generate_and_score
from evaluation.answer_semantic_eval import evaluate_answer
from evaluation.lifecycle_metrics import STAGES, summarize_traces
from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest
from tracing.dataset_trace import record_to_trace
from tracing.engine_trace import artifacts_from_store, retrieval_artifacts, stage_match
from tracing.failure_attribution import attribute_failure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "oracle_experiments"


def log(message: str) -> None:
    print(message, flush=True)


def _sessions(record: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = record.get("conversation") or []
    result: list[dict[str, Any]] = []
    for i, session in enumerate(conversation):
        if isinstance(session, dict):
            sid = str(session.get("session_id") or session.get("id") or f"session_{i}")
            messages = session.get("messages") or session.get("turns") or []
        else:
            sid, messages = f"session_{i}", session
        if isinstance(messages, dict):
            messages = [messages]
        result.append({"session_id": sid, "messages": list(messages or [])})
    return result


def _msg(m: Any) -> tuple[str, str, int | None]:
    if isinstance(m, dict):
        role = str(m.get("role") or m.get("speaker") or "user")
        text = str(m.get("content") or m.get("text") or m.get("message") or "")
        ts = m.get("timestamp")
        return role, text, ts if isinstance(ts, int) else None
    return "user", str(m), None


def _gold_artifacts(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(e["evidence_id"]),
            "content": str(e.get("text") or ""),
            "source_id": e.get("source_id"),
            "turn_id": e.get("turn_id"),
            "role": e.get("role", ""),
            "timestamp": e.get("timestamp"),
            "memory_type": "oracle",
        }
        for e in record.get("gold_evidence", [])
        if e.get("evidence_id") and e.get("text")
    ]


def _stage(gold: list[dict[str, Any]], artifacts: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    match = stage_match(gold, artifacts)
    return {
        "found": match["found"],
        "missing": match["missing"],
        "details": {
            "mode": mode,
            "observed_count": match["observed_count"],
            "match_types": match.get("match_types", {}),
            "match_scores": match.get("match_scores", {}),
            "matches": match["matches"],
        },
    }


def _real_engine(record: dict[str, Any], idx: int) -> tuple[MemoryEngine, str]:
    user_id = f"oracle_{idx}_{uuid.uuid4().hex}"
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / f"oracle_{idx}_{uuid.uuid4().hex}.db"

    log(f"    [engine] initializing MemoryEngine: {db_path.name}")
    t0 = time.perf_counter()
    engine = MemoryEngine(str(db_path))
    log(f"    [engine] initialized in {time.perf_counter() - t0:.2f}s")

    sessions = _sessions(record)
    log(f"    [ingest] {len(sessions)} sessions")
    t0 = time.perf_counter()
    message_count = 0
    for session in sessions:
        messages = []
        for raw in session["messages"]:
            role, text, ts = _msg(raw)
            if text:
                messages.append(AddMessage(role=role, content=text, timestamp=ts))
        if messages:
            message_count += len(messages)
            engine.add(AddRequest(
                request_id=f"{user_id}:{session['session_id']}",
                messages=messages,
                user_id=user_id,
                session_id=session["session_id"],
            ))
    log(f"    [ingest] {message_count} messages in {time.perf_counter() - t0:.2f}s")
    return engine, user_id


def _formation_diagnostics(gold: list[dict[str, Any]], stored: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    semantic = stored["facts"] + stored["relations"] + stored["events"] + stored["rules"] + stored["profiles"]
    raw = stored["raw"]
    diagnostics = []
    for evidence in gold:
        source_id = str(evidence.get("source_id") or evidence.get("source") or "")
        turn_id = str(evidence.get("turn_id") or evidence.get("turn") or "")
        raw_matches = [
            item for item in raw
            if (turn_id and str(item.get("source_turn_id") or item.get("turn_id") or "") == turn_id)
            or (source_id and str(item.get("source_id") or item.get("session_id") or "") == source_id)
        ]
        turn_semantic = [
            item for item in semantic
            if turn_id and str(item.get("source_turn_id") or item.get("turn_id") or "") == turn_id
        ]
        session_semantic = [
            item for item in semantic
            if source_id and str(item.get("source_session_id") or item.get("source_id") or item.get("session_id") or "") == source_id
        ]
        if not raw_matches:
            classification = "F1d_UNKNOWN"
        elif turn_semantic:
            classification = "FORMATION_OK"
        elif session_semantic:
            classification = "F1c_REPRESENTATION_MISMATCH"
        else:
            classification = "F1a_EXTRACTOR_MISS"
        diagnostics.append({
            "gold_evidence_id": evidence.get("evidence_id"),
            "source_session_id": source_id or None,
            "source_turn_id": turn_id or None,
            "source_message_found": bool(raw_matches),
            "source_message_count": len(raw_matches),
            "semantic_memory_count_same_turn": len(turn_semantic),
            "semantic_memory_count_same_session": len(session_semantic),
            "semantic_memory_ids_same_turn": [str(x.get("id")) for x in turn_semantic if x.get("id")],
            "semantic_memory_types_same_turn": [str(x.get("memory_type") or "") for x in turn_semantic],
            "semantic_memory_candidates_same_session": [
                {
                    "id": str(x.get("id")) if x.get("id") else None,
                    "memory_type": str(x.get("memory_type") or ""),
                    "source_turn_id": x.get("source_turn_id") or x.get("turn_id"),
                    "content": str(x.get("content") or "")[:240],
                }
                for x in session_semantic[:20]
            ],
            "classification": classification,
        })
    return diagnostics


def _real_trace_stages(trace: Any, gold: list[dict[str, Any]], engine: MemoryEngine, user_id: str, real: list[dict[str, Any]], search_trace: dict[str, Any]) -> None:
    stored = artifacts_from_store(engine, user_id)
    formation = stored["facts"] + stored["relations"] + stored["events"] + stored["rules"] + stored["profiles"]
    storage = stored["raw"] + formation
    evolution = formation
    trace.formation = type(trace.formation)(**_stage(gold, formation, "real_memory"))
    trace.formation.details["diagnostics"] = _formation_diagnostics(gold, stored)
    trace.formation.details["semantic_memory_counts"] = {
        "total": len(formation),
        "by_type": {
            "fact": len(stored["facts"]), "relation": len(stored["relations"]),
            "event": len(stored["events"]), "rule": len(stored["rules"]),
            "profile": len(stored["profiles"]),
        },
    }
    trace.storage = type(trace.storage)(**_stage(gold, storage, "real_memory"))
    trace.evolution = type(trace.evolution)(**_stage(gold, evolution, "real_memory"))
    retrieval = search_trace.get("retrieval_candidates") or real
    rerank = search_trace.get("reranked") or real
    context = search_trace.get("context") or real
    trace.retrieval = type(trace.retrieval)(**_stage(gold, retrieval, "real_memory_retrieval_candidates"))
    trace.rerank = type(trace.rerank)(**_stage(gold, rerank, "real_memory_reranked"))
    trace.context = type(trace.context)(**_stage(gold, context, "real_memory_context"))


def _oracle_memory_trace(gold: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    stages = {stage: _stage(gold, oracle, "oracle_memory") for stage in ("formation", "storage", "evolution", "rerank")}
    stages["retrieval"] = _stage(gold, oracle, "oracle_memory_bypass")
    stages["context"] = _stage(gold, oracle, "oracle_memory_context")
    return stages


def _oracle_context_trace(gold: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    result = {stage: _stage(gold, [], "oracle_context_not_measured") for stage in ("formation", "storage", "evolution", "retrieval", "rerank")}
    result["context"] = _stage(gold, oracle, "oracle_context")
    return result


def _evidence_sufficiency(question: str, benchmark_answer: str, gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the benchmark Gold Evidence against the benchmark answer.

    The gate must be independent of the Real Memory prediction. Using the model's
    failed answer here would make E0 depend on the very system failure we are trying
    to attribute. The benchmark reference answer is therefore the candidate used by
    the semantic evidence-coverage judge.
    """
    if not gold:
        return {
            "status": "insufficient", "evidence_sufficient": False,
            "confidence": "high", "reason": "No benchmark gold evidence is available.",
            "adjudication": "no_gold_evidence",
        }
    try:
        verdict = evaluate_answer(question, str(benchmark_answer or ""), gold)
    except Exception as exc:
        return {
            "status": "error", "evidence_sufficient": None, "confidence": "low",
            "reason": f"Evidence sufficiency judge failed: {type(exc).__name__}: {exc}",
            "adjudication": "semantic_judge_error",
        }
    sufficient = verdict.get("evidence_sufficient")
    confidence = str(verdict.get("confidence") or "medium").lower()
    if sufficient is False:
        status = "insufficient"
    elif sufficient is True and confidence != "low":
        status = "sufficient"
    else:
        status = "uncertain"
    return {
        "status": status,
        "evidence_sufficient": sufficient,
        "confidence": confidence,
        "reason": verdict.get("reason", ""),
        "adjudication": verdict.get("adjudication", "judge"),
        "adjudication_anchor": verdict.get("adjudication_anchor"),
        "model": verdict.get("model"),
    }


def _answer_failure_type(mode_data: dict[str, Any]) -> str | None:
    answer = mode_data.get("answer") or {}
    if answer.get("correct") is not False:
        return None
    sufficiency = mode_data.get("evidence_sufficiency") or {}
    if sufficiency.get("status") == "insufficient":
        return "E0_EVIDENCE_INSUFFICIENCY"
    if sufficiency and sufficiency.get("status") != "sufficient":
        return "EVAL_EVIDENCE_SUFFICIENCY"
    if mode_data.get("mode") == "oracle_context":
        return "RC_REASONING_CONTROL_FAILURE"
    gold = {str(x.get("evidence_id")) for x in mode_data.get("gold_evidence", []) if x.get("evidence_id")}
    if not gold:
        return "E0_EVIDENCE_INSUFFICIENCY"
    for stage, failure in (("formation", "F1_FORMATION"), ("storage", "F2_STORAGE"), ("evolution", "F3_EVOLUTION"), ("retrieval", "F4_RETRIEVAL"), ("rerank", "F4_RETRIEVAL"), ("context", "F5_CONTEXT")):
        found = {str(x) for x in (mode_data.get(stage) or {}).get("found", [])}
        if not gold.issubset(found):
            return failure
    return "F6_REASONING"


def _score_mode_answer(question: str, reference: str, context: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return generate_and_score(question, reference, context)
    except Exception as exc:
        return {"answer": None, "correct": None, "exact_match": None, "token_f1": None, "threshold": None, "model_error": f"{type(exc).__name__}: {exc}"}


def _answer_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    answers = [(r.get("modes", {}).get(mode, {}).get("answer") or {}) for r in rows]
    scored = [a for a in answers if a.get("correct") is not None]
    correct = sum(1 for a in scored if a.get("correct") is True)
    return {
        "count": len(answers), "scored": len(scored), "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "exact_match": sum(float(a.get("exact_match", 0.0)) for a in scored) / len(scored) if scored else None,
        "token_f1": sum(float(a.get("token_f1", 0.0)) for a in scored) / len(scored) if scored else None,
        "model_errors": sum(1 for a in answers if a.get("model_error")),
    }


def _first_answer_failure_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        failure = row.get("modes", {}).get("real_memory", {}).get("answer_failure_type")
        if failure:
            counts[failure] = counts.get(failure, 0) + 1
    return counts


def run_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    trace = record_to_trace(record)
    gold = [e.__dict__ for e in trace.gold_evidence]
    oracle = _gold_artifacts(record)

    log("    [real] building memory")
    engine, user_id = _real_engine(record, idx)
    log("    [real] reading lifecycle artifacts")
    t0 = time.perf_counter()
    log("    [real] retrieval")
    results = engine.search(SearchRequest(query=record["question"], user_id=user_id, top_k=10, multi_hop=True))
    real = retrieval_artifacts(results)
    log(f"    [real] retrieval done in {time.perf_counter() - t0:.2f}s; artifacts={len(real)}")

    _real_trace_stages(trace, gold, engine, user_id, real, engine.last_search_trace)
    search_trace = engine.last_search_trace
    real_context = search_trace.get("context") or real
    real_answer = _score_mode_answer(record["question"], record.get("answer", ""), real_context)
    oracle_memory_answer = _score_mode_answer(record["question"], record.get("answer", ""), oracle)
    oracle_context_answer = _score_mode_answer(record["question"], record.get("answer", ""), oracle)

    gate = _evidence_sufficiency(record["question"], record.get("answer", ""), gold)
    trace.answer = {**real_answer, "mode": "real_memory", "answer_scoring": "token_f1_plus_temporal_phrase_rules"}
    trace.failure_type = attribute_failure({**trace.to_dict(), "evidence_sufficiency": gate})

    real_mode = {
        "formation": trace.formation.to_dict(), "storage": trace.storage.to_dict(),
        "evolution": trace.evolution.to_dict(), "retrieval": trace.retrieval.to_dict(),
        "rerank": trace.rerank.to_dict(), "context": trace.context.to_dict(),
        "failure_type": trace.failure_type, "answer": real_answer,
        "evidence_sufficiency": gate,
    }
    oracle_memory = _oracle_memory_trace(gold, oracle)
    oracle_memory.update({
        "answer": {**oracle_memory_answer, "mode": "oracle_memory", "answer_scoring": "token_f1_plus_temporal_phrase_rules"},
        "evidence_sufficiency": gate,
    })
    oracle_context = _oracle_context_trace(gold, oracle)
    oracle_context.update({
        "answer": {**oracle_context_answer, "mode": "oracle_context", "answer_scoring": "token_f1_plus_temporal_phrase_rules"},
        "evidence_sufficiency": gate,
    })

    row = {
        "question_id": trace.question_id, "question": trace.question,
        "modes": {"real_memory": real_mode, "oracle_memory": oracle_memory, "oracle_context": oracle_context},
        "gold_evidence_count": len(gold), "gold_evidence": gold,
    }
    for mode in ("real_memory", "oracle_memory", "oracle_context"):
        row["modes"][mode]["answer_failure_type"] = _answer_failure_type({"gold_evidence": gold, "mode": mode, **row["modes"][mode]})
    return row


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mode_trace(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return {"gold_evidence": row.get("gold_evidence", []), **row["modes"][mode]}


def _context_only_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"count": 0, "stage_recall": {}, "first_loss_distribution": {"not_applicable": 0}}
    total = {stage: 0.0 for stage in STAGES}
    for row in rows:
        trace = _mode_trace(row, "oracle_context")
        gold = {str(x.get("evidence_id")) for x in trace["gold_evidence"] if x.get("evidence_id")}
        for stage in STAGES:
            found = {str(x) for x in (trace.get(stage) or {}).get("found", [])}
            total[stage] += len(gold & found) / len(gold) if gold else 1.0
    return {"count": count, "stage_recall": {stage: total[stage] / count for stage in STAGES}, "first_loss_distribution": {"not_applicable": count}}


def _build_summary(dataset: str, limit: int, rows: list[dict[str, Any]], errors: int) -> dict[str, Any]:
    gate_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("modes", {}).get("real_memory", {}).get("evidence_sufficiency", {}).get("status") or "unknown")
        gate_counts[status] = gate_counts.get(status, 0) + 1
    return {
        "dataset": dataset, "limit": limit, "records": len(rows), "errors": errors,
        "real_memory": summarize_traces([_mode_trace(r, "real_memory") for r in rows]) if rows else {},
        "oracle_memory": summarize_traces([_mode_trace(r, "oracle_memory") for r in rows]) if rows else {},
        "oracle_context": _context_only_summary(rows),
        "evidence_sufficiency": {"status_distribution": gate_counts, "gate": "E0 before F1-F6; uncertain/error cases are EVAL"},
        "answer_scoring": {
            "method": "local_ollama_token_f1_plus_temporal_phrase_rules",
            "real_memory": _answer_summary(rows, "real_memory"),
            "oracle_memory": _answer_summary(rows, "oracle_memory"),
            "oracle_context": _answer_summary(rows, "oracle_context"),
            "first_answer_failure_distribution": _first_answer_failure_distribution(rows),
        },
        "note": (
            "Evidence lifecycle and answer scoring are reported separately. "
            "Before F1-F6 attribution, Gold Evidence is checked for question-answer sufficiency "
            "using the benchmark reference answer rather than the Real Memory prediction. "
            "Insufficient evidence is E0; uncertain or judge-error cases are EVAL. "
            "F6 is assigned only when the gate is sufficient, all gold evidence reaches final context, "
            "and the benchmark answer scorer still marks the answer incorrect. Oracle Context remains "
            "a reasoning control and its own wrong answer is RC_REASONING_CONTROL_FAILURE."
        ),
    }


def run(dataset: str, limit: int) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / "comparison.jsonl"
    progress_path = OUT / "progress.json"
    summary_path = OUT / "summary.json"
    jsonl.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    errors = 0
    started = time.perf_counter()

    log("=" * 72)
    log(f"Oracle experiment started: dataset={dataset}, limit={limit}")
    log(f"Output: {OUT}")
    log("=" * 72)
    _write_progress(progress_path, {"status": "running", "dataset": dataset, "limit": limit, "completed": 0, "errors": 0, "elapsed_seconds": 0.0})

    for idx, record in enumerate(iter_normalized(dataset, limit=limit)):
        number = idx + 1
        log(f"\n[{number}/{limit}] START question_id={record.get('question_id')}")
        t0 = time.perf_counter()
        try:
            row = run_record(record, idx)
            rows.append(row)
            _write_jsonl(jsonl, row)
            failure = row["modes"]["real_memory"].get("answer_failure_type")
            log(f"[{number}/{limit}] DONE {time.perf_counter() - t0:.2f}s failure={failure}")
        except Exception as exc:
            errors += 1
            log(f"[{number}/{limit}] ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
        _write_progress(progress_path, {
            "status": "running", "dataset": dataset, "limit": limit,
            "completed": number, "errors": errors,
            "elapsed_seconds": time.perf_counter() - started,
        })

    summary = _build_summary(dataset, limit, rows, errors)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, {
        "status": "completed", "dataset": dataset, "limit": limit,
        "completed": len(rows), "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo_refined_public")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    run(args.dataset, args.limit)


if __name__ == "__main__":
    main()
