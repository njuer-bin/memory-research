"""Run evidence-boundary control experiments with an explicit E0 gate."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.loader import iter_normalized
from evaluation.answer_generation import generate_and_score
from evaluation.answer_semantic_eval import evaluate_answer
from evaluation.gate import evaluate_evidence_gate
from evaluation.lifecycle_metrics import STAGES, summarize_traces
from memory_engine.analyzer import Fact
from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest, fingerprint, new_id
from tracing.dataset_trace import record_to_trace
from tracing.engine_trace import artifacts_from_store, retrieval_artifacts, stage_match
from tracing.real_stages import (
    active_rows,
    fill_real_stages,
    stage_artifact_sets,
    stage_dict,
)
from evaluation.attribution import attribute_failure, control_failure_label

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


def _oracle_timestamp(ts: Any) -> int:
    """Best-effort deterministic timestamp for oracle-injected memories."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return int(ts)
    text = str(ts or "").strip()
    if not text:
        return 0
    digits = re.fullmatch(r"\d{9,13}", text)
    if digits:
        return int(text)
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B, %Y", "%B %d, %Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return int(datetime.strptime(text, fmt).timestamp() * 1000)
        except ValueError:
            continue
    # Unknown format: keep insertion order stable without inventing a date.
    return 0


def _oracle_memory_engine(gold: list[dict[str, Any]], idx: int) -> tuple[MemoryEngine, str]:
    """Engine pre-loaded with gold evidence as already-formed memories.

    Oracle Memory makes formation/storage/evolution perfect by construction:
    gold evidence is inserted directly as active facts with embeddings, then
    the *same* retrieval/rerank/context pipeline runs as for Real Memory. Any
    oracle_memory failure is therefore attributable to retrieval, rerank or
    context assembly -- never to memory formation. This is what separates it
    from Oracle Context, which injects the same evidence straight into the
    answer context, bypassing the pipeline (audit finding C6: the two
    conditions used to share one identical context and were indistinguishable).
    """
    user_id = f"oracle_mem_{idx}_{uuid.uuid4().hex}"
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / f"oracle_mem_{idx}_{uuid.uuid4().hex}.db"
    engine = MemoryEngine(str(db_path))

    for e in gold:
        text = str(e.get("text") or e.get("content") or "")
        if not text:
            continue
        ts = _oracle_timestamp(e.get("timestamp"))
        fact = Fact(
            id=new_id("fact"),
            user_id=user_id,
            subject="user",
            predicate="oracle_evidence",
            object=text[:200],
            content=text,
            timestamp=ts,
            fingerprint=fingerprint(user_id, "oracle_evidence", text),
            valid_from=ts,
            valid_to=None,
            status="active",
            source="user",
            source_session_id=e.get("source_id") or e.get("source"),
            source_turn_id=e.get("turn_id") or e.get("turn"),
        )
        engine.store.insert_fact(fact)
        vector = engine.embedder.embed(text)
        engine.store.embed(fact.id, user_id, vector)
        engine.vector_index.add(user_id, fact.id, vector)
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
    """Fill the Real Memory lifecycle stages via the shared measurement module."""
    fill_real_stages(
        trace, gold, engine, user_id, real, search_trace,
        formation_diagnostics=_formation_diagnostics,
    )


def _oracle_memory_trace(
    gold: list[dict[str, Any]],
    engine: MemoryEngine,
    user_id: str,
    search_trace: dict[str, Any],
    real: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure the oracle-memory engine through the same lifecycle stages.

    Formation/storage/evolution are expected to be perfect by construction
    (gold was injected as active facts); measuring them anyway validates the
    injection. Retrieval/rerank/context are the informative stages: they show
    whether the pipeline can surface evidence that is perfectly in memory.
    """
    stored = artifacts_from_store(engine, user_id)
    sets = stage_artifact_sets(stored)
    stages = {
        "formation": stage_dict(gold, sets["formation"], "oracle_memory_formation"),
        "storage": stage_dict(gold, sets["storage"], "oracle_memory_storage"),
        "evolution": stage_dict(gold, sets["evolution"], "oracle_memory_evolution"),
    }
    stages["evolution"]["details"]["superseded_facts"] = (
        len(stored["facts"]) - len(active_rows(stored["facts"]))
    )
    retrieval = search_trace.get("retrieval_candidates") or real
    rerank = search_trace.get("reranked") or real
    context = search_trace.get("context") or real
    stages["retrieval"] = stage_dict(gold, retrieval, "oracle_memory_retrieval_candidates")
    stages["rerank"] = stage_dict(gold, rerank, "oracle_memory_reranked")
    stages["context"] = stage_dict(gold, context, "oracle_memory_context")
    return stages


def _oracle_context_trace(gold: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle Context bypasses the whole lifecycle: only context is measured.

    The unmeasured stages carry ``measured: False`` so the attribution
    protocol skips them instead of reading empty ``found`` lists as total
    lifecycle failure.
    """
    result = {
        stage: {**stage_dict(gold, [], "oracle_context_not_measured"), "measured": False}
        for stage in ("formation", "storage", "evolution", "retrieval", "rerank")
    }
    result["context"] = stage_dict(gold, oracle, "oracle_context")
    return result


def _evidence_sufficiency(question: str, benchmark_answer: str, gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the benchmark Gold Evidence against the benchmark answer.

    Delegates to the shared E0 gate (evaluation.gate). The gate is independent
    of the Real Memory prediction: using the model's failed answer here would
    make E0 depend on the very system failure we are trying to attribute.
    """
    return evaluate_evidence_gate(question, benchmark_answer, gold)


def _judge_evidence(gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize gold evidence for the semantic judge, preserving metadata.

    Source/turn/timestamp metadata is part of the evidence: the judge prompt
    uses timestamps to resolve relative temporal expressions (yesterday, last
    week, ...). Note that ``evaluate_answer`` reads the ``content`` key, while
    lifecycle traces carry ``text``; passing trace items directly used to send
    an *empty* evidence block to the judge (see docs/AUDIT-2026-10-18.md).
    """
    return [
        {
            "id": str(e.get("evidence_id") or ""),
            "content": str(e.get("text") or e.get("content") or ""),
            "source_id": e.get("source_id") or e.get("source"),
            "turn_id": e.get("turn_id") or e.get("turn"),
            "timestamp": e.get("timestamp"),
        }
        for e in gold
        if (e.get("text") or e.get("content"))
    ]


def _semantic_verdict(question: str, answer_text: str, judge_gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge the *model answer* semantically (distinct from the E0 gate).

    The E0 gate judges the benchmark reference answer; this judges the model's
    own prediction. Only this verdict may feed F6 attribution.
    """
    if not answer_text:
        return {"semantic_correct": None, "semantic_error": "empty_answer"}
    try:
        verdict = evaluate_answer(question, answer_text, judge_gold)
    except Exception as exc:
        return {"semantic_correct": None, "semantic_error": f"{type(exc).__name__}: {exc}"}
    return {
        "semantic_correct": verdict["correct"],
        "semantic_judge_correct": verdict["judge_correct"],
        "semantic_evidence_sufficient": verdict["evidence_sufficient"],
        "semantic_confidence": verdict["confidence"],
        "semantic_reason": verdict["reason"],
        "semantic_adjudication": verdict["adjudication"],
        "semantic_adjudication_anchor": verdict["adjudication_anchor"],
        "semantic_judge_model": verdict["model"],
    }


def _commit_hash() -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _score_mode_answer(question: str, reference: str, context: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return generate_and_score(question, reference, context)
    except Exception as exc:
        return {"answer": None, "correct": None, "exact_match": None, "token_f1": None, "threshold": None, "model_error": f"{type(exc).__name__}: {exc}"}


def _answer_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Report lexical and semantic accuracy side by side.

    Lexical metrics are kept for reproducibility only; attribution uses the
    semantic verdict (see evaluation/attribution.py).
    """
    answers = [(r.get("modes", {}).get(mode, {}).get("answer") or {}) for r in rows]
    scored = [a for a in answers if a.get("correct") is not None]
    lex_correct = sum(1 for a in scored if a.get("correct") is True)
    sem = [a for a in answers if a.get("semantic_correct") is not None]
    sem_correct = sum(1 for a in sem if a.get("semantic_correct") is True)
    return {
        "count": len(answers),
        "lexical": {
            "scored": len(scored),
            "correct": lex_correct,
            "accuracy": lex_correct / len(scored) if scored else None,
            "exact_match": sum(float(a.get("exact_match", 0.0)) for a in scored) / len(scored) if scored else None,
            "token_f1": sum(float(a.get("token_f1", 0.0)) for a in scored) / len(scored) if scored else None,
        },
        "semantic": {
            "scored": len(sem),
            "correct": sem_correct,
            "accuracy": sem_correct / len(sem) if sem else None,
        },
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
    judge_gold = _judge_evidence(gold)
    # Formation-extractor diagnostics (Study 2F ablation arm; empty for rules).
    extraction_stats = dict(getattr(getattr(engine, "analyzer", None), "stats", None) or {})
    extraction_backend = getattr(engine, "extractor_backend", "rule")

    # Oracle Memory: gold injected as formed memories, then the SAME pipeline.
    log("    [oracle-memory] injecting gold as formed memories")
    oracle_engine, oracle_user = _oracle_memory_engine(gold, idx)
    oracle_results = oracle_engine.search(
        SearchRequest(query=record["question"], user_id=oracle_user, top_k=10, multi_hop=True)
    )
    oracle_retrieved = retrieval_artifacts(oracle_results)
    oracle_memory_context = oracle_engine.last_search_trace.get("context") or oracle_retrieved
    log(f"    [oracle-memory] context items={len(oracle_memory_context)}")

    # Oracle Context: gold evidence injected straight into the answer context,
    # bypassing retrieval entirely. Same evidence, different delivery path.
    oracle_context_items = oracle

    real_answer = _score_mode_answer(record["question"], record.get("answer", ""), real_context)
    oracle_memory_answer = _score_mode_answer(record["question"], record.get("answer", ""), oracle_memory_context)
    oracle_context_answer = _score_mode_answer(record["question"], record.get("answer", ""), oracle_context_items)
    for ans in (real_answer, oracle_memory_answer, oracle_context_answer):
        ans.update(_semantic_verdict(record["question"], ans.get("answer") or "", judge_gold))

    gate = _evidence_sufficiency(record["question"], record.get("answer", ""), judge_gold)

    real_mode = {
        "formation": trace.formation.to_dict(), "storage": trace.storage.to_dict(),
        "evolution": trace.evolution.to_dict(), "retrieval": trace.retrieval.to_dict(),
        "rerank": trace.rerank.to_dict(), "context": trace.context.to_dict(),
        "answer": {**real_answer, "mode": "real_memory"},
        "evidence_sufficiency": gate,
    }
    real_mode["answer_failure_type"] = attribute_failure(
        {"gold_evidence": gold, **real_mode},
        semantic_correct=real_answer.get("semantic_correct"),
        gate=gate,
        gold_evidence=gold,
    )

    oracle_memory = _oracle_memory_trace(
        gold, oracle_engine, oracle_user, oracle_engine.last_search_trace, oracle_retrieved
    )
    oracle_memory.update({
        "answer": {**oracle_memory_answer, "mode": "oracle_memory"},
        "evidence_sufficiency": gate,
    })
    oracle_memory["answer_failure_type"] = attribute_failure(
        {"gold_evidence": gold, **oracle_memory},
        semantic_correct=oracle_memory_answer.get("semantic_correct"),
        gate=gate,
        gold_evidence=gold,
    )

    oracle_context = _oracle_context_trace(gold, oracle)
    oracle_context.update({
        "answer": {**oracle_context_answer, "mode": "oracle_context"},
        "evidence_sufficiency": gate,
    })
    oracle_context["answer_failure_type"] = control_failure_label(
        "oracle_context", semantically_wrong=oracle_context_answer.get("semantic_correct") is False
    )

    row = {
        "question_id": trace.question_id, "question": trace.question,
        "gold_answer": record.get("answer"),
        "modes": {"real_memory": real_mode, "oracle_memory": oracle_memory, "oracle_context": oracle_context},
        "gold_evidence_count": len(gold), "gold_evidence": gold,
        "extraction": {"backend": extraction_backend, "stats": extraction_stats},
        "code_commit": _commit_hash(),
    }
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


def _extraction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate formation-extractor diagnostics across records (Study 2F)."""
    backends = {str(r.get("extraction", {}).get("backend", "rule")) for r in rows}
    backend = backends.pop() if len(backends) == 1 else (sorted(backends) or ["rule"])
    keys = ("llm_calls", "parse_failures", "fallback_messages", "extracted_messages",
            "empty_messages", "facts")
    totals = {k: 0 for k in keys}
    for r in rows:
        stats = r.get("extraction", {}).get("stats") or {}
        for k in keys:
            totals[k] += int(stats.get(k, 0) or 0)
    return {"backend": backend, "totals": totals}


def _build_summary(dataset: str, limit: int, rows: list[dict[str, Any]], errors: int) -> dict[str, Any]:
    gate_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("modes", {}).get("real_memory", {}).get("evidence_sufficiency", {}).get("status") or "unknown")
        gate_counts[status] = gate_counts.get(status, 0) + 1
    extraction = _extraction_summary(rows)
    return {
        "dataset": dataset, "limit": limit, "records": len(rows), "errors": errors,
        "extraction": extraction,
        "real_memory": summarize_traces([_mode_trace(r, "real_memory") for r in rows]) if rows else {},
        "oracle_memory": summarize_traces([_mode_trace(r, "oracle_memory") for r in rows]) if rows else {},
        "oracle_context": _context_only_summary(rows),
        "evidence_sufficiency": {"status_distribution": gate_counts, "gate": "E0 before F1-F6; uncertain/error cases are EVAL"},
        "answer_scoring": {
            "method": "lexical (EM/token-F1, reported only) + semantic judge (attribution input)",
            "lexical_note": "Lexical correctness is NOT an attribution input; it produces false negatives on concise correct answers.",
            "real_memory": _answer_summary(rows, "real_memory"),
            "oracle_memory": _answer_summary(rows, "oracle_memory"),
            "oracle_context": _answer_summary(rows, "oracle_context"),
            "first_answer_failure_distribution": _first_answer_failure_distribution(rows),
        },
        "models": {
            "answer_model": os.getenv("ANSWER_MODEL", "qwen2.5:7b"),
            "semantic_judge_model": os.getenv("SEMANTIC_JUDGE_MODEL", os.getenv("ANSWER_MODEL", "qwen2.5:7b")),
            "extractor_backend": extraction.get("backend", "rule"),
            "code_commit": _commit_hash(),
        },
        "note": (
            "Evidence lifecycle and answer scoring are reported separately. "
            "Before F1-F6 attribution, Gold Evidence is checked for question-answer sufficiency "
            "using the benchmark reference answer rather than the Real Memory prediction. "
            "Insufficient evidence is E0; uncertain or judge-error cases are EVAL. "
            "F6_REASONING_CANDIDATE is assigned only when the gate is sufficient, all gold evidence "
            "reaches final context, and the SEMANTIC evaluator (not the lexical scorer) marks the "
            "model answer incorrect. Oracle Context remains a reasoning control and its own wrong "
            "answer is RC_REASONING_CONTROL_FAILURE."
        ),
    }


def _warmup_embedder() -> None:
    """Fail fast when the embedding backend is unavailable.

    With the strict embedding policy a single failed request raises instead of
    silently degrading, so a dead backend would otherwise only surface minutes
    into the first record. Warm up once and abort the whole run up front.
    """
    from memory_engine.vector import EmbeddingProvider

    provider = EmbeddingProvider()
    t0 = time.perf_counter()
    vec = provider.embed("embedding backend warmup probe")
    log(f"[warmup] embedding backend OK: model={provider.model} dim={len(vec)} "
        f"in {time.perf_counter() - t0:.2f}s")


def _load_prior_results(jsonl: Path, progress_path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load existing rows and recorded error ids for an append/repair run."""
    rows: dict[str, dict[str, Any]] = {}
    error_ids: set[str] = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("question_id"))] = row  # last occurrence wins
    if progress_path.exists():
        try:
            prior = json.loads(progress_path.read_text(encoding="utf-8"))
            error_ids = {str(x) for x in (prior.get("error_ids") or [])}
        except (ValueError, OSError):
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

    started = time.perf_counter()

    log("=" * 72)
    mode = "repair" if selected else ("append" if append else "fresh")
    log(f"Oracle experiment started: dataset={dataset}, limit={limit}, mode={mode}"
        + (f", ids={sorted(selected)}" if selected else ""))
    log(f"Output: {out}")
    log("=" * 72)
    _warmup_embedder()
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

    # Rewrite deduplicated: prior rows keep their order, re-run rows stay in
    # place, genuinely new rows are appended. A crashed prior run may have
    # left duplicates; the dict above collapses them (last occurrence wins).
    merged = list(rows_by_id.values())
    with jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _build_summary(dataset, limit, merged, len(error_ids))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, {
        "status": "completed", "dataset": dataset, "limit": limit,
        "completed": len(merged), "errors": len(error_ids),
        "error_ids": sorted(error_ids),
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo_refined_public")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--out",
        default=None,
        help="Result directory (default: results/oracle_experiments). "
        "Use one directory per study so raw results stay separable.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated question_ids to (re)run; requires --append. "
        "Use this to repair individual failed records inside an existing result directory.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into an existing result directory instead of starting fresh; "
        "existing rows for --ids are replaced and the summary is rebuilt.",
    )
    args = parser.parse_args()
    run(args.dataset, args.limit, Path(args.out) if args.out else None,
        ids=args.ids, append=args.append)


if __name__ == "__main__":
    main()
