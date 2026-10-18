"""Single-record LoCoMo provenance validation before large experiments.

The gate verifies that Gold Evidence references survive normalization and
MemoryEngine ingestion with the same benchmark session/turn provenance.
It intentionally fails closed: a session/turn mismatch is a provenance
alignment error, not an F1-F6 memory failure.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from datasets.loader import iter_normalized
from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest
from tracing.engine_trace import artifacts_from_store

ROOT = Path(__file__).resolve().parents[1]


def _sessions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("conversation") or [])


def _message(raw: Any) -> tuple[str, str, int | None]:
    if isinstance(raw, dict):
        role = str(raw.get("role") or raw.get("speaker") or "user")
        text = str(raw.get("content") or raw.get("text") or raw.get("message") or "")
        ts = raw.get("timestamp")
        return role, text, ts if isinstance(ts, int) else None
    return "user", str(raw), None


def _build_engine(record: dict[str, Any]) -> tuple[MemoryEngine, str]:
    db_path = ROOT / "data" / f"provenance_gate_{uuid.uuid4().hex}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    user_id = f"provenance_gate_{uuid.uuid4().hex}"
    engine = MemoryEngine(str(db_path))
    for session in _sessions(record):
        sid = str(session.get("session_id") or session.get("id") or "")
        messages: list[AddMessage] = []
        for raw in session.get("messages") or []:
            role, text, timestamp = _message(raw)
            if text.strip():
                messages.append(AddMessage(role=role, content=text, timestamp=timestamp))
        if messages:
            engine.add(AddRequest(
                request_id=f"{user_id}:{sid}",
                messages=messages,
                user_id=user_id,
                session_id=sid,
            ))
    return engine, user_id


def _artifact_provenance(item: dict[str, Any]) -> tuple[str, str]:
    source = str(
        item.get("source_turn_id")
        or item.get("turn_id")
        or item.get("source_id")
        or item.get("session_id")
        or ""
    )
    if ":turn_" in source:
        return source.split(":turn_", 1)[0], source
    source_id = str(
        item.get("source_session_id")
        or item.get("source_id")
        or item.get("session_id")
        or ""
    )
    return source_id, source


def _check_stage(stage: str, gold: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for evidence in gold:
        expected_source = str(evidence.get("source_id") or "")
        expected_turn = str(evidence.get("turn_id") or "")
        if not expected_turn:
            errors.append({
                "evidence_id": evidence.get("evidence_id"),
                "stage": stage,
                "error": "gold_evidence_missing_turn_provenance",
            })
            continue
        candidates = []
        for item in artifacts:
            actual_source, actual_turn = _artifact_provenance(item)
            if actual_turn == expected_turn:
                candidates.append(item)
                continue
            if expected_source and actual_source == expected_source:
                candidates.append(item)
        exact = [item for item in candidates if _artifact_provenance(item)[1] == expected_turn]
        if exact:
            continue
        if candidates:
            actual_source, actual_turn = _artifact_provenance(candidates[0])
            errors.append({
                "evidence_id": evidence.get("evidence_id"),
                "expected_source_id": expected_source,
                "expected_turn_id": expected_turn,
                "actual_source_id": actual_source,
                "actual_turn_id": actual_turn,
                "stage": stage,
                "error": "missing_or_mismatched_turn_provenance",
            })
        else:
            errors.append({
                "evidence_id": evidence.get("evidence_id"),
                "expected_source_id": expected_source,
                "expected_turn_id": expected_turn,
                "actual_turn_id": None,
                "stage": stage,
                "error": "gold_evidence_not_found_in_stage",
            })
    return errors


def run(question_id: str) -> dict[str, Any]:
    record = next(
        (r for r in iter_normalized("locomo_refined_public") if str(r.get("question_id")) == question_id),
        None,
    )
    if record is None:
        raise KeyError(f"LoCoMo question not found: {question_id}")

    gold = list(record.get("gold_evidence") or [])
    engine, user_id = _build_engine(record)
    stored = artifacts_from_store(engine, user_id)
    formation = stored["facts"] + stored["relations"] + stored["events"] + stored["rules"] + stored["profiles"]
    raw = stored["raw"]

    provenance_errors: dict[str, list[dict[str, Any]]] = {}
    for stage, artifacts in (("formation", formation), ("storage", raw + formation), ("evolution", formation)):
        errors = _check_stage(stage, gold, artifacts)
        if errors:
            provenance_errors[stage] = errors

    result = {
        "gate": "LOCOMO_SINGLE_SAMPLE_PROVENANCE_E2E",
        "question_id": question_id,
        "conversation_id": record.get("conversation_id"),
        "gold_evidence_count": len(gold),
        "gold_evidence": [
            {
                "evidence_id": e.get("evidence_id"),
                "source_id": e.get("source_id"),
                "turn_id": e.get("turn_id"),
                "text": e.get("text"),
            }
            for e in gold
        ],
        "status": "FAIL" if provenance_errors else "PASS",
        "provenance_errors": provenance_errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if provenance_errors:
        raise AssertionError(
            "LoCoMo provenance alignment failed: "
            + json.dumps(provenance_errors, ensure_ascii=False)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", required=True)
    args = parser.parse_args()
    run(args.question_id)


if __name__ == "__main__":
    main()
