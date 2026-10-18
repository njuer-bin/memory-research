"""Benchmark-aware adapters for datasets with known schemas."""
from __future__ import annotations
from typing import Any
from .normalize import _first, _evidence_items


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or message.get("text") or message.get("message") or "")
    return str(message or "")


def _message_has_answer(message: Any) -> bool:
    return isinstance(message, dict) and bool(message.get("has_answer"))


def _answer_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " / ".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "value"):
            if value.get(key) is not None:
                return str(value[key])
    return str(value)


def _dia_session_id(dia_id: Any) -> str:
    """Return the benchmark session prefix from a LoCoMo dia_id.

    Examples: ``D1:3`` -> ``D1`` and ``D10:2`` -> ``D10``.
    Plain ids such as ``D1`` are also accepted.
    """
    value = str(dia_id or "").strip()
    if not value:
        return ""
    return value.split(":", 1)[0]


def _locomo_sessions(conversation: Any) -> list[dict[str, Any]]:
    """Normalize the public LoCoMo conversation object to the runner schema.

    The public conversation file stores sessions under ``sessions`` and does
    not provide a reliable explicit session id. The benchmark's ``dia_id``
    (for example ``D1:3``) is authoritative for provenance. If a container
    session carries a conflicting/generated id, prefer the unique session id
    encoded by its non-empty messages instead of silently renumbering it.
    """
    if not isinstance(conversation, dict):
        return []
    raw_sessions = conversation.get("conversation") or conversation.get("sessions") or []
    if isinstance(raw_sessions, dict):
        raw_sessions = [
            {"session_id": sid, "messages": messages}
            for sid, messages in raw_sessions.items()
        ]
    result: list[dict[str, Any]] = []
    for session_idx, session in enumerate(raw_sessions if isinstance(raw_sessions, list) else []):
        if not isinstance(session, dict):
            continue
        messages = session.get("messages") or session.get("turns") or []
        if isinstance(messages, dict):
            messages = [messages]
        messages = list(messages) if isinstance(messages, list) else []

        explicit_sid = session.get("session_id") or session.get("id")
        explicit_sid = str(explicit_sid).strip() if explicit_sid is not None else ""

        benchmark_sids: list[str] = []
        seen_sids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict) or not _message_text(message).strip():
                continue
            sid = _dia_session_id(message.get("dia_id"))
            if sid and sid not in seen_sids:
                seen_sids.add(sid)
                benchmark_sids.append(sid)

        # Benchmark dia_id provenance wins over a stale/generated container id.
        if len(benchmark_sids) == 1:
            sid = benchmark_sids[0]
        elif explicit_sid:
            sid = explicit_sid
        else:
            sid = f"session_{session_idx}"

        result.append({
            "session_id": sid,
            "date": session.get("date") or session.get("date_time"),
            "messages": messages,
        })
    return result


def _locomo_turns(conversation: Any) -> list[dict[str, Any]]:
    """Flatten LoCoMo sessions while preserving benchmark session provenance."""
    raw_sessions = (
        _locomo_sessions(conversation)
        if isinstance(conversation, dict)
        else conversation if isinstance(conversation, list)
        else []
    )
    turns: list[dict[str, Any]] = []
    for session_idx, session in enumerate(raw_sessions):
        if not isinstance(session, dict):
            continue
        container_source_id = str(session.get("session_id") or session.get("id") or f"session_{session_idx}")
        session_timestamp = session.get("timestamp") or session.get("date") or session.get("date_time")
        messages = session.get("messages") or session.get("turns") or []
        if isinstance(messages, dict):
            messages = [messages]
        # Keep one stable source prefix for every ingested turn in this session.
        # LoCoMo dia_ids may vary per message (D1:1, D1:2, ...) or be plain
        # ids (D1, D3, ...); derive the session prefix from the first non-empty
        # message's dia_id, and fall back to the container session id when absent.
        nonempty_messages = [
            message for message in (messages if isinstance(messages, list) else [])
            if _message_text(message).strip()
        ]
        first_dia_id = next(
            (
                str(message.get("dia_id") or "").strip()
                for message in nonempty_messages
                if isinstance(message, dict) and str(message.get("dia_id") or "").strip()
            ),
            "",
        )
        session_source_id = _dia_session_id(first_dia_id) or container_source_id

        engine_turn_idx = 0
        for msg_idx, message in enumerate(messages if isinstance(messages, list) else []):
            text = _message_text(message).strip()
            if not text:
                continue
            if isinstance(message, dict):
                benchmark_turn_id = str(
                    message.get("turn_id")
                    or message.get("dia_id")
                    or message.get("message_id")
                    or message.get("id")
                    or f"turn_{msg_idx}"
                )
                # Keep source_id stable within a session. benchmark_turn_id
                # preserves the message-level dia_id for exact evidence resolution.
                source_id = session_source_id
                role = str(message.get("role") or message.get("speaker") or "")
                timestamp = message.get("timestamp") or message.get("session_date_time") or session_timestamp
            else:
                benchmark_turn_id = f"turn_{msg_idx}"
                source_id = container_source_id
                role = "user"
                timestamp = session_timestamp
            turns.append({
                "turn_id": f"{source_id}:turn_{engine_turn_idx}",
                "source_id": source_id,
                "benchmark_turn_id": benchmark_turn_id,
                "text": text,
                "role": role,
                "timestamp": timestamp,
            })
            engine_turn_idx += 1
    return turns


def _locomo_evidence(raw_evidence: Any, *, example_id: str, conversation: Any) -> list[dict[str, Any]]:
    turns = _locomo_turns(conversation)
    by_turn = {t["turn_id"]: t for t in turns}
    by_benchmark_turn = {t["benchmark_turn_id"]: t for t in turns}
    by_text = {t["text"].strip(): t for t in turns}
    if raw_evidence is None:
        return []
    if isinstance(raw_evidence, (str, int, float)):
        raw_evidence = [raw_evidence]
    elif isinstance(raw_evidence, dict):
        raw_evidence = [raw_evidence]
    result = []
    for idx, item in enumerate(raw_evidence if isinstance(raw_evidence, list) else []):
        if isinstance(item, dict):
            evidence_id = _first(item, "evidence_id", "memory_id", default=f"{example_id}:e{idx}")
            turn_ref = _first(item, "turn_id", "turn", "dia_id", "dialogue_id", "source_turn_id", "message_id")
            source_ref = _first(item, "source_id", "session_id", "source", "doc_id")
            text = _first(item, "text", "content", "evidence", "memory", default="")
        else:
            evidence_id, turn_ref, source_ref, text = f"{example_id}:e{idx}", str(item), None, ""
        turn = by_turn.get(str(turn_ref)) if turn_ref is not None else None
        if turn is None and turn_ref is not None:
            turn = by_benchmark_turn.get(str(turn_ref))
        if turn is None and str(source_ref or "") in by_turn:
            turn = by_turn[str(source_ref)]
        if turn is None and str(text).strip() in by_text:
            turn = by_text[str(text).strip()]
        if turn is not None:
            result.append({
                "evidence_id": str(evidence_id),
                "source_id": turn["source_id"] or None,
                "turn_id": turn["turn_id"],
                "timestamp": turn.get("timestamp"),
                "text": turn["text"],
                "role": turn.get("role", ""),
                "granularity": "turn",
            })
        else:
            result.append({
                "evidence_id": str(evidence_id),
                "source_id": str(source_ref) if source_ref is not None else None,
                "turn_id": str(turn_ref) if turn_ref is not None else None,
                "timestamp": item.get("timestamp") if isinstance(item, dict) else None,
                "text": str(text or ""),
                "role": str(item.get("role") or item.get("speaker") or "") if isinstance(item, dict) else "",
                "granularity": "unresolved",
            })
    return result


def normalize_longmemeval(raw: dict[str, Any], *, dataset: str, index: int) -> dict[str, Any]:
    qid = str(raw.get("question_id", index))
    session_ids = raw.get("haystack_session_ids", []) or []
    dates = raw.get("haystack_dates", []) or []
    sessions = raw.get("haystack_sessions", []) or []
    answer_ids = {str(x) for x in (raw.get("answer_session_ids", []) or [])}
    conversation = []
    evidence = []
    for i, session in enumerate(sessions):
        sid = str(session_ids[i]) if i < len(session_ids) else f"session_{i}"
        date = dates[i] if i < len(dates) else None
        messages = list(session or []) if isinstance(session, list) else [session]
        conversation.append({"session_id": sid, "date": date, "messages": messages})
        answer_turns = [(j, m) for j, m in enumerate(messages) if _message_has_answer(m)]
        if sid in answer_ids:
            if answer_turns:
                for j, message in answer_turns:
                    text = _message_text(message)
                    if text:
                        evidence.append({
                            "evidence_id": f"{qid}:{sid}:turn_{j}", "source_id": sid,
                            "turn_id": f"{sid}:turn_{j}", "timestamp": date,
                            "text": text,
                            "role": str(message.get("role") or message.get("speaker") or "") if isinstance(message, dict) else "",
                            "granularity": "turn",
                        })
            else:
                evidence.append({
                    "evidence_id": f"{qid}:{sid}", "source_id": sid, "timestamp": date,
                    "text": "\n".join(_message_text(m) for m in messages if _message_text(m)),
                    "role": "", "granularity": "session",
                })
    qtype = raw.get("question_type", "unknown")
    if str(qid).endswith("_abs"):
        qtype = f"{qtype},abstention"
    return {
        "dataset": dataset, "family": "LongMemEval", "split": None,
        "example_id": f"{dataset}:{qid}", "conversation_id": qid,
        "question_id": qid, "question": str(raw.get("question", "")),
        "answer": raw.get("answer"), "gold_evidence": evidence,
        "conversation": conversation, "task_type": [qtype],
        "metadata": {
            "question_date": raw.get("question_date"), "answer_session_ids": list(answer_ids),
            "gold_evidence_granularity": "turn" if any(e.get("granularity") == "turn" for e in evidence) else "session",
            "raw": raw, "normalization": "longmemeval_v2_turn_level",
        },
    }


def normalize_locomo_refined(raw: dict[str, Any], *, index: int, conversation: dict[str, Any] | None = None) -> dict[str, Any]:
    qid = str(_first(raw, "qa_id", "question_id", "id", default=index))
    sample_id = _first(raw, "sample_id", "conversation_id")
    conversation = conversation or {}
    normalized_conversation = _locomo_sessions(conversation)
    answer = raw.get("answer")
    raw_evidence = _first(raw, "evidence", "gold_evidence", "supporting_evidence")
    evidence = _locomo_evidence(
        raw_evidence,
        example_id=f"locomo_refined_public:{qid}",
        conversation=normalized_conversation,
    )
    task = [str(raw.get("category"))] if raw.get("category") is not None else []
    return {
        "dataset": "locomo_refined_public", "family": "LoCoMo-Refined", "split": "public",
        "example_id": f"locomo_refined_public:{qid}",
        "conversation_id": str(sample_id) if sample_id is not None else None,
        "question_id": qid, "question": str(raw.get("question", "")),
        "answer": _answer_text(answer), "gold_evidence": evidence,
        "conversation": normalized_conversation, "task_type": task,
        "metadata": {
            "conversation_idx": raw.get("conversation_idx"), "qa_index": raw.get("qa_index"),
            "raw": raw, "normalization": "locomo_refined_v4_provenance_safe_sessions",
            "evidence_resolution": {
                "resolved_turns": sum(1 for e in evidence if e.get("granularity") == "turn"),
                "unresolved": sum(1 for e in evidence if e.get("granularity") == "unresolved"),
            },
        },
    }
