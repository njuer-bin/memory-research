"""Semantic answer evaluation for open-ended long-term QA.

The evaluator separates evidence sufficiency from answer correctness. The LLM judge
handles ambiguous cases, while deterministic checks protect clear lexical and temporal
cases from judge false-negatives.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from evaluation.answer_generation import score_answer

BASE_URL = os.getenv("ANSWER_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("SEMANTIC_JUDGE_MODEL", os.getenv("ANSWER_MODEL", "qwen2.5:7b"))
TEMPERATURE = float(os.getenv("SEMANTIC_JUDGE_TEMPERATURE", "0"))
TIMEOUT = float(os.getenv("SEMANTIC_JUDGE_TIMEOUT_SECONDS", "180"))
MIN_ANCHOR_TOKENS = int(os.getenv("SEMANTIC_MIN_ANCHOR_TOKENS", "2"))
_RELATIVE_TEMPORAL = re.compile(
    r"\b(?:today|yesterday|tomorrow|this week|last week|next week|"
    r"this month|last month|next month|last|next)\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:today|yesterday|tomorrow|this week|last week|next week|"
    r"this month|last month|next month)\b",
    re.I,
)


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for i, item in enumerate(evidence):
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        metadata: list[str] = []
        for label, key in (("source", "source_id"), ("turn", "turn_id"), ("timestamp", "timestamp")):
            value = item.get(key)
            if value is not None and str(value).strip():
                metadata.append(f"{label}={value}")
        suffix = f" ({', '.join(metadata)})" if metadata else ""
        rows.append(f"[{i + 1}] {content}{suffix}")
    return "\n".join(rows)


def _prompt(question: str, candidate: str, evidence: list[dict[str, Any]]) -> str:
    return f"""You are an answer-quality evaluator for a research experiment.

The candidate answer below is the benchmark reference answer. Do not assume it is
supported by the evidence. First decide whether the Gold Evidence is sufficient to
answer the exact question and support the reference answer.

Evidence metadata is part of the evidence. Timestamps may resolve relative temporal
expressions such as yesterday, last week, last Saturday, or next month.

Mark evidence_sufficient=false when the evidence describes a different entity/event,
only implies the requested relationship, or cannot establish the requested attribute.
Do not infer that two differently named events are the same.

If evidence_sufficient=true, judge whether the reference answer is correct. Accept
concise and semantically equivalent answers. Use confidence high/medium/low and use
low for genuine ambiguity.

Return ONLY JSON:
{{"correct": true, "evidence_sufficient": true, "confidence": "high", "reason": "brief reason"}}

Question:
{question}

Benchmark reference answer:
{candidate}

Gold Evidence:
{_evidence_text(evidence)}
"""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _meaningful_exact_anchor(candidate: str, evidence: list[dict[str, Any]]) -> str | None:
    candidate_tokens = _tokens(candidate)
    if len(candidate_tokens) < MIN_ANCHOR_TOKENS:
        return None
    for item in evidence:
        evidence_tokens = _tokens(str(item.get("content") or ""))
        if len(evidence_tokens) < MIN_ANCHOR_TOKENS:
            continue
        max_len = min(len(candidate_tokens), 12)
        for size in range(max_len, MIN_ANCHOR_TOKENS - 1, -1):
            spans = {tuple(candidate_tokens[i:i + size]) for i in range(len(candidate_tokens) - size + 1)}
            for i in range(len(evidence_tokens) - size + 1):
                span = tuple(evidence_tokens[i:i + size])
                if span in spans and any(len(t) >= 3 or re.search(r"[\u4e00-\u9fff]", t) for t in span):
                    return " ".join(span)
    return None


def _deterministic_support(candidate: str, evidence: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Return (kind, detail) only for strong evidence-coverage cases."""
    anchor = _meaningful_exact_anchor(candidate, evidence)
    if anchor:
        return "deterministic_lexical", anchor

    for item in evidence:
        content = str(item.get("content") or "")
        if not _RELATIVE_TEMPORAL.search(content):
            continue
        context = [{"timestamp": item.get("timestamp")}]
        for match in _RELATIVE_TEMPORAL.finditer(content):
            phrase = match.group(0)
            scored = score_answer(phrase, candidate, context=context)
            if scored.get("temporal_equivalent"):
                return "deterministic_temporal", phrase
    return None


def _reason_indicates_insufficient_evidence(reason: str) -> bool:
    text = reason.strip().lower()
    patterns = (
        r"only mentions? a different",
        r"mentions? a different (?:event|entity|place|person|thing)",
        r"does not (?:provide|contain|establish) (?:information|evidence)",
        r"not (?:the )?(?:same|requested)",
        r"different (?:event|entity|place|person)",
        r"does not specify the (?:local|specific|requested)",
    )
    return bool(text) and any(re.search(pattern, text) for pattern in patterns)


def evaluate_answer(question: str, candidate: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    deterministic = _deterministic_support(candidate, evidence)
    if deterministic:
        kind, detail = deterministic
        return {
            "correct": True,
            "judge_correct": True,
            "evidence_sufficient": True,
            "confidence": "high",
            "reason": f"Deterministic evidence-coverage check: {kind}={detail}.",
            "adjudication": kind,
            "adjudication_anchor": detail,
            "model": MODEL,
            "base_url": BASE_URL,
        }

    payload = {
        "model": MODEL,
        "prompt": _prompt(question, candidate, evidence),
        "stream": False,
        "format": "json",
        "options": {"temperature": TEMPERATURE},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"semantic judge unavailable at {BASE_URL} (model={MODEL}): {exc}") from exc

    raw = str(body.get("response") or "").strip()
    if not raw:
        raise RuntimeError("semantic judge returned an empty response")
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"semantic judge returned invalid JSON: {raw!r}") from exc
    if not isinstance(verdict.get("correct"), bool):
        raise RuntimeError(f"semantic judge missing boolean 'correct': {raw!r}")
    if not isinstance(verdict.get("evidence_sufficient"), bool):
        raise RuntimeError(f"semantic judge missing boolean 'evidence_sufficient': {raw!r}")

    evidence_sufficient = verdict["evidence_sufficient"]
    judge_correct = verdict["correct"]
    confidence = str(verdict.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    reason = str(verdict.get("reason") or "").strip()
    if evidence_sufficient and _reason_indicates_insufficient_evidence(reason):
        evidence_sufficient = False
        judge_correct = False

    final_correct = judge_correct
    adjudication = "judge"
    anchor = None
    if evidence_sufficient and not judge_correct and confidence != "low":
        anchor = _meaningful_exact_anchor(candidate, evidence)
        if anchor:
            final_correct = True
            adjudication = "lexical_anchor_override"

    return {
        "correct": final_correct,
        "judge_correct": judge_correct,
        "evidence_sufficient": evidence_sufficient,
        "confidence": confidence,
        "reason": reason,
        "adjudication": adjudication,
        "adjudication_anchor": anchor,
        "model": MODEL,
        "base_url": BASE_URL,
    }
