"""Benchmark answer generation and scoring helpers."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any

from evaluation.answer_metrics import exact_match, token_f1


BASE_URL = os.getenv("ANSWER_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("ANSWER_MODEL", "qwen2.5:7b")
TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0"))
TIMEOUT = float(os.getenv("ANSWER_TIMEOUT_SECONDS", "180"))
F1_THRESHOLD = float(os.getenv("ANSWER_F1_THRESHOLD", "0.5"))


_DATE_PATTERNS = (
    re.compile(r"\b(?P<day>\d{1,2})\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December),?\s+(?P<year>\d{4})\b", re.I),
    re.compile(r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b", re.I),
    re.compile(r"\b(?P<year>\d{4})[-/]\s*(?P<month>\d{1,2})[-/]\s*(?P<day>\d{1,2})\b"),
)
_MONTH_YEAR_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
    re.I,
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _extract_date(text: str) -> date | None:
    text = str(text or "")
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            groups = match.groupdict()
            month = groups["month"]
            if not month.isdigit():
                month = datetime.strptime(month[:3], "%b").month
            return date(int(groups["year"]), int(month), int(groups["day"]))
        except (TypeError, ValueError):
            continue
    return None


def _extract_month_period(text: str) -> tuple[date, date] | None:
    """Parse an explicit month/year such as ``June 2023`` as a calendar period."""
    match = _MONTH_YEAR_PATTERN.search(str(text or ""))
    if not match:
        return None
    try:
        month = datetime.strptime(match.group("month")[:3], "%b").month
        year = int(match.group("year"))
        start = date(year, month, 1)
        return start, date(year, month, monthrange(year, month)[1])
    except (TypeError, ValueError):
        return None


def _relative_date(text: str, anchor: date | None) -> date | None:
    if anchor is None:
        return None
    normalized = _normalize(text).strip(" .!?;:")
    offsets = {
        "today": 0,
        "yesterday": -1,
        "tomorrow": 1,
    }
    if normalized in offsets:
        return anchor + timedelta(days=offsets[normalized])
    return None


def _temporal_period(text: str, anchor: date | None) -> tuple[date, date] | None:
    """Resolve common benchmark-relative periods against a known anchor date."""
    if anchor is None:
        return None
    normalized = _normalize(text).strip(" .!?;:")

    if normalized == "this week":
        start = anchor - timedelta(days=anchor.weekday())
        return start, start + timedelta(days=6)
    if normalized == "last week":
        start = anchor - timedelta(days=anchor.weekday() + 7)
        return start, start + timedelta(days=6)
    if normalized == "next week":
        start = anchor - timedelta(days=anchor.weekday() - 7)
        return start, start + timedelta(days=6)

    if normalized in {"this month", "last month", "next month"}:
        offset = {"this month": 0, "last month": -1, "next month": 1}[normalized]
        year = anchor.year + (anchor.month - 1 + offset) // 12
        month = (anchor.month - 1 + offset) % 12 + 1
        start = date(year, month, 1)
        end = date(year, month, monthrange(year, month)[1])
        return start, end

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    match = re.fullmatch(r"(last|next) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", normalized)
    if match:
        direction, weekday_name = match.groups()
        target = weekdays[weekday_name]
        delta = (target - anchor.weekday()) % 7
        delta = delta - 7 if direction == "last" else delta + 7
        resolved = anchor + timedelta(days=delta)
        return resolved, resolved

    return None


def _context_anchor_date(context: list[dict[str, Any]]) -> date | None:
    """Find a benchmark/event date from context without inventing one."""
    for item in context:
        for key in ("timestamp", "event_timestamp", "date_time", "datetime"):
            anchor = _extract_date(str(item.get(key) or ""))
            if anchor:
                return anchor
    return None


def _temporal_equivalent(prediction: str, reference: str, anchor: date | None) -> bool:
    """Return True only for explicitly resolvable relative/date pairs."""
    pred_date = _extract_date(prediction)
    ref_date = _extract_date(reference)
    pred_relative = _relative_date(prediction, anchor)
    ref_relative = _relative_date(reference, anchor)
    pred_period = _temporal_period(prediction, anchor)
    ref_period = _temporal_period(reference, anchor)
    pred_month = _extract_month_period(prediction)
    ref_month = _extract_month_period(reference)

    if pred_date and ref_relative:
        return pred_date == ref_relative
    if ref_date and pred_relative:
        return pred_relative == ref_date
    if pred_date and ref_period:
        return ref_period[0] <= pred_date <= ref_period[1]
    if ref_date and pred_period:
        return pred_period[0] <= ref_date <= pred_period[1]
    if pred_period and ref_period:
        return pred_period == ref_period
    if pred_period and ref_month:
        return pred_period == ref_month
    if ref_period and pred_month:
        return ref_period == pred_month
    return False


def _phrase_match(prediction: str, reference: str) -> bool:
    """Accept a clear reference phrase contained in a fuller direct answer."""
    if not prediction or not reference or prediction == reference:
        return False
    if len(reference.split()) < 2:
        return False
    return reference in prediction


def _prompt(question: str, context: list[dict[str, Any]]) -> str:
    evidence = chr(10).join(
        "[{}] {}".format(i + 1, str(item.get("content") or "").strip())
        for i, item in enumerate(context)
        if str(item.get("content") or "").strip()
    )
    return (
        "Answer the question using only the evidence below. "
        "Do not invent facts. Give the shortest direct answer possible. "
        "Do not mention the evidence or your reasoning."
        + chr(10) + chr(10)
        + "Question: " + question
        + chr(10) + chr(10)
        + "Evidence:" + chr(10) + evidence
        + chr(10) + chr(10)
        + "Answer:"
    )


def generate_answer(question: str, context: list[dict[str, Any]]) -> str:
    payload = {
        "model": MODEL,
        "prompt": _prompt(question, context),
        "stream": False,
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
        raise RuntimeError(
            f"answer model unavailable at {BASE_URL} (model={MODEL}): {exc}"
        ) from exc
    answer = str(body.get("response") or "").strip()
    if not answer:
        raise RuntimeError("answer model returned an empty response")
    return answer


def score_answer(
    prediction: str,
    reference: str,
    *,
    context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pred = _normalize(prediction)
    ref = _normalize(reference)
    em = exact_match(pred, ref)
    f1 = token_f1(pred, ref)
    anchor_date = _context_anchor_date(context or [])
    pred_date = _extract_date(pred)
    ref_date = _extract_date(ref)
    temporal_equivalent = _temporal_equivalent(pred, ref, anchor_date)
    phrase_match = _phrase_match(pred, ref)

    explicit_date_mismatch = bool(pred_date and ref_date and pred_date != ref_date)
    if explicit_date_mismatch:
        correct = False
    else:
        correct = bool(
            em == 1.0
            or f1 >= F1_THRESHOLD
            or temporal_equivalent
            or phrase_match
        )

    return {
        "exact_match": em,
        "token_f1": f1,
        "threshold": F1_THRESHOLD,
        "temporal_equivalent": temporal_equivalent,
        "temporal_anchor_date": anchor_date.isoformat() if anchor_date else None,
        "phrase_match": phrase_match,
        "explicit_date_mismatch": explicit_date_mismatch,
        "correct": correct,
    }


def generate_and_score(
    question: str,
    reference: str,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    answer = generate_answer(question, context)
    return {
        "answer": answer,
        **score_answer(answer, reference, context=context),
        "model": MODEL,
        "base_url": BASE_URL,
    }
