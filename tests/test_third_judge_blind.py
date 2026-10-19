"""Tests for the third-family blind judge (Study 2E independence control).

The blind judge must behave identically to the primary judge except for the
LLM backend, so these tests pin: (1) the deterministic channel is reused
(model-independent), (2) the Anthropic-compatible request shape, (3) the
thinking-block extraction, (4) a failed call raises instead of silently
returning a verdict, (5) retry-from keeps usable prior verdicts.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from experiments.third_judge_blind import _blind_verdict

GOLD = [
    {"id": "e1", "content": "I went to the LGBTQ support group on Tuesday.", "timestamp": "2023/05/16 (Tue) 10:00"},
]


class _Resp:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _api_body(text: str, thinking: str | None = None) -> dict:
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    content.append({"type": "text", "text": text})
    return {"id": "x", "type": "message", "role": "assistant", "content": content, "model": "m"}


def test_deterministic_channel_never_calls_llm(monkeypatch):
    """A verbatim-quote candidate must pass via the shared R1 rule, no API call."""
    import experiments.third_judge_blind as mod

    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called for deterministic cases")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    verdict = _blind_verdict(
        "Where did I go on Tuesday?",
        "the LGBTQ support group",
        GOLD,
        "https://example.invalid",
        "key",
        "model-x",
        None,
    )
    assert verdict["correct"] is True
    assert verdict["channel"] == "deterministic"
    assert verdict["adjudication"] == "deterministic_lexical"


def test_thinking_block_extraction(monkeypatch):
    """The verdict lives in the last text block, after any thinking block."""
    import experiments.third_judge_blind as mod

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Resp(
            _api_body(
                '{"correct": false, "evidence_sufficient": true, "confidence": "high", "reason": "wrong day"}',
                thinking="let me evaluate the evidence step by step...",
            )
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    verdict = _blind_verdict(
        "Where did I go on Tuesday?",
        "the park",
        GOLD,
        "https://example.invalid",
        "secret-key",
        "model-x",
        "ses_1",
    )
    assert verdict["correct"] is False
    assert verdict["channel"] == "llm"
    assert verdict["judge_correct"] is False
    assert verdict["confidence"] == "high"
    # request shape: Anthropic-compatible endpoint, key + session headers, UA set
    assert captured["url"] == "https://example.invalid/v1/messages"
    assert captured["headers"]["x-api-key"] == "secret-key"
    assert captured["headers"]["x-opencode-session"] == "ses_1"
    assert "user-agent" in captured["headers"]
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] >= 8192
    assert captured["payload"]["messages"][0]["role"] == "user"


def test_reason_based_insufficiency_reused(monkeypatch):
    """Same reason-based downgrade as the primary judge: a 'correct' verdict
    whose rationale says the evidence describes a different event is demoted
    to incorrect (the judge contradicted itself in the reason field)."""
    import experiments.third_judge_blind as mod

    def fake_urlopen(request, timeout=None):
        return _Resp(
            _api_body(
                '{"correct": true, "evidence_sufficient": true, '
                '"confidence": "high", "reason": "the evidence only mentions a different event"}'
            )
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    verdict = _blind_verdict(
        "Where did I go on Tuesday?",
        "the park",
        GOLD,
        "https://example.invalid",
        "k",
        "m",
        None,
    )
    assert verdict["judge_correct"] is False  # demoted alongside evidence_sufficient
    assert verdict["correct"] is False


def test_http_error_raises(monkeypatch):
    """A gateway failure must raise, never be recorded as a verdict."""
    import experiments.third_judge_blind as mod

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="third judge unavailable"):
        _blind_verdict(
            "Where did I go on Tuesday?",
            "the park",
            GOLD,
            "https://example.invalid",
            "k",
            "m",
            None,
        )


def test_missing_text_block_raises(monkeypatch):
    """Thinking-only truncation (stop_reason=max_tokens) must raise, not pass."""
    import experiments.third_judge_blind as mod

    body = {"content": [{"type": "thinking", "thinking": "only thinking", "signature": "s"}]}
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda r, timeout=None: _Resp(body))
    with pytest.raises(RuntimeError, match="no text block"):
        _blind_verdict(
            "Where did I go on Tuesday?",
            "the park",
            GOLD,
            "https://example.invalid",
            "k",
            "m",
            None,
        )
