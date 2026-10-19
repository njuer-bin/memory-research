"""LLM formation extractor (Study 2F ablation arm): parsing, fallback, batching."""
from __future__ import annotations

import json

import pytest

from memory_engine.llm_analyzer import LLMMemoryAnalyzer


class _Resp:
    def __init__(self, status: int = 200, body: str = "", payload: dict | None = None):
        self.status_code = status
        self.text = body
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        return {"response": self.text}


def _analyzer(monkeypatch, handler):
    a = LLMMemoryAnalyzer(model="stub-model")
    monkeypatch.setattr(a, "_post", handler)
    return a


def _msgs(n: int) -> list[dict]:
    return [
        {
            "user_id": "u1",
            "content": f"I live in city{i} and I like tea.",
            "timestamp": 1700000000000 + i,
            "source": "user",
            "source_raw_id": f"raw{i}",
            "source_session_id": "s1",
            "source_turn_id": f"s1:turn_{i}",
        }
        for i in range(n)
    ]


def test_valid_json_with_fences_is_parsed(monkeypatch):
    payload = [
        {"i": 0, "facts": [{"predicate": "residence", "object": "city0",
                            "content": "I live in city0"}],
         "relations": [], "events": [], "rules": [], "profiles": [], "correction": False},
        {"i": 1, "facts": [], "relations": [], "events": [], "rules": [], "profiles": [],
         "correction": False},
    ]
    body = "```json\n" + json.dumps(payload) + "\n```"

    def handler(prompt):
        return body

    a = _analyzer(monkeypatch, handler)
    out = a.analyze_many(_msgs(2))
    assert len(out) == 2
    assert out[0]["facts"][0].predicate == "residence"
    assert out[0]["facts"][0].object == "city0"
    assert out[0]["facts"][0].source_raw_id == "raw0"
    assert out[1]["facts"] == []
    assert a.stats["facts"] == 1
    assert a.stats["fallback_messages"] == 0


def test_parse_failure_falls_back_to_rule_analyzer(monkeypatch):
    def handler(prompt):
        return "I refuse to output JSON."

    a = _analyzer(monkeypatch, handler)
    out = a.analyze_many(_msgs(2))
    # rule analyzer must still extract "I live in city0" from message 0
    assert out[0]["facts"], "rule fallback should extract residence fact"
    assert out[0]["facts"][0].object.startswith("city")
    assert a.stats["parse_failures"] == 1
    assert a.stats["fallback_messages"] == 2


def test_missing_index_is_empty_extraction_not_fallback(monkeypatch):
    # The extractor may omit messages with nothing durable; a missing index
    # is an empty extraction (counted as skipped), NOT a rule fallback.
    payload = [{"i": 0, "facts": [{"predicate": "residence", "object": "city0",
                                   "content": "I live in city0"}]}]

    def handler(prompt):
        return json.dumps(payload)  # message 1 omitted

    a = _analyzer(monkeypatch, handler)
    out = a.analyze_many(_msgs(2))
    assert a.stats["skipped_messages"] == 1
    assert a.stats["fallback_messages"] == 0
    assert a.stats["parse_failures"] == 0
    assert out[0]["facts"][0].object == "city0"
    assert out[1]["facts"] == []  # empty, not rule-extracted


def test_batching_groups_by_four(monkeypatch):
    calls = []

    def handler(prompt):
        calls.append(prompt)
        n = prompt.count("\n[") + (1 if prompt.startswith("[") else 0)
        # count messages by counting "[i]" markers in the prompt
        import re
        idx = re.findall(r"\[(\d+)\] \(", prompt)
        return json.dumps([
            {"i": int(i), "facts": [], "relations": [], "events": [], "rules": [],
             "profiles": [], "correction": False} for i in idx
        ])

    a = _analyzer(monkeypatch, handler)
    out = a.analyze_many(_msgs(9))
    assert len(out) == 9
    assert len(calls) == 3  # 4 + 4 + 1
    assert a.stats["llm_calls"] == 0  # _post is stubbed; calls counted via handler


def test_temporal_normalization_is_deterministic(monkeypatch):
    # The deterministic temporal parser is shared by both extractor arms (it
    # resolves Chinese temporal expressions); the LLM never assigns timestamps.
    payload = [{"i": 0, "facts": [{"predicate": "residence", "object": "Seattle",
                                   "content": "我去年搬到了西雅图"}]}]

    def handler(prompt):
        return json.dumps(payload)

    a = _analyzer(monkeypatch, handler)
    out = a.analyze_many([{
        "user_id": "u1", "content": "我去年搬到了西雅图",
        "timestamp": 1700000000000, "source": "user", "source_raw_id": "raw0",
        "source_session_id": "s1", "source_turn_id": "s1:turn_0",
    }])
    fact = out[0]["facts"][0]
    # valid_from must come from the deterministic temporal parser, not the LLM
    assert fact.valid_from is not None and fact.valid_from <= 1700000000000
    assert fact.temporal_text == "去年"


def test_single_message_wrapper_matches_contract(monkeypatch):
    payload = [{"i": 0, "facts": [{"predicate": "name", "object": "Ann",
                                   "content": "My name is Ann"}]}]

    def handler(prompt):
        return json.dumps(payload)

    a = _analyzer(monkeypatch, handler)
    out = a.analyze("u1", "My name is Ann", 1700000000000)
    assert out["facts"][0].predicate == "name"
    assert "profiles" in out and "temporal" in out and "correction" in out
