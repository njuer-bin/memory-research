from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import requests

from .analyzer import Event, Fact, Profile, Relation, Rule
from .models import fingerprint, new_id
from .temporal_parser import normalize_temporal


EXTRACTOR_SYSTEM = """You are a memory-formation extractor for a long-term memory system.
Given conversation messages, extract structured memory items about the user.

Rules:
- Only extract durable, checkable facts (attributes, states, events, preferences, relations).
- Skip greetings, questions, chit-chat, and transient dialogue acts.
- Use short snake_case predicates (e.g. residence, workplace, occupation, like, dislike, birthday, name, language, origin, school, degree, companion, plan, event).
- Copy the exact supporting phrase from the message as "content" (verbatim substring, do not paraphrase).
- If a message contains nothing durable, return empty lists for it.
- Reply with JSON only, no prose."""


_NL = chr(10)


def _extractor_user_prompt(batch: list[dict[str, Any]]) -> str:
    lines = []
    for i, m in enumerate(batch):
        lines.append(f"[{i}] ({m['source']}) {m['content']}")
    joined = _NL.join(lines)
    return (
        "Extract memory items from each message below." + _NL
        + "Reply with a JSON array with one object per message index, in order:" + _NL
        + '[{"i":0,"facts":[{"predicate":"...","object":"...","content":"verbatim phrase"}],'
        + '"relations":[{"predicate":"...","object":"...","content":"..."}],'
        + '"events":[{"event":"...","content":"..."}],'
        + '"rules":[{"rule":"...","content":"..."}],'
        + '"profiles":[{"key":"...","value":"...","content":"..."}],'
        + '"correction":false}]' + _NL + _NL
        + "Messages:" + _NL + joined
    )


class LLMMemoryAnalyzer:
    """LLM-driven formation extractor (Study 2F ablation arm).

    Drop-in replacement for the rule-based MemoryAnalyzer: same analyze()
    contract, same output dataclasses, same governance downstream. Only the
    formation coverage differs, which is exactly the variable the ablation
    isolates. Deterministic temporal normalization is shared with the rule
    analyzer; the LLM never assigns timestamps.

    Failure policy: a batch the LLM cannot be parsed into valid JSON after one
    repair retry falls back to the rule analyzer for that batch only, and the
    fallback is counted (reported in the study summary). Nothing is silently
    dropped.
    """

    BATCH = 4  # messages per LLM call

    def __init__(self, model: Optional[str] = None, rule_analyzer=None):
        self.model = model or os.getenv("EXTRACTOR_MODEL", "qwen2.5:7b")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        self.timeout = float(os.getenv("EXTRACTOR_TIMEOUT", "180"))
        self.retries = max(1, int(os.getenv("EXTRACTOR_RETRIES", "2")))
        # Lazy import avoids a cycle when the rule analyzer is the default path.
        from .analyzer import MemoryAnalyzer
        self.rule = rule_analyzer or MemoryAnalyzer()
        self.stats = {"llm_calls": 0, "parse_failures": 0, "fallback_messages": 0,
                      "extracted_messages": 0, "facts": 0, "empty_messages": 0,
                      "skipped_messages": 0}

    # ------------------------------------------------------------------ LLM

    def _post(self, prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "system": EXTRACTOR_SYSTEM,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 2048},
                    },
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"extractor HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                self.stats["llm_calls"] += 1
                return data.get("response", "") or ""
            except Exception as exc:  # transient network / timeout
                last_err = exc
                if attempt + 1 < self.retries:
                    continue
        raise RuntimeError(f"extractor model {self.model!r} failed: {last_err}")

    @staticmethod
    def _parse_json(text: str) -> list[dict[str, Any]]:
        cleaned = text.strip()
        # strip markdown fences the model may add despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON array in extractor output")
        payload = json.loads(cleaned[start:end + 1])
        if not isinstance(payload, list):
            raise ValueError("extractor payload is not a JSON array")
        return payload

    # ------------------------------------------------------------ interface

    def analyze_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze a list of prepared messages.

        Each item: {user_id, content, timestamp, source, source_raw_id,
        source_session_id, source_turn_id}. Returns the same dict shape as
        MemoryAnalyzer.analyze() per message.
        """
        results: list[Optional[dict[str, Any]]] = [None] * len(items)
        pending: list[int] = list(range(len(items)))

        for bstart in range(0, len(items), self.BATCH):
            batch_idx = pending[bstart:bstart + self.BATCH]
            batch = [items[i] for i in batch_idx]
            try:
                raw = self._post(_extractor_user_prompt(batch))
                parsed = self._parse_json(raw)
                by_index = {}
                for entry in parsed:
                    if isinstance(entry, dict) and isinstance(entry.get("i"), int):
                        by_index[entry["i"]] = entry
                if not by_index:
                    raise ValueError("no indexable entries in extractor output")
                for local, item in enumerate(batch):
                    entry = by_index.get(local)
                    if entry is None:
                        # The extractor omits messages with nothing durable;
                        # that is an empty extraction, not a failure.
                        self.stats["skipped_messages"] += 1
                        entry = {}
                    results[batch_idx[local]] = self._build(item, entry)
            except Exception:
                # loud-but-contained: rule fallback for this batch only
                self.stats["parse_failures"] += 1
                self.stats["fallback_messages"] += len(batch)
                for local, item in enumerate(batch):
                    results[batch_idx[local]] = self.rule.analyze(
                        item["user_id"], item["content"], item["timestamp"],
                        source=item.get("source", "user"),
                        source_raw_id=item.get("source_raw_id"),
                        source_session_id=item.get("source_session_id"),
                        source_turn_id=item.get("source_turn_id"),
                    )
        return [r for r in results if r is not None]

    def analyze(self, user_id: str, content: str, timestamp: int, source: str = "user",
                source_raw_id: Optional[str] = None, source_session_id: Optional[str] = None,
                source_turn_id: Optional[str] = None) -> dict[str, Any]:
        """Single-message convenience wrapper (same contract as the rule analyzer)."""
        return self.analyze_many([{
            "user_id": user_id, "content": content, "timestamp": timestamp,
            "source": source, "source_raw_id": source_raw_id,
            "source_session_id": source_session_id, "source_turn_id": source_turn_id,
        }])[0]

    # ------------------------------------------------------------- building

    def _build(self, item: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
        user_id = item["user_id"]
        ts = item["timestamp"]
        source = item.get("source", "user")
        raw_id = item.get("source_raw_id")
        session_id = item.get("source_session_id")
        turn_id = item.get("source_turn_id")

        temporal = normalize_temporal(item["content"], ts)
        facts: list[Fact] = []
        relations: list[Relation] = []
        events: list[Event] = []
        rules: list[Rule] = []
        profiles: list[Profile] = []

        def _clean(v: Any) -> str:
            return str(v or "").strip(" ，,。；;")

        for f in entry.get("facts") or []:
            if not isinstance(f, dict):
                continue
            predicate = _clean(f.get("predicate"))
            obj = _clean(f.get("object"))
            content = _clean(f.get("content")) or f"{predicate} {obj}"
            if not predicate or not obj:
                continue
            facts.append(Fact(
                id=new_id("fact"), user_id=user_id, subject="user",
                predicate=predicate, object=obj, content=content, timestamp=ts,
                fingerprint=fingerprint(user_id, "fact", "user", predicate, obj),
                valid_from=temporal.start if temporal.start is not None else ts,
                valid_to=temporal.end, temporal_text=temporal.text, source=source,
                source_raw_id=raw_id, source_session_id=session_id, source_turn_id=turn_id,
            ))
            profiles.append(Profile(
                user_id=user_id, key=predicate, value=obj, content=content,
                timestamp=ts, source_raw_id=raw_id, source_session_id=session_id,
                source_turn_id=turn_id,
            ))

        for r in entry.get("relations") or []:
            if not isinstance(r, dict):
                continue
            predicate = _clean(r.get("predicate"))
            obj = _clean(r.get("object"))
            content = _clean(r.get("content"))
            if not predicate or not obj:
                continue
            relations.append(Relation(
                id=new_id("rel"), user_id=user_id, subject="user",
                predicate=predicate, object=obj, content=content, timestamp=ts,
                fingerprint=fingerprint(user_id, "rel", "user", predicate, obj),
                source_raw_id=raw_id, source_session_id=session_id, source_turn_id=turn_id,
            ))

        for e in entry.get("events") or []:
            if not isinstance(e, dict):
                continue
            name = _clean(e.get("event"))
            content = _clean(e.get("content")) or item["content"].strip()
            if not name:
                continue
            events.append(Event(
                id=new_id("event"), user_id=user_id, event=name, content=content,
                timestamp=ts,
                fingerprint=fingerprint(user_id, "event", content, name),
                event_start=temporal.start, event_end=temporal.end,
                temporal_text=temporal.text, source_raw_id=raw_id,
                source_session_id=session_id, source_turn_id=turn_id,
            ))

        for r in entry.get("rules") or []:
            if not isinstance(r, dict):
                continue
            value = _clean(r.get("rule"))
            content = _clean(r.get("content"))
            if not value:
                continue
            rules.append(Rule(
                id=new_id("rule"), user_id=user_id, rule=value, content=content,
                timestamp=ts,
                fingerprint=fingerprint(user_id, "rule", value),
                source_raw_id=raw_id, source_session_id=session_id, source_turn_id=turn_id,
            ))

        n_items = len(facts) + len(relations) + len(events) + len(rules)
        self.stats["facts"] += len(facts)
        if n_items:
            self.stats["extracted_messages"] += 1
        else:
            self.stats["empty_messages"] += 1

        return {
            "facts": facts, "relations": relations, "events": events,
            "rules": rules, "profiles": profiles, "temporal": temporal,
            "correction": bool(entry.get("correction")),
        }
