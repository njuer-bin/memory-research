"""Embedding backend failure policy: never degrade silently."""
from __future__ import annotations

import pytest

from memory_engine.vector import EmbeddingProvider, EmbeddingUnavailableError


def _provider(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return EmbeddingProvider()


def _always_fail(*args, **kwargs):
    raise RuntimeError("ollama unreachable")


class _Resp:
    def __init__(self, status: int = 200, body: str = ""):
        self.status_code = status
        self.text = body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)

    def json(self):
        return {"embedding": [3.0, 4.0]}


def test_ollama_failure_raises_instead_of_silent_hash(monkeypatch):
    provider = _provider(
        monkeypatch,
        USE_OLLAMA_EMBEDDING="1",
        EMBEDDING_ALLOW_FALLBACK="0",
        EMBEDDING_RETRIES="2",
        EMBEDDING_RETRY_BACKOFF="0",
        OLLAMA_EMBEDDING_MODEL="bge-m3:latest",
    )
    monkeypatch.setattr("memory_engine.vector.requests.post", _always_fail)
    with pytest.raises(EmbeddingUnavailableError, match="failed after 2 attempt"):
        provider.embed("hello")


def test_retries_exhaust_before_giving_up(monkeypatch):
    provider = _provider(
        monkeypatch, EMBEDDING_RETRIES="3", EMBEDDING_RETRY_BACKOFF="0"
    )
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("busy")

    monkeypatch.setattr("memory_engine.vector.requests.post", flaky)
    with pytest.raises(EmbeddingUnavailableError):
        provider.embed("hello")
    assert calls["n"] == 3


def test_hash_fallback_only_when_explicitly_opted_in(monkeypatch):
    provider = _provider(
        monkeypatch,
        USE_OLLAMA_EMBEDDING="1",
        EMBEDDING_ALLOW_FALLBACK="1",
        EMBEDDING_RETRIES="1",
        EMBEDDING_RETRY_BACKOFF="0",
    )
    monkeypatch.setattr("memory_engine.vector.requests.post", _always_fail)
    vec = provider.embed("hello world")
    assert len(vec) == provider.dim


def test_offline_mode_uses_hash_without_network(monkeypatch):
    provider = _provider(monkeypatch, USE_OLLAMA_EMBEDDING="0")

    def unreachable(*args, **kwargs):
        raise AssertionError("network must not be touched in offline mode")

    monkeypatch.setattr("memory_engine.vector.requests.post", unreachable)
    vec = provider.embed("hello world")
    assert len(vec) == provider.dim


def test_successful_response_is_normalized(monkeypatch):
    provider = _provider(monkeypatch)
    monkeypatch.setattr("memory_engine.vector.requests.post", lambda *a, **k: _Resp())
    assert provider.embed("hello") == [0.6, 0.8]


def test_response_without_vector_is_an_error(monkeypatch):
    provider = _provider(monkeypatch, EMBEDDING_RETRIES="1", EMBEDDING_RETRY_BACKOFF="0")

    class _Empty:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": []}

    monkeypatch.setattr("memory_engine.vector.requests.post", lambda *a, **k: _Empty())
    with pytest.raises(EmbeddingUnavailableError):
        provider.embed("hello")


def test_cosine_rejects_mixed_dimensions():
    from memory_engine.vector import cosine

    with pytest.raises(ValueError, match="different dimensions"):
        cosine([1.0] * 1024, [1.0] * 384)


class _TooLong:
    """Backend stub that rejects inputs longer than `limit` chars."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        text = kwargs["json"]["prompt"]
        if len(text) > self.limit:
            return _Resp(status=500, body='{"error":"the input length exceeds the context length"}')
        return _Resp()


def test_context_length_error_triggers_deterministic_truncation(monkeypatch):
    provider = _provider(monkeypatch, EMBEDDING_RETRIES="3", EMBEDDING_RETRY_BACKOFF="0")
    stub = _TooLong(limit=4000)
    monkeypatch.setattr("memory_engine.vector.requests.post", stub)
    long_text = "x" * 10000
    vec = provider.embed(long_text)
    assert len(vec) == 2  # normalized [3.0, 4.0] stub vector
    # 10000 -> 5000 -> 2500 (fits at 4000): three posts, one per candidate length
    assert stub.calls == 3


def test_context_length_error_does_not_consume_retries(monkeypatch):
    provider = _provider(monkeypatch, EMBEDDING_RETRIES="3", EMBEDDING_RETRY_BACKOFF="0")
    stub = _TooLong(limit=10)
    monkeypatch.setattr("memory_engine.vector.requests.post", stub)
    with pytest.raises(EmbeddingUnavailableError, match="context window"):
        provider.embed("y" * 100000)
    # halving from 100000 down to MIN_EMBED_CHARS (512): log2(100000/512) ~ 8 steps
    assert stub.calls <= 10


def test_truncation_only_for_over_long_inputs(monkeypatch):
    provider = _provider(monkeypatch)
    stub = _TooLong(limit=4000)
    monkeypatch.setattr("memory_engine.vector.requests.post", stub)
    assert provider.embed("short text") == [0.6, 0.8]
    assert stub.calls == 1  # no truncation round-trips for normal inputs
