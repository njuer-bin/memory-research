from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Optional

import requests

from .store import tokenize


class EmbeddingUnavailableError(RuntimeError):
    """The configured embedding backend could not serve a request.

    Experiments must fail loudly instead of silently degrading to hash
    embeddings: hash vectors are semantically random, so mixing them with
    real vectors either crashes the vector index (dimension mismatch) or,
    worse, silently corrupts retrieval scores without any visible error.
    """


class _ContextLengthExceeded(Exception):
    """The backend rejected an input as longer than its context window.

    Deterministic content property (e.g. a pasted paper excerpt whose dense
    technical vocabulary tokenizes past bge-m3's 8192-token window); retrying
    the same text cannot help, the input must be shortened.
    """


MIN_EMBED_CHARS = 512


class EmbeddingProvider:
    """Ollama embedding backend with an explicit, non-silent failure policy.

    Modes:
      - USE_OLLAMA_EMBEDDING=1 (default): real backend. A transiently failing
        request is retried (EMBEDDING_RETRIES, EMBEDDING_TIMEOUT,
        EMBEDDING_RETRY_BACKOFF). Inputs the backend rejects for exceeding its
        context window are deterministically halved until they fit (logged).
        If it still fails, EmbeddingUnavailableError is raised UNLESS
        EMBEDDING_ALLOW_FALLBACK=1 explicitly opts into hash degradation.
      - USE_OLLAMA_EMBEDDING=0: offline mode; stable hash embeddings, no network.

    The historical default (silent hash fallback on any Ollama error) caused
    silent measurement contamination in the Phase 2 studies and was removed.
    Truncation only ever affects retrieval-side embeddings; evidence matching
    and attribution run on content, never on vector similarity.
    """

    def __init__(self, dim=384):
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
        self.dim = dim
        self.use_ollama = os.getenv("USE_OLLAMA_EMBEDDING", "1") == "1"
        self.allow_fallback = os.getenv("EMBEDDING_ALLOW_FALLBACK", "0") == "1"
        self.retries = max(1, int(os.getenv("EMBEDDING_RETRIES", "3")))
        self.timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))
        self.backoff = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "0.5"))

    def embed(self, text: str) -> list[float]:
        if not self.use_ollama:
            return self._hash_embed(text)
        candidate = text
        while True:
            try:
                return self._post_with_retries(candidate)
            except _ContextLengthExceeded:
                if len(candidate) <= MIN_EMBED_CHARS:
                    raise EmbeddingUnavailableError(
                        f"input still exceeds the context window of {self.model!r} "
                        f"after truncating to {len(candidate)} chars"
                    ) from None
                candidate = candidate[: max(MIN_EMBED_CHARS, len(candidate) // 2)]
                print(
                    f"[embedding] input exceeds context window of {self.model!r}; "
                    f"retrying with {len(candidate)} chars (truncated)",
                    flush=True,
                )
            except EmbeddingUnavailableError:
                if self.allow_fallback:
                    return self._hash_embed(text)
                raise

    def _post_with_retries(self, text: str) -> list[float]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._post_once(text)
            except _ContextLengthExceeded:
                raise  # content property: retrying the same text cannot help
            except Exception as exc:  # network / HTTP / decode failure
                last_error = exc
            if attempt + 1 < self.retries and self.backoff > 0:
                time.sleep(self.backoff * (attempt + 1))
        raise EmbeddingUnavailableError(
            f"embedding backend model={self.model!r} url={self.url} failed "
            f"after {self.retries} attempt(s): {last_error}"
        ) from last_error

    def _post_once(self, text: str) -> list[float]:
        r = requests.post(
            self.url,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        if r.status_code in (400, 500) and "context length" in (r.text or "").lower():
            raise _ContextLengthExceeded(r.text[:200])
        r.raise_for_status()
        vec = r.json().get("embedding")
        if vec:
            return self._normalize([float(x) for x in vec])
        raise EmbeddingUnavailableError(
            f"embedding response from {self.model!r} contained no vector"
        )

    def _hash_embed(self, text: str):
        v = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return v
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        return self._normalize(v)

    @staticmethod
    def _normalize(v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n else v


def cosine(a,b):
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(
            f"cosine() received vectors of different dimensions ({len(a)} vs {len(b)}); "
            "mixed embedding backends corrupt similarity scores"
        )
    dot = sum(a[i]*b[i] for i in range(len(a)))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0
