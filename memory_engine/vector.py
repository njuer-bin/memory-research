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


class EmbeddingProvider:
    """Ollama embedding backend with an explicit, non-silent failure policy.

    Modes:
      - USE_OLLAMA_EMBEDDING=1 (default): real backend. A failing request is
        retried (EMBEDDING_RETRIES, EMBEDDING_TIMEOUT, EMBEDDING_RETRY_BACKOFF).
        If it still fails, EmbeddingUnavailableError is raised UNLESS
        EMBEDDING_ALLOW_FALLBACK=1 explicitly opts into hash degradation.
      - USE_OLLAMA_EMBEDDING=0: offline mode; stable hash embeddings, no network.

    The historical default (silent hash fallback on any Ollama error) caused
    silent measurement contamination in the Phase 2 studies and was removed.
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
        if self.use_ollama:
            last_error: Exception | None = None
            for attempt in range(self.retries):
                try:
                    r = requests.post(
                        self.url,
                        json={"model": self.model, "prompt": text},
                        timeout=self.timeout,
                    )
                    r.raise_for_status()
                    vec = r.json().get("embedding")
                    if vec:
                        return self._normalize([float(x) for x in vec])
                    last_error = EmbeddingUnavailableError(
                        f"embedding response from {self.model!r} contained no vector"
                    )
                except Exception as exc:  # network / HTTP / decode failure
                    last_error = exc
                if attempt + 1 < self.retries and self.backoff > 0:
                    time.sleep(self.backoff * (attempt + 1))
            if self.allow_fallback:
                return self._hash_embed(text)
            raise EmbeddingUnavailableError(
                f"embedding backend model={self.model!r} url={self.url} failed "
                f"after {self.retries} attempt(s): {last_error}"
            ) from last_error
        return self._hash_embed(text)

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
