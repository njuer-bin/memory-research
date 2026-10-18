from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

import requests

from .store import tokenize


class EmbeddingProvider:
    """
    优先使用 Ollama（可配置 OLLAMA_EMBEDDING_MODEL）。
    Ollama 不可用时使用稳定 hash embedding，保证本地测试不依赖外部服务。
    """

    def __init__(self, dim=384):
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
        self.dim = dim
        self.use_ollama = os.getenv("USE_OLLAMA_EMBEDDING", "1") == "1"

    def embed(self, text: str) -> list[float]:
        if self.use_ollama:
            try:
                r = requests.post(
                    self.url,
                    json={"model": self.model, "prompt": text},
                    timeout=15,
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if vec:
                    return self._normalize([float(x) for x in vec])
            except Exception:
                pass
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
    n = min(len(a),len(b))
    dot = sum(a[i]*b[i] for i in range(n))
    na = math.sqrt(sum(x*x for x in a[:n]))
    nb = math.sqrt(sum(x*x for x in b[:n]))
    return dot / (na*nb) if na and nb else 0.0
