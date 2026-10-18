from __future__ import annotations

import threading
from typing import Iterable

import numpy as np


class MemoryVectorIndex:
    """
    Process-level dense vector index.

    SQLite remains the durable source of truth. This index only keeps normalized
    vectors in RAM so Search does not repeatedly JSON-decode hundreds of vectors.
    Data written by Add is inserted immediately, preserving Add -> Search
    consistency within the running API process.
    """

    def __init__(self, store):
        self.store = store
        self._lock = threading.RLock()
        self._loaded_users: set[str] = set()
        self._vectors: dict[str, dict[str, np.ndarray]] = {}
        self._matrix: dict[str, np.ndarray] = {}
        self._ids: dict[str, list[str]] = {}

    def _ensure_user(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._loaded_users:
                return

            rows = self.store.embeddings(user_id)
            vectors: dict[str, np.ndarray] = {}
            for memory_id, vector in rows:
                arr = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    arr = arr / norm
                vectors[memory_id] = arr

            self._vectors[user_id] = vectors
            self._rebuild_matrix(user_id)
            self._loaded_users.add(user_id)

    def _rebuild_matrix(self, user_id: str) -> None:
        vectors = self._vectors.setdefault(user_id, {})
        ids = list(vectors.keys())
        self._ids[user_id] = ids
        if not ids:
            self._matrix[user_id] = np.empty((0, 0), dtype=np.float32)
            return

        dim = max(len(vectors[mid]) for mid in ids)
        matrix = np.zeros((len(ids), dim), dtype=np.float32)
        for row, mid in enumerate(ids):
            arr = vectors[mid]
            matrix[row, :len(arr)] = arr
        self._matrix[user_id] = matrix

    def add(self, user_id: str, memory_id: str, vector: list[float]) -> None:
        # If this user already has persisted vectors from before the current
        # process started, load them before inserting the new vector. Otherwise
        # the first Add after startup could hide historical vectors from Search.
        self._ensure_user(user_id)

        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if not np.all(np.isfinite(arr)):
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm

        with self._lock:
            self._vectors.setdefault(user_id, {})[memory_id] = arr
            self._rebuild_matrix(user_id)

    def search(
        self,
        user_id: str,
        query_vector: list[float],
        ids: Iterable[str] | None = None,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        self._ensure_user(user_id)

        with self._lock:
            matrix = self._matrix.get(user_id)
            all_ids = self._ids.get(user_id, [])
            if matrix is None or matrix.size == 0:
                return []

            q = np.asarray(query_vector, dtype=np.float32)
            qnorm = float(np.linalg.norm(q))
            if qnorm <= 0:
                return []
            q = q / qnorm

            allowed = set(ids) if ids is not None else None
            positions = [
                i for i, mid in enumerate(all_ids)
                if allowed is None or mid in allowed
            ]
            if not positions:
                return []

            sub = matrix[positions]
            # Stored vectors and q are normalized, so dot product is cosine.
            scores = sub @ q
            limit = min(max(1, top_k), len(positions))
            if limit < len(positions):
                local = np.argpartition(-scores, limit - 1)[:limit]
                local = local[np.argsort(-scores[local])]
            else:
                local = np.argsort(-scores)

            return [
                (all_ids[positions[int(i)]], float(scores[int(i)]))
                for i in local
            ]
