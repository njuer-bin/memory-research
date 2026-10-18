"""Audit embedding dimensions across leftover engine databases.

A silent fallback to the hash embedder (dim=384) while queries are served by
the real backend (bge-m3, dim=1024) either crashes the vector index with a
matmul dimension mismatch or, when both sides fall back, silently corrupts
retrieval scores. This tool scans every *.db under data/ and reports the
dimension histogram per database so a finished run can be verified clean.

Usage:
    python experiments/audit_embedding_dims.py [--data-dir data] [--expect 1024]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--expect", type=int, default=1024)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dbs = sorted(data_dir.glob("*.db"))
    if not dbs:
        print(f"No databases found under {data_dir}")
        return

    bad: list[str] = []
    total_vectors = 0
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT vector FROM embeddings").fetchall()
        except sqlite3.Error as exc:
            print(f"{db.name}: unreadable ({exc})")
            con.close()
            continue
        con.close()
        dims: Counter[int] = Counter()
        for (vector,) in rows:
            try:
                dims[len(json.loads(vector))] += 1
            except (TypeError, ValueError):
                dims[-1] += 1
        total_vectors += sum(dims.values())
        unexpected = {d: n for d, n in dims.items() if d != args.expect}
        flag = "  <-- UNEXPECTED DIMS" if unexpected else ""
        print(f"{db.name}: {sum(dims.values())} vectors, dims={dict(dims)}{flag}")
        if unexpected:
            bad.append(db.name)

    print()
    print(f"Scanned {len(dbs)} databases, {total_vectors} vectors.")
    if bad:
        print(f"FAIL: {len(bad)} databases contain non-{args.expect}-dim vectors: {bad}")
        raise SystemExit(1)
    print(f"OK: all vectors are {args.expect}-dim (no silent hash fallback).")


if __name__ == "__main__":
    main()
