"""Bootstrap confidence intervals for per-stage recall (Study 2B/2C/2F).

The paper reports point estimates of stage recall; reviewers asked for
uncertainty. This script recomputes the per-question stage coverage from
comparison.jsonl (gold evidence covered at each lifecycle stage) and attaches
a paired bootstrap percentile CI (10,000 resamples of question indices),
using the same resampling convention as experiments/paired_stats.py.

Usage:
    python experiments/stage_recall_ci.py --input results/study2b_longmemeval_oracle_n50/comparison.jsonl [--mode real_memory]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


STAGES = ["formation", "storage", "evolution", "retrieval", "rerank", "context"]


def per_question_recall(rows: list[dict[str, Any]], mode: str) -> dict[str, list[float]]:
    """Per-question recall of gold evidence at each stage (one value per question)."""
    out: dict[str, list[float]] = {s: [] for s in STAGES}
    for row in rows:
        gold = {str(e.get("evidence_id")) for e in row.get("gold_evidence", []) if e.get("evidence_id")}
        trace = (row.get("modes") or {}).get(mode) or {}
        if not gold:
            continue
        for stage in STAGES:
            found = {str(x) for x in (trace.get(stage) or {}).get("found", [])}
            out[stage].append(len(gold & found) / len(gold))
    return out


def bootstrap_ci(values: list[float], repeats: int = 10000, seed: int = 20261019) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[max(0, int(0.025 * repeats))]
    hi = means[min(repeats - 1, int(0.975 * repeats))]
    return sum(values) / n, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="real_memory")
    ap.add_argument("--repeats", type=int, default=10000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    recalls = per_question_recall(rows, args.mode)
    n = len(recalls["formation"])
    print(f"mode={args.mode}  questions={n}  bootstrap={args.repeats}")
    print(f"{'stage':<12} {'recall':>8} {'95% CI':>20}")
    for stage in STAGES:
        point, lo, hi = bootstrap_ci(recalls[stage], args.repeats)
        print(f"{stage:<12} {point:>8.3f}   [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
