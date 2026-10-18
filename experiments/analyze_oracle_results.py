"""Analyze an existing oracle-experiment comparison.jsonl without re-running the LLM.

This is a post-hoc diagnostic step. It:
- prints every sample's question/reference/predictions;
- isolates F6 candidates;
- checks Real-vs-Oracle inversions;
- sweeps token-F1 thresholds using already generated predictions;
- reports bootstrap confidence intervals for accuracy at each threshold.

No model calls are made.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("_status", "ok") == "ok":
                rows.append(row)
    return rows


def answer(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return row["modes"][mode].get("answer") or {}


def threshold_accuracy(rows: list[dict[str, Any]], mode: str, threshold: float) -> float | None:
    vals = []
    for row in rows:
        a = answer(row, mode)
        f1 = a.get("token_f1")
        em = a.get("exact_match")
        if f1 is None or em is None:
            continue
        vals.append(float(em == 1.0 or f1 >= threshold))
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(
    values: list[float],
    repeats: int = 5000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(repeats):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo = estimates[max(0, int(0.025 * repeats))]
    hi = estimates[min(repeats - 1, int(0.975 * repeats))]
    return lo, hi


def stage_recall(row: dict[str, Any], mode: str) -> dict[str, float]:
    gold = {
        str(x["evidence_id"])
        for x in row.get("gold_evidence", [])
        if x.get("evidence_id")
    }
    result = {}
    for stage in ("formation", "storage", "evolution", "retrieval", "rerank", "context"):
        found = {
            str(x)
            for x in (row["modes"][mode].get(stage) or {}).get("found", [])
        }
        result[stage] = len(gold & found) / len(gold) if gold else 1.0
    return result


def print_sample_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== SAMPLE-LEVEL ANSWER INSPECTION ===")
    for i, row in enumerate(rows, 1):
        qid = row.get("question_id")
        print(f"\n[{i}] {qid}")
        print("Q:", row.get("question", ""))
        print("Gold:", row.get("gold_answer", "(not stored)"))
        for mode in ("real_memory", "oracle_memory", "oracle_context"):
            a = answer(row, mode)
            print(
                f"{mode}: correct={a.get('correct')} "
                f"EM={a.get('exact_match')} F1={a.get('token_f1')} "
                f"answer={a.get('answer')!r}"
            )
        real = row["modes"]["real_memory"]
        print("failure:", real.get("answer_failure_type"))
        print("stage recall:", stage_recall(row, "real_memory"))


def print_f6(rows: list[dict[str, Any]]) -> None:
    print("\n=== F6 CANDIDATES ===")
    candidates = [
        r for r in rows
        if r["modes"]["real_memory"].get("answer_failure_type") == "F6_REASONING"
    ]
    if not candidates:
        print("None")
        return
    for row in candidates:
        print(f"\n{row.get('question_id')}: {row.get('question')}")
        print("Gold evidence:")
        for e in row.get("gold_evidence", []):
            print(" -", e.get("text", ""))
        for mode in ("real_memory", "oracle_memory", "oracle_context"):
            a = answer(row, mode)
            print(
                f"{mode}: {a.get('answer')!r}; "
                f"EM={a.get('exact_match')}, F1={a.get('token_f1')}, "
                f"correct={a.get('correct')}"
            )


def print_inversions(rows: list[dict[str, Any]]) -> None:
    print("\n=== ORACLE INVERSIONS ===")
    for row in rows:
        real = answer(row, "real_memory")
        oracle = answer(row, "oracle_context")
        if real.get("correct") is True and oracle.get("correct") is False:
            kind = "REAL_CORRECT_ORACLE_WRONG"
        elif real.get("correct") is False and oracle.get("correct") is True:
            kind = "REAL_WRONG_ORACLE_CORRECT"
        else:
            continue
        print(
            f"{kind}: {row.get('question_id')} | "
            f"Q={row.get('question')!r} | "
            f"real={real.get('answer')!r} | oracle={oracle.get('answer')!r}"
        )


def print_threshold_sweep(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> None:
    print("\n=== TOKEN-F1 THRESHOLD SWEEP (POST-HOC) ===")
    print("threshold\t" + "\t".join(modes))
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        vals = []
        for mode in modes:
            a = threshold_accuracy(rows, mode, threshold)
            vals.append(f"{a:.3f}" if a is not None else "NA")
        print(f"{threshold:.2f}\t" + "\t".join(vals))

    # Confidence intervals are most useful after choosing a protocol.
    threshold = 0.50
    print("\nBootstrap 95% CI at threshold=0.50:")
    for mode in modes:
        values = []
        for row in rows:
            a = answer(row, mode)
            if a.get("token_f1") is None:
                continue
            values.append(float(a.get("exact_match") == 1.0 or a.get("token_f1") >= threshold))
        lo, hi = bootstrap_ci(values)
        acc = sum(values) / len(values) if values else None
        print(f"{mode}: accuracy={acc:.3f} CI=[{lo:.3f}, {hi:.3f}]" if acc is not None else f"{mode}: NA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/oracle_experiments/comparison.jsonl",
        help="Path to comparison.jsonl",
    )
    args = parser.parse_args()
    rows = load_rows(Path(args.input))
    print(f"Loaded {len(rows)} successful rows from {args.input}")
    if not rows:
        return

    print_sample_table(rows)
    print_f6(rows)
    print_inversions(rows)
    print_threshold_sweep(rows, ("real_memory", "oracle_memory", "oracle_context"))


if __name__ == "__main__":
    main()
