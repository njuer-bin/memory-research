"""Re-judge stored answers with a different judge model (Study 2E).

Reads a comparison.jsonl produced by run_oracle_experiments, re-runs ONLY the
semantic verdict on the SAME stored answer texts with the judge model selected
via SEMANTIC_JUDGE_MODEL, and writes a new comparison.jsonl with updated
semantic fields. Answer generation and retrieval are untouched, so any
disagreement between the two runs is attributable to the judge rather than to
pipeline variance.

Usage:
    SEMANTIC_JUDGE_MODEL=gemma3:12b python experiments/rejudge_answers.py \
        --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
        --output results/study2e_judge_gemma/comparison.jsonl

Study 2E uses a deterministic seeded subsample (--limit 20 --seed 20261018,
same seed convention as build_study_set.py) so the kappa estimate is
reproducible without re-judging all records.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.run_oracle_experiments import _judge_evidence, _semantic_verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="source comparison.jsonl")
    parser.add_argument("--output", required=True, help="re-judged comparison.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="re-judge only a deterministic seeded subsample of this many rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261018,
        help="seed for the subsample (matches the study-set builder)",
    )
    args = parser.parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None and args.limit < len(rows):
        rng = random.Random(args.seed)
        keep = set(rng.sample(range(len(rows)), args.limit))
        rows = [row for i, row in enumerate(rows) if i in keep]
        print(f"Subsample: kept {len(rows)} rows (seed={args.seed})")

    out_lines = []
    n_verdicts = 0
    for row in rows:
        if row.get("_status", "ok") != "ok":
            out_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        judge_gold = _judge_evidence(row.get("gold_evidence") or [])
        for mode, mode_data in (row.get("modes") or {}).items():
            ans = mode_data.get("answer")
            if not ans or not ans.get("answer"):
                continue
            verdict = _semantic_verdict(
                row.get("question", ""), ans["answer"], judge_gold
            )
            ans.update(verdict)
            n_verdicts += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))

    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Re-judged {n_verdicts} verdicts across {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    main()
