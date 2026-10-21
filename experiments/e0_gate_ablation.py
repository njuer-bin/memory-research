"""Anti-circular E0 gate ablation (revision round 4, Phase 0).

Answers reviewer P5 ("if you remove a design, what changes"): quantify how
much the anti-circular evidence-sufficiency gate (attribution.py) contributes
to the failure attribution. The gate labels answers whose GOLD evidence
cannot answer the question as E0_EVIDENCE_INSUFFICIENT (a benchmark/question
limitation, NOT a memory-system failure). Removing it would let those items
fall through to earliest_failed_stage and be mis-attributed to a lifecycle
stage (overwhelmingly formation).

This is a pure counterfactual on STORED artifacts: no model calls. The trace
(per-stage ``found``) and gold evidence are already in the comparison.jsonl,
so we re-run attribute_failure with the gate forced sufficient and count the
label transitions.

Usage:
    python experiments/e0_gate_ablation.py \
        --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
        --out results/study2b_longmemeval_oracle_n50/e0_gate_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow `python experiments/e0_gate_ablation.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.attribution import attribute_failure

MODES = ("real_memory", "oracle_memory", "oracle_context")
FORCED_GATE = {"status": "sufficient"}  # disable the E0 anti-circular gate


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def recompute(row: dict[str, Any], mode: str) -> dict[str, Any]:
    md = (row.get("modes") or {}).get(mode) or {}
    answer = md.get("answer") or {}
    semantic_correct = answer.get("semantic_correct")
    old_label = md.get("answer_failure_type") or (
        None if semantic_correct is True else "MISSING"
    )
    if semantic_correct is not True:
        new_label = attribute_failure(
            md,  # trace carries per-stage found/measured
            semantic_correct=semantic_correct,
            gate=FORCED_GATE,
            gold_evidence=row.get("gold_evidence") or [],
        )
    else:
        new_label = None  # correct -> no failure either way
    return {
        "qid": row.get("question_id"),
        "mode": mode,
        "semantic_correct": semantic_correct,
        "old_label": old_label,
        "new_label": new_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="comparison.jsonl of a study")
    parser.add_argument("--out", required=True, help="ablation artifact (.json)")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    print(f"Loaded {len(rows)} rows from {args.input}")

    transitions: dict[str, Counter] = {m: Counter() for m in MODES}
    summary: dict[str, Any] = {}
    for mode in MODES:
        e0_total = 0
        e0_to_formation = 0
        formation_with = 0
        formation_without = 0
        for row in rows:
            r = recompute(row, mode)
            transitions[mode][f"{r['old_label']} -> {r['new_label']}"] += 1
            if mode == "real_memory":
                if r["old_label"] == "E0_EVIDENCE_INSUFFICIENT":
                    e0_total += 1
                    if r["new_label"] == "F1_FORMATION":
                        e0_to_formation += 1
                if r["old_label"] == "F1_FORMATION":
                    formation_with += 1
                if r["new_label"] == "F1_FORMATION":
                    formation_without += 1
        summary[mode] = {
            "e0_total": e0_total,
            "e0_to_formation": e0_to_formation,
            "formation_with_gate": formation_with,
            "formation_without_gate": formation_without,
        }

    for mode in MODES:
        print(f"\n=== {mode} ===")
        for k, v in sorted(transitions[mode].items()):
            print(f"  {k}: {v}")
        s = summary[mode]
        if mode == "real_memory":
            print(
                f"  E0 items (would be mis-attributed): {s['e0_total']}; "
                f"  of which -> F1_FORMATION: {s['e0_to_formation']}"
            )
            print(
                f"  Formation-dominant count: with gate = {s['formation_with_gate']}, "
                f"without gate = {s['formation_without_gate']} "
                f"(+{s['formation_without_gate'] - s['formation_with_gate']})"
            )

    artifact = {
        "study": "e0_gate_ablation",
        "description": (
            "Counterfactual: attribute_failure re-run with the E0 anti-circular "
            "evidence-sufficiency gate forced sufficient, on stored traces only."
        ),
        "forced_gate": FORCED_GATE,
        "summary": summary,
        "transitions": {m: dict(c) for m, c in transitions.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote ablation artifact -> {out}")


if __name__ == "__main__":
    main()
