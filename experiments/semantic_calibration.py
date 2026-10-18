"""Post-hoc semantic evaluation of an existing oracle experiment.

The calibration layer separates:
- F1-F5: lifecycle evidence loss
- E0: Gold Evidence is insufficient/mismatched for the exact question
- F6_REASONING_CANDIDATE: evidence is sufficient and reaches final context, but
  the answer is still semantically incorrect
- EVAL_SEMANTIC_UNCERTAIN: the evaluator cannot support a stable attribution

It does not rerun memory ingestion or answer generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.answer_semantic_eval import evaluate_answer


MODES = ("real_memory", "oracle_memory", "oracle_context")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("_status", "ok") == "ok":
                rows.append(row)
    return rows


def judge_context(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(e.get("evidence_id") or ""),
            "content": str(e.get("text") or ""),
        }
        for e in row.get("gold_evidence", [])
        if e.get("text")
    ]


def first_missing_stage(real: dict[str, Any], gold_ids: set[str]) -> str | None:
    for stage, label in (
        ("formation", "F1_FORMATION"),
        ("storage", "F2_STORAGE"),
        ("evolution", "F3_EVOLUTION"),
        ("retrieval", "F4_RETRIEVAL"),
        ("rerank", "F4_RETRIEVAL"),
        ("context", "F5_CONTEXT"),
    ):
        found = {str(x) for x in (real.get(stage) or {}).get("found", [])}
        if not gold_ids.issubset(found):
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/oracle_experiments/comparison.jsonl")
    parser.add_argument(
        "--output",
        default="results/oracle_experiments/comparison_semantic.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="results/oracle_experiments/semantic_summary.json",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        mode: {"scored": 0, "correct": 0, "incorrect": 0, "errors": 0}
        for mode in MODES
    }
    attribution: dict[str, int] = {}
    f6_candidates: list[dict[str, Any]] = []
    e0_cases: list[dict[str, Any]] = []
    eval_uncertain: list[dict[str, Any]] = []

    with output.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            question = str(row.get("question") or "")
            evidence = judge_context(row)
            new_row = json.loads(json.dumps(row))
            print(f"[{i}/{len(rows)}] {row.get('question_id')}", flush=True)

            for mode in MODES:
                answer = new_row["modes"][mode].get("answer") or {}
                candidate = str(answer.get("answer") or "")
                try:
                    verdict = evaluate_answer(question, candidate, evidence)
                    answer.update(
                        {
                            "semantic_correct": verdict["correct"],
                            "semantic_judge_correct": verdict["judge_correct"],
                            "semantic_evidence_sufficient": verdict[
                                "evidence_sufficient"
                            ],
                            "semantic_confidence": verdict["confidence"],
                            "semantic_reason": verdict["reason"],
                            "semantic_adjudication": verdict["adjudication"],
                            "semantic_adjudication_anchor": verdict[
                                "adjudication_anchor"
                            ],
                            "semantic_judge_model": verdict["model"],
                        }
                    )
                    counts[mode]["scored"] += 1
                    if verdict["correct"]:
                        counts[mode]["correct"] += 1
                    else:
                        counts[mode]["incorrect"] += 1
                except Exception as exc:
                    answer["semantic_correct"] = None
                    answer["semantic_error"] = f"{type(exc).__name__}: {exc}"
                    counts[mode]["errors"] += 1
                new_row["modes"][mode]["answer"] = answer

            real = new_row["modes"]["real_memory"]
            answer = real["answer"]
            semantic_correct = answer.get("semantic_correct")
            evidence_sufficient = answer.get("semantic_evidence_sufficient")
            confidence = answer.get("semantic_confidence")

            if semantic_correct is True:
                failure = None
            elif semantic_correct is None:
                failure = "EVAL_SEMANTIC_UNCERTAIN"
            elif evidence_sufficient is False:
                failure = "E0_EVIDENCE_INSUFFICIENT"
            elif confidence == "low":
                failure = "EVAL_SEMANTIC_UNCERTAIN"
            else:
                gold_ids = {
                    str(e.get("evidence_id"))
                    for e in new_row.get("gold_evidence", [])
                    if e.get("evidence_id")
                }
                if not gold_ids:
                    failure = "E0_EVIDENCE_INSUFFICIENT"
                else:
                    failure = first_missing_stage(real, gold_ids)
                    if failure is None:
                        failure = "F6_REASONING_CANDIDATE"

            real["semantic_answer_failure_type"] = failure
            if failure:
                attribution[failure] = attribution.get(failure, 0) + 1

            if failure == "F6_REASONING_CANDIDATE":
                f6_candidates.append(
                    {
                        "question_id": row.get("question_id"),
                        "question": question,
                        "candidate": str(answer.get("answer") or ""),
                        "judge_correct": answer.get("semantic_judge_correct"),
                        "evidence_sufficient": answer.get(
                            "semantic_evidence_sufficient"
                        ),
                        "confidence": answer.get("semantic_confidence"),
                        "reason": str(answer.get("semantic_reason") or ""),
                        "gold_evidence": evidence,
                    }
                )
            elif failure == "E0_EVIDENCE_INSUFFICIENT":
                e0_cases.append(
                    {
                        "question_id": row.get("question_id"),
                        "question": question,
                        "candidate": str(answer.get("answer") or ""),
                        "reason": str(answer.get("semantic_reason") or ""),
                        "gold_evidence": evidence,
                    }
                )
            elif failure == "EVAL_SEMANTIC_UNCERTAIN":
                eval_uncertain.append(
                    {
                        "question_id": row.get("question_id"),
                        "question": question,
                        "candidate": str(answer.get("answer") or ""),
                        "reason": str(answer.get("semantic_reason") or ""),
                        "confidence": answer.get("semantic_confidence"),
                    }
                )

            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            out.flush()

    judge_models = {
        str(row.get("modes", {}).get(mode, {}).get("answer", {}).get("semantic_judge_model"))
        for row in rows
        for mode in MODES
        if row.get("modes", {}).get(mode, {}).get("answer", {}).get("semantic_judge_model")
    }

    summary = {
        "input": args.input,
        "output": args.output,
        "records": len(rows),
        "judge_models": sorted(judge_models),
        "f6_candidates": f6_candidates,
        "e0_cases": e0_cases,
        "evaluation_uncertain_cases": eval_uncertain,
        "modes": counts,
        "real_memory_semantic_failure_distribution": attribution,
        "note": (
            "Failure attribution is conservative. F6 is only assigned when the "
            "semantic evaluator considers the evidence sufficient, confidence is "
            "not low, and all Gold Evidence reaches final context. E0 captures "
            "evidence/question mismatch or insufficient evidence. A narrow lexical "
            "anchor adjudicator repairs obvious semantic-judge false negatives "
            "only after evidence sufficiency is established."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== F6 CANDIDATES ===", flush=True)
    for item in f6_candidates:
        print(f"[{item['question_id']}] {item['question']}", flush=True)
        print(f"  candidate: {item['candidate']}", flush=True)
        print(f"  judge: {item['reason']}", flush=True)

    print("\n=== E0 EVIDENCE INSUFFICIENCY ===", flush=True)
    for item in e0_cases:
        print(f"[{item['question_id']}] {item['question']}", flush=True)
        print(f"  candidate: {item['candidate']}", flush=True)
        print(f"  reason: {item['reason']}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
