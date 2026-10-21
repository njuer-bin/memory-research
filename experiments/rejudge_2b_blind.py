"""Cross-family blind re-judge of Study 2B (n=200) stored answers.

Reviewer P0-1: the n=200 main run used deepseek-v4.1-flash as BOTH the answer
model and the semantic judge (self-judging). This script re-judges the SAME
stored answer texts with a different family (qwen2.5:7b, local Ollama) on a
deterministic seeded subsample, blind to condition and to the original verdict.

The judge sees only: question, candidate answer, gold evidence -- exactly the
inputs of evaluation.answer_semantic_eval.evaluate_answer, so the pipeline
(deterministic pre-check -> LLM judge -> narrow repair) is identical and only
the judge family changes.

Outputs:
  results/study2b_judge_qwen_n200/rejudge_blind.json   (per-record verdicts)
  results/study2b_judge_qwen_n200/rejudge_report.txt   (kappa, flips, headline movement)

Usage (from repo root):
  SEMANTIC_JUDGE_MODEL=qwen2.5:7b python -m experiments.rejudge_2b_blind \
      --input results/study2b_longmemeval_n200_deepseek/comparison.jsonl \
      --outdir results/study2b_judge_qwen_n200 --limit 30 --seed 20261018
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.answer_semantic_eval import evaluate_answer  # noqa: E402

MODES = ("real_memory", "oracle_memory", "oracle_context")


def _judge_evidence(gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same normalization as run_oracle_experiments._judge_evidence."""
    return [
        {
            "id": str(e.get("evidence_id") or ""),
            "content": str(e.get("text") or e.get("content") or ""),
            "source_id": e.get("source_id") or e.get("source"),
            "turn_id": e.get("turn_id") or e.get("turn"),
            "timestamp": e.get("timestamp"),
        }
        for e in gold
        if (e.get("text") or e.get("content"))
    ]


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    yes_a = sum(1 for x in a if x) / n
    yes_b = sum(1 for y in b if y) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit < len(rows):
        rng = random.Random(args.seed)
        keep = set(rng.sample(range(len(rows)), args.limit))
        rows = [r for i, r in enumerate(rows) if i in keep]
    print(f"Blind re-judge: {len(rows)} records x {len(MODES)} conditions (seed={args.seed})")

    records: list[dict[str, Any]] = []
    n_calls = 0
    for r in rows:
        qid = r["question_id"]
        question = r["question"]
        gold = _judge_evidence(r.get("gold_evidence", []))
        rec: dict[str, Any] = {"question_id": qid, "question": question, "modes": {}}
        for m in MODES:
            a = r["modes"].get(m, {}).get("answer") or {}
            text = str(a.get("answer") or "")
            orig = a.get("semantic_correct")
            orig_model = a.get("model")
            entry: dict[str, Any] = {
                "original_verdict": orig,
                "original_judge_model": orig_model,
                "answer": text[:400],
            }
            if not text.strip():
                entry.update(qwen_verdict=None, qwen_error="empty_answer")
            else:
                try:
                    v = evaluate_answer(question, text, gold)
                    n_calls += 1
                    entry.update(
                        qwen_verdict=v["correct"],
                        qwen_judge_correct=v["judge_correct"],
                        qwen_evidence_sufficient=v["evidence_sufficient"],
                        qwen_confidence=v["confidence"],
                        qwen_adjudication=v["adjudication"],
                        qwen_reason=str(v.get("reason") or "")[:300],
                        qwen_model=v["model"],
                    )
                except Exception as exc:  # noqa: BLE001
                    entry.update(qwen_verdict=None, qwen_error=f"{type(exc).__name__}: {exc}")
            rec["modes"][m] = entry
        records.append(rec)
        print(f"  {qid}: " + ", ".join(
            f"{m.split('_')[0]}={rec['modes'][m]['original_verdict']}->{rec['modes'][m].get('qwen_verdict')}"
            for m in MODES))

    (outdir / "rejudge_blind.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    lines: list[str] = []
    log = lines.append
    log("=== STUDY 2B (n=200) CROSS-FAMILY BLIND RE-JUDGE ===")
    log(f"subsample: {len(records)} records, seed={args.seed}, judge=qwen2.5:7b (local Ollama)")
    log(f"API calls: {n_calls}")
    log("")
    # full-sample decided counts for extrapolation
    full = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    full_correct: dict[str, int] = {}
    full_decided: dict[str, int] = {}
    for m in MODES:
        vs = [r2["modes"][m]["answer"].get("semantic_correct") for r2 in full]
        full_decided[m] = sum(1 for v in vs if v is not None)
        full_correct[m] = sum(1 for v in vs if v is True)
    log("=== 1. AGREEMENT PER CONDITION (deepseek vs qwen2.5:7b) ===")
    for m in MODES:
        pairs = [(rec["modes"][m]["original_verdict"], rec["modes"][m].get("qwen_verdict"))
                 for rec in records]
        both = [(x, y) for x, y in pairs if x is not None and y is not None]
        agree = sum(1 for x, y in both if x == y)
        k = cohens_kappa([x for x, _ in both], [y for _, y in both]) if both else None
        qw_abstain = sum(1 for _, y in pairs if y is None)
        log(f"{m:16s} both-decided n={len(both):2d}: raw agreement {agree}/{len(both)}"
            f" = {agree / len(both) if both else float('nan'):.3f}"
            + (f", Cohen's kappa = {k:.3f}" if k is not None else ", kappa undefined")
            + f"; qwen abstentions on subsample = {qw_abstain}")
    log("")
    log("=== 2. FLIP DIRECTIONS (where both decided) ===")
    for m in MODES:
        ff = sum(1 for rec in records
                 if rec["modes"][m]["original_verdict"] is True
                 and rec["modes"][m].get("qwen_verdict") is False)
        ft = sum(1 for rec in records
                 if rec["modes"][m]["original_verdict"] is False
                 and rec["modes"][m].get("qwen_verdict") is True)
        log(f"{m:16s} deepseek-correct -> qwen-wrong: {ff:2d}; deepseek-wrong -> qwen-correct: {ft:2d}")
    log("")
    log("=== 3. HEADLINE MOVEMENT ===")
    log("subsample accuracy under each judge (decided rows only):")
    for m in MODES:
        d_sub = [(rec["modes"][m]["original_verdict"], rec["modes"][m].get("qwen_verdict"))
                 for rec in records if rec["modes"][m]["original_verdict"] is not None]
        acc_d = sum(1 for x, _ in d_sub if x) / len(d_sub) if d_sub else float("nan")
        dq = [(x, y) for x, y in d_sub if y is not None]
        acc_q = sum(1 for _, y in dq if y) / len(dq) if dq else float("nan")
        log(f"{m:16s} deepseek {acc_d:.3f} (n={len(d_sub)}) vs qwen {acc_q:.3f} (n={len(dq)})")
    log("")
    log("estimated FULL-sample accuracy under qwen (subsample flip rates applied")
    log("to the full decided set; assumes the subsample is exchangeable):")
    for m in MODES:
        ff = sum(1 for rec in records
                 if rec["modes"][m]["original_verdict"] is True
                 and rec["modes"][m].get("qwen_verdict") is False)
        ft = sum(1 for rec in records
                 if rec["modes"][m]["original_verdict"] is False
                 and rec["modes"][m].get("qwen_verdict") is True)
        est = (full_correct[m] - ff + ft) / full_decided[m] if full_decided[m] else float("nan")
        log(f"{m:16s} headline {full_correct[m]}/{full_decided[m]} = "
            f"{full_correct[m]/full_decided[m]:.3f} -> qwen-estimated {est:.3f}")
    log("")
    log("=== 4. PER-RECORD DETAIL ===")
    for rec in records:
        log(f"{rec['question_id']}: " + "; ".join(
            f"{m}: ds={rec['modes'][m]['original_verdict']} qwen={rec['modes'][m].get('qwen_verdict')}"
            + (f" (conf={rec['modes'][m].get('qwen_confidence')})"
               if rec['modes'][m].get('qwen_verdict') is not None else "")
            for m in MODES))

    (outdir / "rejudge_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\n[written] {outdir/'rejudge_blind.json'}")
    print(f"[written] {outdir/'rejudge_report.txt'}")


if __name__ == "__main__":
    main()
