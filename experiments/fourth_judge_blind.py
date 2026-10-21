"""Fourth-family blind judge for Study 2E disagreement cases (revision round 4).

Independence control for the author adjudication: a fourth judge model from
yet another family (Mistral, run locally via Ollama) re-judges the SAME
stored answers under the SAME prompt and the SAME deterministic rules as
the primary judge, but strictly blind: it sees only the question, the
model answer, and the gold evidence. It does not see the qwen/gemma/
deepseek verdicts, the author adjudication, or the fact that these records
were selected because the two primary judges disagreed.

Families so far: qwen2.5:7b (primary), gemma3:12b (2E reliability),
deepseek-v4.1-flash (third blind tiebreaker via API gateway), and now
mistral:7b (fourth blind adjudicator, local).

What is reused from evaluation/answer_semantic_eval.py (byte-identical
prompt, identical deterministic channels, identical post-judge repair):
- _prompt
- _deterministic_support / _full_candidate_anchor /
  _reason_indicates_insufficient_evidence
These are model-independent and stay identical so the comparison is about
the judge model only. What changes: the LLM backend (local Ollama
/api/generate instead of the gateway).

Usage:
    python experiments/fourth_judge_blind.py \
        --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
        --out results/study2e_judge_gemma/fourth_judge_blind.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.answer_semantic_eval import (
    TEMPERATURE,
    _deterministic_support,
    _full_candidate_anchor,
    _prompt,
    _reason_indicates_insufficient_evidence,
)
from experiments.run_oracle_experiments import _judge_evidence

SUBSAMPLE_SEED = 20261018  # identical to Study 2E / third_judge_blind.py


def _blind_verdict(
    question: str,
    candidate: str,
    judge_gold: list[dict[str, Any]],
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Fourth-family verdict with the same rule stack as the primary judge."""
    deterministic = _deterministic_support(candidate, judge_gold)
    if deterministic:
        kind, detail = deterministic
        return {
            "correct": True,
            "judge_correct": True,
            "evidence_sufficient": True,
            "confidence": "high",
            "reason": f"Deterministic evidence-coverage check: {kind}={detail}.",
            "adjudication": kind,
            "adjudication_anchor": detail,
            "model": model,
            "channel": "deterministic",
        }

    def call(reinforce: bool) -> dict[str, Any]:
        prompt = _prompt(question, candidate, judge_gold)
        if reinforce:
            # One bounded reinforcement for models that echo the payload
            # instead of the verdict object (observed: mistral:7b returned
            # the answer text as JSON on 3/60 records).
            prompt = (
                prompt
                + "\n\nIMPORTANT: Do not repeat the question, the reference "
                "answer, or the evidence. Your entire reply must be exactly "
                "one JSON object with the keys correct, evidence_sufficient, "
                "confidence, reason."
            )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": TEMPERATURE},
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"fourth judge unavailable at {base_url} (model={model}): {exc}") from exc

        raw = str(body.get("response") or "").strip()
        if not raw:
            raise RuntimeError("fourth judge returned an empty response")
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"fourth judge returned invalid JSON: {raw!r}") from exc
        if not isinstance(verdict.get("correct"), bool):
            raise RuntimeError(f"fourth judge missing boolean 'correct': {raw!r}")
        if not isinstance(verdict.get("evidence_sufficient"), bool):
            raise RuntimeError(f"fourth judge missing boolean 'evidence_sufficient': {raw!r}")
        return verdict

    verdict = call(reinforce=False)


def _blind_verdict_reinforced(
    question: str,
    candidate: str,
    judge_gold: list[dict[str, Any]],
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Reinforced retry wrapper (same rule stack, payload-echo-safe prompt)."""
    deterministic = _deterministic_support(candidate, judge_gold)
    if deterministic:
        return _blind_verdict(question, candidate, judge_gold, base_url, model)

    import urllib.request as _ur
    import json as _json

    prompt = _prompt(question, candidate, judge_gold) + (
        "\n\nIMPORTANT: Do not repeat the question, the reference answer, or "
        "the evidence. Your entire reply must be exactly one JSON object with "
        "the keys correct, evidence_sufficient, confidence, reason."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": TEMPERATURE},
    }
    request = _ur.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _ur.urlopen(request, timeout=300) as response:
        body = _json.loads(response.read().decode("utf-8"))
    raw = str(body.get("response") or "").strip()
    verdict = _json.loads(raw)
    if not isinstance(verdict.get("correct"), bool) or not isinstance(
        verdict.get("evidence_sufficient"), bool
    ):
        raise RuntimeError(f"reinforced retry still invalid: {raw[:200]!r}")

    evidence_sufficient = verdict["evidence_sufficient"]
    judge_correct = verdict["correct"]
    confidence = str(verdict.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    reason = str(verdict.get("reason") or "").strip()
    if evidence_sufficient and _reason_indicates_insufficient_evidence(reason):
        evidence_sufficient = False
        judge_correct = False
    final_correct = judge_correct
    adjudication = "judge"
    anchor = None
    if evidence_sufficient and not judge_correct and confidence != "low":
        anchor = _full_candidate_anchor(candidate, judge_gold)
        if anchor:
            final_correct = True
            adjudication = "lexical_anchor_override"
    return {
        "correct": final_correct,
        "judge_correct": judge_correct,
        "evidence_sufficient": evidence_sufficient,
        "confidence": confidence,
        "reason": reason,
        "adjudication": adjudication,
        "adjudication_anchor": anchor,
        "model": model,
        "channel": "llm",
    }

    evidence_sufficient = verdict["evidence_sufficient"]
    judge_correct = verdict["correct"]
    confidence = str(verdict.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    reason = str(verdict.get("reason") or "").strip()
    if evidence_sufficient and _reason_indicates_insufficient_evidence(reason):
        evidence_sufficient = False
        judge_correct = False

    final_correct = judge_correct
    adjudication = "judge"
    anchor = None
    if evidence_sufficient and not judge_correct and confidence != "low":
        anchor = _full_candidate_anchor(candidate, judge_gold)
        if anchor:
            final_correct = True
            adjudication = "lexical_anchor_override"

    return {
        "correct": final_correct,
        "judge_correct": judge_correct,
        "evidence_sufficient": evidence_sufficient,
        "confidence": confidence,
        "reason": reason,
        "adjudication": adjudication,
        "adjudication_anchor": anchor,
        "model": model,
        "channel": "llm",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="source comparison.jsonl (Study 2B)")
    parser.add_argument("--out", required=True, help="blind-judge artifact path (.json)")
    parser.add_argument("--limit", type=int, default=20, help="seeded subsample size (matches Study 2E)")
    parser.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    parser.add_argument("--model", default="mistral:7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--retry-from",
        default=None,
        help="existing artifact; verdicts with a fourth.correct value are kept, "
        "only missing/failed ones are re-requested and merged back",
    )
    args = parser.parse_args()

    prior: dict[tuple[str, str], dict[str, Any]] = {}
    if args.retry_from:
        prev = json.loads(Path(args.retry_from).read_text(encoding="utf-8"))
        for v in prev.get("verdicts") or []:
            if (v.get("fourth") or {}).get("correct") is not None:
                prior[(v["qid"], v["mode"])] = v
        print(f"Retry mode: {len(prior)} usable prior verdicts kept")

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    keep = set(rng.sample(range(len(rows)), args.limit))
    rows = [row for i, row in enumerate(rows) if i in keep]
    print(f"Blind subsample: {len(rows)} rows (seed={args.seed})")

    verdicts = []
    for row in rows:
        if row.get("_status", "ok") != "ok":
            continue
        judge_gold = _judge_evidence(row.get("gold_evidence") or [])
        for mode, mode_data in (row.get("modes") or {}).items():
            ans = mode_data.get("answer")
            if not ans or not ans.get("answer"):
                continue
            key = (row["question_id"], mode)
            if key in prior:
                verdicts.append(prior[key])
                print(f"  {row['question_id']}/{mode}: kept prior verdict")
                continue
            record = {
                "qid": row["question_id"],
                "mode": mode,
                "qwen": ans.get("semantic_correct"),
                "answer": ans["answer"],
                "question": row.get("question", ""),
            }
            try:
                verdict = _blind_verdict(
                    row.get("question", ""),
                    ans["answer"],
                    judge_gold,
                    args.base_url,
                    args.model,
                )
            except Exception as exc:  # a failed call is recorded, never silently dropped
                # One bounded reinforced retry for payload-echo failures
                # (mistral:7b returned the answer text as JSON on 3/60).
                try:
                    verdict = _blind_verdict_reinforced(
                        row.get("question", ""),
                        ans["answer"],
                        judge_gold,
                        args.base_url,
                        args.model,
                    )
                    print(f"  RETRY-REINFORCED {row['question_id']}/{mode}: ok")
                except Exception as exc2:
                    verdict = {"correct": None, "error": f"{type(exc).__name__}: {exc}", "model": args.model}
                    print(f"  ERROR {row['question_id']}/{mode}: {exc} (reinforced retry: {exc2})")
            record["fourth"] = verdict
            verdicts.append(record)
            print(
                f"  {row['question_id']}/{mode}: fourth={verdict.get('correct')} "
                f"qwen={record['qwen']} ({verdict.get('channel', 'llm')})"
            )
            time.sleep(0.1)

    # Attach the other three families + author truth for the agreement
    # report only. The fourth judge never saw any of these.
    gemma_path = Path(args.input).parent.parent / "study2e_judge_gemma" / "comparison.jsonl"
    gemma_by_key: dict[tuple[str, str], bool | None] = {}
    if gemma_path.exists():
        e_rows = [
            json.loads(line)
            for line in gemma_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rng2 = random.Random(args.seed)
        keep2 = set(rng2.sample(range(len(e_rows)), args.limit))
        for i, er in enumerate(e_rows):
            if i not in keep2:
                continue
            for mode, mode_data in (er.get("modes") or {}).items():
                ans = mode_data.get("answer") or {}
                if ans.get("answer"):
                    gemma_by_key[(er["question_id"], mode)] = ans.get("semantic_correct")

    third_path = Path(args.out).parent / "third_judge_blind.json"
    deepseek_by_key: dict[tuple[str, str], bool | None] = {}
    if third_path.exists():
        third = json.loads(third_path.read_text(encoding="utf-8"))
        for v in third.get("verdicts") or []:
            t = v.get("third") or {}
            if t.get("correct") is not None:
                deepseek_by_key[(v["qid"], v["mode"])] = bool(t["correct"])

    adj_path = Path(args.out).parent / "adjudication.json"
    truth_by_key: dict[tuple[str, str], bool | None] = {}
    if adj_path.exists():
        adj = json.loads(adj_path.read_text(encoding="utf-8"))
        for case in adj.get("cases") or []:
            truth = case.get("truth")
            if truth == "correct":
                truth_by_key[(case["qid"], case["mode"])] = True
            elif truth == "incorrect":
                truth_by_key[(case["qid"], case["mode"])] = False
            else:  # borderline: excluded from strict match counts
                truth_by_key[(case["qid"], case["mode"])] = None

    for v in verdicts:
        key = (v["qid"], v["mode"])
        v["gemma"] = gemma_by_key.get(key)
        v["deepseek"] = deepseek_by_key.get(key)
        v["author_truth"] = truth_by_key.get(key)

    artifact = {
        "study": "2E-fourth-judge-blind",
        "description": (
            "Fourth-family (Mistral, local Ollama) blind re-judge of the Study 2E "
            "seeded 20-record subsample; identical prompt and deterministic rules "
            "as the primary judge; blind to all other families' verdicts and to "
            "the author adjudication."
        ),
        "seed": args.seed,
        "fourth_judge": {"model": args.model, "base_url": args.base_url},
        "verdicts": verdicts,
        "n_verdicts": len(verdicts),
        "errors": sum(1 for v in verdicts if (v.get("fourth") or {}).get("correct") is None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(verdicts)} blind verdicts ({artifact['errors']} errors) -> {out}")


if __name__ == "__main__":
    main()
