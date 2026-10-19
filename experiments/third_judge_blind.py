"""Third-family blind judge for Study 2E disagreement cases (revision round 3).

Independence control for the author adjudication: a third judge model from a
different family (DeepSeek, via an Anthropic-compatible gateway) re-judges the
SAME stored answers under the SAME prompt and the SAME deterministic rules as
the primary judge, but strictly blind: it sees only the question, the model
answer, and the gold evidence. It does not see the qwen/gemma verdicts, the
author adjudication, or the fact that these records were selected because the
two primary judges disagreed.

What is reused from evaluation/answer_semantic_eval.py (byte-identical prompt,
identical deterministic channels, identical post-judge repair):
- _prompt: the exact judge prompt (so any verdict difference is attributable to
  the model family, not to prompt drift)
- _deterministic_support / _full_candidate_anchor / _reason_indicates_insufficient_evidence:
  the R1 lexical and R2 temporal pre-judge rules and the post-judge anchor
  override. These are model-independent and must stay identical for the
  comparison to be about the judge model only.

What changes: the LLM backend. Credentials come from the environment and are
never written to any artifact:
- BLIND_JUDGE_BASE_URL  (e.g. https://opencode.ai/zen/go)
- BLIND_JUDGE_API_KEY
- BLIND_JUDGE_MODEL     (e.g. deepseek-v4.1-flash)
- BLIND_JUDGE_SESSION   (gateway session header, if required)

Usage:
    BLIND_JUDGE_BASE_URL=... BLIND_JUDGE_API_KEY=... BLIND_JUDGE_MODEL=deepseek-v4.1-flash \
        python experiments/third_judge_blind.py \
        --input results/study2b_longmemeval_oracle_n50/comparison.jsonl \
        --out results/study2e_judge_gemma/third_judge_blind.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.answer_semantic_eval import (
    TEMPERATURE,
    _deterministic_support,
    _full_candidate_anchor,
    _prompt,
    _reason_indicates_insufficient_evidence,
)
from experiments.run_oracle_experiments import _judge_evidence

SUBSAMPLE_SEED = 20261018  # identical to Study 2E / build_study_set.py convention


def _blind_verdict(
    question: str,
    candidate: str,
    judge_gold: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    session: str | None,
) -> dict[str, Any]:
    """Third-family verdict with the same rule stack as the primary judge."""
    # Model-independent deterministic channels, identical to the primary judge.
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

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        # The gateway sits behind a WAF that rejects the default
        # Python-urllib user agent (error 1010); identify the tool honestly.
        "User-Agent": "study2e-blind-judge/1.0 (research reproducibility script)",
    }
    if session:
        headers["x-opencode-session"] = session
    payload = {
        "model": model,
        # Thinking models spend the budget on the reasoning block first; 2048
        # left no room for the verdict text (observed stop_reason=max_tokens).
        "max_tokens": 8192,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": _prompt(question, candidate, judge_gold)}],
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"third judge unavailable at {base_url} (model={model}): {exc}") from exc

    # Thinking models return a thinking block plus a text block; the verdict is
    # in the last text block (same contract as the Ollama JSON-only channel).
    text_blocks = [
        str(block.get("text") or "")
        for block in body.get("content") or []
        if block.get("type") == "text"
    ]
    raw = "\n".join(part for part in text_blocks if part.strip()).strip()
    if not raw:
        raise RuntimeError(f"third judge returned no text block: {body!r:.300}")
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"third judge returned invalid JSON: {raw!r}") from exc
    if not isinstance(verdict.get("correct"), bool):
        raise RuntimeError(f"third judge missing boolean 'correct': {raw!r}")
    if not isinstance(verdict.get("evidence_sufficient"), bool):
        raise RuntimeError(f"third judge missing boolean 'evidence_sufficient': {raw!r}")

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
    parser.add_argument(
        "--retry-from",
        default=None,
        help="existing artifact; verdicts with a third.correct value are kept, "
        "only missing/failed ones are re-requested and merged back",
    )
    args = parser.parse_args()

    import os

    base_url = os.getenv("BLIND_JUDGE_BASE_URL")
    api_key = os.getenv("BLIND_JUDGE_API_KEY")
    model = os.getenv("BLIND_JUDGE_MODEL")
    session = os.getenv("BLIND_JUDGE_SESSION")
    if not (base_url and api_key and model):
        raise SystemExit(
            "Set BLIND_JUDGE_BASE_URL, BLIND_JUDGE_API_KEY, BLIND_JUDGE_MODEL (session header optional)."
        )

    prior: dict[tuple[str, str], dict[str, Any]] = {}
    if args.retry_from:
        prev = json.loads(Path(args.retry_from).read_text(encoding="utf-8"))
        for v in prev.get("verdicts") or []:
            if (v.get("third") or {}).get("correct") is not None:
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
                "qwen_confidence": ans.get("semantic_confidence"),
                "gemma": None,  # filled by the caller-side comparison against study 2E rows
                "answer": ans["answer"],
                "question": row.get("question", ""),
            }
            try:
                verdict = _blind_verdict(
                    row.get("question", ""),
                    ans["answer"],
                    judge_gold,
                    base_url,
                    api_key,
                    model,
                    session,
                )
            except Exception as exc:  # a failed call is recorded, never silently dropped
                verdict = {"correct": None, "error": f"{type(exc).__name__}: {exc}", "model": model}
                print(f"  ERROR {row['question_id']}/{mode}: {exc}")
            record["third"] = verdict
            verdicts.append(record)
            print(
                f"  {row['question_id']}/{mode}: third={verdict.get('correct')} "
                f"qwen={record['qwen']} ({verdict.get('channel', 'llm')})"
            )
            time.sleep(0.3)  # gentle pacing for the shared gateway

    # attach gemma verdicts from the Study 2E re-judged rows (same subsample)
    e2e_path = Path(args.input).parent.parent / "study2e_judge_gemma" / "comparison.jsonl"
    gemma_by_key = {}
    if e2e_path.exists():
        e_rows = [
            json.loads(line)
            for line in e2e_path.read_text(encoding="utf-8").splitlines()
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
    for v in verdicts:
        v["gemma"] = gemma_by_key.get((v["qid"], v["mode"]))

    artifact = {
        "study": "2E-third-judge-blind",
        "description": (
            "Third-family (DeepSeek) blind re-judge of the Study 2E seeded 20-record "
            "subsample; identical prompt and deterministic rules as the primary judge; "
            "blind to both primary verdicts and to the author adjudication."
        ),
        "seed": args.seed,
        "third_judge": {"model": model, "base_url_host": base_url.split("//")[-1].split("/")[0]},
        "verdicts": verdicts,
        "n_verdicts": len(verdicts),
        "errors": sum(1 for v in verdicts if (v.get("third") or {}).get("correct") is None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(verdicts)} blind verdicts ({artifact['errors']} errors) -> {out}")


if __name__ == "__main__":
    main()
