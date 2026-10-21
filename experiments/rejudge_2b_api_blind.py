"""Third API-family blind re-judge of Study 2B (n=200) -- judge standoff breaker.

Reviewer round-4 concern: the cross-family re-judge of Study 2B left a 1-1
standoff. qwen2.5:7b agrees with the primary deepseek-v4.1-flash judge only at
chance level per condition (kappa -0.21/+0.17/+0.14), and the paired Real-vs-OM
delta on the subsample flips sign (deepseek-repaired +0.250 -> qwen -0.080).
Neither judge is obviously right, so the headline movement is genuinely
undetermined.

This script adds a THIRD judge family -- an API-scale model from a family
different from BOTH deepseek and qwen (Claude or GPT class, reached through an
Anthropic-compatible gateway) -- on the SAME seeded subsample (seed 20261018,
30 records x 3 conditions = 90 stored answers), strictly blind: the judge sees
only the question, the candidate answer, and the gold evidence.

The verdict pipeline is byte-identical to the primary judge: the prompt and all
deterministic channels are imported from evaluation.answer_semantic_eval (via
experiments.third_judge_blind._blind_verdict), so any difference is attributable
to the judge model family only. What changes is the LLM backend -- an
API-scale model instead of a local 7B model -- which also keeps the local
memory footprint at zero (no Ollama model is loaded).

Outputs:
  results/study2b_judge_api_n200/rejudge_api_blind.json  (per-record verdicts)
  results/study2b_judge_api_n200/rejudge_api_report.txt  (three-way comparison)

Usage (from repo root):
  BLIND_JUDGE_BASE_URL="https://<anthropic-compatible-gateway>" \
  BLIND_JUDGE_API_KEY="<key>" \
  BLIND_JUDGE_MODEL="<claude-or-gpt-family-model-id>" \
  BLIND_JUDGE_SESSION="ses_2b_apijudge_001" \
    python -m experiments.rejudge_2b_api_blind \
      --input results/study2b_longmemeval_n200_deepseek/comparison.jsonl \
      --qwen results/study2b_judge_qwen_n200/rejudge_blind.json \
      --outdir results/study2b_judge_api_n200 --limit 30 --seed 20261018
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.third_judge_blind import _blind_verdict  # noqa: E402  (shared rule stack)
from experiments.sensitivity_analysis import corrected_verdict  # noqa: E402  (EM repair)

MODES = ("real_memory", "oracle_memory", "oracle_context")
LABEL = {"real_memory": "Real", "oracle_memory": "OM", "oracle_context": "OC"}
SUBSAMPLE_SEED = 20261018  # identical to rejudge_2b_blind.py / Study 2E convention


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(1 for x in a if x) / n
    pb = sum(1 for y in b if y) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    if exp == 1.0:
        return float("nan")
    return (obs - exp) / (1 - exp)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial tail, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/study2b_longmemeval_n200_deepseek/comparison.jsonl")
    ap.add_argument("--qwen", default="results/study2b_judge_qwen_n200/rejudge_blind.json")
    ap.add_argument("--outdir", default="results/study2b_judge_api_n200")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    ap.add_argument(
        "--retry-from",
        default=None,
        help="existing artifact; verdicts with an api.correct value are kept, "
        "only missing/failed ones are re-requested and merged back",
    )
    args = ap.parse_args()

    import os

    base_url = os.getenv("BLIND_JUDGE_BASE_URL")
    api_key = os.getenv("BLIND_JUDGE_API_KEY")
    model = os.getenv("BLIND_JUDGE_MODEL")
    session = os.getenv("BLIND_JUDGE_SESSION")
    if not (base_url and api_key and model):
        raise SystemExit(
            "Set BLIND_JUDGE_BASE_URL, BLIND_JUDGE_API_KEY, BLIND_JUDGE_MODEL "
            "(BLIND_JUDGE_SESSION optional). The model must be from a family "
            "different from both deepseek and qwen (Claude or GPT class)."
        )

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    keep = set(rng.sample(range(len(full)), args.limit))
    rows = [r for i, r in enumerate(full) if i in keep]
    print(f"API-family blind re-judge: {len(rows)} records x {len(MODES)} conditions "
          f"(seed={args.seed}, model={model})")

    # qwen re-judge verdicts for the three-way comparison (same seeded subsample)
    qwen_by_qid: dict[str, dict[str, Any]] = {}
    qwen_path = Path(args.qwen)
    if qwen_path.exists():
        for rec in json.loads(qwen_path.read_text(encoding="utf-8")):
            qwen_by_qid[rec["question_id"]] = rec["modes"]
        print(f"qwen comparison verdicts loaded: {len(qwen_by_qid)} records")
    else:
        print(f"WARNING: {qwen_path} not found; report will skip the qwen comparison")

    prior: dict[tuple[str, str], dict[str, Any]] = {}
    if args.retry_from:
        prev = json.loads(Path(args.retry_from).read_text(encoding="utf-8"))
        # this script's own artifact format: a list of per-record dicts with a
        # "modes" mapping (each mode entry carries the "api" verdict)
        for rec in prev if isinstance(prev, list) else prev.get("verdicts") or []:
            if "modes" in rec:
                for m, entry in (rec.get("modes") or {}).items():
                    if (entry.get("api") or {}).get("correct") is not None:
                        prior[(rec["question_id"], m)] = entry
            elif (rec.get("api") or {}).get("correct") is not None:
                prior[(rec["qid"], rec["mode"])] = rec
        print(f"Retry mode: {len(prior)} usable prior verdicts kept")

    from experiments.run_oracle_experiments import _judge_evidence  # noqa: E402

    records: list[dict[str, Any]] = []
    n_calls = 0
    for r in rows:
        qid = r["question_id"]
        question = r.get("question", "")
        gold = _judge_evidence(r.get("gold_evidence", []))
        rec: dict[str, Any] = {"question_id": qid, "question": question, "modes": {}}
        for m in MODES:
            a = r["modes"].get(m, {}).get("answer") or {}
            text = str(a.get("answer") or "")
            entry: dict[str, Any] = {
                "deepseek_repaired": corrected_verdict(r, m),
                "qwen_verdict": (qwen_by_qid.get(qid, {}).get(m) or {}).get("qwen_verdict"),
                "answer": text[:400],
            }
            key = (qid, m)
            if key in prior:
                entry["api"] = prior[key]["api"]
                rec["modes"][m] = entry
                continue
            if not text.strip():
                entry["api"] = {"correct": None, "error": "empty_answer", "model": model}
            else:
                try:
                    v = _blind_verdict(question, text, gold, base_url, api_key, model, session)
                    n_calls += 1
                    entry["api"] = {
                        "correct": v["correct"],
                        "judge_correct": v["judge_correct"],
                        "evidence_sufficient": v["evidence_sufficient"],
                        "confidence": v["confidence"],
                        "adjudication": v["adjudication"],
                        "reason": str(v.get("reason") or "")[:300],
                        "model": v["model"],
                        "channel": v["channel"],
                    }
                except Exception as exc:  # noqa: BLE001  (a failed call is recorded, never dropped)
                    entry["api"] = {"correct": None, "error": f"{type(exc).__name__}: {exc}",
                                    "model": model}
                    print(f"  ERROR {qid}/{m}: {exc}")
            rec["modes"][m] = entry
            print(f"  {qid}/{LABEL[m]}: ds={entry['deepseek_repaired']} "
                  f"qwen={entry['qwen_verdict']} api={entry['api'].get('correct')}")
            time.sleep(0.3)  # gentle pacing for the shared gateway
        records.append(rec)

    (outdir / "rejudge_api_blind.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    lines: list[str] = []
    log = lines.append
    log("=== STUDY 2B (n=200) THIRD API-FAMILY BLIND RE-JUDGE (STANDOFF BREAKER) ===")
    log(f"subsample: {len(records)} records, seed={args.seed}")
    log(f"api judge: {model} via {base_url.split('//')[-1].split('/')[0]} "
        f"(family distinct from deepseek and qwen)")
    log(f"API calls: {n_calls}")
    log("baselines: deepseek-v4.1-flash (repaired: exact-match counted correct), "
        "qwen2.5:7b (local Ollama blind re-judge)")
    log("")

    def tripairs(m: str) -> list[tuple[str, bool, bool]]:
        """(qid, deepseek, api) rows where both decided."""
        out = []
        for rec in records:
            d = rec["modes"][m]["deepseek_repaired"]
            a = (rec["modes"][m].get("api") or {}).get("correct")
            if d is not None and a is not None:
                out.append((rec["question_id"], bool(d), bool(a)))
        return out

    def qpairs(m: str) -> list[tuple[str, bool, bool]]:
        """(qid, qwen, api) rows where both decided."""
        out = []
        for rec in records:
            q = rec["modes"][m]["qwen_verdict"]
            a = (rec["modes"][m].get("api") or {}).get("correct")
            if q is not None and a is not None:
                out.append((rec["question_id"], bool(q), bool(a)))
        return out

    log("=== 1. PER-CONDITION AGREEMENT ===")
    for m in MODES:
        dp = tripairs(m)
        qp = qpairs(m)
        if dp:
            agree_d = sum(1 for _, x, y in dp if x == y)
            kd = cohens_kappa([x for _, x, _ in dp], [y for _, _, y in dp])
            log(f"{LABEL[m]:5s} vs deepseek: n={len(dp):2d}  agree {agree_d}/{len(dp)}"
                f" = {agree_d / len(dp):.3f}  kappa={kd:+.3f}")
        else:
            log(f"{LABEL[m]:5s} vs deepseek: no co-decided rows")
        if qp:
            agree_q = sum(1 for _, x, y in qp if x == y)
            kq = cohens_kappa([x for _, x, _ in qp], [y for _, _, y in qp])
            log(f"{LABEL[m]:5s} vs qwen:     n={len(qp):2d}  agree {agree_q}/{len(qp)}"
                f" = {agree_q / len(qp):.3f}  kappa={kq:+.3f}")
        else:
            log(f"{LABEL[m]:5s} vs qwen:     no co-decided rows")
        api_abstain = sum(1 for rec in records
                          if (rec["modes"][m].get("api") or {}).get("correct") is None)
        log(f"{LABEL[m]:5s} api abstentions on subsample: {api_abstain}")
    log("")

    log("=== 2. STANDOFF RESOLUTION (rows where deepseek and qwen disagree) ===")
    for m in MODES:
        side_ds = side_qw = both_decided = 0
        for rec in records:
            d = rec["modes"][m]["deepseek_repaired"]
            q = rec["modes"][m]["qwen_verdict"]
            a = (rec["modes"][m].get("api") or {}).get("correct")
            if d is None or q is None or d == q:
                continue
            both_decided += 1
            if a is None:
                continue
            if bool(a) == bool(d):
                side_ds += 1
            elif bool(a) == bool(q):
                side_qw += 1
        log(f"{LABEL[m]:5s} ds-vs-qwen discordant rows: {both_decided}; "
            f"api sides with deepseek: {side_ds}, with qwen: {side_qw}")
    log("")

    log("=== 3. PAIRED CONTRASTS UNDER EACH JUDGE (exact McNemar) ===")
    for a, b in (("real_memory", "oracle_memory"),
                 ("real_memory", "oracle_context"),
                 ("oracle_memory", "oracle_context")):
        for side, name in (
            (lambda rec, m=a: rec["modes"][m]["deepseek_repaired"], "deepseek-repaired"),
            (lambda rec, m=a: rec["modes"][m]["qwen_verdict"], "qwen"),
            (lambda rec, m=a: (rec["modes"][m].get("api") or {}).get("correct"), "api-family"),
        ):
            bd = cd = n = 0
            dsum = 0.0
            for rec in records:
                va = side(rec, a)
                vb = side(rec, b)
                if va is None or vb is None:
                    continue
                va, vb = bool(va), bool(vb)
                n += 1
                dsum += vb - va
                if vb and not va:
                    bd += 1
                elif va and not vb:
                    cd += 1
            p = mcnemar_exact(bd, cd)
            log(f"  {name:18s} {LABEL[a]}-vs-{LABEL[b]}: n={n:3d}  delta={dsum / n:+.3f}  "
                f"b={bd} c={cd}  exact p={p:.4f}" if n else
                f"  {name:18s} {LABEL[a]}-vs-{LABEL[b]}: no co-decided pairs")
        log("")

    log("=== 4. ESTIMATED FULL-SAMPLE MOVEMENT UNDER THE API JUDGE ===")
    log("(subsample flip rates applied to the repaired full-sample decided sets;")
    log("assumes the subsample is exchangeable):")
    for m in MODES:
        full_dec = [corrected_verdict(r, m) for r in full]
        full_dec = [v for v in full_dec if v is not None]
        n_full = len(full_dec)
        corr_full = sum(1 for v in full_dec if v)
        ft = fr = 0
        n_pair = 0
        for rec in records:
            d = rec["modes"][m]["deepseek_repaired"]
            a = (rec["modes"][m].get("api") or {}).get("correct")
            if d is None or a is None:
                continue
            n_pair += 1
            if a and not d:
                ft += 1
            elif d and not a:
                fr += 1
        if n_pair:
            est = corr_full + (ft - fr) / n_pair * n_full
            log(f"{LABEL[m]:5s} headline {corr_full}/{n_full} = {corr_full / n_full:.3f} -> "
                f"api-estimated ~{est:.0f}/{n_full} = {est / n_full:.3f} "
                f"(+{ft / n_pair * n_full:.1f}/-{fr / n_pair * n_full:.1f} points-estimate)")
        else:
            log(f"{LABEL[m]:5s} no co-decided rows for extrapolation")
    log("")

    log("=== 5. PER-RECORD DETAIL ===")
    for rec in records:
        log(f"{rec['question_id']}: " + "; ".join(
            f"{LABEL[m]}: ds={rec['modes'][m]['deepseek_repaired']} "
            f"qwen={rec['modes'][m]['qwen_verdict']} "
            f"api={(rec['modes'][m].get('api') or {}).get('correct')}"
            for m in MODES))

    (outdir / "rejudge_api_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n[written] {outdir/'rejudge_api_blind.json'}")
    print(f"[written] {outdir/'rejudge_api_report.txt'}")


if __name__ == "__main__":
    main()
