"""Analyze the cross-family blind re-judge of Study 2B (n=200) against the
*repaired* verdicts (exact-match answers counted correct via the
deterministic channel; see experiments/sensitivity_analysis.py section 7).

Inputs:
  results/study2b_longmemeval_n200_deepseek/comparison.jsonl   (original run)
  results/study2b_judge_qwen_n200/rejudge_blind.json           (qwen blind verdicts)

Output:
  results/study2b_judge_qwen_n200/rejudge_report.txt

No model calls. Agreement statistics are computed pairwise on rows where
both judges produced a verdict; paired McNemar uses the exact binomial
test (same rule as experiments/paired_stats.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MODES = ("real_memory", "oracle_memory", "oracle_context")
LABEL = {"real_memory": "Real", "oracle_memory": "OM", "oracle_context": "OC"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial tail, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, 2 * tail)


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


def load(source: Path, rejudge: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = json.loads(rejudge.read_text(encoding="utf-8"))
    return rows, {rec["question_id"]: rec for rec in recs}


def repaired_verdict(row: dict[str, Any], mode: str) -> bool | None:
    """Deterministic exact-match channel first; then the semantic verdict."""
    a = row["modes"].get(mode, {}).get("answer") or {}
    if a.get("exact_match"):
        return True
    sc = a.get("semantic_correct")
    return None if sc is None else bool(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/study2b_longmemeval_n200_deepseek/comparison.jsonl")
    ap.add_argument("--rejudge", default="results/study2b_judge_qwen_n200/rejudge_blind.json")
    ap.add_argument("--out", default="results/study2b_judge_qwen_n200/rejudge_report.txt")
    args = ap.parse_args()

    rows, recs = load(Path(args.source), Path(args.rejudge))
    row_by_qid = {r["question_id"]: r for r in rows}
    recs = list(recs.values())
    lines: list[str] = []

    def log(msg: str = "") -> None:
        lines.append(msg)

    log("Cross-family blind re-judge of Study 2B (n=200), repaired verdicts")
    log("=" * 72)
    log(f"subsample: {len(recs)} records x 3 conditions = {len(recs) * 3} stored answers")
    log("primary judge: deepseek-v4.1-flash (answer model, same family)")
    log("re-judge:      qwen2.5:7b (local Ollama), strictly blind")
    log("verdicts compared against the REPAIRED deepseek verdicts")
    log("(exact-match answers counted correct via the deterministic channel).")
    log("")

    # ---- per-condition agreement -------------------------------------------
    log("Per-condition agreement (rows where both judges decided):")
    for m in MODES:
        pairs: list[tuple[bool, bool]] = []
        for rec in recs:
            row = row_by_qid[rec["question_id"]]
            v0 = repaired_verdict(row, m)
            v1 = rec["modes"][m].get("qwen_verdict")
            if v0 is None or v1 is None:
                continue
            pairs.append((bool(v0), bool(v1)))
        n = len(pairs)
        agree = sum(1 for x, y in pairs if x == y)
        k = cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
        ff = sum(1 for x, y in pairs if x and not y)   # deepseek-true -> qwen-false
        ft = sum(1 for x, y in pairs if y and not x)   # deepseek-false -> qwen-true
        log(f"  {LABEL[m]:5s} n={n:3d}  agree {agree}/{n} = {agree / n:.3f}  "
            f"kappa={k:+.3f}  flips: ds-wrong->qwen-correct={ft}, ds-correct->qwen-wrong={ff}")
    log("")

    # ---- paired contrasts under each judge ---------------------------------
    log("Paired contrasts on the subsample (both conditions decided):")
    for a, b in (("real_memory", "oracle_memory"),
                 ("real_memory", "oracle_context"),
                 ("oracle_memory", "oracle_context")):
        for side, name in (
            (lambda rec: (repaired_verdict(row_by_qid[rec["question_id"]], a),
                          repaired_verdict(row_by_qid[rec["question_id"]], b)), "deepseek-repaired"),
            (lambda rec: (rec["modes"][a].get("qwen_verdict"), rec["modes"][b].get("qwen_verdict")), "qwen"),
        ):
            bd = cd = n = 0
            dsum = 0.0
            for rec in recs:
                va, vb = side(rec)
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
                f"b={bd} c={cd}  exact p={p:.4f}")
        log("")

    # ---- estimated full-sample movement ------------------------------------
    log("Estimated full-sample movement (subsample flip rates applied to the")
    log("repaired full-sample decided sets):")
    for m in MODES:
        decided = [repaired_verdict(r, m) for r in rows]
        decided = [v for v in decided if v is not None]
        n_full = len(decided)
        corr_full = sum(1 for v in decided if v)
        ft_rate = ft_rt = 0.0
        n_pair = 0
        for rec in recs:
            row = row_by_qid[rec["question_id"]]
            v0 = repaired_verdict(row, m)
            v1 = rec["modes"][m].get("qwen_verdict")
            if v0 is None or v1 is None:
                continue
            n_pair += 1
            if v1 and not v0:
                ft_rate += 1
            elif v0 and not v1:
                ft_rt += 1
        est = corr_full + ft_rate / n_pair * n_full - ft_rt / n_pair * n_full
        log(f"  {LABEL[m]:5s} {corr_full}/{n_full} = {corr_full / n_full:.3f} -> "
            f"~{est:.0f}/{n_full} = {est / n_full:.3f} under qwen "
            f"(+{ft_rate / n_pair * n_full:.1f}/-{ft_rt / n_pair * n_full:.1f} points-estimate)")
    log("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
