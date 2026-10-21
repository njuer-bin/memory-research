"""Analyze the Mem0 external-validity study (Study 3A) against Study 2B.

Reads results/study3a_mem0_n50/comparison.jsonl and reports everything the
paper integration needs:

- three-condition semantic accuracy (+ exact McNemar and paired bootstrap
  via experiments/paired_stats.py conventions)
- per-stage recall chain for real_memory and oracle_memory
- failure-label distribution (real_memory) and E0 gate distribution
- per-question-type breakdown (LongMemEval types, from the source file)
- side-by-side comparison with the Study 2B n=50 protocol subset (in-house
  rule extractor, same questions, same answer/judge model)
- Mem0 extraction stats (memories formed, updated, deleted)

No model calls; pure measurement over stored artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.lifecycle_metrics import STAGES  # noqa: E402

N50_PILOT = ROOT / "results" / "study2b_longmemeval_oracle_n50" / "comparison.jsonl"
SOURCE = ROOT / "datasets" / "data" / "longmemeval_oracle.json"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def verdict(row: dict[str, Any], mode: str) -> bool | None:
    return (row.get("modes", {}).get(mode, {}).get("answer") or {}).get("semantic_correct")


def acc(rows: list[dict[str, Any]], mode: str) -> str:
    scored = [verdict(r, mode) for r in rows]
    evaluated = [v for v in scored if v is not None]
    if not evaluated:
        return "n/a"
    correct = sum(1 for v in evaluated if v is True)
    return f"{correct}/{len(evaluated)} = {correct / len(evaluated):.3f}"


def mcnemar(rows: list[dict[str, Any]], mode_a: str, mode_b: str) -> str:
    b = c = 0
    for r in rows:
        va, vb = verdict(r, mode_a), verdict(r, mode_b)
        if va is None or vb is None:
            continue
        if va is True and vb is False:
            b += 1
        elif va is False and vb is True:
            c += 1
    n = b + c
    if n == 0:
        return "no discordant pairs"
    # exact binomial two-sided p
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    p = min(1.0, p)
    return f"b={b}, c={c}, p={p:.4f}"


def boot_diff(rows: list[dict[str, Any]], mode_a: str, mode_b: str,
              reps: int = 10000, seed: int = 42) -> str:
    diffs = []
    for r in rows:
        va, vb = verdict(r, mode_a), verdict(r, mode_b)
        if va is not None and vb is not None:
            diffs.append((1.0 if va else 0.0) - (1.0 if vb else 0.0))
    if not diffs:
        return "n/a"
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)
    )
    point = sum(diffs) / n
    return f"{point:+.3f}  95% CI [{means[int(0.025 * reps)]:+.3f}, {means[int(0.975 * reps)]:+.3f}]"


def stage_recall(rows: list[dict[str, Any]], mode: str) -> dict[str, float]:
    totals = {s: 0.0 for s in STAGES}
    for r in rows:
        gold = {str(x["evidence_id"]) for x in r.get("gold_evidence", []) if x.get("evidence_id")}
        if not gold:
            continue
        trace = r["modes"][mode]
        for s in STAGES:
            data = trace.get(s) or {}
            if data.get("measured") is False:
                totals[s] += 1.0  # not applicable: count as perfect (not measured)
                continue
            found = {str(x) for x in (data.get("found") or [])}
            totals[s] += len(gold & found) / len(gold)
    n = len(rows) or 1
    return {s: totals[s] / n for s in STAGES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "results" / "study3a_mem0_n50" / "comparison.jsonl"))
    parser.add_argument("--pilot", default=str(N50_PILOT))
    args = parser.parse_args()

    rows = load(Path(args.input))
    print(f"Mem0 Study 3A: {len(rows)} records\n")

    print("=== THREE-CONDITION SEMANTIC ACCURACY ===")
    for mode in ("real_memory", "oracle_memory", "oracle_context"):
        print(f"  {mode:16s}: {acc(rows, mode)}")

    print("\n=== CONDITION CONTRASTS (exact McNemar / paired bootstrap) ===")
    for a, b in (("real_memory", "oracle_memory"), ("real_memory", "oracle_context"),
                 ("oracle_memory", "oracle_context")):
        print(f"  {a} vs {b}: {mcnemar(rows, a, b)}")
        print(f"    bootstrap diff: {boot_diff(rows, a, b)}")

    print("\n=== STAGE RECALL (real_memory) ===")
    for s, v in stage_recall(rows, "real_memory").items():
        print(f"  {s:10s}: {v:.3f}")
    print("=== STAGE RECALL (oracle_memory) ===")
    for s, v in stage_recall(rows, "oracle_memory").items():
        print(f"  {s:10s}: {v:.3f}")

    print("\n=== FAILURE LABELS (real_memory) ===")
    counts = Counter(str(r["modes"]["real_memory"].get("answer_failure_type") or "None") for r in rows)
    for label, n in counts.most_common():
        print(f"  {label:28s}: {n}  ({n / len(rows):.1%})")

    print("\n=== E0 GATE DISTRIBUTION ===")
    gates = Counter(str(r["modes"]["real_memory"].get("evidence_sufficiency", {}).get("status") or "unknown") for r in rows)
    for status, n in gates.most_common():
        print(f"  {status:12s}: {n}")

    print("\n=== MEM0 EXTRACTION STATS ===")
    stats = Counter()
    for r in rows:
        for k, v in (r.get("extraction", {}).get("stats") or {}).items():
            stats[k] += int(v or 0)
    for k in ("messages_ingested", "llm_calls", "facts", "memories_updated", "memories_deleted"):
        print(f"  {k:18s}: {stats[k]}")

    # per-type
    print("\n=== PER QUESTION TYPE ===")
    if SOURCE.exists():
        src = json.loads(SOURCE.read_text(encoding="utf-8"))
        qtype = {str(x.get("question_id")): x.get("question_type", "unknown") for x in src if isinstance(x, dict)}
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            qt = qtype.get(str(r["question_id"]), "unknown")
            a = agg.setdefault(qt, {"n": 0, "real": [0, 0], "om": [0, 0], "oc": [0, 0], "fail": Counter()})
            a["n"] += 1
            for key, mode in (("real", "real_memory"), ("om", "oracle_memory"), ("oc", "oracle_context")):
                v = verdict(r, mode)
                if v is not None:
                    a[key][1] += 1
                    a[key][0] += 1 if v else 0
            a["fail"][str(r["modes"]["real_memory"].get("answer_failure_type") or "None")] += 1
        for qt in sorted(agg):
            a = agg[qt]
            def fmt(k):
                c, e = a[k]
                return f"{c}/{e}={c / e:.3f}" if e else "-"
            fails = ", ".join(f"{lbl}x{n}" for lbl, n in sorted(a["fail"].items()) if lbl != "None") or "-"
            print(f"  {qt:26s} n={a['n']:3d}  Real {fmt('real'):>12s}  OcMem {fmt('om'):>12s}  OcCtx {fmt('oc'):>12s}  | {fails}")

    # pilot comparison
    pilot_path = Path(args.pilot)
    if pilot_path.exists():
        pilot = load(pilot_path)
        print(f"\n=== SIDE-BY-SIDE vs STUDY 2B n=50 PILOT (in-house rule extractor, same questions) ===")
        print(f"  {'condition':16s} {'Mem0 (3A)':>16s} {'in-house (2B n50)':>18s}")
        for mode in ("real_memory", "oracle_memory", "oracle_context"):
            print(f"  {mode:16s} {acc(rows, mode):>16s} {acc(pilot, mode):>18s}")
        print("\n  real_memory stage recall:")
        m_stage = stage_recall(rows, "real_memory")
        p_stage = stage_recall(pilot, "real_memory")
        for s in STAGES:
            print(f"    {s:10s}: Mem0 {m_stage[s]:.3f}   in-house {p_stage[s]:.3f}")
        print("\n  real_memory failure labels:")
        p_counts = Counter(str(r["modes"]["real_memory"].get("answer_failure_type") or "None") for r in pilot)
        for label in sorted(set(counts) | set(p_counts)):
            print(f"    {label:28s}: Mem0 {counts.get(label, 0):3d}   in-house {p_counts.get(label, 0):3d}")
        m_stored = sum(int((r.get("extraction", {}).get("stats") or {}).get("facts", 0) or 0) for r in rows)
        print(f"\n  memories formed: Mem0 {m_stored} across {len(rows)} records "
              f"(in-house rule extractor: 335 items on the same 50)")


if __name__ == "__main__":
    main()
