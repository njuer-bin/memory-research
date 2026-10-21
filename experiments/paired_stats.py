"""Paired statistics for the three-condition comparison.

Compares real_memory vs oracle_memory vs oracle_context on the SAME records
(within-subject design) using:
- exact McNemar test (binomial on discordant pairs);
- paired bootstrap percentile CIs for accuracy differences;
- failure-label distribution per condition;
- optional judge-agreement mode comparing two judge runs on shared records
  (Cohen's kappa) for the judge-reliability study.

No model calls are made; it only reads comparison.jsonl files.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

# Allow both `python -m experiments.paired_stats` and
# `python experiments/paired_stats.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODES = ("real_memory", "oracle_memory", "oracle_context")
FAILURE_LABELS = (
    "F1_FORMATION",
    "F2_STORAGE",
    "F3_EVOLUTION",
    "F4_RETRIEVAL",
    "F5_RERANKING",
    "F6_REASONING_CANDIDATE",
    "E0_EVIDENCE_INSUFFICIENT",
    "EVAL_SEMANTIC_UNCERTAIN",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("_status", "ok") == "ok":
                rows.append(row)
    return rows


def answer(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return row["modes"].get(mode, {}).get("answer") or {}


def correct_flag(row: dict[str, Any], mode: str) -> bool | None:
    """Semantic verdict only; lexical metrics are never attribution input.

    The answer dict carries two flags: ``correct`` is the lexical verdict
    (EM/F1/temporal/phrase) reported for reproducibility only, and
    ``semantic_correct`` is the judge verdict that feeds attribution. This
    function deliberately reads the latter.
    """
    a = answer(row, mode)
    c = a.get("semantic_correct")
    return None if c is None else bool(c)


def paired_binary(rows: list[dict[str, Any]], mode: str) -> list[bool | None]:
    return [correct_flag(r, mode) for r in rows]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0

    def binom_pmf(k: int, n: int, p: float) -> float:
        return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))

    k = min(b, c)
    tail = sum(binom_pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def accuracy(values: list[bool | None]) -> tuple[int, int]:
    """Returns (correct, decided) — uncertain rows are excluded from the rate."""
    decided = [v for v in values if v is not None]
    return sum(1 for v in decided if v), len(decided)


def paired_bootstrap_diff(
    a: list[bool | None],
    b: list[bool | None],
    repeats: int = 10000,
    seed: int = 42,
) -> tuple[float | None, float | None, float | None]:
    """Percentile CI for acc(b) - acc(a) over paired record resampling."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None, None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(repeats):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        acc_a = sum(1 for x, _ in sample if x) / n
        acc_b = sum(1 for _, y in sample if y) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    lo = diffs[max(0, int(0.025 * repeats))]
    hi = diffs[min(repeats - 1, int(0.975 * repeats))]
    point = (
        sum(1 for _, y in pairs if y) / n - sum(1 for x, _ in pairs if x) / n
    )
    return point, lo, hi


def build_conversation_map(dataset: str) -> dict[str, str]:
    """Map question_id -> conversation_id for a normalized study dataset.

    Used for cluster-robust bootstrap resampling (reviewer P4): questions
    inside one conversation are correlated, so the resampling unit must be
    the conversation, not the question.
    """
    from datasets.loader import iter_normalized

    mapping: dict[str, str] = {}
    for r in iter_normalized(dataset):
        q = r.get("question_id")
        c = (
            r.get("conversation_id")
            or r.get("sample_id")
            or r.get("cid")
            or r.get("conversation")
        )
        if q is not None:
            mapping[str(q)] = str(c) if c is not None else "NA"
    return mapping


def paired_bootstrap_diff_clustered(
    a: list[bool | None],
    b: list[bool | None],
    qids: list[str],
    cid_of: dict[str, str],
    repeats: int = 10000,
    seed: int = 42,
) -> tuple[float | None, float | None, float | None]:
    """Cluster-robust percentile CI for acc(b) - acc(a).

    Resamples whole conversations (clusters) with replacement; all
    within-cluster question pairs are carried together, so intra-conversation
    correlation is respected (unlike paired_bootstrap_diff, which resamples
    questions independently).
    """
    triples = [
        (x, y, cid_of.get(q))
        for x, y, q in zip(a, b, qids)
        if x is not None and y is not None
    ]
    clusters: dict[str, list[tuple[bool, bool]]] = {}
    for x, y, c in triples:
        clusters.setdefault(c, []).append((x, y))
    cluster_ids = list(clusters.keys())
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return None, None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(repeats):
        chosen = [rng.choice(cluster_ids) for _ in range(n_clusters)]
        xs, ys = [], []
        for cid in chosen:
            for x, y in clusters[cid]:
                xs.append(x)
                ys.append(y)
        if not xs:
            continue
        diffs.append(sum(ys) / len(ys) - sum(xs) / len(xs))
    if not diffs:
        return None, None, None
    diffs.sort()
    lo = diffs[max(0, int(0.025 * repeats))]
    hi = diffs[min(repeats - 1, int(0.975 * repeats))]
    point = sum(y for _, y, _ in triples) / len(triples) - sum(
        x for x, _, _ in triples
    ) / len(triples)
    return point, lo, hi


def failure_distribution(rows: list[dict[str, Any]], mode: str) -> dict[str, int]:
    dist = {label: 0 for label in FAILURE_LABELS}
    for row in rows:
        label = row["modes"].get(mode, {}).get("answer_failure_type")
        if label is None:
            continue
        dist[label] = dist.get(label, 0) + 1
    return dist


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
            for x in (row["modes"].get(mode, {}).get(stage) or {}).get("found", [])
        }
        result[stage] = len(gold & found) / len(gold) if gold else 1.0
    return result


def mean_stage_recall(rows: list[dict[str, Any]], mode: str) -> dict[str, float]:
    stages = ("formation", "storage", "evolution", "retrieval", "rerank", "context")
    totals = {s: 0.0 for s in stages}
    count = 0
    for row in rows:
        recall = stage_recall(row, mode)
        for s in stages:
            totals[s] += recall[s]
        count += 1
    return {s: totals[s] / count for s in stages} if count else totals


def type_of(row: dict[str, Any]) -> str:
    # LongMemEval: question_type; LoCoMo: category (raw). Same key order as
    # build_study_set.type_of so study sets and stats agree.
    raw = (row.get("metadata") or {}).get("raw") or {}
    value = (
        row.get("question_type")
        or raw.get("question_type")
        or row.get("category")
        or raw.get("category")
        or row.get("task_type")
        or "NA"
    )
    if isinstance(value, list):
        value = value[0] if value else "NA"
    return str(value)


def print_type_breakdown(rows: list[dict[str, Any]], types: dict[str, str]) -> None:
    """Per-question-type table (paper Table: accuracy + failure mix)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(types.get(row["question_id"], "NA"), []).append(row)
    print("\n=== BREAKDOWN BY QUESTION TYPE ===")
    header = f"{'type':28s} {'n':>3s} {'real':>6s} {'oracleM':>7s} {'oracleC':>7s}  failures"
    print(header)
    for t in sorted(groups):
        group = groups[t]
        cells = []
        for mode in MODES:
            flags = [correct_flag(r, mode) for r in group]
            decided = [f for f in flags if f is not None]
            acc = sum(1 for f in decided if f) / len(decided) if decided else float("nan")
            cells.append(f"{acc:.3f}")
        fails: dict[str, int] = {}
        for r in group:
            label = r["modes"]["real_memory"].get("answer_failure_type")
            if label:
                fails[label] = fails.get(label, 0) + 1
        fail_str = ", ".join(f"{k}x{v}" for k, v in sorted(fails.items())) or "-"
        print(f"{t:28s} {len(group):3d} {cells[0]:>6s} {cells[1]:>7s} {cells[2]:>7s}  {fail_str}")


def cohens_kappa(a: list[bool | None], b: list[bool | None]) -> tuple[float | None, int]:
    """Kappa on rows where both judges returned a verdict."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None, 0
    observed = sum(1 for x, y in pairs if x == y) / n
    yes_a = sum(1 for x, _ in pairs if x) / n
    yes_b = sum(1 for _, y in pairs if y) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if expected == 1.0:
        return None, n
    return (observed - expected) / (1 - expected), n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="comparison.jsonl of a study")
    parser.add_argument(
        "--judge-input",
        default=None,
        help="optional second comparison.jsonl with a different judge model, "
        "aligned by question_id, for the judge-agreement study",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="optional dataset id to join question types for the per-type "
        "breakdown (e.g. longmemeval_oracle_n50, locomo_n50)",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument(
        "--cluster-by-conversation",
        action="store_true",
        help="resample whole conversations (clusters) instead of questions for "
        "the paired bootstrap CI; requires --dataset to map question_id->conversation",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    print(f"Loaded {len(rows)} successful rows from {args.input}")
    if not rows:
        return

    print("\n=== CONDITION ACCURACY (semantic verdict) ===")
    verdicts = {mode: paired_binary(rows, mode) for mode in MODES}
    for mode in MODES:
        correct, decided = accuracy(verdicts[mode])
        rate = correct / decided if decided else float("nan")
        uncertain = sum(1 for v in verdicts[mode] if v is None)
        print(
            f"{mode:16s} {correct}/{decided} = {rate:.3f} "
            f"(uncertain={uncertain})"
        )

    print("\n=== PAIRED TESTS (exact McNemar + paired bootstrap) ===")
    for treatment in ("oracle_memory", "oracle_context"):
        control = verdicts["real_memory"]
        treated = verdicts[treatment]
        decided_pairs = [
            (c, t) for c, t in zip(control, treated)
            if c is not None and t is not None
        ]
        b = sum(1 for c, t in decided_pairs if not c and t)  # only real wrong
        c = sum(1 for c, t in decided_pairs if c and not t)  # only oracle wrong
        p = mcnemar_exact(b, c)
        point, lo, hi = paired_bootstrap_diff(
            control, treated, repeats=args.bootstrap_repeats
        )
        print(
            f"real_memory vs {treatment}: "
            f"discordant b={b} (only real wrong) c={c} (only oracle wrong), "
            f"exact p={p:.4f}"
        )
        if point is not None:
            print(
                f"  delta_acc = {point:+.3f}, "
                f"paired bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]"
            )
        if args.cluster_by_conversation and args.dataset:
            qids = [r["question_id"] for r in rows]
            cid_of = build_conversation_map(args.dataset)
            n_conv = len({cid_of.get(q) for q in qids})
            print(f"  conversations (clusters) = {n_conv} over {len(qids)} questions")
            cpoint, clo, chi = paired_bootstrap_diff_clustered(
                control, treated, qids, cid_of, repeats=args.bootstrap_repeats
            )
            if cpoint is not None:
                print(
                    f"  cluster-robust 95% CI (by conversation) = "
                    f"[{clo:+.3f}, {chi:+.3f}]"
                )

    print("\n=== FAILURE DISTRIBUTION (real_memory) ===")
    dist = failure_distribution(rows, "real_memory")
    total = sum(dist.values())
    for label, count in dist.items():
        if count:
            print(f"{label:28s} {count:3d} ({count / total:.1%})")
    print(f"{'(no failure / correct)':28s} {len(rows) - total:3d}")

    print("\n=== MEAN STAGE RECALL (gold evidence found) ===")
    for mode in MODES:
        recall = mean_stage_recall(rows, mode)
        print(
            f"{mode:16s} "
            + " ".join(f"{s}={recall[s]:.3f}" for s in recall)
        )

    if args.dataset:
        from datasets.loader import iter_normalized

        types = {r["question_id"]: type_of(r) for r in iter_normalized(args.dataset)}
        print_type_breakdown(rows, types)

    if args.judge_input:
        print("\n=== JUDGE AGREEMENT ===")
        alt_rows = {
            r["question_id"]: r for r in load_rows(Path(args.judge_input))
        }
        primary: list[bool | None] = []
        alternate: list[bool | None] = []
        for row in rows:
            alt = alt_rows.get(row["question_id"])
            if alt is None:
                continue
            for mode in MODES:
                primary.append(correct_flag(row, mode))
                alternate.append(correct_flag(alt, mode))
        pairs = [
            (x, y) for x, y in zip(primary, alternate)
            if x is not None and y is not None
        ]
        kappa, n = cohens_kappa(primary, alternate)
        agree = sum(1 for x, y in pairs if x == y)
        print(f"n paired verdicts = {n}")
        if pairs:
            print(f"raw agreement = {agree}/{len(pairs)} ({agree / len(pairs):.3f})")
        else:
            print("no overlapping decided verdicts")
        if kappa is not None:
            print(f"Cohen's kappa = {kappa:.3f}")
        else:
            print("Cohen's kappa = undefined (degenerate marginals)")


if __name__ == "__main__":
    main()
