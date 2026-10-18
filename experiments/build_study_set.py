"""Build a stratified study subset from a dataset.

Stratification is two-level: the primary stratum is the conversation (or
session group) so that no single conversation dominates the sample and
within-conversation correlation is spread across the study; the secondary
stratum is the benchmark question/task type so the type mix matches the
population. Allocation uses the largest-remainder method with a
one-per-present-type minimum; selection inside each stratum is a seeded
shuffle, so rebuilding with the same seed reproduces the same subset.

Two storage modes:

- ``--raw`` (default for LoCoMo): the subset stores the benchmark's own
  question records (question, answer, evidence messages) as jsonl, and the
  loader joins the conversations from the full local dataset at read time.
  The committed subset stays small and carries no full conversation text.
- normalized mode: the subset stores already-normalized records (self-
  contained, used when each record carries its own haystack, e.g. LongMemEval).

Usage:
    python -m experiments.build_study_set --dataset locomo_refined_public \
        --n 50 --seed 20261018 --out locomo_n50 --raw
    python -m experiments.build_study_set --dataset longmemeval_oracle \
        --n 50 --seed 20261018 --out longmemeval_oracle_n50 \
        --primary question_type
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from datasets.loader import iter_normalized, iter_raw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "data"
MANIFEST = ROOT / "datasets" / "manifest.json"

FAMILIES = {
    "locomo_refined_public": "LoCoMo-Refined",
    "longmemeval_oracle": "LongMemEval",
    "longmemeval_s_sample10": "LongMemEval",
}


def largest_remainder(shares: dict[str, float], n: int) -> dict[str, int]:
    """Integer allocation summing to n, proportional to shares."""
    total = sum(shares.values())
    if total <= 0:
        return {k: 0 for k in shares}
    exact = {k: v / total * n for k, v in shares.items()}
    base = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:remainder]:
        base[k] += 1
    return base


def allocate_with_minimum(
    by_type: dict[str, list[Any]], alloc: int
) -> dict[str, int]:
    """Proportional allocation, but every non-empty stratum keeps >= 1 slot
    when the conversation's allocation allows it. Without this, small
    question types (e.g. LoCoMo temporal reasoning, ~5% of the population)
    are routinely sampled at zero even in conversations that have them.
    """
    shares = {k: float(len(v)) for k, v in by_type.items() if v}
    result = largest_remainder(shares, alloc)
    non_empty = [t for t, rows in by_type.items() if rows]
    if alloc >= len(non_empty):
        for t in non_empty:
            if result.get(t, 0) == 0:
                donor = max(result, key=lambda k: (result[k], k))
                if result[donor] > 1:
                    result[donor] -= 1
                    result[t] = 1
    return result


def conversation_of(row: dict[str, Any]) -> str:
    qid = str(row.get("qa_id") or row.get("question_id") or "")
    if "#" in qid:
        return qid.split("#", 1)[0]
    for key in ("sample_id", "conversation_id", "conversationId"):
        if row.get(key):
            return str(row[key])
    return qid


def type_of(row: dict[str, Any]) -> str:
    """Benchmark question/task type of a record (raw or normalized).

    LongMemEval carries it at ``metadata.raw.question_type``; LoCoMo exposes
    ``category`` (raw) or ``task_type`` (normalized). ``task_type`` is checked
    last because the generic normalizer fills it with dataset *capabilities*,
    which are not question types.
    """
    raw = (row.get("metadata") or {}).get("raw") or {}
    value = (
        row.get("question_type")
        or raw.get("question_type")
        or row.get("category")
        or row.get("task_type")
        or "NA"
    )
    if isinstance(value, list):
        value = value[0] if value else "NA"
    return str(value)


def build(
    dataset: str, n: int, seed: int, primary: str, raw: bool
) -> tuple[list[dict], dict]:
    rows = list(iter_raw(dataset)) if raw else list(iter_normalized(dataset))

    if primary == "question_type":
        # Datasets whose records each carry their own haystack (LongMemEval)
        # have no meaningful conversation grouping; the question type is the
        # natural primary stratum there.
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            groups[type_of(row)].append(row)
        group_label = "question_type"
    else:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[conversation_of(row)].append(row)
        group_label = "conversation"

    group_alloc = largest_remainder(
        {k: float(len(v)) for k, v in groups.items()}, n
    )

    rng = random.Random(seed)
    selected: list[dict] = []
    strata_plan: dict[str, Any] = {}
    for group, alloc in sorted(group_alloc.items()):
        pool = list(groups[group])
        rng.shuffle(pool)
        if primary == "question_type":
            selected.extend(pool[:alloc])
            strata_plan[group] = {"_total": alloc}
        else:
            by_type: dict[str, list[dict]] = collections.defaultdict(list)
            for row in pool:
                by_type[type_of(row)].append(row)
            type_alloc = allocate_with_minimum(by_type, alloc)
            strata_plan[group] = dict(sorted(type_alloc.items()))
            for t, take in sorted(type_alloc.items()):
                selected.extend(by_type[t][:take])

    if primary == "question_type":
        method = (
            "stratified proportional sample; primary stratum question type "
            "(records each carry their own haystack, so conversation grouping "
            "is not meaningful); seeded random selection within stratum"
        )
    else:
        method = (
            "stratified proportional sample; primary stratum conversation, "
            "secondary stratum question/task type; largest-remainder allocation "
            "with a minimum of one question per type present in a conversation "
            "when the allocation allows"
        )
    recipe = {
        "method": method,
        "source_dataset": dataset,
        "storage": "raw benchmark question records (conversations joined at load time)"
        if raw
        else "normalized records (self-contained)",
        "population": len(rows),
        "n": len(selected),
        "seed": seed,
        "primary_stratum": group_label,
        "group_allocation": {k: group_alloc[k] for k in sorted(group_alloc)},
        "strata": strata_plan,
    }
    return selected, recipe


def register(out_name: str, recipe: dict, *, raw: bool) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [d for d in manifest["datasets"] if d.get("id") != out_name]
    entry: dict[str, Any] = {
        "id": out_name,
        "family": FAMILIES.get(recipe["source_dataset"], "study_subset"),
        "status": "ready",
        "capabilities": ["study_subset"],
        "sampling": recipe,
    }
    if raw:
        entry["source"] = f"datasets/data/{out_name}.jsonl"
        entry["format"] = "jsonl"
        entry["conversations"] = (
            "datasets/data/locomo_refined_dataset/public/conversations.jsonl"
        )
        entry["provenance_note"] = (
            "Question-level subset; full conversations are joined from the "
            "local LoCoMo-Refined copy at load time (verify redistribution "
            "terms before publishing derived data)."
        )
    else:
        entry["source"] = f"datasets/data/{out_name}.json"
        entry["format"] = "json"
        entry["pre_normalized"] = True
    entries.append(entry)
    manifest["datasets"] = entries
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="source dataset id")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--out", required=True, help="study set name under datasets/data/")
    parser.add_argument(
        "--primary",
        choices=("conversation", "question_type"),
        default="conversation",
        help="primary stratification stratum",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="store raw benchmark question records as jsonl instead of "
        "normalized records (conversations are joined at load time)",
    )
    args = parser.parse_args()

    selected, recipe = build(
        args.dataset, args.n, args.seed, primary=args.primary, raw=args.raw
    )

    if args.raw:
        out_path = DATA / f"{args.out}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in selected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stale = DATA / f"{args.out}.json"
        if stale.exists():
            stale.unlink()
    else:
        out_path = DATA / f"{args.out}.json"
        out_path.write_text(
            json.dumps(selected, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    register(args.out, recipe, raw=args.raw)

    print(f"Wrote {len(selected)} records -> {out_path}")
    for group, alloc in recipe["group_allocation"].items():
        print(f"  {group}: {alloc}")


if __name__ == "__main__":
    main()
