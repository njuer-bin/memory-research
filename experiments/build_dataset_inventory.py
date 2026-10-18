"""Build unified dataset inventories and normalized samples.

Usage:
    python -m experiments.build_dataset_inventory
    python -m experiments.build_dataset_inventory --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.loader import iter_normalized, load_manifest
from datasets.validate import inventory


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dataset_inventory"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def build(limit: int | None = None) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = load_manifest()["datasets"]
    report: dict[str, Any] = {"limit": limit, "datasets": []}

    for item in datasets:
        dataset_id = item["id"]
        entry: dict[str, Any] = {"id": dataset_id, "family": item["family"]}
        try:
            entry["inventory"] = inventory(dataset_id, limit=limit)
            samples = []
            for record in iter_normalized(dataset_id, limit=min(limit or 5, 5)):
                samples.append({
                    "example_id": record.get("example_id"),
                    "question_id": record.get("question_id"),
                    "conversation_id": record.get("conversation_id"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "gold_evidence": record.get("gold_evidence", []),
                    "task_type": record.get("task_type", []),
                })
            entry["sample"] = samples
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["datasets"].append(entry)

    path = RESULTS / "inventory.json"
    path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = build(args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
