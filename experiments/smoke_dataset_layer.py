"""Run lightweight dataset-layer smoke checks and emit trace-ready samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.loader import iter_normalized, load_manifest
from datasets.validate import validate_manifest
from tracing.dataset_trace import record_to_trace


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dataset_inventory" / "trace_samples.json"


def run(limit: int = 3) -> dict:
    errors = validate_manifest()
    result = {"manifest_errors": errors, "datasets": []}

    for item in load_manifest()["datasets"]:
        dataset_id = item["id"]
        entry = {"id": dataset_id, "family": item["family"], "status": "ok", "samples": []}
        try:
            for record in iter_normalized(dataset_id, limit=limit):
                trace = record_to_trace(record)
                entry["samples"].append(
                    {
                        "example_id": record.get("example_id"),
                        "question_id": trace.question_id,
                        "question": trace.question,
                        "gold_evidence_count": len(trace.gold_evidence),
                        "gold_evidence_ids": sorted(trace.gold_ids()),
                        "first_loss_stage_before_pipeline": trace.first_loss_stage(),
                    }
                )
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        result["datasets"].append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
