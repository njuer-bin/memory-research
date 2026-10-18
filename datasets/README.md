# Unified datasets layer

This directory is the single entry point for benchmark data used by the memory-failure-attribution project.

## Registry

manifest.json records every dataset currently checked into the repository, its format, capabilities, and loading status.

Current families:
- LoCoMo-Refined — 10 conversations / 1,382 questions in the public split.
- LongMemEval — sample and oracle variants for long-term memory evaluation.
- BEAM — 100K parquet benchmark plus a checked-in preview.
- Local sample — small smoke-test conversation.

## Unified record

Every adapter targets schema.json:

question -> gold answer -> gold evidence -> conversation -> task type -> metadata

Raw benchmark fields are preserved under metadata.raw, so benchmark-specific information is not lost.

## Loading

    from datasets.loader import iter_normalized
    for record in iter_normalized("locomo_refined_public", limit=10):
        print(record["example_id"], record["question"])

BEAM parquet is loaded lazily and requires pyarrow.

## Research role

This layer separates benchmark data from failure attribution. Once a record is normalized, the same tracing/evaluation code can be used across LoCoMo, LongMemEval, BEAM, and future datasets without changing the metrics.

## Data policy

The repository may contain benchmark files added by the project owner. Before publishing or redistributing a benchmark, verify its license and redistribution terms. For restricted datasets, keep only manifests, adapters, download instructions, and derived statistics.
