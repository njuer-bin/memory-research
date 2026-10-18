# Experiments
Planned tracks:
- real_system: normal conversation -> memory -> retrieval -> context -> answer.
- oracle_memory: inject gold evidence into memory to isolate downstream loss.
- oracle_context: provide gold evidence directly to the answer model.
- ablation: remove temporal state, provenance, reliability, conflict resolution, or entity relations.
- baselines: vector RAG, BM25, hybrid retrieval, and compatible memory baselines.

Every experiment should emit a machine-readable trace so failure attribution is reproducible.
