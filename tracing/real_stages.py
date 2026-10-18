"""Shared real-memory lifecycle stage measurement.

Both the oracle experiment runner and the diagnostic lifecycle probe measure
the Real Memory condition through this single implementation. Stage
definitions follow docs/PROTOCOL.md:

- formation: everything the extractor produced (incl. later-superseded facts)
- storage:   raw messages + formation
- evolution: post-governance current belief state (active facts + the rest)
- retrieval / rerank / context: from the engine search trace

Evidence matching goes through tracing.engine_trace.stage_match, which
validates provenance matches against content preservation (audit C3).
"""
from __future__ import annotations

from typing import Any

from tracing.engine_trace import artifacts_from_store, stage_match


def active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that survive governance (facts can be superseded; nothing else can)."""
    return [r for r in rows if str(r.get("status") or "active") == "active"]


def semantic_memories(stored: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return (
        stored["facts"]
        + stored["relations"]
        + stored["events"]
        + stored["rules"]
        + stored["profiles"]
    )


def stage_artifact_sets(
    stored: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Formation / storage / evolution artifact lists for a stored memory set."""
    formation = semantic_memories(stored)
    return {
        "formation": formation,
        "storage": stored["raw"] + formation,
        "evolution": active_rows(stored["facts"])
        + stored["relations"]
        + stored["events"]
        + stored["rules"]
        + stored["profiles"],
    }


def stage_dict(
    gold: list[dict[str, Any]], artifacts: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    match = stage_match(gold, artifacts)
    return {
        "found": match["found"],
        "missing": match["missing"],
        "details": {
            "mode": mode,
            "observed_count": match["observed_count"],
            "match_types": match.get("match_types", {}),
            "match_scores": match.get("match_scores", {}),
            "provenance_only": match.get("provenance_only", []),
            "matches": match["matches"],
        },
    }


def fill_real_stages(
    trace: Any,
    gold: list[dict[str, Any]],
    engine: Any,
    user_id: str,
    real: list[dict[str, Any]],
    search_trace: dict[str, Any],
    *,
    formation_diagnostics: Any = None,
) -> None:
    """Fill a MemoryTrace's six lifecycle stages for the Real Memory condition."""
    stored = artifacts_from_store(engine, user_id)
    sets = stage_artifact_sets(stored)

    trace.formation = type(trace.formation)(
        **stage_dict(gold, sets["formation"], "real_memory")
    )
    if formation_diagnostics is not None:
        trace.formation.details["diagnostics"] = formation_diagnostics(gold, stored)
    trace.formation.details["semantic_memory_counts"] = {
        "total": len(sets["formation"]),
        "by_type": {
            "fact": len(stored["facts"]),
            "relation": len(stored["relations"]),
            "event": len(stored["events"]),
            "rule": len(stored["rules"]),
            "profile": len(stored["profiles"]),
        },
    }
    trace.storage = type(trace.storage)(
        **stage_dict(gold, sets["storage"], "real_memory")
    )
    trace.evolution = type(trace.evolution)(
        **stage_dict(gold, sets["evolution"], "real_memory")
    )
    trace.evolution.details["superseded_facts"] = (
        len(stored["facts"]) - len(active_rows(stored["facts"]))
    )
    trace.evolution.details["measured"] = True

    retrieval = search_trace.get("retrieval_candidates") or real
    rerank = search_trace.get("reranked") or real
    context = search_trace.get("context") or real
    trace.retrieval = type(trace.retrieval)(
        **stage_dict(gold, retrieval, "real_memory_retrieval_candidates")
    )
    trace.rerank = type(trace.rerank)(
        **stage_dict(gold, rerank, "real_memory_reranked")
    )
    trace.context = type(trace.context)(
        **stage_dict(gold, context, "real_memory_context")
    )
