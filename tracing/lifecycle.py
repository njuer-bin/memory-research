"""Evidence matching across memory-lifecycle stages."""
from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    return re.sub(r"\s+", "", value)


def _text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or "")
    return str(item or "")


def _tokens(text: str) -> set[str]:
    # Character bigrams make the matcher useful for Chinese without requiring
    # a heavyweight tokenizer; word tokens are retained for Latin text.
    value = normalize_text(text)
    chars = {value[i:i + 2] for i in range(max(0, len(value) - 1))}
    words = set(re.findall(r"[a-z0-9_]+", value))
    return chars | words


def _semantic_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _source_match(evidence: dict[str, Any], artifact: Any) -> bool:
    """Match provenance without allowing session identity to override a turn id."""
    if not isinstance(artifact, dict):
        return False

    evidence_turn = evidence.get("turn_id") or evidence.get("source_turn_id")
    artifact_turn = artifact.get("turn_id") or artifact.get("source_turn_id")
    if evidence_turn:
        return bool(artifact_turn and str(evidence_turn) == str(artifact_turn))

    evidence_raw = evidence.get("source_raw_id") or evidence.get("raw_id")
    artifact_raw = artifact.get("source_raw_id") or artifact.get("raw_id") or artifact.get("id")
    if evidence_raw:
        return bool(artifact_raw and str(evidence_raw) == str(artifact_raw))

    source = evidence.get("source_id") or evidence.get("source")
    artifact_source = artifact.get("source_id") or artifact.get("session_id")
    return bool(source and artifact_source and str(source) == str(artifact_source))


def match_evidence(
    gold: Iterable[dict[str, Any]],
    observed: Iterable[Any],
    *,
    semantic_threshold: float = 0.55,
) -> dict[str, Any]:
    """Match gold evidence using exact, provenance, then semantic evidence.

    Exact text is strongest. Provenance/source matching is used when an
    artifact carries benchmark turn/session lineage. Semantic matching handles
    normal memory formation, where a fact such as "用户居住在杭州" is a
    faithful representation of a longer source turn.
    """
    observed_list = list(observed)
    found: list[str] = []
    missing: list[str] = []
    matches: dict[str, Any] = {}
    match_types: dict[str, str] = {}
    match_scores: dict[str, float] = {}

    for idx, evidence in enumerate(gold):
        evidence_id = str(evidence.get("evidence_id") or f"gold_{idx}")
        gold_text = str(evidence.get("text") or "")
        best = None
        best_type = None
        best_score = 0.0

        # 1. Provenance. For turn-level evidence this must be checked before
        # text matching, otherwise repeated text from another turn can be
        # incorrectly attributed to the gold turn.
        for artifact in observed_list:
            if _source_match(evidence, artifact):
                best, best_type, best_score = artifact, "provenance", 1.0
                break

        # 2. Exact text.
        norm_gold = normalize_text(gold_text)
        if best is None and norm_gold:
            for artifact in observed_list:
                if normalize_text(_text(artifact)) == norm_gold:
                    best, best_type, best_score = artifact, "exact", 1.0
                    break

        # 3. Semantic representation.
        if best is None and gold_text:
            for artifact in observed_list:
                score = _semantic_similarity(gold_text, _text(artifact))
                if score > best_score:
                    best, best_type, best_score = artifact, "semantic", score
            if best_score < semantic_threshold:
                best, best_type, best_score = None, None, 0.0

        if best is None:
            missing.append(evidence_id)
        else:
            found.append(evidence_id)
            matches[evidence_id] = best
            match_types[evidence_id] = best_type
            match_scores[evidence_id] = round(best_score, 4)

    return {
        "found": sorted(found),
        "missing": sorted(missing),
        "matches": matches,
        "match_types": match_types,
        "match_scores": match_scores,
        "observed_count": len(observed_list),
    }


def stage_from_observed(gold: Iterable[dict[str, Any]], observed: Iterable[Any]) -> dict[str, Any]:
    result = match_evidence(gold, observed)
    return {k: v for k, v in result.items() if k != "matches"}
