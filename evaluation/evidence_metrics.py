"""Evidence-level metrics."""
from typing import Iterable, Set

def evidence_recall(gold: Iterable[str], observed: Iterable[str]) -> float:
    g, o = set(gold), set(observed)
    return len(g & o) / len(g) if g else 1.0

def evidence_precision(gold: Iterable[str], observed: Iterable[str]) -> float:
    g, o = set(gold), set(observed)
    return len(g & o) / len(o) if o else (1.0 if not g else 0.0)

def complete_evidence(gold: Iterable[str], observed: Iterable[str]) -> float:
    return float(set(gold).issubset(set(observed)))
