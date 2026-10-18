"""Aggregate failure-attribution metrics."""
from collections import Counter
from typing import Dict, Iterable

def failure_distribution(labels: Iterable[str]) -> Dict[str, float]:
    labels = list(labels)
    counts, total = Counter(labels), len(labels)
    return {k: v / total for k, v in counts.items()} if total else {}

def first_loss_accuracy(predicted: Iterable[str], gold: Iterable[str]) -> float:
    p, g = list(predicted), list(gold)
    return 1.0 if not g and not p else (0.0 if len(p) != len(g) else sum(a == b for a, b in zip(p, g)) / len(g))
