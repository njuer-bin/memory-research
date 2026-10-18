"""Temporal correctness metrics."""
def temporal_accuracy(predictions, references) -> float:
    p, r = list(predictions), list(references)
    if not r: return 1.0 if not p else 0.0
    return 0.0 if len(p) != len(r) else sum(a == b for a, b in zip(p, r)) / len(r)
