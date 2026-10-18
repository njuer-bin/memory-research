"""Provider-independent answer metrics."""
def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())

def token_f1(prediction: str, reference: str) -> float:
    pred, ref = prediction.strip().split(), reference.strip().split()
    if not pred and not ref: return 1.0
    if not pred or not ref: return 0.0
    counts = {}
    for t in pred: counts[t] = counts.get(t, 0) + 1
    overlap, used = 0, {}
    for t in ref:
        if counts.get(t, 0) > used.get(t, 0):
            overlap += 1; used[t] = used.get(t, 0) + 1
    if not overlap: return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)
