"""Four-family judge agreement report for Study 2E (revision round 4).

Reads fourth_judge_blind.json (which already carries qwen, gemma, deepseek
and author-truth columns attached post hoc) and reports:
1. pairwise raw agreement + Cohen's kappa across the four judge families
   on all subsample verdicts;
2. the 16 author-adjudicated disagreement cases: fourth-family vs
   adjudicated truth, and vs each other family;
3. per-mode sensitivity of each family on the subsample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAMILIES = ("qwen", "gemma", "deepseek", "mistral")
MODES = ("real_memory", "oracle_memory", "oracle_context")


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    yes_a = sum(1 for x, _ in pairs if x) / n
    yes_b = sum(1 for _, y in pairs if y) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def pairwise(verdicts: list[dict[str, Any]]) -> list[str]:
    out = ["=== PAIRWISE AGREEMENT (all subsample verdicts) ==="]
    for i, a in enumerate(FAMILIES):
        for b in FAMILIES[i + 1:]:
            pairs = []
            for v in verdicts:
                x, y = v.get(a), v.get(b)
                if isinstance(x, bool) and isinstance(y, bool):
                    pairs.append((x, y))
            if not pairs:
                out.append(f"{a} vs {b}: no overlapping verdicts")
                continue
            agree = sum(1 for x, y in pairs if x == y)
            k = kappa(pairs)
            out.append(
                f"{a:9s} vs {b:9s}: {agree}/{len(pairs)} = {agree/len(pairs):.3f}"
                + (f", kappa = {k:.3f}" if k is not None else ", kappa undefined")
            )
    return out


def confusion(verdicts: list[dict[str, Any]], a: str, b: str) -> dict[tuple[bool, bool], int]:
    counts: dict[tuple[bool, bool], int] = {}
    for v in verdicts:
        x, y = v.get(a), v.get(b)
        if isinstance(x, bool) and isinstance(y, bool):
            counts[(x, y)] = counts.get((x, y), 0) + 1
    return counts


def main() -> None:
    path = Path("results/study2e_judge_gemma/fourth_judge_blind.json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    verdicts = [v for v in artifact["verdicts"] if (v.get("fourth") or {}).get("correct") is not None]
    for v in verdicts:
        v["mistral"] = bool(v["fourth"]["correct"])

    lines: list[str] = []
    lines.append(f"Fourth-family blind judge report ({path.name})")
    lines.append(f"n verdicts with all-family data: {len(verdicts)}, errors: {artifact['errors']}")
    lines.append("")

    lines.extend(pairwise(verdicts))
    lines.append("")
    for a, b in (("qwen", "mistral"), ("gemma", "mistral"), ("deepseek", "mistral")):
        conf = confusion(verdicts, a, b)
        lines.append(f"confusion ({a}, mistral): {dict(sorted(conf.items()))}")
    lines.append("")

    # The 16 author-adjudicated disagreement cases
    adj_cases = [
        v for v in verdicts
        if v.get("author_truth") is not None or (v["qid"], v["mode"]) in {
            ("afdc33df", "real_memory"), ("afdc33df", "oracle_memory")  # borderline
        }
    ]
    lines.append(f"=== THE 16 AUTHOR-ADJUDICATED DISAGREEMENT CASES (n={len(adj_cases)}) ===")
    lines.append(f"{'qid':14s} {'mode':15s} {'qwen':>5s} {'gemma':>5s} {'deep':>5s} {'mist':>5s} {'truth':>10s}")
    for v in adj_cases:
        truth = v.get("author_truth")
        truth_s = {True: "correct", False: "incorrect", None: "borderline"}[truth] if truth is not None else "n/a"
        def cell(x):
            return {True: "T", False: "F", None: "-"}[x] if x is not None else "-"
        lines.append(
            f"{v['qid']:14s} {v['mode']:15s} {cell(v.get('qwen')):>5s} {cell(v.get('gemma')):>5s} "
            f"{cell(v.get('deepseek')):>5s} {cell(v.get('mistral')):>5s} {truth_s:>10s}"
        )
    decidable = [v for v in adj_cases if v.get("author_truth") is not None]
    m = [v for v in decidable if v.get("mistral") is not None]
    if m:
        hit = sum(1 for v in m if v["mistral"] == v["author_truth"])
        lines.append(f"mistral vs author truth (decidable n={len(m)}): {hit}/{len(m)}")
    for fam in ("qwen", "gemma", "deepseek"):
        f = [v for v in adj_cases if v.get(fam) is not None]
        if f:
            hit = sum(1 for v in f if v[fam] == v.get("author_truth"))
            lines.append(f"{fam} vs author truth (n={len(f)}): {hit}/{len(f)}")
    for fam in ("qwen", "gemma", "deepseek"):
        pairs = [(v["mistral"], v[fam]) for v in adj_cases if v.get(fam) is not None]
        if pairs:
            hit = sum(1 for x, y in pairs if x == y)
            lines.append(f"mistral vs {fam} (16-case set, n={len(pairs)}): {hit}/{len(pairs)}")
    lines.append("")

    lines.append("=== SENSITIVITY BY MODE (subsample) ===")
    for mode in MODES:
        rows = [v for v in verdicts if v["mode"] == mode]
        cells = []
        for fam in FAMILIES:
            vals = [v[fam] for v in rows if isinstance(v.get(fam), bool)]
            acc = sum(1 for x in vals if x) / len(vals) if vals else float("nan")
            cells.append(f"{fam}={sum(1 for x in vals if x)}/{len(vals)}={acc:.3f}")
        lines.append(f"{mode:15s} " + "  ".join(cells))

    report = "\n".join(lines)
    print(report)
    Path("results/study2e_judge_gemma/fourth_judge_report.txt").write_text(report + "\n", encoding="utf-8")
    print("\nWrote results/study2e_judge_gemma/fourth_judge_report.txt")


if __name__ == "__main__":
    main()
