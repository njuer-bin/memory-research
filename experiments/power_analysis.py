"""Power analysis / MDE for the exact McNemar primary contrasts (P0-3).

For the two paired primary comparisons of each study
(real_memory vs oracle_memory, real_memory vs oracle_context) this script:

1. recomputes the observed discordant pairs (b = only-real-wrong,
   c = only-oracle-wrong) and the exact two-sided McNemar p-value;
2. simulates the ACHIEVED POWER of the exact test at n=50 under the
   observed discordance structure (b ~ Bin(n, b/n), c ~ Bin(n, c/n),
   exact McNemar decision rule, alpha = 0.05 two-sided);
3. computes the MINIMUM DETECTABLE EFFECT (MDE): holding the observed
   false-fix rate c/n fixed, the smallest recovery rate b/n (and the
   corresponding accuracy difference) that reaches 80% power at n=50;
4. computes the sample size required to detect the OBSERVED effect at
   80% power.

No model calls; reads comparison.jsonl files only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.paired_stats import (  # noqa: E402
    MODES,
    accuracy,
    correct_flag,
    load_rows,
    mcnemar_exact,
)

ALPHA = 0.05
POWER_TARGET = 0.80
SIM_REPEATS = 20000
SEED = 20261019


def discordant(rows: list[dict[str, Any]], control: str, treated: str) -> tuple[int, int, int]:
    """Returns (b, c, n_decided) for the exact McNemar test."""
    b = c = n = 0
    for row in rows:
        x = correct_flag(row, control)
        y = correct_flag(row, treated)
        if x is None or y is None:
            continue
        n += 1
        if not x and y:
            b += 1
        elif x and not y:
            c += 1
    return b, c, n


def power_at(n: int, p10: float, p01: float, alpha: float = ALPHA,
             repeats: int = SIM_REPEATS, seed: int = SEED) -> float:
    """Simulated power of the exact McNemar test at sample size n.

    Discordant-cell counts are drawn independently from their cell
    probabilities (the standard McNemar power approximation: the
    concordant cells absorb the covariance between b and c).
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(repeats):
        b = sum(1 for _ in range(n) if rng.random() < p10)
        c = sum(1 for _ in range(n) if rng.random() < p01)
        if mcnemar_exact(b, c) <= alpha:
            hits += 1
    return hits / repeats


def required_n(p10: float, p01: float, target: float = POWER_TARGET,
               n_max: int = 400) -> int | None:
    """Smallest n (scanned upward) whose simulated power reaches target."""
    n = 20
    while n <= n_max:
        if power_at(n, p10, p01, repeats=8000, seed=SEED) >= target:
            return n
        n += 10
    return None


def mde_for(n: int, p01: float, target: float = POWER_TARGET) -> tuple[float, float]:
    """Smallest recovery rate p10 reaching `target` power at size n.

    Returns (p10, p10 - p01). Scans a fine grid from p01 upward.
    """
    step = 0.005
    p10 = p01 + step
    while p10 <= 0.95:
        if power_at(n, p10, p01, repeats=6000, seed=SEED) >= target:
            return p10, p10 - p01
        p10 += step
    return float("nan"), float("nan")


def analyze_study(label: str, path: Path) -> list[str]:
    rows = load_rows(path)
    out = [f"=== {label} ({path.name}, n={len(rows)}) ==="]
    for treated in ("oracle_memory", "oracle_context"):
        b, c, n = discordant(rows, "real_memory", treated)
        if n == 0:
            out.append(f"  real vs {treated}: no decided pairs")
            continue
        p = mcnemar_exact(b, c)
        rc, rd = accuracy([correct_flag(r, "real_memory") for r in rows])
        tc, td = accuracy([correct_flag(r, treated) for r in rows])
        delta = (tc / td if td else float("nan")) - (rc / rd if rd else float("nan"))
        out.append(
            f"  real vs {treated}: b={b} c={c} n={n} "
            f"acc {rc}/{rd}={rc/rd:.3f} vs {tc}/{td}={tc/td:.3f} "
            f"(delta={delta:+.3f}) exact p={p:.4f}"
        )
        p10, p01 = b / n, c / n
        power = power_at(n, p10, p01)
        out.append(
            f"    achieved power at n={n} under observed structure "
            f"(p10={p10:.3f}, p01={p01:.3f}): {power:.1%}"
        )
        mde_rate, mde_delta = mde_for(n, p01)
        out.append(
            f"    MDE at 80% power, n={n}, p01 fixed at {p01:.3f}: "
            f"p10>={mde_rate:.3f} (delta>={mde_delta:+.3f})"
        )
        req = required_n(p10, p01)
        out.append(
            f"    n required for 80% power at the observed effect "
            f"(p10={p10:.3f}, p01={p01:.3f}): {req if req else '>400'}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study2b", default="results/study2b_longmemeval_oracle_n50/comparison.jsonl")
    parser.add_argument("--study3a", default="results/study3a_mem0_n50/comparison.jsonl")
    parser.add_argument("--out", default="results/power_analysis.txt")
    args = parser.parse_args()

    lines: list[str] = []
    lines.append("Power analysis for exact McNemar primary contrasts (alpha=0.05, two-sided)")
    lines.append(f"Simulation: {SIM_REPEATS} replicates per point, seed={SEED}")
    lines.append("")
    for label, path in (
        ("Study 2B (in-house rule arm)", Path(args.study2b)),
        ("Study 3A (Mem0 external arm)", Path(args.study3a)),
    ):
        if path.exists():
            lines.extend(analyze_study(label, path))
            lines.append("")

    # Symmetric MDE reference: equal false-fix and recovery rates is the
    # classic McNemar grid; report one reference point for the paper.
    ref = mde_for(50, 0.02)
    lines.append(
        f"Reference: symmetric-structure MDE at n=50, p01=0.02 -> "
        f"p10>={ref[0]:.3f} (delta>={ref[1]:+.3f}) at 80% power"
    )

    report = "\n".join(lines)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
