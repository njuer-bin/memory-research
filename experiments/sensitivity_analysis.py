"""Sensitivity analyses for Study 2B (n=200) reviewer items P0-2 / P1-3.

Reviewer concerns addressed:
  P0-2  Differential exclusion of `uncertain` verdicts (real 37 / OM 61 / OC 66)
        inflates accuracy, and more so in oracle conditions. This script
        quantifies the worst/best-case bounds, the paired McNemar under each
        assignment, the missingness cross-tab, and a lexical profile of the
        uncertain rows (are they disproportionately wrong?).
  P1-3  The unreported oracle_memory vs oracle_context contrast inside 2B
        (13 points, n=200) is reported here as exact McNemar b/c/p on the
        pairwise-decided subset.

No model calls; reads comparison.jsonl only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

MODES = ("real_memory", "oracle_memory", "oracle_context")
PAIRS = (
    ("real_memory", "oracle_memory"),
    ("real_memory", "oracle_context"),
    ("oracle_memory", "oracle_context"),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("_status", "ok") == "ok":
                rows.append(row)
    return rows


def verdict(row: dict[str, Any], mode: str) -> bool | None:
    a = row["modes"].get(mode, {}).get("answer") or {}
    c = a.get("semantic_correct")
    return None if c is None else bool(c)


def lexical(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return row["modes"].get(mode, {}).get("answer") or {}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0

    def pmf(k: int, n: int, p: float) -> float:
        return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))

    k = min(b, c)
    tail = sum(pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def paired_accuracy(rows, a: str, b: str, fill: bool | None) -> tuple[int, int, int, int]:
    """Returns (correct_a, correct_b, n, discordant_b_only_a_wrong style b/c).

    fill: how to treat uncertain (None) verdicts. None = drop the pair
    (pairwise deletion); True = count as correct; False = count as wrong.
    """
    ca = cb = 0
    b_disc = c_disc = 0
    n = 0
    for r in rows:
        va, vb = verdict(r, a), verdict(r, b)
        if va is None or vb is None:
            if fill is None:
                continue
            va = fill if va is None else va
            vb = fill if vb is None else vb
        n += 1
        ca += 1 if va else 0
        cb += 1 if vb else 0
        if va and not vb:
            c_disc += 1
        elif vb and not va:
            b_disc += 1
    return ca, cb, n, (b_disc, c_disc)



def corrected_verdict(row: dict[str, Any], mode: str) -> bool | None:
    """Exact-match answers are certainly correct; the n=200 runner left them
    unjudged (semantic_correct=None). Count them True via the narrowest
    deterministic channel; the rest inherit the semantic verdict."""
    a = row["modes"].get(mode, {}).get("answer") or {}
    if a.get("exact_match"):
        return True
    sc = a.get("semantic_correct")
    return None if sc is None else bool(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/study2b_longmemeval_n200_deepseek/comparison.jsonl")
    ap.add_argument("--out", default="results/study2b_longmemeval_n200_deepseek/sensitivity.txt")
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    lines: list[str] = []
    log = lines.append
    log("=== STUDY 2B (n=200) SENSITIVITY ANALYSES ===")
    log(f"Loaded {len(rows)} rows from {args.input}")
    log("")

    # ---------------------------------------------------------------- headline
    log("=== 1. HEADLINE (protocol: uncertain excluded) ===")
    for m in MODES:
        vs = [verdict(r, m) for r in rows]
        decided = [v for v in vs if v is not None]
        unc = sum(1 for v in vs if v is None)
        acc = sum(1 for v in decided if v) / len(decided) if decided else float("nan")
        log(f"{m:16s} {sum(1 for v in decided if v):3d}/{len(decided):3d} = {acc:.3f}  (uncertain={unc}, excluded {unc/len(rows):.1%})")
    log("")

    # ------------------------------------------------- uncertain-as-wrong / -correct
    log("=== 2. BOUNDING SCENARIOS (all n=200 kept) ===")
    log("uncertain-as-WRONG (worst case) vs uncertain-as-CORRECT (best case):")
    for m in MODES:
        vs = [verdict(r, m) for r in rows]
        corr = sum(1 for v in vs if v is True)
        unc = sum(1 for v in vs if v is None)
        log(f"{m:16s} wrong-bounded {corr/len(rows):.3f}   correct-bounded {(corr+unc)/len(rows):.3f}")
    log("")
    log("Paired McNemar under each assignment (n=200 pairs):")
    for fill, name in ((False, "uncertain-as-WRONG"), (True, "uncertain-as-CORRECT")):
        log(f"  -- {name} --")
        for a, b in PAIRS:
            ca, cb, n, (bd, cd) = paired_accuracy(rows, a, b, fill)
            p = mcnemar_exact(bd, cd)
            log(f"  {a:16s} vs {b:16s}: acc {ca/n:.3f} vs {cb/n:.3f} (delta {cb/n-ca/n:+.3f}), "
                f"b={bd} c={cd}, exact p={p:.4f}")
    log("")

    # ----------------------------------------------------- missingness structure
    log("=== 3. MISSINGNESS CROSS-TAB (decidedness pattern) ===")
    pat: dict[tuple[bool, bool, bool], int] = {}
    for r in rows:
        key = tuple(verdict(r, m) is not None for m in MODES)  # type: ignore[assignment]
        pat[key] = pat.get(key, 0) + 1  # type: ignore[index]
    for key in sorted(pat, key=lambda k: -pat[k]):
        names = "/".join(("D" if k else "U") for k in key)
        log(f"  real/OM/OC = {names}: {pat[key]} rows")
    all_dec = [r for r in rows if all(verdict(r, m) is not None for m in MODES)]
    log(f"rows with all three conditions decided: {len(all_dec)}")
    if all_dec:
        accs = {m: sum(1 for r in all_dec if verdict(r, m)) / len(all_dec) for m in MODES}
        log("  accuracy on the full-decision intersection: "
            + ", ".join(f"{m}={accs[m]:.3f}" for m in MODES))
    log("")

    # ------------------------------------------------------- lexical profile
    log("=== 4. ARE UNCERTAIN ROWS DISPROPORTIONATELY WRONG? (lexical profile) ===")
    for m in MODES:
        und = [lexical(r, m) for r in rows if verdict(r, m) is None]
        dec = [lexical(r, m) for r in rows if verdict(r, m) is not None]
        def prof(items: list[dict[str, Any]]) -> str:
            if not items:
                return "n/a"
            f1 = sum(float(x.get("token_f1") or 0) for x in items) / len(items)
            em = sum(1 for x in items if x.get("exact_match")) / len(items)
            ph = sum(1 for x in items if x.get("phrase_match")) / len(items)
            return f"mean_token_f1={f1:.3f} exact_match={em:.1%} phrase_match={ph:.1%}"
        log(f"{m:16s} uncertain(n={len(und):3d}): {prof(und)}")
        log(f"{'':16s} decided  (n={len(dec):3d}): {prof(dec)}")
    log("")

    # ------------------------------------------------- OM vs OC (P1-3, decided subset)
    log("=== 5. ORACLE_MEMORY vs ORACLE_CONTEXT (reviewer P1-3) ===")
    ca, cb, n, (bd, cd) = paired_accuracy(rows, "oracle_memory", "oracle_context", None)
    log(f"pairwise-decided subset n={n}: OM {ca/n:.3f} vs OC {cb/n:.3f} (delta {cb/n-ca/n:+.3f}), "
        f"b={bd} (only OC wrong), c={cd} (only OM wrong), exact p={mcnemar_exact(bd, cd):.4f}")
    # paired bootstrap for this contrast (same seed as paired_stats.py)
    rng = random.Random(42)
    pairs = [(verdict(r, "oracle_memory"), verdict(r, "oracle_context")) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    diffs = []
    nn = len(pairs)
    for _ in range(10000):
        s = [pairs[rng.randrange(nn)] for _ in range(nn)]
        diffs.append(sum(1 for _, y in s if y) / nn - sum(1 for x, _ in s if x) / nn)
    diffs.sort()
    lo, hi = diffs[int(0.025 * 10000)], diffs[int(0.975 * 10000)]
    log(f"paired bootstrap 95% CI for OC-OM: [{lo:+.3f}, {hi:+.3f}]")
    log("")

    # ---------------------------------------------- headline movement estimate
    log("=== 6. HEADLINE MOVEMENT IF UNCERTAIN ARE WRONG (reviewer's estimate check) ===")
    for m in MODES:
        vs = [verdict(r, m) for r in rows]
        corr = sum(1 for v in vs if v is True)
        log(f"{m:16s} {corr/len(rows):.3f} (vs headline {sum(1 for v in vs if v is True)/max(1,sum(1 for v in vs if v is not None)):.3f})")
    log("")
    ca, cb, n, (bd, cd) = paired_accuracy(rows, "real_memory", "oracle_memory", False)
    log(f"real-vs-OM gap under uncertain-as-wrong: {cb/n - ca/n:+.3f} "
        f"(headline {96/139 - 85/163:+.3f} on decided subsets)")


    # ------------------------------------------- corrected verdicts (EM=1 fix)
    log("=== 7. CORRECTED VERDICTS (exact-match counted correct) ===")
    log("The n=200 runner left exact-match answers unjudged (semantic_correct")
    log("= None); these answers equal the gold string, so they are certainly")
    log("correct. Counting them True via a deterministic exact-match channel:")
    for m in MODES:
        vs = [corrected_verdict(r, m) for r in rows]
        dec = [v for v in vs if v is not None]
        corr = sum(1 for v in dec if v)
        log(f"{m:16s} {corr}/{len(dec)} = {corr/len(dec):.3f}  (undecided={len(vs)-len(dec)})")
    log("")
    log("Paired McNemar under corrected verdicts (pairwise-decided):")
    for a, b in PAIRS:
        ca = cb = bd = cd = 0
        n = 0
        for r in rows:
            va, vb = corrected_verdict(r, a), corrected_verdict(r, b)
            if va is None or vb is None:
                continue
            n += 1
            ca += 1 if va else 0
            cb += 1 if vb else 0
            if vb and not va:
                bd += 1
            elif va and not vb:
                cd += 1
        p = mcnemar_exact(bd, cd)
        log(f"  {a:16s} vs {b:16s}: acc {ca/n:.3f} vs {cb/n:.3f} (delta {cb/n-ca/n:+.3f}), "
            f"b={bd} c={cd}, exact p={p:.6f}")
    log("")
    log("EM=1 rows among the excluded (per condition):")
    for m in MODES:
        n_em1 = sum(1 for r in rows
                    if (r["modes"][m].get("answer") or {}).get("exact_match")
                    and (r["modes"][m].get("answer") or {}).get("semantic_correct") is None)
        n_none = sum(1 for r in rows
                     if (r["modes"][m].get("answer") or {}).get("semantic_correct") is None)
        log(f"  {m:16s} {n_em1} of {n_none} excluded rows are exact matches")
    log("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
