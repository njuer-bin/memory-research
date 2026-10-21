#!/usr/bin/env python3
"""Retroactive L3 scoring of surviving Stage-A runs (2026-10-21, review-driven).

Scores ONLY the runs whose result JSONs survive (deepseek 2x4 under tag r0;
qwen stability reruns under tag r1). Ladder runs whose JSONs were overwritten
by later variants are NOT scored here -- their qualitative evidence lives in
results/kwm_stage_a/logs/*.log and the harness-change record.

Scoring rules are taken verbatim from the frozen task-set doc
(docs/KWM-TASKSET-V1-2026-10-21.md, section 3.6 -- family A):

  C-A1 (w=0.4): |Delta_est - 2.1| <= 0.3        (absolute, pp)
  C-A2 (w=0.2): CI width in [0.9, 3.5]          (absent CI -> 0)
  C-A3 (w=0.2): semantic judge -- NOT BUILT -> unjudgeable (excluded)
  C-A4 (w=0.2): tool calls <= 10
  composite = 0.4*s1 + 0.2*s2 + 0.2*s3 + 0.2*s4 ; pass if >= 0.6

Because C-A3 is unjudgeable we report the deterministic subtotal and the
composite bounds [subtotal, subtotal + 0.2], with verdicts:
  pass_guaranteed   (lower bound >= 0.6)
  fail_guaranteed   (upper bound <  0.6)
  unjudgeable       (otherwise)

Provenance (model attribution) is external to the JSONs -- the runner did not
record the model (defect, recorded in the erratum). Attribution:
  r0 tags -> DeepSeek-V4.1-Flash  (results/kwm_stage_a/logs/kwm_cloud.log)
  r1 tags -> Qwen2.5-7B           (results/kwm_stage_a/logs/kwm_stab.log)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "kwm_stage_a"

GOLD_DELTA = 2.1          # family A gold (frozen)
TOL = 0.3                 # C-A1 tolerance (frozen, absolute pp)
CI_LO, CI_HI = 0.9, 3.5   # C-A2 pre-registered CI width band
MAX_CALLS = 10            # C-A4

DELTA_RE = re.compile(r"delta=([-\d.]+)", re.I)
DELTA_RE2 = re.compile(r"DELTA\s+([-\d.]+)")
CI_RE = re.compile(r"CI95\s*\[([-\d.]+),\s*([-\d.]+)\]")


def score_family_a(run: dict) -> dict:
    est = None
    ci = None
    crashed = False
    for tc in run.get("tool_calls") or []:
        res = str(tc.get("result") or "")
        if "[退出码 1]" in res or "ASSERTION FAILED" in res:
            crashed = True
            continue
        m = DELTA_RE.search(res) or DELTA_RE2.search(res)
        if m:
            est = float(m.group(1))
        c = CI_RE.search(res)
        if c:
            lo, hi = float(c.group(1)), float(c.group(2))
            ci = round(hi - lo, 4)
    n_calls = run.get("n_tool_calls", len(run.get("tool_calls") or []))

    s1 = 1 if (est is not None and abs(est - GOLD_DELTA) <= TOL) else 0
    s2 = 1 if (ci is not None and CI_LO <= ci <= CI_HI) else 0
    s4 = 1 if n_calls <= MAX_CALLS else 0
    subtotal = round(0.4 * s1 + 0.2 * s2 + 0.2 * s4, 3)
    if subtotal >= 0.6:
        verdict = "pass_guaranteed"
    elif subtotal + 0.2 < 0.6:
        verdict = "fail_guaranteed"
    else:
        verdict = "unjudgeable(C-A3 decisive)"
    return {
        "estimate": est, "crashed": crashed, "ci_width": ci,
        "dev_vs_gold": None if est is None else round(est - GOLD_DELTA, 4),
        "s1": s1, "s2": s2, "s4": s4, "s3": "UNJUDGEABLE(judge not built)",
        "subtotal_deterministic": subtotal,
        "composite_bounds": [subtotal, round(subtotal + 0.2, 3)],
        "verdict": verdict, "n_tool_calls": n_calls,
    }


def main() -> None:
    out = {}
    hdr = f"{'run':28s} {'model(src)':16s} {'method':13s} {'est':>8s} {'dev':>7s} {'CIw':>6s} {'s1':>2s} {'s2':>2s} {'s4':>2s} {'sub':>5s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(RES.glob("*.json")):
        run = json.loads(p.read_text(encoding="utf-8"))
        if run.get("family") != "A":
            continue
        s = score_family_a(run)
        tag = p.stem
        model = "deepseek-v4.1f" if tag.endswith("_r0") else "qwen2.5:7b"
        methods = "+".join(run.get("methods_chosen") or ["?"])
        est = "CRASH" if s["crashed"] and s["estimate"] is None else (
            f"{s['estimate']:.4f}" if s["estimate"] is not None else "none")
        dev = "n/a" if s["dev_vs_gold"] is None else f"{s['dev_vs_gold']:+.3f}"
        ciw = "n/a" if s["ci_width"] is None else f"{s['ci_width']:.3f}"
        print(f"{tag:28s} {model:16s} {methods:13s} {est:>8s} {dev:>7s} {ciw:>6s} "
              f"{s['s1']:>2d} {s['s2']:>2d} {s['s4']:>2d} {s['subtotal_deterministic']:>5.2f}  {s['verdict']}")
        out[tag] = {"model_attribution": model, "methods_chosen": methods, **s}
    (RES / "l3_scoring_retroactive.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", RES / "l3_scoring_retroactive.json")


if __name__ == "__main__":
    sys.exit(main())
