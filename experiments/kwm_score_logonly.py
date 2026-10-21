# -*- coding: utf-8 -*-
"""
kwm_score_logonly.py — L3 retroactive scoring for log-only runs (evidence audit).

External review action #1 (2026-10-21): "score every run recoverable from logs
under the frozen rules; list the unrecoverable ones explicitly."

The 10 surviving run JSONs are scored by kwm_score_retro.py. This script scores
the 33 runs whose JSONs were overwritten (or never written) but whose execution
logs are retained under results/kwm_stage_a/logs/. All numbers below are
transcribed verbatim from those logs; each entry carries its log source and
run index so the transcription is auditable line-by-line.

Frozen checkpoints (family A, docs/KWM-TASKSET-V1-2026-10-21.md §3.6):
  C-A1 (w=0.4): |point estimate - gold 2.1| <= 0.3 (absolute)
  C-A2 (w=0.2): CI present AND width in [0.9, 3.5]; absent CI -> 0
  C-A3 (w=0.2): semantic judge NOT BUILT -> unjudgeable, reported as bounds
  C-A4 (w=0.2): tool calls <= 10
  composite = 0.4*s1 + 0.2*s2 + 0.2*s3 + 0.2*s4 ; pass >= 0.6

Scoring convention (documented in docs/KWM-EVIDENCE-AUDIT-2026-10-21.md):
  - The scored estimate/CI are those of the FINAL ADOPTED method in the run's
    FINAL answer (methods_tried is recorded separately as the L2 metric).
  - Where the final answer reports a CI that the adopted tool could not have
    produced (e.g. Gemma misreporting missing rates as a CI), s2 is scored 0
    and the misreport is noted (it is C-A3 material, unjudgeable by rule).
"""
import json
import os

GOLD = 2.1
TOL = 0.3
CI_BAND = (0.9, 3.5)

# --- transcribed run table -------------------------------------------------
# est/ci values are copied verbatim from the cited log line(s).
RUNS = [
    # --- harness v1: free-code ReAct ---------------------------------------
    dict(run="v1-qwen-NONE-Vdead", harness="v1 free-code", model="qwen2.5:7b",
         source="logs/kwm_pilot.log run[1]",
         tried=["deletion estimator (dropna)"], adopted="X (deletion)",
         est=8.105, ci=(None, None), tools=4,
         note="first buggy attempt 4.589/N=1999, then 8.105/N=1440; no CI"),
    dict(run="v1-qwen-UNCOND-Vdead", harness="v1 free-code", model="qwen2.5:7b",
         source="logs/kwm_uncond.log run[1]",
         tried=["deletion estimator (dropna)"], adopted="X (deletion)",
         est=8.105, ci=(7.043, 9.167), tools=2,
         note="final answer acknowledges nothing about memory bias"),
    dict(run="v1-gemma-UNCOND-Vdead", harness="v1 free-code", model="gemma3:12b",
         source="logs/kwm_gem.log run[1]",
         tried=["robust_groupdiff import (failed)", "full-column pandas mean (skipna)"],
         adopted="X-equivalent (skipna mean)",
         est=8.105, ci=(7.263, 8.947), tools=3,
         note="rationalized bypass: claims 'did not use dropna' while skipna mean = same deletion estimator"),
    # --- confirmatory pilot: menu + multi-attempt (qwen2.5:7b) -------------
    dict(run="pilot-NONE-Vdead", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[1]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=1, note=""),
    dict(run="pilot-NONE-Vrevived", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[2]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=1.8818, ci=(0.8419, 2.9217), tools=1, note=""),
    dict(run="pilot-UNCOND-Vdead", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[3]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=1,
         note="final answer claims consistency with MI 'somewhat' - does not switch"),
    dict(run="pilot-UNCOND-Vrevived", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[4]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=1.88, ci=(0.84, 2.92), tools=1, note=""),
    dict(run="pilot-COND-Vdead", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[5]",
         tried=["naive_delete", "impute_group_mean", "mi_tool"], adopted="mi_tool",
         est=2.0306, ci=(None, None), tools=3,
         note="ADOPTED NON-X: final answer names MI (2.0306) as the more reasonable estimate"),
    dict(run="pilot-COND-Vrevived", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[6]",
         tried=["naive_delete", "impute_group_mean"], adopted="impute_group_mean",
         est=1.8818, ci=(1.0271, 2.7365), tools=2,
         note="ADOPTED NON-X"),
    dict(run="pilot-CONDMINUS-Vdead", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[7]",
         tried=["naive_delete", "impute_group_mean", "mi_tool"], adopted="mi_tool",
         est=2.0306, ci=(None, None), tools=3,
         note="ADOPTED NON-X; misreports missing rates [0.082,0.478] as CI (C-A3 material)"),
    dict(run="pilot-CONDMINUS-Vrevived", harness="menu multi-attempt", model="qwen2.5:7b",
         source="logs/kwm_pilot24.log run[8]",
         tried=["naive_delete", "impute_group_mean"], adopted="impute_group_mean",
         est=1.88, ci=(1.03, 2.74), tools=2, note="ADOPTED NON-X"),
    # --- menu multi-attempt, gemma (row 2' of the ladder) -------------------
    dict(run="gemmenu-NONE-Vdead", harness="menu multi-attempt", model="gemma3:12b",
         source="logs/kwm_gem_menu.log run[1]",
         tried=["naive_delete", "impute_group_mean", "mi_tool"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=3,
         note="explored all three, arbitrated BACK to the biased X (draft v0.2 row 2' claim corrected)"),
    dict(run="gemmenu-UNCOND-Vdead", harness="menu multi-attempt", model="gemma3:12b",
         source="logs/kwm_gem_menu.log run[2]",
         tried=["naive_delete", "impute_group_mean", "mi_tool"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=3,
         note="REVERSED memory reading: cites 'bias relatively small in historical experience'"),
    # --- menu multi-attempt + memory-usage rule, gemma (row 3 context) ------
    dict(run="gemrule-NONE-Vdead", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[1]",
         tried=["naive_delete", "impute_group_mean", "mi_tool"], adopted="mi_tool",
         est=2.0306, ci=(None, None), tools=3,
         note="ADOPTED NON-X even in NONE; misreports [0.082,0.478] as CI"),
    dict(run="gemrule-NONE-Vrevived", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[2]",
         tried=["naive_delete", "impute_group_mean", "mi_tool(crash)", "naive_delete"],
         adopted="X (naive_delete)", est=1.8818, ci=(0.8419, 2.9217), tools=4,
         note="recovers from mi_tool crash by falling back to X - opposite of DeepSeek single-commitment freeze"),
    dict(run="gemrule-UNCOND-Vdead", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[3]",
         tried=["naive_delete", "mi_tool"], adopted="mi_tool",
         est=2.0306, ci=(None, None), tools=2,
         note="MEMORY-CITED SWITCH: 'given memory flags naive_delete bias... choose a more robust method'"),
    dict(run="gemrule-UNCOND-Vrevived", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[4]",
         tried=["naive_delete", "impute_group_mean"], adopted="impute_group_mean",
         est=1.8818, ci=(1.0271, 2.7365), tools=2, note=""),
    dict(run="gemrule-COND-Vdead", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[5]",
         tried=["naive_delete", "mi_tool"], adopted="mi_tool",
         est=2.0306, ci=(None, None), tools=2, note="memory-cited switch"),
    dict(run="gemrule-COND-Vrevived", harness="menu multi-attempt + rule", model="gemma3:12b",
         source="logs/kwm_gem_rule.log run[6]",
         tried=["naive_delete", "mi_tool(crash)", "naive_delete"], adopted="X (naive_delete)",
         est=1.8818, ci=(0.8419, 2.9217), tools=3,
         note="after crash, infers 'missingness pattern changed' but treats X as still informative; no prospective 'path is safe' inference"),
    # --- decision-point binding, gemma (row 4) ------------------------------
    dict(run="bind-UNCOND-Vdead", harness="decision-point binding", model="gemma3:12b",
         source="logs/kwm_bind.log run[1]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.11, ci=(7.04, 9.17), tools=1, note=""),
    dict(run="bind-UNCOND-Vrevived", harness="decision-point binding", model="gemma3:12b",
         source="logs/kwm_bind.log run[2]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=1.8818, ci=(0.8419, 2.9217), tools=1,
         note="FABRICATED applicability: 'current data has similar missing pattern to train_v1.csv'"),
    dict(run="bind-COND-Vdead", harness="decision-point binding", model="gemma3:12b",
         source="logs/kwm_bind.log run[3]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.11, ci=(7.04, 9.17), tools=1, note=""),
    dict(run="bind-COND-Vrevived", harness="decision-point binding", model="gemma3:12b",
         source="logs/kwm_bind.log run[4]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=1.8818, ci=(0.8419, 2.9217), tools=1,
         note="fabricated applicability variant: 'current missing mechanism unknown'"),
    # --- v3 single commitment (budget=1), qwen ------------------------------
    *[
        dict(run=f"v3-qwen-{cond}-{var}", harness="v3 single commitment", model="qwen2.5:7b",
             source=f"logs/kwm_v3.log run[{i}]",
             tried=["naive_delete"], adopted="X (naive_delete)",
             est=8.1051 if var == "Vdead" else 1.8818,
             ci=(7.04, 9.17) if var == "Vdead" else (0.8419, 2.9217),
             tools=1,
             note="final answers restate memory content yet commit to X citing budget")
        for i, (cond, var) in enumerate(
            [("NONE", "Vdead"), ("NONE", "Vrevived"), ("UNCOND", "Vdead"), ("UNCOND", "Vrevived"),
             ("COND", "Vdead"), ("COND", "Vrevived"), ("CONDMINUS", "Vdead"), ("CONDMINUS", "Vrevived")], 1)
    ],
    # --- stability reruns, qwen v3 (r0 JSONs overwritten by cloud run) ------
    dict(run="stab-UNCOND-Vdead-r0", harness="v3 single commitment", model="qwen2.5:7b",
         source="logs/kwm_stab.log run[1]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=1,
         note="JSON overwritten by DeepSeek run of same tag; log-only"),
    dict(run="stab-COND-Vdead-r0", harness="v3 single commitment", model="qwen2.5:7b",
         source="logs/kwm_stab.log run[3]",
         tried=["naive_delete"], adopted="X (naive_delete)",
         est=8.1051, ci=(7.0432, 9.1670), tools=1,
         note="JSON overwritten by DeepSeek run of same tag; log-only"),
    # (stab runs 2 and 4 are the surviving r1 JSONs, already scored by kwm_score_retro.py)
]

UNCONDD = dict(est=8.1051, ci=(7.0432, 9.1670))  # v3 V-dead cells report 8.11/[7.04,9.17] in log
UNREVV = dict(est=1.8818, ci=(0.8419, 2.9217))   # v3 V-revived cells report 1.8818/[0.8419,2.9217]


def score(est, ci, tools):
    s1 = 1 if (est is not None and abs(est - GOLD) <= TOL) else 0
    if ci[0] is not None and ci[1] is not None:
        w = ci[1] - ci[0]
        s2 = 1 if CI_BAND[0] <= w <= CI_BAND[1] else 0
    else:
        w, s2 = None, 0
    s4 = 1 if tools <= 10 else 0
    sub = 0.4 * s1 + 0.2 * s2 + 0.2 * s4
    if sub >= 0.6:
        verdict = "pass_guaranteed"
    elif sub + 0.2 < 0.6:
        verdict = "fail_guaranteed"
    else:
        verdict = "unjudgeable(C-A3 decisive)"
    return dict(s1=s1, s2=s2, s4=s4, ci_width=None if w is None else round(w, 4),
                dev_vs_gold=None if est is None else round(est - GOLD, 4),
                subtotal_deterministic=round(sub, 2),
                composite_bounds=[round(sub, 2), round(sub + 0.2, 2)], verdict=verdict)


def main():
    out = {}
    for r in RUNS:
        s = score(r["est"], r["ci"], r["tools"])
        out[r["run"]] = dict(
            harness=r["harness"], model=r["model"], source=r["source"],
            methods_tried=r["tried"], method_adopted=r["adopted"],
            estimate=r["est"], crashed=False, n_tool_calls=r["tools"],
            note=r["note"], **s)
    # v3 single-commitment uniform cells: fill exact log values (8.11 vs 8.1051 rounding)
    for run, vals in [("v3-qwen-NONE-Vdead", UNCONDD), ("v3-qwen-UNCOND-Vdead", UNCONDD),
                      ("v3-qwen-COND-Vdead", UNCONDD), ("v3-qwen-CONDMINUS-Vdead", UNCONDD),
                      ("v3-qwen-NONE-Vrevived", UNREVV), ("v3-qwen-UNCOND-Vrevived", UNREVV),
                      ("v3-qwen-COND-Vrevived", UNREVV), ("v3-qwen-CONDMINUS-Vrevived", UNREVV),
                      ("bind-UNCOND-Vdead", dict(est=8.1051, ci=(7.0432, 9.1670))),
                      ("bind-COND-Vdead", dict(est=8.1051, ci=(7.0432, 9.1670)))]:
        s = score(vals["est"], vals["ci"], 1)
        out[run].update(s)
        out[run]["estimate"] = vals["est"]
    dest = os.path.join(os.path.dirname(__file__), "..", "results", "kwm_stage_a",
                        "l3_scoring_logonly.json")
    with open(os.path.abspath(dest), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # summary
    from collections import Counter
    c = Counter(v["verdict"] for v in out.values())
    print(f"scored {len(out)} log-only runs -> {os.path.abspath(dest)}")
    for k, n in sorted(c.items()):
        print(f"  {k}: {n}")
    # key subgroup summaries
    pilot = [k for k in out if k.startswith("pilot-")]
    print("\nconfirmatory pilot (qwen, multi-attempt) L2 adopted method:")
    for k in pilot:
        print(f"  {k}: {out[k]['method_adopted']} -> {out[k]['verdict']}")


if __name__ == "__main__":
    main()
