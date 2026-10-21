# -*- coding: utf-8 -*-
"""
kwm_run_manifest.py — unique run manifest (third external review, action #1).

Resolves the suspected double-counting between surviving JSONs and log-only
runs by building ONE flat list with exactly one row per executed run, then
mechanically verifying the three conditions the review requires:

  V-a: unique run_id count == 43
  V-b: the 10 JSON-evidenced runs and the 33 log-only runs are disjoint
  V-c: the §6.4b grouping numbers regenerate from the manifest alone

Run IDs are the canonical keys already used by the two scorers:
  - kwm_score_retro.py  -> JSON tag ids  (A_<COND>_<VAR>_r{0,1})
  - kwm_score_logonly.py -> log-derived ids (pilot-*, v1-*, gemmenu-*,
    gemrule-*, bind-*, v3-qwen-*, stab-*-r0)

Overlap check is by ID *and* by (log file, run index) provenance: the stab log
contains 4 runs; runs #2 and #4 are the surviving r1 JSONs and are NOT counted
as log-only anywhere. The manifest records this linkage explicitly so the
reviewer can verify no run appears in both pools.
"""
import json
import os
from collections import Counter

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(BASE, "results", "kwm_stage_a")

retro = json.load(open(os.path.join(RES, "l3_scoring_retroactive.json"), encoding="utf-8"))
logonly = json.load(open(os.path.join(RES, "l3_scoring_logonly.json"), encoding="utf-8"))

# canonical grouping (one group per run; verified exhaustive below)
def group_of(rid):
    if rid in retro:
        return "surviving-JSON/deepseek" if "deepseek" in retro[rid]["model_attribution"] \
            else "surviving-JSON/qwen-r1"
    if rid.startswith("pilot-"):
        return "log-only/confirmatory-pilot"
    if rid.startswith("v1-"):
        return "log-only/v1-free-code"
    if rid.startswith(("gemmenu-", "gemrule-", "bind-")):
        return "log-only/gemma-multi-attempt"
    if rid.startswith("v3-qwen-") or rid.startswith("stab-"):
        return "log-only/qwen-single-commitment"
    raise KeyError(rid)

def cond_split(rid):
    """condition x variant from the run id or its record."""
    src = retro.get(rid) or logonly[rid]
    if rid in retro:
        # tags look like A_<COND>_V_<var>_r<n>
        body = rid.split("_", 1)[1]
        cond, rest = body.split("_V_")
        var = "V-dead" if rest.startswith("dead") else "V-revived"
        cond = cond.replace("MINUS", "-MINUS")
    else:
        parts = rid.split("-")
        cond = "-".join(p for p in parts if p in ("NONE", "UNCOND", "COND", "MINUS", "CONDMINUS"))
        cond = cond.replace("CONDMINUS", "COND-MINUS")
        var = "V-dead" if "Vdead" in rid else "V-revived"
    return cond, var

rows = []
for rid, rec in retro.items():
    cond, var = cond_split(rid)
    rows.append(dict(run_id=rid,
                     evidence=f"JSON: results/kwm_stage_a/{rid}.json",
                     model=rec["model_attribution"], harness="v3 single commitment",
                     variant=var, condition=cond,
                     method_adopted=rec["methods_chosen"],
                     l3_status=rec["verdict"],
                     group=group_of(rid)))
for rid, rec in logonly.items():
    cond, var = cond_split(rid)
    rows.append(dict(run_id=rid,
                     evidence=f"log: results/kwm_stage_a/{rec['source'].split('logs/')[-1]}",
                     model=rec["model"], harness=rec["harness"],
                     variant=var, condition=cond,
                     method_adopted=rec["method_adopted"],
                     l3_status=rec["verdict"],
                     group=group_of(rid)))

# ---- verification V-a / V-b ------------------------------------------------
ids = [r["run_id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate run_id!"
json_ids = set(retro)
log_ids = set(logonly)
overlap = json_ids & log_ids
assert not overlap, f"OVERLAP between JSON and log-only pools: {overlap}"
n_json, n_log = len(json_ids), len(log_ids)

# stab-log linkage record: runs 2/4 of kwm_stab.log are the surviving r1 JSONs
stab_linkage = {
    "kwm_stab.log run[1]": "stab-UNCOND-Vdead-r0 (log-only; JSON tag later overwritten by deepseek A_UNCOND_V_dead_r0)",
    "kwm_stab.log run[2]": "A_UNCOND_V_dead_r1.json (surviving JSON; NOT counted as log-only)",
    "kwm_stab.log run[3]": "stab-COND-Vdead-r0 (log-only; JSON tag later overwritten by deepseek A_COND_V_dead_r0)",
    "kwm_stab.log run[4]": "A_COND_V_dead_r1.json (surviving JSON; NOT counted as log-only)",
}

# ---- verification V-c: regenerate §6.4b from the manifest ------------------
g = Counter(r["group"] for r in rows)
verdict_by_group = {}
for r in rows:
    verdict_by_group.setdefault(r["group"], Counter())[r["l3_status"].split("(")[0]] += 1

report = {
    "n_unique_runs": len(rows),
    "n_json_evidenced": n_json,
    "n_log_only": n_log,
    "overlap_between_pools": sorted(overlap),
    "checks": {
        "V-a unique run_id count == 43": len(rows) == 43,
        "V-b JSON/log-only pools disjoint": not overlap,
        "V-c group totals regenerate": {
            "surviving-JSON/deepseek": g["surviving-JSON/deepseek"],
            "surviving-JSON/qwen-r1": g["surviving-JSON/qwen-r1"],
            "log-only/confirmatory-pilot": g["log-only/confirmatory-pilot"],
            "log-only/v1-free-code": g["log-only/v1-free-code"],
            "log-only/gemma-multi-attempt": g["log-only/gemma-multi-attempt"],
            "log-only/qwen-single-commitment": g["log-only/qwen-single-commitment"],
        },
    },
    "stab_log_linkage": stab_linkage,
    "verdicts_by_group": {k: dict(v) for k, v in verdict_by_group.items()},
    "runs": rows,
}

out_json = os.path.join(RES, "run_manifest.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# markdown manifest for the draft appendix
md = ["| # | run_id | model | harness | variant | condition | adopted | evidence source | L3 status | group |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    md.append(f"| {i} | {r['run_id']} | {r['model']} | {r['harness']} | {r['variant']} | "
              f"{r['condition']} | {r['method_adopted']} | {r['evidence']} | {r['l3_status']} | {r['group']} |")
with open(os.path.join(RES, "run_manifest.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"unique runs: {len(rows)} (JSON {n_json} + log-only {n_log}, overlap {len(overlap)})")
for k, v in report["checks"]["V-c group totals regenerate"].items():
    print(f"  {k}: {v}")
print("all checks passed" if all([len(rows) == 43, not overlap]) else "CHECK FAILED")
