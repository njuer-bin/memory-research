"""KWM 阶段 A runner：2×4×2 矩阵编排 + 沙箱 + 记忆注入/检索 + 结果落盘。

用法：
  python kwm_run_stage_a.py --families A B C --conditions NONE UNCOND COND COND_MINUS \
        --variants V_dead V_revived --repeats 2
  python kwm_run_stage_a.py --pilot        # 仅家族 A：4 条件 × 2 变体 × 1 重复
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kwm_common import DATA_DIR, TASKS_DIR, RESULTS_DIR, KWMMemory, dump_json, CONDITIONS, VARIANTS, ROOT

import kwm_agent as agent_mod

FAMILY_DATA = {
    "A": ("train_v1.csv", "train_v2.csv", None),              # (V_dead 文件, V_revived 文件, 陷阱工具)
    "B": ("sales_v1.csv", "sales_v2.csv", "legacy_agg.py"),
    "C": ("db_v1.sqlite", "db_v2.sqlite", None),
}
TRAP_TOOL = {"A": "robust_groupdiff.py", "B": "legacy_agg.py", "C": None}


def setup_sandbox(family: str, variant: str, sandbox: Path) -> None:
    sandbox.mkdir(parents=True, exist_ok=True)
    dead_f, rev_f, _ = FAMILY_DATA[family]
    src = DATA_DIR / (dead_f if variant == "V_dead" else rev_f)
    shutil.copy(src, sandbox / src.name)
    tool = TRAP_TOOL[family]
    if tool:
        tool_src = (ROOT / "tools" / tool) if family == "A" else (ROOT / "scripts" / tool)
        shutil.copy(tool_src, sandbox / tool)


def get_memory_text(family: str, condition: str) -> tuple[str | None, str | None]:
    """返回 (注入到 agent 的记忆文本, 命中条件标签)。NONE 返回 (None, None)。"""
    if condition == "NONE":
        return None, None
    rec_path = TASKS_DIR / f"family_{family.lower()}" / f"records_{condition.lower()}.json"
    with open(rec_path, encoding="utf-8") as f:
        content = json.load(f)["content"]
    # 检索：以任务描述为 query（单条注入下即返回该记录）
    with open(TASKS_DIR / f"family_{family.lower()}/task_description.json", encoding="utf-8") as f:
        query = json.load(f)["description"]
    mem = KWMMemory()
    mem.inject(family, condition, content, variant=None)
    ret = mem.retrieve(query, family=family)
    return (ret["context_content"], condition if ret["enters_context"] else None)


def run_one(family: str, condition: str, variant: str, repeat: int, do_memory: bool = True) -> dict:
    tag = f"{family}_{condition}_{variant}_r{repeat}"
    sandbox = RESULTS_DIR / tag
    if sandbox.exists():
        shutil.rmtree(sandbox)
    setup_sandbox(family, variant, sandbox)
    dead_f, rev_f, _ = FAMILY_DATA[family]
    data_file = dead_f if variant == "V_dead" else rev_f
    memory_text = None
    hit_condition = None
    if do_memory and condition != "NONE":
        memory_text, hit_condition = get_memory_text(family, condition)
    agent = agent_mod.KWMAgent(family, variant, condition, memory_text, sandbox, data_file)
    res = agent.run()
    res["memory_injected"] = condition != "NONE"
    res["memory_hit_condition"] = hit_condition
    res["sandbox"] = str(sandbox)
    dump_json(RESULTS_DIR / f"{tag}.json", res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--pilot", action="store_true", help="家族 A：4 条件×2 变体×1 重复")
    args = ap.parse_args()

    if args.pilot:
        args.families, args.conditions, args.variants, args.repeats = ["A"], CONDITIONS, VARIANTS, 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for fam in args.families:
        for cond in args.conditions:
            for var in args.variants:
                for rep in range(args.repeats):
                    r = run_one(fam, cond, var, rep)
                    n += 1
                    print(f"[{n}] {fam}/{cond}/{var}/r{rep} tools={r['n_tool_calls']} "
                          f"mem={'Y' if r['memory_injected'] else 'N'}")
    print(f"完成 {n} 次 agent 运行，结果见 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
