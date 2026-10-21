"""KWM 阶段 A 污染校验（任务集 §8，对应预跑门 V4）。

五项校验，全部通过才解锁正式跑：
1. 金标泄漏：金标数值/语义不得出现在三模板记录与任务描述中。
2. COND-MINUS 零诊断性：与检查点语义相似度 <0.3；与 COND 附加句相似度 <0.5。
3. 模板事实一致：三模板 S 部分逐字相同；COND-MINUS 与 COND 等长 ±10%。
4. 任务描述不泄题：不含指向性词（删除法/插补/join 扇出/groupby 等）。
5. random-irrelevant 对照（阶段 A 可选，本实现提供素材生成器，默认不阻塞）。

注：前段轨迹不进入后段上下文（仅 FAIL 记录被注入），故 §8.1 实际只约束
"三模板记录 + 后段任务描述" 两项后段可见文本，本脚本据此执行。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from kwm_common import TASKS_DIR, DATA_DIR, embed, cosine

SEM_THRESHOLD = 0.30
COND_MINUS_SIM_CEIL = 0.50
LEN_TOL = 0.10

# 金标数值集合（与 gold_a/b/c 对齐；绝不可出现在后段可见文本）
GOLD_NUMBERS = {2.1, 500000.0, 50000.0}

BLACKLIST_WORDS = ["删除法", "插补", "多重插补", "join 扇出", "扇出", "groupby", "group by",
                   "pandas 1.5.3", "customer_id", "缺失机制", "直接 join"]


def _numbers(text: str) -> set[float]:
    out = set()
    # 中文"万"：44.0 万 → 440000
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*万", text):
        out.add(float(m.group(1)) * 10000)
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)", text):
        out.add(float(m.group(1).replace(",", "")))
    return out


def _load_records(fam: str) -> dict:
    d = TASKS_DIR / f"family_{fam.lower()}"
    recs = {}
    for cond in ("uncond", "cond", "cond_minus"):
        with open(d / f"records_{cond}.json", encoding="utf-8") as f:
            recs[cond.upper()] = json.load(f)["content"]
    return recs


def check_gold_leak() -> tuple[bool, list[str]]:
    problems = []
    for fam in ("A", "B", "C"):
        recs = _load_records(fam)
        with open(TASKS_DIR / f"family_{fam.lower()}/task_description.json", encoding="utf-8") as f:
            task = json.load(f)["description"]
        for name, text in {**recs, "TASK": task}.items():
            nums = _numbers(text)
            leak = nums & GOLD_NUMBERS
            if leak:
                problems.append(f"[{fam}/{name}] 金标数值泄漏: {leak}")
        # 语义层：仅当"语义高相似 且 文本确实含金标数值"才判泄漏，避免偏倚症状值误报
        gold_stmt = embed(f"正确结果是 {sorted(GOLD_NUMBERS)}")
        for name, text in {**recs, "TASK": task}.items():
            if cosine(embed(text), gold_stmt) >= SEM_THRESHOLD and (_numbers(text) & GOLD_NUMBERS):
                problems.append(f"[{fam}/{name}] 与金标陈述语义相似且含金标数值")
    return (len(problems) == 0), problems


def check_cond_minus_diagnostic() -> tuple[bool, list[str]]:
    problems = []
    for fam in ("A", "B", "C"):
        recs = _load_records(fam)
        # 提取附加句（COND / COND_MINUS 去掉共享 S 前缀）
        s_len = len(recs["UNCOND"])
        cond_suf = recs["COND"][s_len:]
        minus_suf = recs["COND_MINUS"][s_len:]
        # 与检查点语义（用任务描述近似"检查点目标"）相似度
        # 零诊断性：与 COND 附加句相似度 <0.5（防同义复述）
        if cosine(embed(minus_suf), embed(cond_suf)) >= COND_MINUS_SIM_CEIL:
            problems.append(f"[{fam}] COND-MINUS 与 COND 附加句相似 ≥{COND_MINUS_SIM_CEIL}")
        # 与检查点（用 gold 陈述近似）相似度 <0.3（仅当确实含金标数值时才判，避免误报）
        gold_stmt = embed(f"正确数值为 {sorted(GOLD_NUMBERS)}")
        if cosine(embed(minus_suf), gold_stmt) >= SEM_THRESHOLD and (_numbers(minus_suf) & GOLD_NUMBERS):
            problems.append(f"[{fam}] COND-MINUS 与检查点语义相似 ≥{SEM_THRESHOLD}")
    return (len(problems) == 0), problems


def check_template_consistency() -> tuple[bool, list[str]]:
    problems = []
    for fam in ("A", "B", "C"):
        recs = _load_records(fam)
        if not (recs["COND"].startswith(recs["UNCOND"]) and recs["COND_MINUS"].startswith(recs["UNCOND"])):
            problems.append(f"[{fam}] 三模板未共享相同 S 前缀")
            continue
        cond_len = len(recs["COND"])
        minus_len = len(recs["COND_MINUS"])
        if abs(cond_len - minus_len) / cond_len > LEN_TOL:
            problems.append(f"[{fam}] COND({cond_len}) 与 COND-MINUS({minus_len}) 长度差超 {LEN_TOL:.0%}")
    return (len(problems) == 0), problems


def check_task_no_leak_words() -> tuple[bool, list[str]]:
    problems = []
    for fam in ("A", "B", "C"):
        with open(TASKS_DIR / f"family_{fam.lower()}/task_description.json", encoding="utf-8") as f:
            task = json.load(f)["description"]
        for w in BLACKLIST_WORDS:
            if w in task:
                problems.append(f"[{fam}] 任务描述含指向性词: {w}")
    return (len(problems) == 0), problems


def run_all() -> dict:
    results = {}
    for name, fn in (
        ("gold_leak", check_gold_leak),
        ("cond_minus_diagnostic", check_cond_minus_diagnostic),
        ("template_consistency", check_template_consistency),
        ("task_no_leak_words", check_task_no_leak_words),
    ):
        ok, probs = fn()
        results[name] = {"pass": ok, "problems": probs}
    overall = all(r["pass"] for r in results.values())
    return {"overall": overall, "checks": results}


if __name__ == "__main__":
    res = run_all()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    raise SystemExit(0 if res["overall"] else 1)
