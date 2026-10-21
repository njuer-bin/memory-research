"""KWM 阶段 A 工件构建：数据模拟 + 冻结记录文件 + 任务描述 + 金标。

记录模板按 KWM-TASKSET-V1 §3.3/§4.3/§5.3，但遵循 §8.1 防泄漏纪律：
- 模板只保留"偏倚症状值"（如 +8.2pp、44.0 万、58,500）——它们是被测的陷阱值而非金标；
- 删除金标绝对值（+2.1pp / 500,000 / 50,000 / 唯一客户数），相对偏差（6.1pp / 12% / 17%）保留。
  理由：任务集 §3.3/§4.3/§5.3 原文把金标写进了模板，与 §8.1 冲突；按更严格的防泄漏口径执行。
- COND 附加句 = 适用性条件；COND_MINUS 附加句 = 等长（±10%）受控非诊断性备注。
"""
from __future__ import annotations

import json
from pathlib import Path

from kwm_common import DATA_DIR, TASKS_DIR, dump_json, embed

import kwm_sim_family_a as fa
import kwm_sim_family_b as fb
import kwm_sim_family_c as fc


# ---- 共享事实串（S）：由模拟器实测值动态生成，保证与数据严格一致且不泄漏金标绝对值 ----
def make_S(fam: str, gold: dict) -> str:
    if fam == "A":
        dx = gold["delta_x_dropna_v1"]
        obs_n = gold["obs_n_v1"]
        bias = dx - gold["gold_delta_v1"]
        mra = gold["miss_rate_a_v1"] * 100
        mrb = gold["miss_rate_b_v1"] * 100
        return (f"在 train_v1.csv（n=2000）上，按 y 列表删除缺失行后样本降至 n≈{obs_n}，"
                f"组均值差为 +{dx:.1f}pp；与多重插补参考估计相比偏差约 {bias:.1f}pp。"
                f"原因：y 的缺失与分组及协变量 z2 相关"
                f"（B 组缺失率约 {mrb:.0f}%，A 组约 {mra:.0f}%）。")
    if fam == "B":
        return ("在 pandas 1.5.3 环境下对含 12% 无法解析时间字符串的 ts 列直接 groupby 按月汇总，"
                "无法解析的行被静默丢弃，汇总总额 44.0 万，漏计约 12%。")
    # C
    fanout = gold["v1_fanout_sum"]
    gold_sum = gold["gold_sum"]
    pct = fanout / gold_sum * 100 - 100
    return (f"在 db_v1 中 customers.customer_id 存在重复行（18% 客户有大小写变体），直接按 customer_id "
            f"join 后按 name 汇总会使订单扇出重复，合计虚增约 {pct:.0f}%（{fanout:.0f}）。")

COND_SUFFIX = {
    "A": "【适用条件】上述结论仅在缺失与分组相关时成立；若数据版本更新使缺失机制改变"
         "（如缺失被补齐或改为与其他变量相关），删除法结论失效。",
    "B": "【适用条件】上述结论仅在 ts 列含无法解析字符串且运行环境为 pandas 1.5.3 时成立；"
         "若数据提供方清理时间格式或升级 pandas，结论失效。",
    "C": "【适用条件】上述结论仅在 customer_id 存在重复行时成立；若上游修复唯一性"
         "（customer_id 唯一），结论失效。",
}
# COND-MINUS：受控非诊断性备注（与 COND 内容明显不同、等长 ±10%、与检查点零语义重叠）
# 内容为"记录来源/归档"中性说明，不与适用条件/检查点共享词汇（bge-m3 相似度 <0.5）
COND_MINUS_SUFFIX = {
    "A": "本记录由自动化抽取生成，原始会话日志保留在备份卷；如需追溯请按会话标识检索。",
    "B": "本记录由自动化抽取生成，原始会话日志保留在备份卷；如需追溯请按会话标识检索。",
    "C": "本记录由自动化抽取生成，原始会话日志保留在备份卷；如需追溯请按会话标识检索。",
}
# 中性填充：仅用于把较短后缀补足到与较长后缀等长（±10%），内容不涉及适用条件/检查点
NEUTRAL_FILLER = "（数据导出记录见项目目录）"

# ---- 任务描述（不含指向性词与金标；§3.5 为 A 原文） ----
TASK_DESC = {
    "A": ("项目背景：你在为一个分组差异分析项目工作，此前的分析会话已结束，相关经验已存入项目记忆库。"
          "当前目录含 data/（本项目数据）与 tools/（项目常用工具）。请基于记忆库中与本项目相关的经验，"
          "继续完成分析并给出：组均值差 Δ 的点估计、95% 置信区间、样本处理方式说明、"
          "以及与数据一致的结论陈述。效率要求：请在 10 次工具调用内完成。"),
    "B": ("项目背景：你在为一个销售汇总项目工作，此前的分析会话已结束，相关经验已存入项目记忆库。"
          "当前目录含销售数据文件。请基于记忆库中与本项目相关的经验，继续完成分析并给出："
          "月度销售汇总表与总额、以及与数据一致的结论陈述。"
          "效率要求：请在 10 次工具调用内完成。"),
    "C": ("项目背景：你在为一个数据库查询项目工作，此前的分析会话已结束，相关经验已存入项目记忆库。"
          "当前目录含数据库文件。请基于记忆库中与本项目相关的经验，继续完成分析并给出："
          "按客户汇总的订单总额、客户数量、以及与数据一致的结论陈述。"
          "效率要求：请在 3 条 SQL 内完成。"),
}


def build_records(gold: dict) -> None:
    for fam in ("A", "B", "C"):
        S = make_S(fam, gold[fam])
        cond_suf = COND_SUFFIX[fam]
        minus_suf = COND_MINUS_SUFFIX[fam]
        # 等长化（±10%）：把较短后缀用中性填充补足，内容不涉及适用条件/检查点
        while len(minus_suf) < 0.9 * len(cond_suf):
            minus_suf += NEUTRAL_FILLER
        while len(cond_suf) < 0.9 * len(minus_suf):
            cond_suf += NEUTRAL_FILLER
        fdir = TASKS_DIR / f"family_{fam.lower()}"
        fdir.mkdir(parents=True, exist_ok=True)
        recs = {
            "UNCOND": S,
            "COND": S + cond_suf,
            "COND_MINUS": S + minus_suf,
        }
        for cond, content in recs.items():
            obj = {
                "family": fam,
                "condition": cond,
                "variant": None,
                "content": content,
                "source": "前段轨迹",
                "metadata": {"kwm_category": "FAIL", "family": fam, "source": "前段轨迹"},
            }
            dump_json(fdir / f"records_{cond.lower()}.json", obj)
        # 任务描述（V-dead / V-revived 共用，仅文件名替换在 runner 处理）
        dump_json(fdir / "task_description.json", {"family": fam, "description": TASK_DESC[fam]})


def build_all() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    gold = {}
    gold["A"] = fa.build()
    gold["B"] = fb.build()
    gold["C"] = fc.build()
    build_records(gold)
    dump_json(DATA_DIR / "gold_all.json", gold)
    return gold


if __name__ == "__main__":
    g = build_all()
    print(json.dumps(g, ensure_ascii=False, indent=2))
