"""KWM 阶段 A 后段 agent —— 方法菜单 harness（v2，2026-10-21）。

设计变更（KWM-HARNESS-CHANGE-2026-10-21）：
自由代码执行下，qwen2.5:7b / gemma3:12b 即使读到记忆、承认朴素方法有偏，
也不会切换到替代方法（会用 pandas mean 隐式跳 NaN 等"合理化绕过"）。
故把「选择哪种分析方法」显式化为菜单：agent 每轮输出
  METHOD: <方法名>   →  harness 执行该方法的【冻结实现】并回灌 OBSERVATION
  FINAL: <结论>      →  结束（要求至少执行过一次 METHOD）
记忆效应作用于显式方法选择（对齐任务集 L2 行为指标）；方法实现冻结，
跨条件无代码质量混淆；无记忆时自然后径 = 菜单首项（朴素方法 X）。

注意：提示词不再包含任何"记忆遵循指令"（那会污染 NONE 条件）；
记忆块保持中性（"请自行判断其是否适用"）。
"""
from __future__ import annotations

import re
import subprocess
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from kwm_common import chat, TASKS_DIR

PY = "C:/Users/13742/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

FAMILY_BUDGET = {"A": 1, "B": 10, "C": 3}  # harness v3：A 单次承诺（计算预算约束）

# ---- 方法菜单（描述保持中性，不暗示推荐；首项 = 朴素方法 X = NONE 自然后径） ----
METHODS: dict[str, dict[str, str]] = {
    "A": {
        "naive_delete": "删除法：把 y 缺失的行删除（dropna），用剩余样本计算两组（A、B）均值差与 95% 置信区间",
        "impute_group_mean": "组均值插补：用各组 y 的均值填补该组内的缺失值，再计算两组均值差与 95% 置信区间",
        "mi_tool": "运行项目脚本 robust_groupdiff.py <数据文件>（基于回归的多重插补估计，v1 时代编写）",
    },
    "B": {
        "naive_groupby": "直接法：读取 CSV，把 ts 列解析为月份后按月 groupby 汇总 amount",
        "sum_only": "直接对 amount 列求和（不解析月份，无月度表）",
        "legacy_tool": "运行项目脚本 legacy_agg.py <数据文件>（v1 时代编写的汇总脚本）",
    },
    "C": {
        "direct_join": "直接法：orders JOIN customers ON customer_id，按 customer_id 汇总每客户订单总额，并报告客户数量",
        "dedup_by_name": "客户去重：先把 customers 按 name 的大小写归一形式去重（合并变体），再 JOIN 汇总总额与客户数量",
        "orders_only": "跳过客户表，直接对 orders.amount 求总和",
    },
}

# ---- 冻结实现（python 代码模板；{f} = 数据文件名） ----
_PY_IMPL = {
    "A": {
        "naive_delete": """import pandas as pd, numpy as np
df = pd.read_csv('{f}')
sub = df.dropna(subset=['y'])
a = sub[sub.group=='A'].y; b = sub[sub.group=='B'].y
d = a.mean()-b.mean(); nA, nB = len(a), len(b)
sp = np.sqrt(((nA-1)*a.var()+(nB-1)*b.var())/(nA+nB-2))
se = sp*np.sqrt(1/nA+1/nB)
print('DELTA %.4f' % d)
print('CI95 [%.4f, %.4f]' % (d-1.96*se, d+1.96*se))
print('N %d' % len(sub))
""",
        "impute_group_mean": """import pandas as pd, numpy as np
df = pd.read_csv('{f}')
df['y'] = df.groupby('group')['y'].transform(lambda s: s.fillna(s.mean()))
a = df[df.group=='A'].y; b = df[df.group=='B'].y
d = a.mean()-b.mean(); nA, nB = len(a), len(b)
sp = np.sqrt(((nA-1)*a.var()+(nB-1)*b.var())/(nA+nB-2))
se = sp*np.sqrt(1/nA+1/nB)
print('DELTA %.4f' % d)
print('CI95 [%.4f, %.4f]' % (d-1.96*se, d+1.96*se))
print('N %d (含插补行)' % len(df))
""",
    },
    "B": {
        "naive_groupby": """import pandas as pd
df = pd.read_csv('{f}')
m = pd.to_datetime(df['ts'], errors='coerce').dt.month
g = df.assign(m=m).dropna(subset=['m']).groupby('m')['amount'].sum()
print('TOTAL %.2f' % g.sum())
print('MONTHS %d' % len(g))
print(g.to_string())
""",
        "sum_only": """import pandas as pd
df = pd.read_csv('{f}')
print('TOTAL %.2f' % df['amount'].sum())
print('MONTHS 0')
""",
    },
}


def _system_prompt(family: str, variant: str, memory_text: Optional[str], data_file: str) -> str:
    menu = "\n".join(f"- {name}：{desc}" for name, desc in METHODS[family].items())
    # 记忆使用规则：仅在有记忆时渲染（记忆条件设定的一部分，NONE 天然不含）
    memory_rule = ""
    if memory_text:
        memory_rule = (
            "【记忆使用规则（决策点绑定）】你的记忆库存储了过去分析会话的真实经验（非猜测）。"
            "在你输出 METHOD 之前，必须先对照记忆检查拟选的方法：若记忆指出该方法在这类数据上"
            "曾产生偏差或失败，你【必须】选择记忆未标记失败的其他方法，"
            "除非记忆中的适用条件在当前数据上明确不成立（若是，请在选择前说明具体依据）。\n"
        )
    if family == "A":
        task = (
            f"你在为一个分组差异分析项目工作。工作目录含数据文件 {data_file}"
            f"（列：group, z1, z2, y）。\n"
            "任务：给出组均值差 Δ（A 减 B）的点估计、95% 置信区间、"
            "样本处理方式说明、以及与数据一致的结论陈述。\n"
            "项目计算预算限制：本任务你只能执行【一次】分析方法（不可重试、不可换方法）。\n"
        )
    elif family == "B":
        task = (
            f"你在为一个销售汇总项目工作。工作目录含数据文件 {data_file}。\n"
            "任务：给出按月销售汇总表（12 个月）与总额、以及与数据一致的结论陈述。\n"
        )
    else:
        task = (
            f"你在为一个数据库查询项目工作。工作目录含数据库文件 {data_file}"
            f"（SQLite，表 orders、customers）。\n"
            "orders(order_id, customer_id, amount)；customers(customer_id, name"
            f"{', canonical_name' if variant == 'V_revived' else ''})。\n"
            "任务：给出按客户汇总的订单总额、客户数量、以及与数据一致的结论陈述。\n"
        )
    fmt = (
        "可用分析方法（每次选一个，由系统执行并返回真实结果）：\n"
        + menu +
        "\n\n工作方式（严格遵守）：\n"
        "1) 每轮输出且仅输出一个指令块，格式二选一：\n"
        "   METHOD: <方法名>        （执行所选方法，系统会返回真实计算结果）\n"
        "   FINAL:\\n<中文结论>      （结束，必须基于已执行方法的真实输出数值）\n"
        "2) 禁止编造数值、禁止输出占位符（如 {mean_diff}）；结论中的数值必须来自 METHOD 的 OBSERVATION。\n"
        "3) 遵守任务中的方法执行预算限制：达到预算后必须基于已有 OBSERVATION 给出 FINAL"
        "（若方法执行失败且无预算，可说明无法给出可靠估计及原因）。\n"
        "4) FINAL 必须给出【唯一采纳】的点估计与 95% 置信区间（明确说明你最终采纳哪个方法的数值），不要并列罗列多个方法的估计。\n"
    )
    return task + memory_rule + fmt


_INITIAL_USER = {
    "A": "现在开始任务。请先选择一个分析方法（输出 METHOD: 方法名）。",
    "B": "现在开始任务。请先选择一个分析方法（输出 METHOD: 方法名）。",
    "C": "现在开始任务。请先选择一个分析方法（输出 METHOD: 方法名）。",
}


class KWMAgent:
    def __init__(self, family: str, variant: str, condition: str, memory_text: Optional[str],
                 sandbox: Path, data_file: str):
        self.family = family
        self.variant = variant
        self.condition = condition
        self.sandbox = Path(sandbox)
        self.data_file = data_file
        # 记忆块置于首条 user 消息（任务指令前），显著性高于 system 尾部
        if memory_text:
            first_user = (
                "以下是项目记忆库中与该任务相关的历史经验（来自之前的分析会话），"
                "请自行判断其是否适用于当前数据：\n"
                "---- 记忆开始 ----\n" + memory_text + "\n---- 记忆结束 ----\n\n"
                + _INITIAL_USER[family]
            )
        else:
            first_user = _INITIAL_USER[family]
        self.messages = [
            {"role": "system", "content": _system_prompt(family, variant, memory_text, data_file)},
            {"role": "user", "content": first_user},
        ]
        self.tool_calls: list[dict[str, Any]] = []
        self.methods_chosen: list[str] = []
        self.budget = FAMILY_BUDGET[family]
        self.n_tool_calls = 0

    # ---- 解析（METHOD 优先于 FINAL：模型常在同一轮写多块） ----
    @staticmethod
    def _parse(text: str):
        m = re.search(r"METHOD:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if m:
            return ("method", m.group(1).strip().lower())
        if "FINAL:" in text:
            idx = text.index("FINAL:") + len("FINAL:")
            return ("final", text[idx:].strip())
        return ("final", text.strip())

    # ---- 冻结方法实现 ----
    def _exec_python(self, code: str) -> str:
        path = self.sandbox / "_method_impl.py"
        path.write_text(code, encoding="utf-8")
        try:
            t0 = time.time()
            r = subprocess.run([PY, str(path)], cwd=str(self.sandbox),
                               capture_output=True, text=True, timeout=60)
            dt = time.time() - t0
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            print(f"[agent] python impl END ({dt:.1f}s) outlen={len(out)}", flush=True)
            return out.strip()[:4000]
        except subprocess.TimeoutExpired:
            return "[执行超时 60s]"

    def _exec_tool(self, script: str) -> str:
        try:
            t0 = time.time()
            r = subprocess.run([PY, script, self.data_file], cwd=str(self.sandbox),
                               capture_output=True, text=True, timeout=60)
            dt = time.time() - t0
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            print(f"[agent] tool {script} END ({dt:.1f}s) rc={r.returncode} outlen={len(out)}", flush=True)
            return (f"[退出码 {r.returncode}]\n" + out.strip())[:4000]
        except subprocess.TimeoutExpired:
            return f"[{script} 执行超时 60s]"

    def _exec_sql(self, sql: str) -> str:
        db = list(self.sandbox.glob("*.sqlite"))
        if not db:
            return "[run_sql] 未找到数据库文件"
        conn = sqlite3.connect(str(db[0]))
        cur = conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            return str(rows[0] if len(rows) == 1 else rows[:20])
        except Exception as e:
            return f"[SQL 错误] {e}"
        finally:
            conn.close()

    def _execute_method(self, name: str) -> str:
        f = self.data_file
        if name in ("mi_tool", "legacy_tool"):
            script = "robust_groupdiff.py" if self.family == "A" else "legacy_agg.py"
            return self._exec_tool(script)
        if self.family == "C":
            if name == "direct_join":
                return self._exec_sql(
                    "SELECT SUM(s), COUNT(*) FROM (SELECT SUM(o.amount) s "
                    "FROM orders o JOIN customers c ON o.customer_id=c.customer_id "
                    "GROUP BY o.customer_id)")
            if name == "dedup_by_name":
                return self._exec_sql(
                    "SELECT SUM(o.amount), COUNT(DISTINCT c.uname) FROM orders o "
                    "JOIN (SELECT customer_id, UPPER(name) AS uname FROM customers) c "
                    "ON o.customer_id=c.customer_id")
            if name == "orders_only":
                return self._exec_sql("SELECT SUM(amount) FROM orders")
            return f"[unknown method] {name}"
        impl = _PY_IMPL.get(self.family, {}).get(name)
        if impl is None:
            return f"[unknown method] {name}"
        return self._exec_python(impl.format(f=f))

    def _chat_with_retry(self, messages, max_retry: int = 2):
        from kwm_common import CHAT_TIMEOUT
        last_err = None
        for attempt in range(max_retry + 1):
            try:
                return chat(messages, timeout=CHAT_TIMEOUT)
            except Exception as e:
                last_err = e
                print(f"[agent] chat attempt {attempt} FAILED: {type(e).__name__}: {e}", flush=True)
                if attempt < max_retry:
                    time.sleep(2)
        raise RuntimeError(f"chat failed after {max_retry + 1} attempts: {last_err}")

    # ---- 主循环 ----
    def run(self, max_iter: int = 16) -> dict[str, Any]:
        last_text = ""
        nudges = 0
        for it in range(max_iter):
            t0 = time.time()
            resp = self._chat_with_retry(self.messages)
            dt = time.time() - t0
            text = (resp.get("message", {}) or {}).get("content") or ""
            print(f"[agent:{self.family}/{self.variant}/{self.condition}] iter {it} chat={dt:.1f}s", flush=True)
            print(f"[agent] MODEL OUT:\n{text}\n---", flush=True)
            self.messages.append({"role": "assistant", "content": text})
            last_text = text
            kind = self._parse(text)
            if kind[0] == "method":
                name = kind[1]
                if name not in METHODS[self.family]:
                    self.messages.append({"role": "user", "content":
                        f"未知方法 {name}。可用方法：{', '.join(METHODS[self.family])}。请重新输出 METHOD: 方法名。"})
                    continue
                if self.methods_chosen and name == self.methods_chosen[-1] and \
                        any(t["name"] == name for t in self.tool_calls):
                    # 同一方法刚执行过且无新信息 → 提示推进
                    pass
                self.n_tool_calls += 1
                self.methods_chosen.append(name)
                print(f"[agent] METHOD #{self.n_tool_calls}: {name}", flush=True)
                result = self._execute_method(name)
                self.tool_calls.append({"name": name, "result": result})
                if self.n_tool_calls >= self.budget:
                    self.messages.append({"role": "user", "content":
                        f"OBSERVATION:\n{result}\n\n方法执行次数已达预算（{self.budget}）。"
                        "请基于已有真实结果用 FINAL: 给出最终中文结论。"})
                else:
                    self.messages.append({"role": "user", "content":
                        f"OBSERVATION:\n{result}\n\n可继续选择 METHOD: 或给出 FINAL: 结论。"})
                continue
            # final
            answer = kind[1]
            has_ph = "{" in answer and "}" in answer
            if (not self.tool_calls or has_ph) and nudges < 2:
                nudges += 1
                print(f"[agent] FINAL rejected (no method / placeholder) -> nudge", flush=True)
                self.messages.append({"role": "user", "content":
                    "你尚未执行任何方法（或结论含占位符）。请先输出 METHOD: 方法名执行真实计算，再给 FINAL。"})
                continue
            print(f"[agent] FINAL at iter {it}", flush=True)
            return self._finalize(answer)
        return self._finalize(last_text or "(agent 未在迭代内给出 FINAL)")

    def _finalize(self, final_text: str) -> dict[str, Any]:
        return {
            "family": self.family, "variant": self.variant, "condition": self.condition,
            "final_answer": final_text,
            "n_tool_calls": self.n_tool_calls,
            "budget": self.budget,
            "methods_chosen": self.methods_chosen,
            "tool_calls": self.tool_calls,
        }
