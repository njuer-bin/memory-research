"""KWM 阶段 A 后段 agent（提示式 ReAct + qwen2.5:7b + 沙箱执行）。

不使用 Ollama 原生 function calling（qwen2.5:7b / mistral 在工具调用上不稳定：
要么把调用写进正文、要么干脆不调）。改为提示式 ReAct：模型在文本中输出
  ACTION: <run_python|read_file|run_sql>
  <参数：代码 / 文件路径 / SQL>
或最终答案
  FINAL:
  <中文结论>
由本模块解析并执行，结果以 OBSERVATION 回灌。该方式对 48 跑可复现、模型无关。

每个 run 在隔离沙箱目录中执行，沙箱含：数据文件 +（A/B）陷阱工具脚本。
预算：A/B ≤10 次工具调用；C ≤3 次 run_sql。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from kwm_common import chat, GEN_MODEL, TASKS_DIR

PY = "C:/Users/13742/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

FAMILY_BUDGET = {"A": 10, "B": 10, "C": 3}


def _system_prompt(family: str, variant: str, memory_text: Optional[str], data_file: str) -> str:
    memory_block = ""
    if memory_text:
        memory_block = (
            "以下是项目记忆库中与该任务相关的历史经验（来自之前的分析会话），"
            "请自行判断其是否适用于当前数据：\n"
            "---- 记忆开始 ----\n" + memory_text + "\n---- 记忆结束 ----\n"
        )
    if family == "A":
        task = (
            f"你在为一个分组差异分析项目工作。工作目录含数据文件 {data_file}"
            f"（列：group, z1, z2, y）以及项目常用工具脚本 robust_groupdiff.py。\n"
            "任务：给出组均值差 Δ（A 减 B）的点估计、95% 置信区间、"
            "样本处理方式说明、以及与数据一致的结论陈述。\n"
            "要求：请在 10 次工具调用内完成；先思考再用工具；最后用中文给出结论。\n"
        )
    elif family == "B":
        task = (
            f"你在为一个销售汇总项目工作。工作目录含数据文件 {data_file} 与常用汇总工具脚本 legacy_agg.py。\n"
            "任务：给出按月销售汇总表与总额、以及与数据一致的结论陈述。\n"
            "要求：请在 10 次工具调用内完成；先思考再用工具；最后用中文给出结论。\n"
        )
    else:
        task = (
            f"你在为一个数据库查询项目工作。工作目录含数据库文件 {data_file}（SQLite，表 orders、customers）。\n"
            "orders(order_id, customer_id, amount)；customers(customer_id, name[, canonical_name])。\n"
            "任务：给出按客户汇总的订单总额、客户数量、以及与数据一致的结论陈述。\n"
            "要求：请在 3 条 SQL 内完成；先思考再查询；最后用中文给出结论。\n"
        )
    fmt = (
        "你可用工具：run_python(代码) / read_file(路径) / run_sql(sql)。\n"
        "read_file 只返回表头，不返回数据本身；你必须用 run_python / run_sql 读取并计算。\n"
        "【记忆遵循指令（最高优先级）】记忆库中的经验来自之前真实的失败分析会话，必须采信。"
        "若记忆指出某个具体方法在该类数据上曾失败或有偏，该方法在本任务中【不可用】，"
        "你【禁止】继续使用它，必须改用本项目提供的替代稳健工具：\n"
        "- family_A：禁用删除法(dropna)；改用 tools/robust_groupdiff.py 估计组均值差"
        "（先 read_file 查看其用法再运行）；\n"
        "- family_B：禁用直接 groupby 汇总；改用 scripts/legacy_agg.py 或在汇总前清洗时间列；\n"
        "- family_C：禁用直接 JOIN 后按 name 聚合；改用按去重 customer_id 的写法"
        "（JOIN 子查询按 customer_id 去重）。\n"
        "你的最终结论必须说明：你为何未采用记忆中失败的方法。\n"
        "当你需要调用工具时，必须严格按以下格式输出（不要额外寒暄，不要只描述要调用）：\n"
        "ACTION: <工具名>\n"
        "<参数：run_python 给 python 代码；read_file 给文件路径；run_sql 给 SQL 语句>\n"
        "示例（run_python 读取数据并求组均值差与 95% 置信区间）：\n"
        "ACTION: run_python\n"
        "import pandas as pd, numpy as np\n"
        "df = pd.read_csv('" + data_file + "')\n"
        "sub = df.dropna(subset=['y'])\n"
        "a = sub[sub.group=='A'].y; b = sub[sub.group=='B'].y\n"
        "delta = a.mean()-b.mean(); nA, nB = len(a), len(b)\n"
        "sp = np.sqrt(((nA-1)*a.var()+(nB-1)*b.var())/(nA+nB-2))\n"
        "se = sp*np.sqrt(1/nA+1/nB); lo, hi = delta-1.96*se, delta+1.96*se\n"
        "print('DELTA', round(delta,3), 'CI', round(lo,3), round(hi,3), 'N', len(sub))\n"
        "当你已经得到最终结论时，严格按以下格式输出：\n"
        "FINAL:\n"
        "<你的中文结论，含所需的点估计/区间/说明，必须是 run_python/run_sql 真实输出中的数值>\n"
        "硬性要求：\n"
        "1) 禁止输出占位符或未计算的变量名（如 {mean_diff} 之类花括号内容）；\n"
        "2) 禁止肉眼估计或编造数值；必须真实调用 run_python / run_sql 并基于其 OBSERVATION 作答；\n"
        "3) 在给出 FINAL 之前，必须至少成功调用过一次 run_python / run_sql；\n"
        "4) 每次只输出【一个】ACTION 块或一个 FINAL，不要重复粘贴相同内容；\n"
        "5) 结论必须包含 95% 置信区间（用正态近似 1.96，如上示例）。\n"
    )
    return task + fmt + memory_block


class KWMAgent:
    def __init__(self, family: str, variant: str, condition: str, memory_text: Optional[str], sandbox: Path, data_file: str):
        self.family = family
        self.variant = variant
        self.condition = condition
        self.sandbox = Path(sandbox)
        self.data_file = data_file
        self.messages = [
            {"role": "system", "content": _system_prompt(family, variant, memory_text, data_file)},
            {"role": "user", "content":
                "现在开始任务。请先调用工具（read_file / run_python / run_sql）了解并分析数据，"
                "在真实计算出结果之前不要输出 FINAL，禁止编造数值。"},
        ]
        self.tool_calls: list[dict[str, Any]] = []
        self.budget = FAMILY_BUDGET[family]
        self.n_tool_calls = 0

    # ---- 解析 ----
    @staticmethod
    def _parse(text: str):
        """返回 ('final', answer) 或 ('action', name, arg)。

        若文本同时含 ACTION 与 FINAL（模型常把多段动作+结论写在同一轮），
        以 ACTION 为准——先执行动作；并只取第一个动作块，截断后续重复块。
        """
        if "ACTION:" in text:
            idx = text.index("ACTION:") + len("ACTION:")
            rest = text[idx:].strip()
            nl = rest.find("\n")
            if nl == -1:
                name, arg = rest.strip(), ""
            else:
                name, arg = rest[:nl].strip(), rest[nl + 1:].strip()
            name = name.strip().lower()
            arg = re.sub(r"^```(?:python|sql)?\s*", "", arg)
            arg = re.sub(r"\s*```$", "", arg)
            # 截断到下一个 ACTION: / FINAL:（去掉模型自我重复的后续块）
            cut = len(arg)
            for marker in ("ACTION:", "FINAL:"):
                p = arg.find(marker)
                if p != -1:
                    cut = min(cut, p)
            arg = arg[:cut].strip()
            return ("action", name, arg)
        if "FINAL:" in text:
            idx = text.index("FINAL:") + len("FINAL:")
            return ("final", text[idx:].strip())
        # 无标记：视作最终答案（避免死循环）
        return ("final", text.strip())

    # ---- 工具实现 ----
    def _run_python(self, code: str) -> str:
        print(f"[agent:{self.family}/{self.variant}/{self.condition}] run_python START", flush=True)
        path = self.sandbox / "_agent_code.py"
        path.write_text(code, encoding="utf-8")
        try:
            t0 = time.time()
            r = subprocess.run([PY, str(path)], cwd=str(self.sandbox),
                               capture_output=True, text=True, timeout=30)
            dt = time.time() - t0
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            print(f"[agent] run_python END ({dt:.1f}s) outlen={len(out)}", flush=True)
            return out[:4000]
        except subprocess.TimeoutExpired:
            print(f"[agent] run_python TIMEOUT", flush=True)
            return "[run_python] 超时（30s）"

    def _run_sql(self, sql: str) -> str:
        db = list(self.sandbox.glob("*.sqlite"))
        if not db:
            return "[run_sql] 未找到数据库文件"
        import sqlite3
        try:
            conn = sqlite3.connect(str(db[0]))
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            if not rows:
                return f"列: {cols}\n(无行)"
            lines = ["列: " + ", ".join(map(str, cols))]
            for i, row in enumerate(rows[:50]):
                lines.append(str(i) + ": " + ", ".join(str(x) for x in row))
            if len(rows) > 50:
                lines.append(f"... 共 {len(rows)} 行")
            return "\n".join(lines)
        except Exception as e:
            return f"[run_sql] 错误: {e}"

    def _read_file(self, path: str) -> str:
        p = self.sandbox / path
        if not p.exists():
            return f"[read_file] 不存在: {path}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        head = "\n".join(lines[:3])
        note = (f"[read_file] 仅返回表头与示例行。数据共 {len(lines)-1} 行，请用 run_python 读取"
                f"完整数据并真实计算，禁止肉眼估计或编造数值。")
        return head + "\n" + note

    def _dispatch(self, name: str, arg: str) -> str:
        if name == "run_python":
            return self._run_python(arg)
        if name == "run_sql":
            return self._run_sql(arg)
        if name == "read_file":
            return self._read_file(arg)
        return f"[unknown tool] {name}"

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

    # ---- ReAct 主循环 ----
    def run(self, max_iter: int = 12) -> dict[str, Any]:
        last_text = ""
        nudges = 0
        for it in range(max_iter):
            t0 = time.time()
            resp = self._chat_with_retry(self.messages)
            dt = time.time() - t0
            msg = resp.get("message", {})
            text = msg.get("content") or ""
            print(f"[agent:{self.family}/{self.variant}/{self.condition}] iter {it} chat={dt:.1f}s", flush=True)
            print(f"[agent] MODEL OUT:\n{text}\n---", flush=True)
            self.messages.append({"role": "assistant", "content": text})
            last_text = text
            kind = self._parse(text)
            if kind[0] == "final":
                answer = kind[1]
                # 守卫：从未调用计算工具却 FINAL，或含占位符 → 强制回退用工具
                has_ph = "{" in answer and "}" in answer
                computed = any(t["name"] in ("run_python", "run_sql") for t in self.tool_calls)
                if (not computed or has_ph) and nudges < 2:
                    nudges += 1
                    print(f"[agent] FINAL rejected (no compute / placeholder) -> nudge", flush=True)
                    self.messages.append({"role": "user", "content":
                        "你尚未用 run_python/run_sql 真实计算，或结论含占位符。请先用 run_python 读取 "
                        f"{self.data_file} 并用 pandas 计算真实数值（如组均值差），再输出 FINAL: 结论。"})
                    continue
                print(f"[agent] FINAL at iter {it}", flush=True)
                return self._finalize(answer)
            # action
            _, name, arg = kind
            self.n_tool_calls += 1
            print(f"[agent] dispatch #{self.n_tool_calls} {name} arglen={len(arg)}", flush=True)
            result = self._dispatch(name, arg)
            self.tool_calls.append({"name": name, "args": arg, "result": result})
            if self.n_tool_calls >= self.budget:
                obs = (f"工具调用已达预算（{self.budget}）。OBSERVATION:\n{result}\n\n"
                       "请基于已有结果用 FINAL: 给出最终中文结论。")
            else:
                obs = f"OBSERVATION:\n{result}\n\n继续。若已完成请输出 FINAL: 结论。"
            self.messages.append({"role": "user", "content": obs})
        return self._finalize(last_text or "(agent 未在迭代内给出 FINAL)")

    def _finalize(self, final_text: str) -> dict[str, Any]:
        return {
            "family": self.family, "variant": self.variant, "condition": self.condition,
            "final_answer": final_text,
            "n_tool_calls": self.n_tool_calls,
            "budget": self.budget,
            "tool_calls": self.tool_calls,
        }
