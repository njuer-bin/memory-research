"""KWM 阶段 A 公共模块：冻结常量、Ollama 客户端、KWM 记忆注入/检索层。

设计说明（与任务集 KWM-TASKSET-V1-2026-10-21.md 对齐）：
- 记忆注入不能走 memory_engine.engine.add()：其 rule-based 抽取器（analyzer.py）
  只认人称事实/关系/规则正则，FAIL 记录（中文统计/代码文本）不会被抽成 fact，
  检索不到原文。故此处自建轻量记忆层：直接以冻结原文做稠密向量（bge-m3）检索，
  等价于 memory_engine 的"Real Memory 稠密检索"方式，且保留精确内容与 metadata、
  以及 L1（命中/进入上下文）轨迹。
- 生成模型：qwen2.5:7b（native tools 能力，Ollama）。
- 嵌入模型：bge-m3:latest（任务集 §8 污染校验指定；维度 1024）。
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
TASKS_DIR = ROOT / "tasks"
DATA_DIR = ROOT / "data" / "kwm_stage_a"
TOOLS_DIR = ROOT / "tools"
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results" / "kwm_stage_a"

# ---- 冻结常量（来自任务集 §3-§5、§10） ----
OLLAMA_URL = os.getenv("KWM_OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.getenv("KWM_GEN_MODEL", "qwen2.5:7b")
GEN_TEMPERATURE = 0.0
CHAT_TIMEOUT = float(os.getenv("KWM_CHAT_TIMEOUT", "90.0"))
EMBED_MODEL = os.getenv("KWM_EMBED_MODEL", "bge-m3:latest")
EMBED_DIM = 1024

# 检索：进入上下文的 top-k（与 last_search_trace.context 同语义）
RETRIEVAL_TOP_K = 3
# L1 "命中" 的最低相似度阈值（低于则视为未检索到相关记忆）
L1_HIT_THRESHOLD = 0.30

# 复现目标值（任务集 §12 预跑门 V1；±1% 容差，实测后校准冻结）
TARGET = {
    "A": {
        "delta_x1_dropna": 8.2,      # v1 删除法组均值差（偏）
        "gold_delta": 2.1,           # 金标（完整数据真实 Δ）≈ MI 参考
        "v2_dropna_bias": 0.3,       # v2 删除法 Δ 与金标差 ≤0.3pp
        "v1_dropna_bias": 5.0,       # v1 删除法 Δ 与金标差 >5pp
    },
    "B": {"v1_sum": 440_000, "gold_sum": 500_000},
    "C": {"v1_sum": 58_500, "gold_sum": 50_000},
}

CONDITIONS = ["NONE", "UNCOND", "COND", "COND_MINUS"]
VARIANTS = ["V_dead", "V_revived"]

# 任务得分权重（§3.6 / §4.4 / §5.4）
SCORE_WEIGHTS = {"estimate": 0.4, "ci": 0.2, "semantic": 0.2, "budget": 0.2}
PASS_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Ollama 生成客户端（native tools）
# ---------------------------------------------------------------------------
API_BASE = os.getenv("KWM_API_BASE", "").rstrip("/")       # 云端生成模型（可选）
API_KEY = os.getenv("KWM_API_KEY", "")
API_PROTOCOL = os.getenv("KWM_API_PROTOCOL", "openai")     # "openai" | "anthropic"


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    *,
    model: str = GEN_MODEL,
    temperature: float = GEN_TEMPERATURE,
    timeout: float = CHAT_TIMEOUT,
) -> dict[str, Any]:
    """生成模型调用。KWM_API_BASE 设置时走 OpenAI 兼容 /chat/completions，否则 Ollama。

    返回统一为 Ollama 风格 {'message': {'content': ...}}。
    """
    if API_BASE and API_PROTOCOL == "anthropic":
        system_text = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
        convo = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": model,
            "max_tokens": 4000,
            "temperature": temperature,
            "system": system_text,
            "messages": convo,
            "stream": False,
        }
        # opencode zen 端点：Cloudflare UA 检查 + x-opencode-session 路由 + x-api-key 凭证
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-cli/2.0.0 (external, cli)",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "x-opencode-session": str(uuid.uuid4()),
        }
        req = urllib.request.Request(
            f"{API_BASE}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic chat unavailable at {API_BASE}: {exc}") from exc
        parts = [b.get("text") or "" for b in (data.get("content") or []) if b.get("type") == "text"]
        return {"message": {"content": "".join(parts)}}
    if API_BASE:
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
        req = urllib.request.Request(
            f"{API_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API chat unavailable at {API_BASE}: {exc}") from exc
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {"message": {"content": content}}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama chat unavailable at {OLLAMA_URL}: {exc}") from exc


def embed(text: str, *, model: str = EMBED_MODEL, timeout: float = 60.0) -> list[float]:
    """调用 Ollama /api/embeddings（bge-m3）。"""
    payload = {"model": model, "prompt": text}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            vec = json.loads(resp.read().decode("utf-8")).get("embedding")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama embed unavailable at {OLLAMA_URL}: {exc}") from exc
    if not vec:
        raise RuntimeError(f"embedding response from {model!r} contained no vector")
    v = [float(x) for x in vec]
    # L2 归一化（与 memory_engine.vector.cosine 一致）
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(a[i] * b[i] for i in range(len(a)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# KWM 记忆层：精确注入冻结原文 + 稠密检索 + L1 轨迹
# ---------------------------------------------------------------------------
@dataclass
class MemoryRecord:
    rid: str
    family: str
    condition: str
    content: str
    variant: Optional[str] = None
    source: str = "前段轨迹"
    vec: list[float] = field(default_factory=list)


class KWMMemory:
    """注入 FAIL 记录并按后段查询做稠密检索。

    L1 轨迹定义（与任务集 §6 / §17.1 对齐）：
      - hit：top_k 内存在本家族 FAIL 记录且相似度 ≥ L1_HIT_THRESHOLD
      - enters_context：hit 且被注入到 agent 最终 prompt（此处取 top-1 命中）
    """

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self._embed_cache: dict[str, list[float]] = {}

    def _e(self, text: str) -> list[float]:
        if text not in self._embed_cache:
            self._embed_cache[text] = embed(text)
        return self._embed_cache[text]

    def reset(self) -> None:
        self.records = []

    def inject(self, family: str, condition: str, content: str, variant: Optional[str] = None) -> None:
        self.records.append(
            MemoryRecord(
                rid=f"{family}_{condition}_{len(self.records)}",
                family=family,
                condition=condition,
                content=content,
                variant=variant,
                vec=self._e(content),
            )
        )

    def retrieve(self, query: str, *, family: Optional[str] = None) -> dict[str, Any]:
        qv = self._e(query)
        scored = []
        for r in self.records:
            if family and r.family != family:
                continue
            sim = cosine(qv, r.vec)
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:RETRIEVAL_TOP_K]
        # 命中：top_k 内存在 FAIL 记录且相似度 ≥ 阈值
        hit_rec = None
        for sim, r in top:
            if sim >= L1_HIT_THRESHOLD:
                hit_rec = r
                break
        context_rec = hit_rec  # 进入上下文 = 最高相似度命中（top-1）
        return {
            "query": query,
            "top": [{"rid": r.rid, "family": r.family, "condition": r.condition,
                     "sim": round(s, 4), "content": r.content} for s, r in top],
            "hit": hit_rec is not None,
            "enters_context": context_rec is not None,
            "context_content": context_rec.content if context_rec else None,
            "context_condition": context_rec.condition if context_rec else None,
        }


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (DATA_DIR, TASKS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
