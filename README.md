# Agent Memory Failure Attribution

**From Memory Formation to Reasoning: Diagnosing Information Loss in Long-Term LLM Agents**

研究问题：

> 当一个长期记忆 LLM agent 给出错误答案时，失败**首先**发生在生命周期的哪个边界？

本仓库是 `njuer-bin/memory-failure-attribution` 的研究延续版（v2）：保留原研究的问题定义、生命周期 trace 与 oracle 控制设计，修复外部审计发现的归因仪器与测量有效性问题（见 `docs/AUDIT-2026-10-18.md`），然后在冻结协议下重跑实验并撰写论文。

## 生命周期与失败分类学

    Conversation → Formation → Storage → Evolution → Retrieval → Reranking → Context → Reasoning → Answer

- **F1 Formation** — 证据在对话中存在但未被抽取为可用记忆
- **F2 Storage** — 抽取的记忆未持久化或不可访问
- **F3 Evolution** — 更新、陈旧事实、冲突处理错误（formation 全量 vs governance 后 active 集的差集，见 docs/PROTOCOL.md §5）
- **F4 Retrieval** — 证据在记忆中但未被检索/重排进可用集
- **F5 Context** — 证据在进入最终答案上下文前丢失
- **F6 Reasoning** — 证据到达最终上下文且语义判错（candidate 归因）
- **E0 Evidence Insufficiency** — Gold Evidence 本身不能回答该问题
- **EVAL Semantic Uncertainty** — 评测器不可靠到无法支持归因

## Oracle 控制

1. **Real Memory**：真实对话 → 真实记忆 → 检索 → 上下文 → 答案模型
2. **Oracle Memory**：Gold Evidence 作为已形成记忆（active facts）注入，走**完整**检索/重排/上下文管线
3. **Oracle Context**：Gold Evidence 直接进入最终答案上下文（管线旁路，纯推理控制）

## 当前状态

- [x] 外部审计完成（6 Critical / 6 Important / 5 Minor，见 `docs/AUDIT-2026-10-18.md`）
- [x] Phase 1：修复 C1/C2（归因协议统一为 `evaluation/attribution.py` + 反循环 E0 gate）
- [x] Phase 1：修复 C3–C6（provenance 内容校验 / F3 埋点 / 确定性通道收窄 / oracle 条件区分）
- [ ] Phase 2：冻结协议 → LongMemEval n=50 → 配对统计 + judge 可靠性 → LoCoMo 复跑
- [ ] Phase 3：论文初稿

Phase 1 的协议规则已冻结并写入 `docs/PROTOCOL.md`（归因决策顺序、E0 反循环、
provenance 内容校验、F3 定义、oracle 三条件、R1/R2 窄规则）。冒烟测试
（longmemeval_s_sample10，2 条）已通过：三条件上下文互相独立、code_commit
随行记录。

## 运行

    pip install -r requirements.txt
    pytest tests/ -q
    python -m experiments.run_oracle_experiments --dataset longmemeval_s_sample10 --limit 10

实验环境（Ollama、模型固定、种子）见 `docs/ENVIRONMENT.md`；实验协议与冻结规则见 `docs/PROTOCOL.md`；研究日志见 `docs/RESEARCH_LOG.md`。

## 数据

完整基准数据集**不进入本公开仓库**（再分发条款待核对，见 `datasets/manifest.json`
各条目的 `provenance` / `license_note`）。仓库中只提交小规模研究摘录：

- `datasets/data/longmemeval_s_sample10.json` — LongMemEval-s 10 条摘录（Study 2A）
- `datasets/data/longmemeval_oracle_n50.json` — LongMemEval oracle 50 条分层研究集（Study 2B）
- `datasets/data/locomo_n50.jsonl` — LoCoMo-Refined 50 问研究集（仅问题/证据，对话在加载时从本地全量数据 join）

复现完整研究需自行获取原始数据并按 manifest 的 `provenance` 放置到
`datasets/data/` 下；研究集本身可由脚本以固定种子重建：

    python -m experiments.build_study_set --dataset longmemeval_oracle --n 50 --seed 20261018 --out longmemeval_oracle_n50 --primary question_type
    python -m experiments.build_study_set --dataset locomo_refined_public --n 50 --seed 20261018 --out locomo_n50 --raw
