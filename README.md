# Agent Memory Failure Attribution

**From Memory Formation to Reasoning: Diagnosing Information Loss in Long-Term LLM Agents**

研究问题：

> 当一个长期记忆 LLM agent 给出错误答案时，失败**首先**发生在生命周期的哪个边界？

本仓库是 `njuer-bin/memory-failure-attribution` 的研究延续版（v2）：保留原研究的问题定义、生命周期 trace 与 oracle 控制设计，修复外部审计发现的归因仪器与测量有效性问题（见 `docs/AUDIT-2026-10-18.md`），然后在冻结协议下重跑实验并撰写论文。

## 生命周期与失败分类学

    Conversation → Formation → Storage → Evolution → Retrieval → Reranking → Context → Reasoning → Answer

- **F1 Formation** — 证据在对话中存在但未被抽取为可用记忆
- **F2 Storage** — 抽取的记忆未持久化或不可访问
- **F3 Evolution** — 更新、陈旧事实、冲突处理错误（当前版本：埋点中，见 docs/PROTOCOL.md）
- **F4 Retrieval** — 证据在记忆中但未被检索/重排进可用集
- **F5 Context** — 证据在进入最终答案上下文前丢失
- **F6 Reasoning** — 证据到达最终上下文且语义判错（candidate 归因）
- **E0 Evidence Insufficiency** — Gold Evidence 本身不能回答该问题
- **EVAL Semantic Uncertainty** — 评测器不可靠到无法支持归因

## Oracle 控制

1. **Real Memory**：真实对话 → 真实记忆 → 检索 → 上下文 → 答案模型
2. **Oracle Memory**：Gold Evidence 注入记忆侧路径（控制条件）
3. **Oracle Context**：Gold Evidence 直接进入最终答案上下文（控制条件）

## 当前状态

- [x] 外部审计完成（6 Critical / 6 Important / 5 Minor，见 `docs/AUDIT-2026-10-18.md`）
- [ ] Phase 1：修复 C1/C2（归因协议分裂）、C3–C6（测量有效性）
- [ ] Phase 2：冻结协议 → LongMemEval n=50 → 配对统计 + judge 可靠性 → LoCoMo 复跑
- [ ] Phase 3：论文初稿

## 运行

    pip install -r requirements.txt
    pytest tests/ -q
    python -m experiments.run_oracle_experiments --dataset longmemeval_s_sample10 --limit 10

实验环境（Ollama、模型固定、种子）见 `docs/ENVIRONMENT.md`；实验协议与冻结规则见 `docs/PROTOCOL.md`；研究日志见 `docs/RESEARCH_LOG.md`。

## 数据

基准数据位于 `datasets/data/`（LongMemEval、LoCoMo-Refined 公开样本）。公开发布前需核对各基准的再分发许可，见 `datasets/manifest.json` 的 license_note。
