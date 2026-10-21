# 论文一结果审计记录（止损收尾，2026-10-21）

- **审计对象**：论文一《Where Does Long-Term Agent Memory Actually Fail?》（`paper/main.tex`，revision 3 未提交）
- **审计性质**：v2.0 决定的限时止损收尾第一环——**只做影响核心结论可信度的复核，不扩样本、不改设计、不加新实验**
- **审计方法**：① 独立代码路径从原始 `comparison.jsonl` 重算全部核心数字（不复用 `experiments/` 任何函数）；② 重跑冻结统计脚本核对可复现性；③ 实跑测试套件；④ 核对 commit 链与未提交改动
- **环境**： managed Python 3.13.12 + venv（`~/.workbuddy/binaries/python/envs/default`，补装 pytest/pydantic/numpy/requests）；模型侧无 Ollama 调用（纯后验复核）

---

## 1. 核心数字复核（独立复算 vs 记录值）

数据源：`results/study2b_longmemeval_oracle_n50/comparison.jsonl`（50 行，0 错误）。
复算脚本为本次审计新写的独立代码路径（直接读 JSONL 字段，未 import `experiments/paired_stats.py`）。

| 核心数字 | 记录值（AUDIT-2026-10-18 / paired_stats.txt） | 独立复算值 | 状态 |
|---|---|---|---|
| real_memory 语义正确率 | 31/47 = 0.660 | 31/47 = 0.6596 | ✅ 一致 |
| oracle_memory 语义正确率 | 40/48 = 0.833 | 40/48 = 0.8333 | ✅ 一致 |
| oracle_context 语义正确率 | 36/48 = 0.750 | 36/48 = 0.7500 | ✅ 一致 |
| real vs oracle_memory | b=10, c=2, exact p=0.0386 | b=10, c=2, exact p=0.0386 | ✅ 一致 |
| real vs oracle_context | b=6, c=1, exact p=0.1250 | b=6, c=1, exact p=0.1250 | ✅ 一致 |
| bootstrap CI（real vs OM） | [+0.043, +0.319] | （重跑冻结脚本输出一致，见 §2） | ✅ 一致 |
| 生命周期首失（real） | formation 45/50 = 90%，none 5 | 45/50 行 formation 阶段有 missing | ✅ 一致 |
| 答案失败分布（real） | F1 9、E0 7、EVAL 3 | F1 9、E0 7、EVAL 3 | ✅ 一致 |
| E0 gate 分布 | sufficient 40 / insufficient 8 / error 2 | 40 / 8 / 2 | ✅ 一致 |

**结论：全部核心数字可从原始数据独立复现，无一处偏差。**

### 1.1 口径厘清（论文写作必须按此表述，已逐例追踪）

`evidence_sufficiency` 的 8 个 insufficient 与答案失败的 7 个 E0 不是同一口径，逐例核对后的完整账目：

- **8 个 insufficient**：7 个成为 `E0_EVIDENCE_INSUFFICIENT` 答案失败；1 个（`9a707b81`，baking class 时间证据不明确）模型仍答对（semantic_correct=True，无失败标签）。
- **2 个 error**：均为 judge 超时（`81507db6`、`15745da0_abs`，TimeoutError），按协议归 `EVAL_SEMANTIC_UNCERTAIN`。
- **EVAL 共 3**：2 个 judge 超时 + 1 个真实语义不确定（`1f2b8d4f`）。
- 即：**gate 层 insufficient=8，答案层 E0=7**；论文报告时两个数字都要给且注明定义，不能混用。

---

## 2. 冻结脚本可复现性

重跑 `experiments/paired_stats.py --input results/study2b_longmemeval_oracle_n50/comparison.jsonl`，输出与存档的 `paired_stats.txt` 逐行一致（含分类型 breakdown 与 judge agreement 段）。存档文件未被人为编辑痕迹（与当次运行输出相同）。

---

## 3. 跨研究一致性（四臂对照——核心发现的稳健性证据）

| 研究 | 记忆栈 / 抽取器 | 答案模型 | 首失 formation 占比 | real vs oracle_memory |
|---|---|---|---|---|
| 2B n=50 | 自建正则抽取器管线 | qwen2.5:7b | 45/50 = 90% | 66.0→83.3%，+17.4pp，p=0.0386 |
| 2B n=200 | 自建正则抽取器管线 | deepseek-v4.1-flash | 155/200 = 77.5% | 52.1→69.1%，+11.7pp，p=0.0041 |
| 3A n=50 | **Mem0（外部系统）** | qwen2.5:7b | 40/50 = 80% | 58.3→79.6%，+21.3pp，**p=0.0063** |
| 2F n=50 | 自建管线 + **LLM 抽取器** | qwen2.5:7b | 13/50 = 26% | 70.8→71.4%，+0.6pp，n.s. |

**两条硬结论**：

1. **"形成阶段是主要失败边界"跨三个独立记忆栈复现**（自建 90% / Mem0 80% / deepseek 77.5%），不是单一管线的伪影。
2. **2F 是"瓶颈在抽取器而非基座模型"的因果证据**：同一管线只换抽取器（正则→LLM），formation recall 0.236→0.838，oracle 差距从 +17.4pp 塌缩到 +0.6pp。论文的 extractor-bottleneck 主张由此获得操作级支持（对应最初审计 I6 的"论文中单独报告抽取器覆盖率"——2F 即为该报告）。

---

## 4. Judge 可靠性现状（四家族盲判研究，study2e + study2b_judge）

**四家族两两一致率 / kappa**（n=50 oracle 记录上 60 个裁决）：

| 对 | raw | kappa |
|---|---|---|
| qwen vs mistral | 0.877 | 0.530 |
| qwen vs deepseek | 0.754 | 0.459 |
| gemma vs mistral | 0.895 | 0.345 |
| qwen vs gemma | 0.782 | 0.158 |
| gemma vs deepseek | 0.667 | 0.201 |
| deepseek vs mistral | 0.617 | 0.115 |

**作者裁定（16 个分歧案例）**：deepseek 13/16 最准，qwen 8/14，gemma 3/14，mistral 1/14。主 judge（qwen2.5:7b）中游；mistral/gemma 偏松（correct 率 ~90%），deepseek 偏严（40%）。

**必须写入论文的诚实记录**：

1. 主结果 judge 与答案模型同家族（qwen 判 qwen），虽有四家族盲判 + 作者裁定兜底，**judge 依赖性是已知限制**，Limitations 必须明示。
2. 2B-n200 上交叉家族重判（qwen 重判 deepseek 的 30 记录子样本）出现过 **Real-vs-OM 符号翻转**（deepseek 修复版 +25.0pp p=0.065 vs qwen 重判 −8.0pp p=0.69）——子样本小、非结论性，但论文不得宣称效应方向对 judge 完全不敏感；缓解证据是 n=200 全量的三向边界分析（§4.1）。
3. uncertain 处理已有完整敏感性分析：worst-case（uncertain 判错）下 2B-n200 的 OM 差距降为 +5.5pp（p=0.135）；**精确匹配确定通道**（EM=1 的答案必然正确，非判断）下为 +18.2pp，p=1e-6。论文应主报确定通道修正值 + worst-case 边界。

### 4.1 2B-n200 边界分析存档（已存在于 results/study2b_longmemeval_n200_deepseek/sensitivity.txt）

- uncertain-as-wrong：real 0.425 / OM 0.480 / OC 0.550（OM 差距 +5.5pp，p=0.135，n.s.）
- uncertain-as-correct：0.610 / 0.785 / 0.880（+17.5pp，p<1e-4）
- 精确匹配修正：0.592 / 0.769 / 0.876（+18.2pp，p=1e-6）
- 全决断子集（110 行）：0.627 / 0.755 / 0.845
- OM vs OC（审稿人 P1-3）：pairwise-decided n=120，+8.3pp，p=0.0414，CI [+0.017, +0.158]

---

## 5. 统计功效的诚实记录（论文必须报告）

存档 `results/power_analysis.txt`（20000 次模拟/点，seed=20261019）：

- **2B n=50**：real vs OM 在观测结构下 achieved power 仅 **57.1%**（n=47）——p=0.0386 是**功效不足下的显著**；同效应 80% 功效需 n≈80。
- **3A n=50（Mem0）**：achieved power **84.5%**（p=0.0063）——功效充足。
- 合并解读：两臂方向一致、效应量相近（+17.4 / +21.3pp），合并证据支持主张；但单臂 2B 不得作为唯一证据，论文须报告两臂 + 功效分析。

---

## 6. 测试与可复现性

| 项 | 状态 |
|---|---|
| 测试套件 | **72 passed**（venv 实跑；比 10-18 审计记录的 58 多 14 个，为后续轮次新增的 judge/extractor/边界测试） |
| commit 链 | n=50 结果记录commit `a0fbf66` 是当前 HEAD（`0da190e`）的祖先，结果-代码对应关系完整 |
| requirements | 已含 pytest/pyarrow（I5 部分完成项确认在位） |
| 未提交改动 | `paper/main.tex`（+1267/−517，revision 3）、`paper/*.pdf`（已重编译）、`paper/refs.bib`（+10）、`docs/ENVIRONMENT.md`（+272）、`experiments/paired_stats.py`（+92）——**尚未 commit，见 §8 行动项** |

---

## 7. Related work 差异化定位（写作口径）

### 7.1 最接近竞争者：MemFail（2605.26667，UC Berkeley，2026-05）

MemFail 将记忆系统分解为 summarize/storage/retrieve 三操作，用**对抗性合成数据集**逐操作隔离故障模式，评估四个记忆系统。差异化要点：

| 维度 | MemFail | 本研究 |
|---|---|---|
| 归因单位 | **系统操作**（哪个组件类型出错） | **证据项**（哪条 gold evidence 在哪个边界首次不可用） |
| 数据 | 对抗性合成（Conditional-Facts / Coexisting-Facts 等五个数据集） | 自然基准（LongMemEval 问题 + 真实会话轨迹） |
| 因果控制 | 操作隔离靠数据集对抗设计 | **oracle 三条件**（Real / Oracle Memory / Oracle Context）在同记录上配对，分离记忆侧丢失与下游推理 |
| 推理失败 | "measured only for completeness" | F6 仅作 candidate，且要求 gate sufficient + 全链路到达 + 语义判定错误三条件同时成立 |
| 生命周期粒度 | 三操作 | F1–F6 六阶段 + E0 反循环 gate + EVAL 语义不确定 |

一句话：MemFail 问"哪类**操作**会坏"，本研究问"这条**证据**死在哪"——操作级 vs 证据级，合成对抗 vs 自然基准，单臂诊断 vs oracle 配对因果。

### 7.2 次级近邻（进 related work，不构成竞争）

- **MeClear（2609.09115）**：查询级识别并清除有害记忆（Shapley 归因）。差异：它是**处置**研究（压制/删除），本研究是**归因**研究（首次丢失定位）；其归因是操作级反事实，本研究是证据级生命周期。
- **ACL 2026 经验跟随（2505.16067）**：经验模仿导致的 error propagation / misaligned replay。差异：现象描述 + 增删操作缓解，无生命周期归因、无 oracle 控制；可引作"记忆下游误用"的相邻现象。
- **Useful Memories Become Faulty（2605.12978）**：持续更新中记忆效用先升后降。差异：更新场景的聚合曲线观察，无逐证据归因；其"抽象丢失适用条件"机制可作为 F3 讨论引文。
- **BASM（2608.22339）**：技能记忆的适用边界字段（KWM 论文一的最强近邻，论文一仅需一句带过：其证明边界信息的因果作用，但对象是成功技能、无证据级归因）。

---

## 8. 行动项与剩余风险

**收尾行动项（按序）**：

1. **提交 revision 3**（§6 未提交改动）——`/commit` 走标准流程；这是"可独立发布版本"的最后一步。
2. 论文修订时落实三项诚实报告：① judge 四家族可靠性 + 作者裁定（§4）；② 2B 功效不足 + 3A 功效充足的双臂报告（§5）；③ uncertain 边界分析主报确定通道修正值（§4.1）。
3. Limitation 段落按 §4/§5 清单成文（judge 同家族、n=50 单臂功效、natural benchmark 的 oracle 构造依赖 gold evidence 标注质量）。
4. 论文一定稿即转 tech report + 开源仪器路线（memory_engine/evaluation/tracing 已按 KWM 基础设施标准维护，测试 72 passed 可直接作为 artifact 质量证据）。

**剩余风险（不影响止损决策，记录在案）**：

- R1（中）：judge 同家族 + 子样本符号翻转先例——若审稿人要求换 judge 全量重判，成本以天计，暂不主动做（双臂 + 四家族研究 + 作者裁定已是同等纪律工作的上限配置）。
- R2（低）：LongMemEval 的 gold evidence 标注质量决定 oracle 构造上限——协议中 provenance + 内容双校验已缓解，论文方法节已有说明。
- R3（低）：2F 的 LLM 抽取器与主实验正则抽取器不同后端，跨后端比较只在 formation recall 层面成立，论文不得外推为"任何 LLM 抽取器都消除 F1"。

---

## 附：本次审计复算方式留档

- 独立复算：managed Python 3.13.12 直读 `results/study2b_longmemeval_oracle_n50/comparison.jsonl`，按 `semantic_correct` 字段重算三条件正确率、不一致对 b/c、精确 McNemar（二项尾概率）、首失分布、gate 分布；未调用项目内任何分析代码。
- 可复现性：`python -m experiments.paired_stats --input <same file>` 输出与存档逐行一致。
- 测试：`python -m pytest tests/ -q` → 72 passed（venv：`C:/Users/13742/.workbuddy/binaries/python/envs/default`）。
