# 证据与结论对账（Evidence–Claim Reconciliation Audit）

**日期**：2026-10-21 晚
**触发**：第二轮外部评审行动 #1——"完成一次严格的证据与结论对账：统一确认性与
探索性实验的统计口径，核对 methods_chosen 和最后采纳的方法，把所有能从日志恢复
的运行按冻结规则评分，并将无法恢复或无法评分的部分明确列出。"
**方法**：以保留工件（run JSON + 已入库执行日志）为唯一权威，逐运行重建
L2（尝试序列 / 最终采纳方法）与 L3（冻结检查点评分）；不修改任何冻结规则；
不回溯改写历史文档正文，只追加勘误。

---

## 1. 结论先行

对账发现 **草稿 v0.2 与保留工件存在三处事实矛盾**（均已在 v0.3 中修正）：

| # | v0.2 表述 | 保留工件中的事实 | 修正 |
|---|---|---|---|
| C1 | §6.1：确认性试点"全部八格选择 X" | `kwm_pilot24.log`：NONE/UNCOND 四格承诺 X；**COND 与 COND-MINUS 四格探索后采纳非 X 方法**（V-dead 采纳 mi_tool Δ=2.0306；V-revived 采纳 impute_group_mean Δ=1.8818） | v0.3 §6.1 全表重建 |
| C2 | §6.2 行 2′：Gemma 多次尝试"自主仲裁到正确答案" | `kwm_gem_menu.log`：Gemma 探索全部三个方法后**最终采纳有偏的 naive_delete（8.1051）**，并在 UNCOND 格反向解读记忆（"历史经验中该方法偏差相对较小"） | v0.3 §6.2 修正 |
| C3 | §6.2 行 3 / 止损报告 §3："Gemma 8/8 选 X，行为与 floor 相同" | `kwm_gem_rule.log`：Gemma 在多次尝试+规则下 **V-dead 格明确引用记忆切换到 mi_tool**（UNCOND/COND 两格），NONE 格也采纳 mi_tool；"4/4 选 X + 捏造适用性"实际来自 `kwm_bind.log`（决策点绑定变体） | v0.3 §6.2 重建；止损报告追加勘误 6 |

**关键裁定**：G-core（预注册停机门槛）的 **裁定结果不变、但依据表述必须更换**。
P-A1 预注册对比是 UNCOND vs NONE 的跨变体方向性差异；实测 UNCOND 与 NONE 四格
行为完全一致（全部承诺 X）→ 方向性效应未出现 → 试点仍按预注册规则判定失败。
但"行为层指标不动"这一表述是错的：行为动了，只是**动的位置不在预注册对比上**
（COND/COND-MINUS 移动但 COND-MINUS≡COND，不能归因于条件内容；UNCOND 不动）。
"主要终点 H-A 已被完整检验并得到阴性结论"的表述相应收窄为"确认性试点触发
预注册行为层停机条件"（评审行动 #1 原文建议的措辞）。

---

## 2. 全量运行盘点（家族 A）

阶段 A 共 **43 次 agent 运行**。10 次有存活 JSON（8 DeepSeek + 2 qwen r1，
已由 `kwm_score_retro.py` 评分）；**33 次仅存于日志**（JSON 被同名 tag 覆盖
或从未写盘），本次由 `kwm_score_logonly.py` 按同一冻结规则评分
→ `results/kwm_stage_a/l3_scoring_logonly.json`（每条含日志出处与运行序号，
转录可逐行审计）。

| Harness 变体 | 模型 | 运行数 | 存活 JSON | 日志 |
|---|---|---|---|---|
| v1 自由代码 ReAct | qwen2.5:7b | 2（NONE/UNCOND × V-dead） | 0 | kwm_pilot.log, kwm_uncond.log |
| v1 自由代码 ReAct | gemma3:12b | 1（UNCOND × V-dead） | 0 | kwm_gem.log |
| 确认性试点：菜单+多次尝试 | qwen2.5:7b | 8（2×4 全矩阵） | 0 | kwm_pilot24.log |
| 菜单+多次尝试 | gemma3:12b | 2（V-dead NONE/UNCOND） | 0 | kwm_gem_menu.log |
| 菜单+多次尝试+记忆规则 | gemma3:12b | 6 | 0 | kwm_gem_rule.log |
| 决策点绑定 | gemma3:12b | 4 | 0 | kwm_bind.log |
| v3 单次承诺（budget=1） | qwen2.5:7b | 8（2×4 全矩阵） | 0 | kwm_v3.log |
| v3 稳定性复测 | qwen2.5:7b | 4 | 2（r1） | kwm_stab.log |
| v3 单次承诺 | deepseek-v4.1-flash | 8（2×4 全矩阵） | 8 | kwm_cloud.log |

**唯一性审计（2026-10-21 深夜追加，第三轮评审行动 #1）**：
`experiments/kwm_run_manifest.py` 机械生成 43 行唯一运行清单
（`results/kwm_stage_a/run_manifest.json` + `run_manifest.md`，字段：run_id/模型/
harness/变体×条件/证据源/L3 状态/分组），并机检三项条件：**(V-a)** 唯一 run_id
数 = 43；**(V-b)** 10 个 JSON 证据运行与 33 个仅日志运行两池零重叠——
`kwm_stab.log` 四次运行的归属显式记录（run 2/4 即存活 r1 JSON，仅计入 JSON 池；
run 1/3 为被覆盖的 r0，仅计入 log-only 池）；**(V-c)** 全部分组数字
（8+2 JSON；8 试点 + 3 v1 + 12 Gemma + 10 qwen 单次承诺 log-only）可由清单机械
汇总复现。结论：无重复计数。

**无法恢复 / 无法评分清单**（完整列举）：1. **草稿 v0.2 §6.2 行 2**（"qwen 菜单+多次尝试 4/4 选 X"）：无对应保留日志。
   最接近的工件是确认性试点（kwm_pilot24.log），但其实测为 4/8 承诺 X——
   行 2 可能与试点 V-dead 半区混淆，**该行声明为不可验证**（v0.3 标注）。
2. **草稿 v0.2 §6.2 行 5**（"记忆移至 user 消息首位，Gemma 无效"）：无保留日志，
   **不可验证**（仅 harness 变更记录有文字记载）。
3. 33 个仅日志运行的 OBSERVATION 全文payload 只保留截断摘要；工具完整输出
   （如 impute_group_mean 的完整 stdout）无法二次核验——评分只使用日志中
   明确记录的数值。
4. C-A3（语义 judge）对全部 43 次运行均不可评——judge 未建，按冻结规则以
   区间报告。

---

## 3. 统一口径：尝试序列 ≠ 最终采纳方法

评审指出 `methods_chosen` 的歧义必须消除。自本次对账起，L2 指标统一为两个
字段：**methods_tried**（探索序列）与 **method_adopted**（FINAL 答案中采纳的
方法及其估计值）。评分一律使用 adopted 方法的输出。

## 4. 确认性试点重建（kwm_pilot24.log，qwen2.5:7b，菜单+多次尝试）

| 格 | tried | **adopted** | 估计 | CI 宽 | s1 | s2 | s4 | 得分区间 | 裁定 |
|---|---|---|---|---|---|---|---|---|---|
| NONE / V-dead | X | X | 8.1051 | 2.124 | 0 | 1 | 1 | [0.4, 0.6] | unjudgeable |
| NONE / V-revived | X | X | 1.8818 | 2.080 | 1 | 1 | 1 | [0.8, 1.0] | pass_guaranteed |
| UNCOND / V-dead | X | X | 8.1051 | 2.124 | 0 | 1 | 1 | [0.4, 0.6] | unjudgeable |
| UNCOND / V-revived | X | X | 1.88 | 2.08 | 1 | 1 | 1 | [0.8, 1.0] | pass_guaranteed |
| COND / V-dead | X→impute→mi | **mi_tool** | 2.0306 | — | 1 | 0 | 1 | [0.6, 0.8] | pass_guaranteed |
| COND / V-revived | X→impute | **impute** | 1.8818 | 1.709 | 1 | 1 | 1 | [0.8, 1.0] | pass_guaranteed |
| COND-MINUS / V-dead | X→impute→mi | **mi_tool** | 2.0306 | — | 1 | 0 | 1 | [0.6, 0.8] | pass_guaranteed |
| COND-MINUS / V-revived | X→impute | **impute** | 1.88 | 1.71 | 1 | 1 | 1 | [0.8, 1.0] | pass_guaranteed |

**如何解读（全部为 n=1 观察级陈述）**：

1. **P-A1（UNCOND vs NONE 跨变体方向性差异）未出现**——两条件四格行为一致。
   G-core 依预注册规则判失败：**裁定不变，依据更正**。
2. COND 与 COND-MINUS 在两个变体下**行为完全相同**（H-C 无差异信号），因此
   COND/COND-MINUS 格的移动**不能归因于适用条件内容**；与 UNCOND 的差异同样
   无法与逐格随机波动区分（每格 n=1）。
3. V-dead 下 COND/COND-MINUS（pass_guaranteed 0.6）与 NONE/UNCOND
   （unjudgeable 0.4）的 L3 分层**方向上**与"记忆存在有帮助"一致，但
   (a) COND-MINUS 同步移动否决内容归因；(b) UNCOND 含完整失败事实却不动的
   异常无法解释；(c) 单格 n=1。**不作为任何效应证据，仅如实入档。**
4. 全部 43 次运行中，**没有任何一次**在方法选择前利用 COND 适用条件预判
   "路径已安全"（前瞻性判别为零，与 v0.2 §7.1(3) 一致）。

## 5. 探索性阶梯重建（以保留日志为准）

| 行 | Harness（保留日志） | 模型 | adopted 行为摘要 | 与 v0.2 的差异 |
|---|---|---|---|---|
| 1 | v1 自由代码（pilot/uncond/gem.log） | qwen + gemma | 三次运行均落到删除估计量（Gemma 经 skipna mean 变体绕过并否认） | 一致 |
| 2 | （无保留日志） | qwen | — | **改为不可验证** |
| 2′ | gem_menu.log | gemma | 探索全部三方法后**采纳有偏 X**（UNCOND 格反向解读记忆） | **修正**（v0.2 称"仲裁到正确答案"） |
| 3 | gem_rule.log | gemma | NONE/UNCOND/COND 的 V-dead 格采纳 mi_tool（两格明示记忆引用切换）；V-revived 崩溃后回退 X | **修正**（v0.2 称"8/8 选 X"） |
| 4 | bind.log | gemma | 4/4 承诺 X；V-revived 捏造适用性（"缺失模式相似"） | 一致（v0.2 行 3 的描述实际对应本行） |
| 5 | （无保留日志） | gemma | — | **改为不可验证** |
| 6 | cloud.log | deepseek | 八格机械选 mi_tool；V-dead 4/4 pass、V-revived 4/4 fail_guaranteed | 一致（v0.2 §6.3） |
| — | v3.log + stab.log | qwen | v3 单次承诺下 8/8 + 2 复测全部承诺 X；V-revived 格 0.8 pass、V-dead 格 0.4 unjudgeable | 补充入档 |

## 6. 全谱 L3 汇总（43 次运行）

| 裁定 | 存活 JSON（10） | 仅日志（33） | 合计 |
|---|---|---|---|
| pass_guaranteed | 4（deepseek V-dead） | 18 | 22 |
| fail_guaranteed | 4（deepseek V-revived） | 1（v1 qwen NONE/V-dead，无 CI） | 5 |
| unjudgeable（C-A3 决定性） | 2（qwen r1） | 14 | 16 |

**对草稿主张的最终裁定**：

- "任务成功完全由 模型×变体 决定、记忆条件零方差"——**只在 10 个存活 JSON
  （v3 单次承诺 harness）范围内成立**；跨全部 43 次运行不成立（试点 COND/
  COND-MINUS 格移动；gem_rule 多格切换）。v0.3 收窄表述。
- 稳健存活的核心观察：①UNCOND vs NONE 预注册对比在**所有** harness×模型下
  均未出现方向性差异；②单次承诺（budget=1）下三层模型无一切换；③COND 适用
  条件的前瞻性判别在 43 次运行中零出现；④DeepSeek V-revived 四格确定性失败
  （机械沿用崩溃工具）；⑤Gemma V-revived 崩溃后**能**回退 X（多次尝试预算下）。

## 7. 评分口径备注（评审问题 4 的记录）

- `pass_guaranteed` 底分问题属实：冻结权重下 C-A1+C-A4=0.6 已达门槛，
  无 CI、无语义分也可 pass。这是**指标效度缺陷**，按纪律不在本轮修改冻结规则，
  已写入 v0.3 §8 并纳入协议 v2 的必要条件重构（C-A1 类"点估计入容差"改为
  必要条件，CI/语义/成本作加权加分）。
- qwen V-dead 格（0.4 [0.4,0.6]）的裁定统一为 **unjudgeable（C-A3 决定性）**，
  删除 v0.2 中"倾向失败"的措辞——s3=1.0 时恰过线，s3<1.0 才失败，无先验理由
  设定 s3。

---
**产物**：`experiments/kwm_score_logonly.py`（可审计转录）、
`results/kwm_stage_a/l3_scoring_logonly.json`（33 条评分）、本文档。
**下游**：论文二草稿 v0.3、止损报告勘误 6、协议 v2。
