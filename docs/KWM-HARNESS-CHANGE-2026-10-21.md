# KWM Harness 变更记录：方法菜单化（2026-10-21）

## 变更标识
- 日期：2026-10-21；阶段：A（工件实现 + 预跑）；影响：主实验后段 agent harness。
- 处置依据：用户拍板（2026-10-21 15:57，选项 A「方法菜单 harness」）。

## 触发事实（三级提示逐级强化的证伪过程）
在自由代码执行（提示式 ReAct + run_python）下，UNCOND/V_dead 条件中：
1. qwen2.5:7b：读到记忆、在结论中承认"dropna 有偏"，仍用 dropna（Δ=8.105）。
2. gemma3:12b：同上；即使提示"记忆指出失败的方法禁止使用、必须改用
   robust_groupdiff.py"，仍坚持 dropna；最终一轮改为对全列直接求 mean
   （pandas mean 隐式跳 NaN = 同一删除法），仍得 8.105，并声称"未使用 dropna"。
结论：自由代码执行下本地 7B/12B 模型会"合理化绕过"记忆约束，
「记忆 → 方法切换」操纵无法经纯提示产生。

## 变更内容
1. **方法菜单化**：agent 不再写自由代码。每轮输出 `METHOD: <名称>`（系统执行该方法的
   **冻结实现**并回灌 OBSERVATION）或 `FINAL: <结论>`（须至少执行过一次 METHOD）。
   - 记忆效应作用于显式方法选择（对齐任务集 L2 行为指标：路径选择/显式放弃）。
   - 方法实现冻结 → 跨条件无代码质量混淆；杜绝"隐式跳 NaN"式绕过。
   - `methods_chosen` 序列写入结果 JSON，作为 L2 主行为指标。
2. **移除提示词污染**：删除此前无条件附加的"【记忆遵循指令】禁用 dropna/必须用
   robust_groupdiff"块（它会同时污染 NONE 条件）。记忆块恢复中性措辞
   （"请自行判断其是否适用于当前数据"）。
3. **家族 C 数据修复**（实现与 docstring 意图不符的缺陷）：
   - 原 v2 customers 仍有 1180 行（重复 id 未修复），"v2 直接 join 正确"不成立。
   - 修复后 v2 = 1000 行唯一 customer_id + 90 对跨客户大小写变体名（'ACME0xx'/
     'acme0xx' 属不同 id）+ canonical_name 每客户唯一。
   - v1 修正理解：180 个重复 id，每 id 词干唯一（'ACME0xx'/'acme0xx' 同 id）→
     name 归一去重在 v1 不误合并（Y 完全正确）。
   - 查询语义冻结：X=直接 JOIN 按 id 聚合（v1 扇出 59,280 / v2 正确 50,000+992）；
     Y=按 UPPER(name) 去重后 JOIN（v1 正确 50,000+991 / v2 总额对但客户数 904≠992，
     C-C2 陷阱）；Z=仅汇总 orders（总额对、无客户信息，语义扣分）。
4. **三家族方法菜单冻结**（首项 = 朴素方法 X = NONE 自然后径）：
   - A：naive_delete / impute_group_mean / mi_tool(robust_groupdiff.py)
   - B：naive_groupby / sum_only / legacy_tool(legacy_agg.py)
   - C：direct_join / dedup_by_name / orders_only
5. 冻结实现实测（V_dead / V_revived）：
   - A：naive 8.105(biased)/1.882(ok)；mi_tool 2.031(ok)/assert 崩溃；impute 待跑。
   - B：naive_groupby 440,000(biased)/500,000(ok)；legacy_tool 500,000(ok)/崩溃；
     sum_only 500,000(无月表)。
   - C：direct_join (59,280,991)/(50,000,992)；dedup_by_name (50,000,991)/(50,000,904✗)；
     orders_only 50,000。

## 对预注册设计的影响
- H-A/H-B/H-C 假设与 2×4 矩阵不变；主结果指标（任务成功率加权）不变。
- L1 轨迹指标新增 `methods_chosen`；"显式放弃"判定映射为"选择非 X 方法"。
- V6 门重述：NONE 条件下 agent 首选方法 = X（菜单首项）的比例须显著高于随机
  （3 选 1 基线 33%；门限暂定 ≥2/2 重复且两变体一致，预跑后冻结）。
- 风险记录：菜单本身可能吸引 agent 偏离首项（描述效应）；若 NONE 首选非 X，
  需回改菜单措辞中性化并重跑，不得改数据。

## 附录：2026-10-21 晚间三层模型诊断（harness v3 + 云端协议接入）

### 云端接入
KWM_API_BASE/KWM_API_KEY/KWM_API_PROTOCOL=anthropic → opencode zen 端点
（Cloudflare UA 检查 + x-opencode-session 路由 + x-api-key 凭证；thinking 块解析跳过）。
模型 deepseek-v4.1-flash，单次调用 2–3s。

### 三层诊断结果（家族 A，2×4 全矩阵各 1 次）
| 层 | 模型 | NONE 首选 | 记忆条件行为 |
|---|---|---|---|
| floor | qwen2.5:7b | X（naive_delete） | 承认记忆但不行动（全部 8 格选 X） |
| mid | gemma3:12b | X | 同上 + 反向解读记忆/捏造适用性 |
| ceiling | deepseek-v4.1-flash | mi_tool（最优法） | 全部 8 格选 mi_tool；V_revived 全部因脚本断言崩溃 → 诚实报告无法估计（含 COND，未用适用条件预判） |

### 结构性诊断（阶段 A 核心发现）
P-A1 操纵要求"NONE 自然后径 = X（有偏）"，但这在能力轴两端都不成立：
- 无能模型：不因记忆改变行为（floor：记忆敏感性缺失）；
- 胜任模型：无需记忆即避开 X（ceiling：先验统计知识覆盖了记忆的信息价值）。
即**方法选择型操纵的"记忆价值窗口"在能力轴上是空的**——记忆只能帮助
"不知道删除法有偏但能执行记忆指示"的模型，而这类模型在实测的两端都不存在。
COND 的适用条件判别（V_revived 中推出"X 安全"）在三层模型中均未自发出现
（deepseek 在 COND/V_revived 中仍选 mi_tool 而非先检验缺失机制）。
此为"权重知识 vs 记忆经验"边界的直接证据，与 BASM 的 App. Skill 非单调掉点
观察同向。

### 处置建议（待用户决策）
1. 止损收尾：以"三层诊断"为阶段 A 证伪结论（与论文一止损同构）；
2. 设计重构：把操纵从"方法选择"改为"SOP 冲突"（任务规定标准流程=删除法，
   记忆与之冲突）——测的是记忆 vs 程序遵从，偏离原 KWM 问题；
3. 换任务域：找"权重里没有、只能靠记忆"的信息类型（如项目私有参数），
   重建 NONE 自然后径。

---

## 勘误（2026-10-21 晚）

- 本文档附录表格中 ceiling 行 "全部 8 格选 mi_tool；V_revived 全部因脚本断言崩溃
  → 诚实报告无法估计" 与 "先验统计知识覆盖了记忆的信息价值" 并列，构成矛盾：
  V-revived 中崩溃的工具不是最优法。正确表述：**八格机械选 mi_tool（策略惯性/
  过度泛化），V_dead 通过、V_revived 确定性失败**；"先验知识覆盖"解读撤回。
- runner 缺陷记录：run JSON 未写 model 字段；变体间复用同名 tag 造成覆盖。
  已入库 results/kwm_stage_a/logs/ 保全日志；整改项记入论文二草稿 §7.4(4)。
