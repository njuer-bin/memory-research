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
