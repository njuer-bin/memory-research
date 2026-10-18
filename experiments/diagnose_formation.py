#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_formation.py
=====================
把 LoCoMo n=50 的 formation 失败拆开：区分「真实 F1（抽取器没抽到）」
与「测量伪影（匹配器验证不了语义表征）」。

用法（在仓库根目录）:
    G:\\aconda\\python.exe -m experiments... # 不需要；这是独立脚本
    G:\\aconda\\python.exe diagnose_formation.py --jsonl results/oracle_experiments/comparison.jsonl

或直接:
    G:\\aconda\\python.exe diagnose_formation.py

默认读取 <repo>/results/oracle_experiments/comparison.jsonl。
只读，不修改任何结果文件；不触发 memory 重建或答案生成。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["FORMATION_OK", "F1a_EXTRACTOR_MISS", "F1c_REPRESENTATION_MISMATCH", "F1d_UNKNOWN"]


def _find_diagnostics(node):
    """在任意嵌套结构中稳健地找到 diagnostics 列表（含 classification 字段）。"""
    if isinstance(node, dict):
        if "diagnostics" in node and isinstance(node["diagnostics"], list):
            if not node["diagnostics"] or isinstance(node["diagnostics"][0], dict):
                return node["diagnostics"]
        for v in node.values():
            r = _find_diagnostics(v)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_diagnostics(v)
            if r is not None:
                return r
    return None


def _stage_of(mode_obj, stage):
    if not isinstance(mode_obj, dict):
        return None
    s = mode_obj.get(stage)
    return s if isinstance(s, dict) else None


def _recall(stage_obj, gold_ids):
    if not stage_obj or not gold_ids:
        return None
    found = {str(x) for x in (stage_obj.get("found") or [])}
    return len(gold_ids & found) / len(gold_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="comparison.jsonl 路径（默认 results/oracle_experiments/comparison.jsonl）")
    args = ap.parse_args()

    if args.jsonl:
        path = Path(args.jsonl)
    else:
        path = Path(__file__).resolve().parent / "results" / "oracle_experiments" / "comparison.jsonl"
    if not path.exists():
        print(f"[ERROR] 找不到结果文件: {path}", file=sys.stderr)
        print("        请在仓库根目录运行，或用 --jsonl 指定路径。", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"读取 {len(rows)} 条记录: {path}\n")

    # ---------- 1) formation 诊断分类分布 (real_memory) ----------
    cls_by_mode = defaultdict(Counter)
    crosstab = Counter()          # (classification, source_message_found)
    per_q_f1a = []
    per_q_f1c = []
    per_q_ok = []
    total_diag = 0

    semantic_counts = []

    for r in rows:
        qid = r.get("question_id")
        modes = r.get("modes", {}) or {}
        for mode_name, mode_obj in modes.items():
            diags = _find_diagnostics(_stage_of(mode_obj, "formation"))
            if not diags:
                continue
            for d in diags:
                cls = str(d.get("classification") or "UNKNOWN")
                cls_by_mode[mode_name][cls] += 1
                if mode_name == "real_memory":
                    total_diag += 1
                    smf = bool(d.get("source_message_found"))
                    crosstab[(cls, smf)] += 1
        # real_memory per-question
        rm = modes.get("real_memory", {}) or {}
        fd = _stage_of(rm, "formation")
        if fd:
            det = fd.get("details", {}) or {}
            ds = det.get("diagnostics") or []
            for d in ds:
                c = d.get("classification")
                if c == "F1a_EXTRACTOR_MISS":
                    per_q_f1a.append(qid)
                elif c == "F1c_REPRESENTATION_MISMATCH":
                    per_q_f1c.append(qid)
                elif c == "FORMATION_OK":
                    per_q_ok.append(qid)
            smc = det.get("semantic_memory_counts")
            if isinstance(smc, dict):
                semantic_counts.append(smc.get("total"))

    print("=" * 72)
    print("1) formation.details.diagnostics 分类分布（按模式）")
    print("=" * 72)
    for mode_name in ("real_memory", "oracle_memory", "oracle_context"):
        c = cls_by_mode.get(mode_name)
        if not c:
            continue
        total = sum(c.values())
        print(f"\n[{mode_name}]  证据条目总数={total}")
        for cls in CLASSES:
            n = c.get(cls, 0)
            if n:
                print(f"    {cls:<28} {n:>4}  ({n/total:6.1%})")
        for cls, n in c.items():
            if cls not in CLASSES:
                print(f"    {cls:<28} {n:>4}  ({n/total:6.1%})")

    # ---------- 2) 关键交叉表：分类 × 原始消息是否找到 ----------
    print("\n" + "=" * 72)
    print("2) [real_memory] 交叉表: 分类 × 原始消息是否在同一 turn 找到")
    print("=" * 72)
    print(f"{'classification':<30}{'src_msg_found=True':>20}{'False':>10}")
    for cls in CLASSES:
        t = crosstab.get((cls, True), 0)
        f = crosstab.get((cls, False), 0)
        if t or f:
            print(f"{cls:<30}{t:>20}{f:>10}")

    # ---------- 3) 逐阶段召回（从 found/missing 重算，校验与 summary 一致） ----------
    print("\n" + "=" * 72)
    print("3) [real_memory] 逐阶段证据召回（从 comparison.jsonl 重算）")
    print("=" * 72)
    stages = ["formation", "storage", "evolution", "retrieval", "rerank", "context"]
    for mode_name in ("real_memory", "oracle_memory"):
        acc = {s: [] for s in stages}
        for r in rows:
            modes = r.get("modes", {}) or {}
            mo = modes.get(mode_name, {}) or {}
            gold = r.get("gold_evidence") or []
            gold_ids = {str(g.get("evidence_id")) for g in gold if g.get("evidence_id")}
            for s in stages:
                rc = _recall(_stage_of(mo, s), gold_ids)
                if rc is not None:
                    acc[s].append(rc)
        print(f"\n[{mode_name}]")
        for s in stages:
            if acc[s]:
                avg = sum(acc[s]) / len(acc[s])
                print(f"    {s:<12} {avg:6.3f}   (n={len(acc[s])})")

    # ---------- 4) 一致性红旗 ----------
    print("\n" + "=" * 72)
    print("4) 一致性检查（红旗）")
    print("=" * 72)
    f1a = cls_by_mode.get("real_memory", {}).get("F1a_EXTRACTOR_MISS", 0)
    f1c = cls_by_mode.get("real_memory", {}).get("F1c_REPRESENTATION_MISMATCH", 0)
    ok = cls_by_mode.get("real_memory", {}).get("FORMATION_OK", 0)
    print(f"  - real_memory 证据总条目 = {total_diag}")
    print(f"    FORMATION_OK              = {ok}")
    print(f"    F1a_EXTRACTOR_MISS (真F1?) = {f1a}")
    print(f"    F1c_REPR_MISMATCH (伪影?)  = {f1c}")
    if total_diag:
        print(f"\n  >> 语义表征失配占比 = {f1c/total_diag:.1%}（F1c 通常指向归因/匹配器伪影，而非真实抽取失败）")
        if f1a > 0 and f1c > f1a:
            print("  >> 红旗: F1c > F1a，说明大量 'formation 失败' 更可能是匹配器验证不了语义表征")
    t_true = crosstab.get(("F1a_EXTRACTOR_MISS", True), 0)
    if t_true:
        print(f"\n  >> 红旗: {t_true} 条被标为 F1a 的证据其原始消息其实已找到（src_msg_found=True），")
        print("          这类应重点复查是抽取器未产出语义记忆，还是匹配器未命中。")

    # ---------- 5) 需要人工复查的样例 ----------
    print("\n" + "=" * 72)
    print("5) 建议人工复查的样例（去重前若干个）")
    print("=" * 72)
    print(f"  F1a_EXTRACTOR_MISS 涉及 question: {sorted(set(per_q_f1a))[:15]}{' ...' if len(set(per_q_f1a))>15 else ''}")
    print(f"  F1c_REPR_MISMATCH  涉及 question: {sorted(set(per_q_f1c))[:15]}{' ...' if len(set(per_q_f1c))>15 else ''}")

    print("\n完成。请把上面输出贴回，我可据此判断 F1 中真实 vs 伪影的比例。")
    print("注意：本脚本只读 comparison.jsonl；在完成本项诊断前，不要运行 semantic_calibration。")


if __name__ == "__main__":
    main()
