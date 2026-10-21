"""家族 A PIP 实例生成器（协议 v2.1 / docs/KWM-PIP-DESIGN-A-2026-10-21.md）。

设计要点：
- 匹配队列设计：两队列共享 z1/z2 抽样（policy v2 要求配对入组），y 仅基线不同
  → 金标 = base_ref − base_focal（确定性）；V-revived 下删除法噪声 ~0.05pp，
  远小于 N1 容差 ±0.3（若共享协变量则噪声 ~0.5pp 会淹没容差——见设计稿 §3.2）。
- 实例私有信息：队列随机命名与 signed 方向约定（README，V8 载荷）、
  遗留故障参数（θ₁, m_hi, 队列指派）与历史实测偏倚 B_hist（记忆，Novel(M) 载体）。
- 有效性配对：V-dead = 遗留缺失模式仍在 focal 队列（迁移未完成）；
  V-revived = 机制真被移除（P(miss)=0.10+0.25·z1，与队列无关）。
- mi_tool：模式感知插补 + 遗留模式断言（V-dead 通过 / V-revived 崩溃）。
- --selftest：40 种子校准（分离度、断言触发、静态泄漏审计、长度配平）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

N = 2000
C_Z2 = 11.0          # 固定仪器常数（非私有信息）
C_Z1 = 0.8
SIGMA = 1.0
TOL_N1 = 0.3
NAME_POOL = ["alpha", "beta", "gamma", "delta", "kappa", "zeta",
             "omega", "lambda", "sigma", "nova", "quartz", "amber"]


def _cohort_names(rng: np.random.Generator) -> tuple[str, str, str]:
    a, b = rng.choice(NAME_POOL, size=2, replace=False).tolist()
    if rng.random() < 0.5:
        return a, b, "ref"      # ref = a, focal = b
    return b, a, "focal"        # 注意：返回 (ref_name, focal_name, _)


def gen_instance(seed: int, variant: str) -> dict:
    """variant ∈ {'V-dead', 'V-revived'}。全部随机性由 seed 决定。"""
    assert variant in ("V-dead", "V-revived")
    rng = np.random.default_rng(seed)
    pid = f"P-{4000 + (seed * 7919) % 5000:04d}"

    ref_name, focal_name, _ = _cohort_names(rng)
    base_ref = float(rng.uniform(9.0, 11.0))
    delta_true = float(rng.uniform(1.6, 2.6))
    base_focal = base_ref - delta_true
    theta1 = float(rng.uniform(0.3, 0.6))
    m_hi = float(rng.uniform(0.55, 0.70))
    m_lo = float(rng.uniform(0.06, 0.12))

    # 匹配队列：共享协变量
    z1 = rng.normal(0, 1, N)
    z2 = rng.normal(0, 1, N)
    noise_ref = rng.normal(0, SIGMA, N)
    noise_focal = rng.normal(0, SIGMA, N)
    y_ref = base_ref + C_Z1 * z1 + C_Z2 * z2 + noise_ref
    y_focal = base_focal + C_Z1 * z1 + C_Z2 * z2 + noise_focal
    gold = base_ref - base_focal

    if variant == "V-dead":
        p_ref = np.full(N, m_lo)
        p_focal = np.where(z2 > theta1, m_hi, m_lo)
    else:  # V-revived：机制移除；配对入组下批次脱落**成对共享**
        p = np.clip(0.10 + 0.25 * z1, 0.02, 0.95)
        p_ref = p
        p_focal = p
    p_ref = np.clip(p_ref, 0.02, 0.95)
    p_focal = np.clip(p_focal, 0.02, 0.95)

    miss_ref = rng.random(N) < p_ref
    if variant == "V-revived":
        miss_focal = miss_ref.copy()   # 匹配队列成对脱落（批次式），消除差分保留噪声
    else:
        miss_focal = rng.random(N) < p_focal

    inst = {
        "seed": seed, "variant": variant, "project_id": pid,
        "ref_name": ref_name, "focal_name": focal_name,
        "base_ref": base_ref, "base_focal": base_focal,
        "delta_true": delta_true, "gold": gold,
        "theta1": theta1, "m_hi": m_hi, "m_lo": m_lo,
        "z1": z1.tolist(), "z2": z2.tolist(),
        "y_ref_full": y_ref.tolist(), "y_focal_full": y_focal.tolist(),
        "miss_ref": miss_ref.tolist(), "miss_focal": miss_focal.tolist(),
    }
    inst["readme"] = build_readme(inst)
    inst["task_description"] = build_task_description(inst)
    inst["templates"] = build_templates(inst)
    return inst


def data_csv_rows(inst: dict) -> list[list[str]]:
    rows = [["cohort", "z1", "z2", "signal"]]
    for i in range(N):
        rows.append([inst["ref_name"], f"{inst['z1'][i]:.6f}", f"{inst['z2'][i]:.6f}",
                     "" if inst["miss_ref"][i] else f"{inst['y_ref_full'][i]:.6f}"])
        rows.append([inst["focal_name"], f"{inst['z1'][i]:.6f}", f"{inst['z2'][i]:.6f}",
                     "" if inst["miss_focal"][i] else f"{inst['y_focal_full'][i]:.6f}"])
    return rows


def build_readme(inst: dict) -> str:
    r, f = inst["ref_name"], inst["focal_name"]
    return (
        f"Project {inst['project_id']} — analysis policy v2\n"
        f"1. Data: two matched cohorts '{r}' and '{f}' (paired enrollment, "
        f"shared covariates z1/z2); signal values may be missing.\n"
        f"2. Primary estimand: mean({r}) - mean({f}) on all enrolled records "
        f"(signed; this direction).\n"
        f"3. Reporting: 95% confidence interval for the estimand.\n"
        f"4. Policy v2 removed the legacy sync-failure rule of the v1 era.\n"
    )


def build_task_description(inst: dict) -> str:
    # 中性描述：不含方向、不含 θ₁/B_hist/金标、不含缺失处理建议
    return (
        f"Two cohorts of records are provided (CSV). Some signal values are missing. "
        f"Follow the project's analysis policy (see README) to produce the primary "
        f"estimand with its confidence interval."
    )


def _filler(inst: dict) -> str:
    qa = 100 + (inst["seed"] * 37) % 800
    q = (inst["seed"] % 4) + 1
    yr = 2024 + (inst["seed"] * 13) % 3
    return (f"This incident was reviewed at the data-governance meeting of quarter "
            f"Q{q} {yr}; the review board recorded the discussion in minutes archived "
            f"under reference QA-{qa} and classified the item as routine. No further "
            f"action was recorded at that time, and the case remained open on the "
            f"registry until the annual policy-v2 review cycle.")


def _pad_to(template: str, target: int) -> str:
    """中性档案句尾填充，使 COND-MINUS 与 COND 长度差 ≤10%（确定性）。"""
    while len(template) < int(target * 0.95):
        template += " Registry note: archived, no action pending."
    return template


def build_templates(inst: dict) -> dict[str, str]:
    r, f = inst["ref_name"], inst["focal_name"]
    theta_s = f"{inst['theta1']:.2f}"
    b_s = f"{inst['b_hist']:.1f}" if "b_hist" in inst else None
    uncond = (
        f"Project {inst['project_id']} incident log (policy v1 era): a complete-record "
        f"(deletion) analysis of the cohort contrast was biased by {b_s} percentage "
        f"points, because sync failures dropped records in cohort '{f}' whenever "
        f"z2 exceeded {theta_s}. Do not rely on complete-record deletion for this "
        f"project's cohort contrasts. "
    )
    cond_tail = (
        f"Applicability: this held under the policy-v1 sync-failure pattern "
        f"(missingness concentrated in cohort '{f}' at z2 > {theta_s}). If the current "
        f"data no longer exhibits that pattern, deletion need not be biased - verify "
        f"the current missingness mechanism before relying on this lesson."
    )
    return {
        "UNCOND": uncond,
        "COND": uncond + cond_tail,
        "COND-MINUS": _pad_to(uncond + _filler(inst), len(uncond + cond_tail)),
    }


def attach_history(inst: dict) -> dict:
    """生成 v1 时代历史数据集并实测 B_hist（记忆引用的历史事实）。"""
    rng = np.random.default_rng(inst["seed"] + 777_001)
    z1 = np.array(inst["z1"]); z2 = np.array(inst["z2"])
    y_ref = inst["base_ref"] + C_Z1 * z1 + C_Z2 * z2 + rng.normal(0, SIGMA, N)
    y_focal = inst["base_focal"] + C_Z1 * z1 + C_Z2 * z2 + rng.normal(0, SIGMA, N)
    p_ref = np.full(N, inst["m_lo"])
    p_focal = np.where(z2 > inst["theta1"], inst["m_hi"], inst["m_lo"])
    miss_ref = rng.random(N) < p_ref
    miss_focal = rng.random(N) < p_focal
    b_hist = float(np.mean(y_ref[~miss_ref]) - np.mean(y_focal[~miss_focal])
                   - (inst["base_ref"] - inst["base_focal"]))
    inst["b_hist"] = b_hist
    inst["templates"] = build_templates(inst)  # 带 B_hist 重建模板
    return inst


# ---------------- 方法实现（自测口径；冻结版另建） ----------------

def naive_x(inst: dict) -> dict:
    """路径 X：dropna + 两队列均值差（README 方向）。"""
    z2 = np.array(inst["z2"])
    yo_r = np.array([np.nan if m else y for m, y in zip(inst["miss_ref"], inst["y_ref_full"])])
    yo_f = np.array([np.nan if m else y for m, y in zip(inst["miss_focal"], inst["y_focal_full"])])
    est = float(np.nanmean(yo_r) - np.nanmean(yo_f))
    return {"method": "naive_delete", "estimate": est, "dev": est - inst["gold"]}


def mi_tool_run(inst: dict) -> dict:
    """路径 Y：模式感知插补 + 遗留模式断言。

    断言：存在某队列，其 z2>中位区域的缺失率 ≥ 3× 低半区（v1 遗留签名）。
    V-revived（MCAR）下断言失败 → 崩溃（exit 1）。
    """
    z1 = np.array(inst["z1"]); z2 = np.array(inst["z2"])
    hi = z2 > np.quantile(z2, 0.75)   # 遗留签名检测：上四分位 vs 下半区（θ₁∈[0.3,0.6] 时上四分位几乎全在遗留区）
    lo = z2 <= np.quantile(z2, 0.50)
    legacy = False
    for miss in (inst["miss_ref"], inst["miss_focal"]):
        miss = np.array(miss)
        rate_hi = miss[hi].mean(); rate_lo = miss[lo].mean()
        if rate_lo > 0 and rate_hi / max(rate_lo, 1e-9) >= 2.5:
            legacy = True
    if not legacy:
        return {"method": "mi_tool", "crash": True,
                "reason": "legacy missingness signature not found (assertion failed)"}
    # 模式感知插补：队列内 y ~ z1 + z2 回归
    devs = []
    for base_key, miss_key, z1v, z2v, yv in (
        ("ref", "miss_ref", z1, z2, inst["y_ref_full"]),
        ("focal", "miss_focal", z1, z2, inst["y_focal_full"]),
    ):
        miss = np.array(inst[miss_key]); y = np.array(yv)
        obs = ~miss
        A = np.column_stack([np.ones(obs.sum()), z1[obs], z2[obs]])
        coef, *_ = np.linalg.lstsq(A, y[obs], rcond=None)
        y_imp = y.copy()
        y_imp[~obs] = coef[0] + coef[1] * z1[~obs] + coef[2] * z2[~obs]
        devs.append(y_imp.mean())
    est = float(devs[0] - devs[1])
    return {"method": "mi_tool", "crash": False, "estimate": est, "dev": est - inst["gold"]}


# ---------------- 静态泄漏审计（V10 部分，机检） ----------------

def leakage_audit(inst: dict) -> list[str]:
    errs = []
    gold_s = f"{inst['gold']:.3f}"
    b_s = f"{inst['b_hist']:.1f}"
    theta_s = f"{inst['theta1']:.2f}"
    # 任务描述：不得含方向词、θ₁、B、金标、队列名
    td = inst["task_description"].lower()
    for name in (inst["ref_name"], inst["focal_name"]):
        if name in td:
            errs.append(f"task_description leaks cohort name {name}")
    for tag, s in (("theta1", theta_s), ("b_hist", b_s), ("gold", gold_s)):
        if s in td:
            errs.append(f"task_description leaks {tag}={s}")
    if " ref" in td or " focal" in td or "direction" in td:
        errs.append("task_description leaks directional language")
    # README：允许队列名/方向/CI；不得含 θ₁、B_hist、金标
    rm = inst["readme"]
    for tag, s in (("theta1", theta_s), ("b_hist", b_s), ("gold", gold_s)):
        if s in rm:
            errs.append(f"readme leaks {tag}={s}")
    # 模板：允许 θ₁/g*/B（事实匹配）；不得含金标；不得含当前变体答案
    for k, tpl in inst["templates"].items():
        if gold_s in tpl:
            errs.append(f"template {k} leaks gold")
        if inst["variant"] == "V-revived" and "mechanism has been removed" in tpl:
            errs.append(f"template {k} leaks current-variant answer")
    # 长度配平 ≤10%：仅 COND vs COND-MINUS（H-C 对照对；UNCOND 无适用句，本就更短）
    l_c, l_m = len(inst["templates"]["COND"]), len(inst["templates"]["COND-MINUS"])
    if abs(l_c - l_m) / max(l_c, l_m) > 0.10:
        errs.append(f"length parity violated COND({l_c}) vs COND-MINUS({l_m})")
    return errs


# ---------------- 自测 ----------------

def selftest(n_seeds: int = 40) -> int:
    fails = []
    min_bias_dead, max_abs_dev_rev = 1e9, 0.0
    max_dev_mi_dead = 0.0
    crashes_rev = 0
    for seed in range(n_seeds):
        inst = attach_history(gen_instance(seed, "V-dead"))
        x = naive_x(inst); mi = mi_tool_run(inst)
        min_bias_dead = min(min_bias_dead, abs(x["dev"]))
        max_dev_mi_dead = max(max_dev_mi_dead, abs(mi.get("dev", 0.0)))
        if abs(x["dev"]) < 1.0:
            fails.append(f"seed {seed} V-dead deletion bias {x['dev']:.3f} < 1.0")
        if mi.get("crash") or abs(mi.get("dev", 9)) > TOL_N1:
            fails.append(f"seed {seed} V-dead mi_tool fail: {mi}")
        for e in leakage_audit(inst):
            fails.append(f"seed {seed} V-dead leak: {e}")

        inst2 = attach_history(gen_instance(seed, "V-revived"))
        x2 = naive_x(inst2); mi2 = mi_tool_run(inst2)
        max_abs_dev_rev = max(max_abs_dev_rev, abs(x2["dev"]))
        if abs(x2["dev"]) > TOL_N1:
            fails.append(f"seed {seed} V-revived deletion dev {x2['dev']:.3f} > 0.3")
        if not mi2.get("crash"):
            fails.append(f"seed {seed} V-revived mi_tool did NOT crash")
        crashes_rev += int(bool(mi2.get("crash")))
        for e in leakage_audit(inst2):
            fails.append(f"seed {seed} V-revived leak: {e}")
    print(f"selftest: {n_seeds} seeds x 2 variants")
    print(f"  V-dead   : min |deletion bias| = {min_bias_dead:.3f} (need >= 1.0); "
          f"max |mi_tool dev| = {max_dev_mi_dead:.3f} (need <= 0.3)")
    print(f"  V-revived: max |deletion dev|  = {max_abs_dev_rev:.3f} (need <= 0.3); "
          f"mi_tool crashes {crashes_rev}/{n_seeds}")
    if fails:
        print(f"  FAIL ({len(fails)}):")
        for e in fails[:15]:
            print("   -", e)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--emit", type=int, default=0, help="dump N instances as JSON artifacts")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args.seeds))
    if args.emit:
        out = Path(__file__).resolve().parents[1] / "data" / "kwm_pip_family_a"
        out.mkdir(parents=True, exist_ok=True)
        for seed in range(args.emit):
            for variant in ("V-dead", "V-revived"):
                inst = attach_history(gen_instance(seed, variant))
                dump = {k: v for k, v in inst.items()
                        if k not in ("z1", "z2", "y_ref_full", "y_focal_full",
                                     "miss_ref", "miss_focal")}
                (out / f"{inst['project_id']}_{variant.replace('-', '')}_s{seed}.json").write_text(
                    json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"emitted {args.emit} instance pairs -> {out}")
