"""家族 A：缺失数据机制模拟器（统计域 / T1）。

冻结规格（KWM-TASKSET-V1 §3）：
- n=2000，group∈{A,B} 各 1000；z1,z2~N(0,1)；y = base_g + 0.8·z1 + 0.6·z2 + N(0,1)
  base_A=10.0, base_B=7.9 → 金标 Δ = +2.1（完整数据）
- v1 MAR-by-(group,z2)：P(miss|A)=0.05+0.4·z2，P(miss|B)=0.45+0.4·z2（clip[0.02,0.9]）
  → 删除法（路径 X）有偏，目标 Δ_X1≈+8.2pp
- v2 操纵：缺失机制改为仅依赖 z1 且两组相同 P(miss)=0.1+0.3·z1
  → 删除法对 Δ 无偏（目标偏差 ≤0.3pp）；robust_groupdiff.py 的 v1 断言崩溃
- 输出：train_v1/v2.csv, test_v1/v2.csv, gold_a.json（金标 Δ 与各切片实测）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kwm_common import DATA_DIR, dump_json, embed, RETRIEVAL_TOP_K  # noqa: F401 (embed used elsewhere)


N = 2000
SEED = 20261021
BASE_A = 10.0
BASE_B = 7.9
COEF_Z2 = 11.0          # 放大 z2 在 y 中的权重 → 删除法偏倚显著且可测
SA, IA = 0.1, 0.05       # v1 A：近平直低缺失（≈5%）
SB, IB = 0.8, 0.45       # v1 B：缺失率≈45% 且随 z2 升高（产生偏倚）


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 0.02, 0.95)


def simulate(version: str, seed: int = SEED, n: int = N) -> dict:
    rng = np.random.default_rng(seed)
    half = n // 2
    group = np.array(["A"] * half + ["B"] * half)
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    base = np.where(group == "A", BASE_A, BASE_B)
    noise = rng.normal(0, 1, n)
    y_full = base + 0.8 * z1 + COEF_Z2 * z2 + noise

    if version == "v1":
        p_miss = np.where(group == "A", IA + SA * z2, IB + SB * z2)
    else:  # v2: 仅依赖 z1，两组相同 → 删除法对 Δ 无偏
        p_miss = 0.10 + 0.3 * z1
    p_miss = _clip(p_miss)
    missing = rng.random(n) < p_miss
    y_obs = np.where(missing, np.nan, y_full)

    # 金标：DGP 真实组均值差（确定性，= base_A - base_B = 2.1）
    gold_delta = float(BASE_A - BASE_B)
    # 删除法（路径 X）实测 Δ
    mask = ~np.isnan(y_obs)
    drop_a = y_obs[(group == "A") & mask]
    drop_b = y_obs[(group == "B") & mask]
    delta_x = float(np.nanmean(drop_a) - np.nanmean(drop_b))
    miss_rate_a = float(missing[group == "A"].mean())
    miss_rate_b = float(missing[group == "B"].mean())

    return {
        "version": version, "n": int(n), "group": group.tolist(),
        "z1": z1.tolist(), "z2": z2.tolist(), "y_obs": y_obs.tolist(),
        "missing": missing.tolist(), "p_miss": p_miss.tolist(),
        "gold_delta": gold_delta, "delta_x_dropna": delta_x,
        "miss_rate_a": miss_rate_a, "miss_rate_b": miss_rate_b,
    }


def write_csv(d: dict, path: Path) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "z1", "z2", "y"])
        for g, z1, z2, y, miss in zip(d["group"], d["z1"], d["z2"], d["y_obs"], d["missing"]):
            w.writerow([g, f"{z1:.6f}", f"{z2:.6f}", "" if miss else f"{y:.6f}"])


def build() -> dict:
    out = {}
    for ver in ("v1", "v2"):
        d = simulate(ver)
        write_csv(d, DATA_DIR / f"train_{ver}.csv")
        # held-out test 切片（独立种子，结构相同）
        dt = simulate(ver, seed=SEED + 1)
        write_csv(dt, DATA_DIR / f"test_{ver}.csv")
        out[ver] = {
            "gold_delta": d["gold_delta"],
            "delta_x_dropna": d["delta_x_dropna"],
            "obs_n": int(d["n"] - sum(d["missing"])),
            "miss_rate_a": d["miss_rate_a"],
            "miss_rate_b": d["miss_rate_b"],
        }
    gold = {
        "family": "A",
        "true_delta": BASE_A - BASE_B,
        "gold_delta_v1": out["v1"]["gold_delta"],
        "delta_x_dropna_v1": out["v1"]["delta_x_dropna"],
        "obs_n_v1": out["v1"]["obs_n"],
        "gold_delta_v2": out["v2"]["gold_delta"],
        "delta_x_dropna_v2": out["v2"]["delta_x_dropna"],
        "miss_rate_a_v1": out["v1"]["miss_rate_a"],
        "miss_rate_b_v1": out["v1"]["miss_rate_b"],
        "miss_rate_a_v2": out["v2"]["miss_rate_a"],
        "miss_rate_b_v2": out["v2"]["miss_rate_b"],
    }
    dump_json(DATA_DIR / "gold_a.json", gold)
    return gold


if __name__ == "__main__":
    g = build()
    print(json.dumps(g, ensure_ascii=False, indent=2))
