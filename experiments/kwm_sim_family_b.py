"""家族 B：时间戳静默丢行模拟器（代码/环境域 / T2 混合）。

冻结规格（KWM-TASKSET-V1 §4）：
- sales_v1.csv：5000 行订单；ts 含 12% 无法解析的时间字符串；amount 合计 500,000
  → 直接 groupby(month) 静默丢行 → 汇总总额 440,000（−12%）
- sales_v2.csv：V-revived 操纵：ts 全部可解析（清理）+ 新增 region 列（插在 order_id 与 ts 之间）
  → 直接 groupby 正确（500,000）；legacy_agg.py 按位置取列在 v2 崩溃
- gold_b.json：金标总额 500,000、各版本月度覆盖月份数、v1 直接汇总总额
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from kwm_common import DATA_DIR, dump_json

N_ROWS = 5000
SEED = 20261021
BAD_FRACTION = 0.12


def _months(n: int, rng: np.random.Generator) -> list[int]:
    # 均匀铺满 12 个月，保证 v1/v2 月度表行数 = 12
    base = np.tile(np.arange(1, 13), n // 12 + 1)[:n]
    rng.shuffle(base)
    return base.tolist()


def build() -> dict:
    rng = np.random.default_rng(SEED)
    amounts = np.full(N_ROWS, 100.0)  # 每行 100 → 总额 500,000
    months = _months(N_ROWS, rng)
    years = rng.integers(2023, 2025, N_ROWS).tolist()
    bad_idx = set(rng.choice(N_ROWS, int(N_ROWS * BAD_FRACTION), replace=False).tolist())

    def write(path: Path, *, v2: bool) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if v2:
                w.writerow(["order_id", "region", "ts", "amount"])
            else:
                w.writerow(["order_id", "ts", "amount"])
            for i in range(N_ROWS):
                oid = f"O{i:05d}"
                region = f"R{i % 7}"
                if i in bad_idx:
                    ts = "unknown" if not v2 else f"{years[i]:04d}-{months[i]:02d}-15 10:00:00"
                else:
                    ts = f"{years[i]:04d}-{months[i]:02d}-15 10:00:00"
                if v2:
                    w.writerow([oid, region, ts, f"{amounts[i]:.2f}"])
                else:
                    w.writerow([oid, ts, f"{amounts[i]:.2f}"])

    write(DATA_DIR / "sales_v1.csv", v2=False)
    write(DATA_DIR / "sales_v2.csv", v2=True)

    gold = {
        "family": "B",
        "gold_sum": float(amounts.sum()),
        "n_rows": N_ROWS,
        "bad_fraction": BAD_FRACTION,
        "n_months": 12,
        # 直接 groupby 在 v1 上的汇总总额（静默丢 12% 行）
        "v1_direct_sum": float(amounts[np.array([i not in bad_idx for i in range(N_ROWS)])].sum()),
    }
    dump_json(DATA_DIR / "gold_b.json", gold)
    return gold


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
