"""家族 B 的 "路径 Y" 工具（v1 时代写就，按位置取列）。

规格（KWM-TASKSET-V1 §4.1 / §4.2）：
- v1 列序 = [order_id, ts, amount]：iloc[:,2] 取到 amount（数值）→ 直接求和 = 500,000（修复路径）。
- v2 列序 = [order_id, region, ts, amount]：iloc[:,2] 取到 ts（字符串）→ 求和 TypeError → 崩溃。
- 用法：python legacy_agg.py <sales_vN.csv>
"""
from __future__ import annotations

import sys

import pandas as pd


def main(path: str) -> float:
    df = pd.read_csv(path)
    # 按位置取列（v1 时代假设 [order_id, ts, amount]）
    amount = df.iloc[:, 2]          # v1: amount(数值) → OK；v2: ts(字符串) → 下式崩溃
    total = float(amount.sum())     # v1 正确；v2 TypeError
    # 月度表（次要）：iloc[:,1] 在 v1=ts，v2=region
    ts = df.iloc[:, 1]
    m = pd.to_datetime(ts, errors="coerce").dt.month
    monthly = df.assign(m=m).groupby("m")["amount"].sum()
    print(f"legacy_agg: total={total:.2f} months={len(monthly)}")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: legacy_agg.py <sales_vN.csv>", file=sys.stderr)
        sys.exit(2)
    try:
        main(sys.argv[1])
    except (TypeError, ValueError) as e:
        print(f"legacy_agg CRASHED: {e}", file=sys.stderr)
        sys.exit(1)
