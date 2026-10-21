"""家族 A 的 "路径 Y" 工具（v1 时代写就，含 v1 专用健全性断言）。

规格（KWM-TASKSET-V1 §3.1 / §3.4）：
- 对含缺失 y 的 CSV 做多重插补（M=20）估计组均值差。
- 含 v1 专用断言：assert miss_rate_B > 3*miss_rate_A。
  在 v1 数据上断言通过（B 缺失率远高于 A）；在 v2 数据上两组缺失率相近 → 断言失败 → 崩溃。
- 用法：python robust_groupdiff.py <train_vN.csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path: str) -> float:
    df = pd.read_csv(path)
    g = df["group"].values
    z1 = df["z1"].values.astype(float)
    z2 = df["z2"].values.astype(float)
    y = df["y"].values.astype(float)
    missing = np.isnan(y)

    mr_a = float(missing[g == "A"].mean())
    mr_b = float(missing[g == "B"].mean())

    # ---- v1 专用健全性断言（v2 数据上此处崩溃） ----
    assert mr_b > 3.0 * mr_a, "unexpected data version: missingness pattern changed"

    obs = ~missing
    Xo = np.column_stack([
        (g == "A").astype(float)[obs],
        (g == "B").astype(float)[obs],
        z1[obs], z2[obs],
    ])
    Yo = y[obs]
    beta, _, _, _ = np.linalg.lstsq(Xo, Yo, rcond=None)
    resid = Yo - Xo @ beta
    rmean, rstd = resid.mean(), resid.std()

    rng = np.random.default_rng(0)
    diffs = []
    Xa = np.column_stack([(g == "A").astype(float), (g == "B").astype(float), z1, z2])
    for _ in range(20):
        yi = y.copy()
        yi[missing] = Xa[missing] @ beta + rng.normal(rmean, rstd, int(missing.sum()))
        b2, _, _, _ = np.linalg.lstsq(Xa, yi, rcond=None)
        mean_a = b2[0] + b2[2] * z1[g == "A"].mean() + b2[3] * z2[g == "A"].mean()
        mean_b = b2[1] + b2[2] * z1[g == "B"].mean() + b2[3] * z2[g == "B"].mean()
        diffs.append(mean_a - mean_b)
    est = float(np.mean(diffs))
    print(f"robust_groupdiff: delta={est:.4f} (mr_a={mr_a:.3f}, mr_b={mr_b:.3f})")
    return est


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: robust_groupdiff.py <train_vN.csv>", file=sys.stderr)
        sys.exit(2)
    try:
        main(sys.argv[1])
    except AssertionError as e:
        print(f"robust_groupdiff ASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
