"""家族 A 参数扫描（非对称）：A 近平直低缺失，B 依赖 z2；找偏倚显著且断言通过的机制。"""
from __future__ import annotations
import numpy as np

N = 2000
BASE_A, BASE_B = 10.0, 7.9
SEED = 20261021


def trial(coef_z2, sa, ia, sb, ib, seed=SEED):
    rng = np.random.default_rng(seed)
    half = N // 2
    group = np.array(["A"] * half + ["B"] * half)
    z1 = rng.normal(0, 1, N)
    z2 = rng.normal(0, 1, N)
    base = np.where(group == "A", BASE_A, BASE_B)
    y = base + 0.8 * z1 + coef_z2 * z2 + rng.normal(0, 1, N)
    pa = np.clip(ia + sa * z2, 0.02, 0.95)
    pb = np.clip(ib + sb * z2, 0.02, 0.95)
    ma = rng.random(N) < pa
    mb = rng.random(N) < pb
    miss = np.where(group == "A", ma, mb)
    y_obs = np.where(miss, np.nan, y)
    gold = float(np.nanmean(y[group == "A"]) - np.nanmean(y[group == "B"]))
    m = ~np.isnan(y_obs)
    dx = float(np.nanmean(y_obs[(group == "A") & m]) - np.nanmean(y_obs[(group == "B") & m]))
    mra = float(miss[group == "A"].mean())
    mrb = float(miss[group == "B"].mean())
    return {"gold": gold, "dx_dropna": dx, "bias": dx - gold, "mra": mra, "mrb": mrb,
            "assert_ok": bool(mrb > 3 * mra)}


cands = []
for coef_z2 in (6.0, 10.0, 14.0, 18.0):
    for (ia, sa) in ((0.05, 0.1), (0.03, 0.2)):
        for (ib, sb) in ((0.45, 0.8), (0.45, 1.2), (0.5, 1.0)):
            r = trial(coef_z2, sa, ia, sb, ib)
            if r["assert_ok"] and r["dx_dropna"] > r["gold"]:
                cands.append((coef_z2, ia, sa, ib, sb, r))
cands.sort(key=lambda c: -c[5]["bias"])
print(f"assert_ok candidates: {len(cands)}")
for c in cands[:20]:
    cz, ia, sa, ib, sb, r = c
    print(f"coef_z2={cz} A({ia},{sa}) B({ib},{sb}) -> gold={r['gold']:.2f} "
          f"dx={r['dx_dropna']:.2f} bias={r['bias']:.2f} mra={r['mra']:.3f} mrb={r['mrb']:.3f}")
