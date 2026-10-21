"""家族 C：SQL join 扇出模拟器（数据库域 / T1）。

冻结规格（KWM-TASKSET-V1 §5）：
- db_v1.sqlite：orders(5000 行, amount 合计 50,000)；customers 中 18% 的 customer_id
  有重复行（大小写变体，如 'ACME'/'acme'）
  → 直接 JOIN customers ON customer_id + GROUP BY c.name → 订单扇出 +17% → 合计 58,500
- db_v2.sqlite：V-revived 操纵：上游修复唯一性（customer_id 唯一）+ 新增 canonical_name
  列（每客户唯一）；遗留 name 列保留大小写不一致
  → 直接 join 正确（50,000）；按 name 聚合在 v2 因大小写变体分裂客户 → C-C2 失败
- gold_c.json：金标 orders SUM=50,000、v1 扇出合计、v1/v2 唯一客户数
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from kwm_common import DATA_DIR, dump_json

N_ORDERS = 5000
N_CUSTOMERS = 1000
SEED = 20261021
DUP_FRACTION = 0.18


def build() -> dict:
    rng = np.random.default_rng(SEED)
    # 客户：82% 单行，18% 重复（同 customer_id 两行，name 大小写变体）
    cust_ids = [f"C{i:04d}" for i in range(N_CUSTOMERS)]
    customer_rows = []  # (customer_id, name)
    for i in range(N_CUSTOMERS):
        cid = cust_ids[i]
        if i < int(N_CUSTOMERS * DUP_FRACTION):
            customer_rows.append((cid, f"ACME{i:03d}"))
            customer_rows.append((cid, f"acme{i:03d}"))  # 大小写变体，同 id
        else:
            customer_rows.append((cid, f"CUST{i:03d}"))

    # 订单：从去重客户 id 抽样（每个客户出现一次）→ 18% 的订单引用重复 id，
    # 触发 JOIN 扇出因子 ≈1.18（58,500 / 50,000）
    distinct_ids = list(dict.fromkeys(r[0] for r in customer_rows))
    chosen = rng.choice(distinct_ids, N_ORDERS, replace=True)
    # 均匀 amount，总计 50,000
    amounts = np.full(N_ORDERS, 50_000.0 / N_ORDERS)

    def make_db(path: Path, *, v2: bool) -> dict:
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE orders (order_id TEXT, customer_id TEXT, amount REAL)")
        if v2:
            cur.execute("CREATE TABLE customers (customer_id TEXT, name TEXT, canonical_name TEXT)")
        else:
            cur.execute("CREATE TABLE customers (customer_id TEXT, name TEXT)")
        for i in range(N_ORDERS):
            cur.execute("INSERT INTO orders VALUES (?,?,?)",
                        (f"O{i:05d}", chosen[i], float(amounts[i])))
        for cid, name in customer_rows:
            if v2:
                cur.execute("INSERT INTO customers VALUES (?,?,?)",
                            (cid, name, name.upper()))
            else:
                cur.execute("INSERT INTO customers VALUES (?,?)", (cid, name))
        conn.commit()
        # 实测：错误查询（JOIN + GROUP BY name）
        cur.execute("""
            SELECT SUM(o.amount) FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            GROUP BY c.name
        """)
        rows = cur.fetchall()
        fanout_sum = float(sum(r[0] for r in rows))
        # 正确查询（去重 customer_id）
        cur.execute("""
            SELECT SUM(o.amount) FROM orders o
            JOIN (SELECT customer_id, MIN(name) AS name FROM customers GROUP BY customer_id) c
            ON o.customer_id = c.customer_id
        """)
        correct_sum = float(cur.fetchone()[0])
        uniq = cur.execute("SELECT COUNT(DISTINCT customer_id) FROM customers").fetchone()[0]
        conn.close()
        return {"fanout_sum": fanout_sum, "correct_sum": correct_sum, "unique_customers": uniq}

    v1 = make_db(DATA_DIR / "db_v1.sqlite", v2=False)
    v2 = make_db(DATA_DIR / "db_v2.sqlite", v2=True)

    gold = {
        "family": "C",
        "gold_sum": 50_000.0,
        "n_orders": N_ORDERS,
        "n_customers_total_rows_v1": len(customer_rows),
        "unique_customers_v1": v1["unique_customers"],
        "unique_customers_v2": v2["unique_customers"],
        "v1_fanout_sum": v1["fanout_sum"],
        "v1_correct_sum": v1["correct_sum"],
        "v2_correct_sum": v2["correct_sum"],
    }
    dump_json(DATA_DIR / "gold_c.json", gold)
    return gold


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
