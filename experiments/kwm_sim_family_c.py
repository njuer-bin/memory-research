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

    # ---- v1 客户：82% 单行，18% 重复（同 customer_id 两行，name 大小写变体） ----
    cust_ids = [f"C{i:04d}" for i in range(N_CUSTOMERS)]
    v1_rows = []  # (customer_id, name)
    for i in range(N_CUSTOMERS):
        cid = cust_ids[i]
        if i < int(N_CUSTOMERS * DUP_FRACTION):
            v1_rows.append((cid, f"ACME{i:03d}"))
            v1_rows.append((cid, f"acme{i:03d}"))  # 大小写变体，同 id（同一客户）
        else:
            v1_rows.append((cid, f"CUST{i:03d}"))

    # ---- v2 客户：上游修复唯一性 → 每客户一行（1000 行）；但遗留 name 列在
    # 不同客户之间保留大小写变体冲突（90 对：'ACME0xx'/'acme0xx' 属于不同
    # customer_id）→ 按 name 归一去重会错误合并不同客户（C-C2 陷阱） ----
    N_COLLIDE_PAIRS = 90
    v2_rows = []
    for k in range(N_COLLIDE_PAIRS):
        v2_rows.append((f"C{2 * k:04d}", f"ACME{k:03d}"))
        v2_rows.append((f"C{2 * k + 1:04d}", f"acme{k:03d}"))  # 不同客户，大小写变体名
    for i in range(2 * N_COLLIDE_PAIRS, N_CUSTOMERS):
        v2_rows.append((f"C{i:04d}", f"CUST{i:03d}"))

    # 订单：从各自版本的客户 id 集合抽样
    ids_v1 = list(dict.fromkeys(r[0] for r in v1_rows))
    ids_v2 = [r[0] for r in v2_rows]
    chosen_v1 = rng.choice(ids_v1, N_ORDERS, replace=True)
    chosen_v2 = rng.choice(ids_v2, N_ORDERS, replace=True)
    # 均匀 amount，总计 50,000
    amounts = np.full(N_ORDERS, 50_000.0 / N_ORDERS)

    def make_db(path: Path, *, v2: bool) -> dict:
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE orders (order_id TEXT, customer_id TEXT, amount REAL)")
        rows = v2_rows if v2 else v1_rows
        chosen = chosen_v2 if v2 else chosen_v1
        if v2:
            cur.execute("CREATE TABLE customers (customer_id TEXT, name TEXT, canonical_name TEXT)")
        else:
            cur.execute("CREATE TABLE customers (customer_id TEXT, name TEXT)")
        for i in range(N_ORDERS):
            cur.execute("INSERT INTO orders VALUES (?,?,?)",
                        (f"O{i:05d}", chosen[i], float(amounts[i])))
        for cid, name in rows:
            if v2:
                # canonical_name 每客户唯一（上游修复时生成），遗留 name 保留大小写不一致
                cur.execute("INSERT INTO customers VALUES (?,?,?)",
                            (cid, name, f"CUSTOMER_{cid}"))
            else:
                cur.execute("INSERT INTO customers VALUES (?,?)", (cid, name))
        conn.commit()
        # X（朴素路径）：直接 JOIN 后按 customer_id 聚合（v1 因客户表重复行扇出）
        cur.execute("""
            SELECT SUM(s) FROM (
                SELECT SUM(o.amount) s FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                GROUP BY o.customer_id
            )
        """)
        fanout_sum = float(cur.fetchone()[0])
        # Y（name 归一去重路径）：JOIN 前把客户表按 UPPER(name) 去重
        cur.execute("""
            SELECT SUM(o.amount), COUNT(DISTINCT c.uname)
            FROM orders o
            JOIN (SELECT customer_id, UPPER(name) AS uname FROM customers GROUP BY customer_id) c
            ON o.customer_id = c.customer_id
        """)
        name_dedup_sum, name_dedup_count = cur.fetchone()
        # 正确查询：JOIN 前按 customer_id 去重客户表
        cur.execute("""
            SELECT SUM(s), COUNT(*) FROM (
                SELECT SUM(o.amount) s FROM orders o
                JOIN (SELECT customer_id FROM customers GROUP BY customer_id) c
                ON o.customer_id = c.customer_id
                GROUP BY o.customer_id
            )
        """)
        correct_sum, correct_count = cur.fetchone()
        uniq = cur.execute("SELECT COUNT(DISTINCT customer_id) FROM customers").fetchone()[0]
        conn.close()
        return {"fanout_sum": fanout_sum,
                "name_dedup_sum": float(name_dedup_sum), "name_dedup_count": int(name_dedup_count),
                "correct_sum": float(correct_sum), "correct_count": int(correct_count),
                "unique_customers": uniq}

    v1 = make_db(DATA_DIR / "db_v1.sqlite", v2=False)
    v2 = make_db(DATA_DIR / "db_v2.sqlite", v2=True)

    gold = {
        "family": "C",
        "gold_sum": 50_000.0,
        "gold_customers": N_CUSTOMERS,
        "n_orders": N_ORDERS,
        "n_customers_total_rows_v1": len(v1_rows),
        "unique_customers_v1": v1["unique_customers"],
        "unique_customers_v2": v2["unique_customers"],
        "v1_fanout_sum": v1["fanout_sum"],
        "v1_name_dedup_sum": v1["name_dedup_sum"], "v1_name_dedup_count": v1["name_dedup_count"],
        "v1_correct_sum": v1["correct_sum"], "v1_correct_count": v1["correct_count"],
        "v2_name_dedup_sum": v2["name_dedup_sum"],
        "v2_name_dedup_count": v2["name_dedup_count"],
        "v2_correct_sum": v2["correct_sum"], "v2_correct_count": v2["correct_count"],
    }
    dump_json(DATA_DIR / "gold_c.json", gold)
    return gold


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
