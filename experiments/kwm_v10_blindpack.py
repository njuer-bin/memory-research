"""V10 泄漏审计——人工盲评包（协议 v2.1 / docs/KWM-PIP-DESIGN-A-2026-10-21.md §7-②③）。

目的：人工（≥2 评分者）确认 COND-MINUS 记忆模板不含适用条件语义，
同时以 COND 为正对照、UNCOND 为第二对照（可行动但无条件句），
排除评分者把"任何可行动内容"都判为条件句的松弛偏差。

盲评设计：
- 条件标签对评分者不可见：条目匿名编号 M-01..M-K，固定种子洗牌；
- 构成：n_instances 个种子 × {COND, COND-MINUS} + n_uncond × UNCOND；
- 模板取自 attach_history(gen_instance(seed, 'V-dead'))（模板与 variant 无关）；
- 答案键单独落盘（answer_key.json，评分前不得给评分者）；
- provenance.json 留痕：种子清单、生成器 sha256、git commit、构建时间
  （V10 组成 ③ 的工件留存要求）。

评分判据（在评分前冻结，写死于 THRESHOLDS）：
- COND        : q1(含适用条件语义)=Y，每人 ≥11/12
- COND-MINUS  : q1=N，每人 ≥11/12
- UNCOND      : q1=N，每人 ≥5/6
- 评分者间 q1 一致率 ≥90%（kappa 报告不设门）
任一不满足 → 判 FAIL：修订 filler → 重新生成 → 重新盲评。

用法：
  python kwm_v10_blindpack.py --build                # 生成盲评包
  python kwm_v10_blindpack.py --score r1.csv r2.csv  # 评分
评分者答卷格式（CSV，无表头也可，有表头自动跳过）：item_id,q1,q2
  q1 ∈ {Y,N}：是否含适用条件语义；q2 ∈ {Y,N}：是否含可行动教训/禁令。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwm_pip_family_a import attach_history, gen_instance  # noqa: E402

# ---- 冻结参数（构建后不得更改；更改即视为新盲评轮次） ----
BUILD_SEED = 20261021
N_INSTANCES = 12          # 每条件 12 条（COND / COND-MINUS）
N_UNCOND = 6
THRESHOLDS = {"COND": 11, "COND-MINUS": 11, "UNCOND": 5}   # 每评分者最低正确数（q1）
AGREEMENT_MIN = 0.90      # 评分者间 q1 一致率下限

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "kwm_v10_blindpack"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def build() -> int:
    items = []          # (item_id, text, true_label, seed)
    pool = []
    for seed in range(N_INSTANCES):
        inst = attach_history(gen_instance(seed, "V-dead"))
        pool.append((inst["templates"]["COND"], "COND", seed))
        pool.append((inst["templates"]["COND-MINUS"], "COND-MINUS", seed))
    uncond_seeds = list(range(100, 100 + N_UNCOND))
    for seed in uncond_seeds:
        inst = attach_history(gen_instance(seed, "V-dead"))
        pool.append((inst["templates"]["UNCOND"], "UNCOND", seed))

    import random
    rng = random.Random(BUILD_SEED)
    rng.shuffle(pool)
    for i, (text, label, seed) in enumerate(pool, 1):
        items.append({"item_id": f"M-{i:02d}", "text": text,
                      "true_label": label, "seed": seed})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 评分者答卷模板
    with open(OUT_DIR / "response_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "q1", "q2"])
        for it in items:
            w.writerow([it["item_id"], "", ""])

    # 盲评条目（人类可读）
    lines = ["# V10 盲评条目（匿名）", "",
             "对每条记忆条目回答：q1 = 是否含**适用条件语义**；q2 = 是否含**可行动教训/禁令**。",
             "把结果填入 response_template.csv（或自行建 CSV：item_id,q1,q2）。", ""]
    for it in items:
        lines += [f"## {it['item_id']}", "", "```", it["text"].strip(), "```", ""]
    (OUT_DIR / "blind_items.md").write_text("\n".join(lines), encoding="utf-8")

    # 机器可读盲评条目（不含标签）
    (OUT_DIR / "blind_items.json").write_text(json.dumps(
        [{"item_id": it["item_id"], "text": it["text"]} for it in items],
        ensure_ascii=False, indent=1), encoding="utf-8")

    # 答案键（评分前不得分发）
    (OUT_DIR / "answer_key.json").write_text(json.dumps(
        {it["item_id"]: {"true_label": it["true_label"], "seed": it["seed"]}
         for it in items}, ensure_ascii=False, indent=1), encoding="utf-8")

    # 工件留痕（V10 组成 ③）
    gen_file = Path(__file__).resolve().parent / "kwm_pip_family_a.py"
    provenance = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "build_seed": BUILD_SEED,
        "n_instances": N_INSTANCES, "n_uncond": N_UNCOND,
        "uncond_seeds": uncond_seeds,
        "cond_minus_seeds": list(range(N_INSTANCES)),
        "generator_sha256": hashlib.sha256(gen_file.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "thresholds_frozen": THRESHOLDS,
        "agreement_min": AGREEMENT_MIN,
        "changelog": [
            "2026-10-21 首轮盲评包构建（生成器 300 种子自测全绿后冻结版本）",
        ],
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(
        provenance, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"built {len(items)} blind items -> {OUT_DIR}")
    print(f"  composition: COND x{N_INSTANCES}, COND-MINUS x{N_INSTANCES}, UNCOND x{N_UNCOND}")
    print(f"  answer key + provenance written (do NOT distribute key to raters)")
    return 0


def _read_responses(path: str, valid_ids: set[str]) -> dict[str, tuple[str, str]]:
    resp = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    start = 1 if rows and rows[0] and rows[0][0].strip().lower() == "item_id" else 0
    for row in rows[start:]:
        if not row or not row[0].strip():
            continue
        iid = row[0].strip()
        if iid not in valid_ids:
            raise SystemExit(f"{path}: unknown item_id '{iid}'")
        q1 = row[1].strip().upper()
        q2 = row[2].strip().upper() if len(row) > 2 else ""
        if q1 not in ("Y", "N") or q2 not in ("Y", "N", ""):
            raise SystemExit(f"{path}: bad response for {iid}: q1={q1!r} q2={q2!r}")
        resp[iid] = (q1, q2)
    missing = valid_ids - set(resp)
    if missing:
        raise SystemExit(f"{path}: missing responses for {sorted(missing)}")
    return resp


def _kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(response_paths: list[str]) -> int:
    key = json.loads((OUT_DIR / "answer_key.json").read_text(encoding="utf-8"))
    valid_ids = set(key)
    raters = [_read_responses(p, valid_ids) for p in response_paths]

    print(f"V10 blind scoring — {len(raters)} rater(s), {len(key)} items")
    print(f"  frozen thresholds: {THRESHOLDS}, agreement >= {AGREEMENT_MIN}\n")
    all_pass = True
    for name, resp in zip(response_paths, raters):
        print(f"[{Path(name).name}]")
        for cond, expect in (("COND", "Y"), ("COND-MINUS", "N"), ("UNCOND", "N")):
            ids = [i for i in valid_ids if key[i]["true_label"] == cond]
            correct = sum(1 for i in ids if resp[i][0] == expect)
            need = THRESHOLDS[cond]
            ok = correct >= need
            all_pass &= ok
            print(f"  q1 {cond:<11}: {correct}/{len(ids)} correct (need >= {need}) "
                  f"{'PASS' if ok else 'FAIL'}")
        # UNCOND 的 q2（可行动性）仅描述性报告
        uids = [i for i in valid_ids if key[i]["true_label"] == "UNCOND"]
        q2y = sum(1 for i in uids if resp[i][1] == "Y")
        print(f"  q2 UNCOND actionable=Y: {q2y}/{len(uids)} (descriptive)")
        print()

    if len(raters) >= 2:
        ids = sorted(valid_ids)
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                a = [raters[i][x][0] for x in ids]
                b = [raters[j][x][0] for x in ids]
                agree = sum(x == y for x, y in zip(a, b)) / len(ids)
                k = _kappa(a, b)
                ok = agree >= AGREEMENT_MIN
                all_pass &= ok
                print(f"agreement rater{i+1} vs rater{j+1}: {agree:.3f} "
                      f"(kappa {k:.3f}) {'PASS' if ok else 'FAIL'}")
        print()

    verdict = "PASS" if all_pass else "FAIL"
    print(f"VERDICT: {verdict}")
    if verdict == "FAIL":
        print("  -> 修订 filler / 模板，重新生成盲评包（新构建轮次），再次盲评。")
    # 评分结果留痕（V10 组成 ③）
    (OUT_DIR / f"score_{'_'.join(Path(p).stem for p in response_paths)}.json").write_text(
        json.dumps({"verdict": verdict, "raters": response_paths,
                    "thresholds": THRESHOLDS, "agreement_min": AGREEMENT_MIN,
                    "scored_at": datetime.now().isoformat(timespec="seconds")},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="构建盲评包")
    ap.add_argument("--score", nargs="+", metavar="RATER_CSV", help="评分答卷（≥1，建议 2）")
    args = ap.parse_args()
    if args.build:
        sys.exit(build())
    if args.score:
        sys.exit(score(args.score))
    ap.print_help()
