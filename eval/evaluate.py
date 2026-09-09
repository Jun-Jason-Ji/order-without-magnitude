#!/usr/bin/env python3
"""
CourtSI-Bench 评测脚本(复现论文指标)。

输入预测文件格式(list, 与 qa_bench.json 对齐):
    [{"image_id":..., "question":..., "answer": <gt>, "category": <细类>, "vlm_answer": <pred>}, ...]
默认要求每行显式含 vlm_answer 或 prediction。缺失预测按 0 分并写入 coverage；
GT 自洽检查只能用 --ground_truth_self_check 显式开启。

细类 -> 4 大类映射(论文 Table 2):
    distance_measurement_*     -> distance   (数值, T=15cm)
    localization               -> localization(三元组 (x,y,z), 每维阈值 30cm)
    spatial_counting_*         -> counting   (整数/单字母, 精确匹配)
    relational_reasoning_*     -> relational (单字母 MCQ, 精确匹配)

指标:
    * 精确匹配准确率 (exact-match) 用于 counting / relational / MCQ。
    * T-MRA (Threshold Mean Relative Accuracy) 用于 distance / localization:
        T-MRA = (1/10) * Σ_{θ∈{0.5..0.95}} 1( (|ŷ-y| - T)/|y| < 1-θ )
      其中 distance 的 T=15cm, localization 对 (x,y,z) 三轴分别算 T-MRA 再取平均。

用法:
    python evaluate.py --file_path pred.json --output_folder results --save_per_qa_output
"""
import argparse
import json
import math
import os
import re

# T-MRA 阈值集合 C = {0.5, 0.55, ..., 0.95}
T_MRA_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
# 答案单位是米, 阈值同样用米 (论文 15cm / 30cm)
T_DISTANCE = 0.15   # m, 距离测量
T_LOC = 0.30        # m, 定位(> 15cm*sqrt(3))

# CourtDyn 动力学三组 (下一篇; 见 engine/dynamics_qa.py)。GT 出自单目身高尺,
# 声明分辨率是 ~10% 相对量级, 所以阈值按各量纲的典型值放宽, 不做厘米级声明。
T_SPEED = 0.30      # m/s, 速度 (GT 中位数 ~0.9 m/s)
T_PATH = 0.50       # m,   路程 (GT 中位数 ~1.7 m)
T_TIMING = 0.50     # s,   到达/追及时间 (GT 中位数 ~4.5 s)

# 13 个细类 -> 4 大类
TASK_GROUPS = {
    "distance": "distance",
    "localization": "localization",
    "counting": "counting",
    "relational": "relational",
}
DISTANCE_PREFIX = "distance_measurement"
COUNTING_PREFIX = "spatial_counting"
RELATIONAL_PREFIX = "relational_reasoning"
# 动力学细类前缀 -> 组。放在 relational 判定之前是安全的: CourtDyn 的判别题
# 用的是 relational_reasoning_dyn_* , 仍归 relational (精确匹配 A/B)。
DYNAMICS_PREFIXES = {
    "dynamics_speed": "speed",
    "dynamics_path": "path",
    "dynamics_time": "timing",
}
# 走 T-MRA 的数值组; counting/relational 走精确匹配
TMRA_GROUPS = ("distance", "localization", "speed", "path", "timing")
# 组 -> (阈值 T, floor_v2 下限)。legacy_v1 一律 floor=0。
SCALAR_TMRA_SPEC = {
    "distance": (T_DISTANCE, 0.5),
    "speed": (T_SPEED, 1.0),
    "path": (T_PATH, 2.0),
    "timing": (T_TIMING, 2.0),
}


def map_group(category):
    """把细类归并到 distance/localization/counting/relational (+ CourtDyn 三组)。"""
    c = (category or "").lower()
    if c.startswith(DISTANCE_PREFIX):
        return "distance"
    if c == "localization":
        return "localization"
    if c.startswith(COUNTING_PREFIX):
        return "counting"
    if c.startswith(RELATIONAL_PREFIX):
        return "relational"
    for pref, grp in DYNAMICS_PREFIXES.items():
        if c.startswith(pref):
            return grp
    return "other"


def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_tuple3(s):
    """解析 '(3.35, 2.92, 0.99)' -> [3.35, 2.92, 0.99], 失败返回 None。"""
    if not isinstance(s, str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) >= 3:
        try:
            return [float(x) for x in nums[:3]]
        except ValueError:
            return None
    return None


def to_number(s):
    """从自由文本里抽一个标量数字。"""
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def exact_match(pred, gt):
    """精确匹配(归一化),处理整数与单字母。"""
    p, g = normalize_text(pred), normalize_text(gt)
    if p == g:
        return True
    pn, gn = to_number(pred), to_number(gt)
    if pn is not None and gn is not None and abs(pn - gn) < 1e-6:
        return True
    return False


def t_mra_scalar(pred, gt, T, denominator_floor=0.0):
    """标量 T-MRA。

    denominator_floor=0 保留历史实现（GT=0 时得 0），仅用于复现旧表。
    新冻结的 floor_v2 协议给 distance/localization 分别使用 0.5m/1.0m
    下限，与奖励的相对误差口径一致，也让零坐标轴有定义。
    """
    pn, gn = to_number(pred), to_number(gt)
    if pn is None or gn is None:
        return 0.0
    denominator = max(abs(gn), float(denominator_floor))
    if denominator == 0:
        return 0.0
    rel = (abs(pn - gn) - T) / denominator
    return float(sum(1.0 for theta in T_MRA_THRESHOLDS if rel < 1.0 - theta)) / len(T_MRA_THRESHOLDS)


def t_mra_tuple(pred, gt, T, denominator_floor=0.0):
    """三元组 (x,y,z) 逐维 T-MRA 取平均; 任一解析失败则 0。"""
    pv, gv = parse_tuple3(pred), parse_tuple3(gt)
    if pv is None or gv is None:
        return 0.0
    return sum(t_mra_scalar(pv[i], gv[i], T, denominator_floor)
               for i in range(3)) / 3.0


def score_pair(group, pred, gt, metric_version="legacy_v1"):
    """返回 (exact:0/1, tmra:float or None)。"""
    if metric_version not in {"legacy_v1", "floor_v2"}:
        raise ValueError(f"unknown metric_version: {metric_version}")
    if group in SCALAR_TMRA_SPEC:
        thresh, floor_v2 = SCALAR_TMRA_SPEC[group]
        exact = 1 if exact_match(pred, gt) else 0
        floor = floor_v2 if metric_version == "floor_v2" else 0.0
        tmra = 1.0 if exact else t_mra_scalar(pred, gt, thresh, floor)
        return exact, tmra
    if group == "localization":
        # 精确匹配: 三元组完全相等才算(容忍1e-3)
        pv, gv = parse_tuple3(pred), parse_tuple3(gt)
        exact = 1 if (pv and gv and all(abs(a - b) < 1e-3 for a, b in zip(pv, gv))) else 0
        floor = 1.0 if metric_version == "floor_v2" else 0.0
        tmra = 1.0 if exact else t_mra_tuple(pred, gt, T_LOC, floor)
        return exact, tmra
    # counting / relational / other -> 精确匹配
    return (1 if exact_match(pred, gt) else 0), None


def evaluate(pairs, *, allow_ground_truth_self_check=False,
             metric_version="legacy_v1"):
    """Score predictions and expose coverage/parse failures.

    Missing prediction fields never inherit the ground truth.  The only path
    that copies GT into predictions is the explicit self-check switch.
    """
    records = []
    by_group = {}
    overall_exact = 0
    overall_tmra = 0.0
    numeric_n = 0
    prediction_present_n = 0
    parse_failure_n = 0

    for item in pairs:
        cat = item.get("category", "unknown")
        gt = item.get("answer")
        has_prediction = "vlm_answer" in item or "prediction" in item
        if "vlm_answer" in item:
            pred = item["vlm_answer"]
        elif "prediction" in item:
            pred = item["prediction"]
        elif allow_ground_truth_self_check:
            pred = gt
        else:
            pred = ""
        group = map_group(cat)
        if has_prediction or allow_ground_truth_self_check:
            exact, tmra = score_pair(group, pred, gt, metric_version)
        else:
            exact = 0
            tmra = 0.0 if group in TMRA_GROUPS else None
        prediction_present_n += int(has_prediction)
        parse_failure = bool(
            (group in SCALAR_TMRA_SPEC and to_number(pred) is None)
            or (group == "localization" and parse_tuple3(pred) is None)
        )
        parse_failure_n += int(parse_failure)
        rec = dict(item)
        rec["group"] = group
        rec["exact"] = exact
        rec["tmra"] = tmra
        rec["prediction_present"] = has_prediction
        rec["parse_failure"] = parse_failure
        # 论文定位指标: 3D 欧氏误差 <= 30cm 记为正确
        if group == "localization":
            pv, gv = parse_tuple3(pred), parse_tuple3(gt)
            rec["within_30cm"] = int(
                pv is not None and gv is not None and
                math.dist(pv, gv) <= T_LOC)
        overall_exact += exact
        if tmra is not None:
            overall_tmra += tmra
            numeric_n += 1
        g = by_group.setdefault(group, {"exact": 0, "tmra": 0.0, "n": 0})
        g["exact"] += exact
        g["tmra"] += tmra or 0.0
        g["n"] += 1
        records.append(rec)

    n = len(pairs)
    summary = {
        "n_total": n,
        "metric_version": metric_version,
        "n_prediction_present": prediction_present_n,
        "prediction_coverage": prediction_present_n / n if n else 0.0,
        "n_parse_failure_numeric": parse_failure_n,
        "exact_accuracy": overall_exact / n if n else 0.0,
        "tmra_overall": overall_tmra / numeric_n if numeric_n else None,
    }
    per_group = {}
    for grp, d in by_group.items():
        per_group[grp] = {
            "n": d["n"],
            "exact_accuracy": d["exact"] / d["n"],
            "tmra": (d["tmra"] / d["n"]) if grp in TMRA_GROUPS else None,
        }
    loc_recs = [r for r in records if r["group"] == "localization"]
    if loc_recs:
        per_group["localization"]["acc_at_30cm"] = (
            sum(r["within_30cm"] for r in loc_recs) / len(loc_recs))
    summary["per_group"] = per_group
    return summary, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_path", required=True, help="prediction json (含 answer + vlm_answer + category)")
    ap.add_argument("--output_folder", default="results")
    ap.add_argument("--save_per_qa_output", action="store_true")
    ap.add_argument("--ground_truth_self_check", action="store_true",
                    help="显式用 answer 作为缺失 prediction，仅供 evaluator 自检")
    ap.add_argument("--metric_version", default="legacy_v1",
                    choices=["legacy_v1", "floor_v2"],
                    help="legacy_v1 只复现旧表；新实验须显式用 floor_v2")
    args = ap.parse_args()

    with open(args.file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary, records = evaluate(
        data,
        allow_ground_truth_self_check=args.ground_truth_self_check,
        metric_version=args.metric_version,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_per_qa_output:
        rec_path = os.path.join(args.output_folder, "per_qa.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[eval] per-QA written -> {rec_path}")

    print("[eval] ===== CourtSI-Bench summary =====")
    print(f"  total pairs : {summary['n_total']}")
    print(f"  exact acc   : {summary['exact_accuracy']:.4f}")
    if summary["tmra_overall"] is not None:
        print(f"  T-MRA (num) : {summary['tmra_overall']:.4f}")
    print("  per-group:")
    for grp, d in summary["per_group"].items():
        t = f"tmra={d['tmra']:.4f}" if d["tmra"] is not None else "n/a"
        print(f"    {grp:14s} n={d['n']:5d} acc={d['exact_accuracy']:.4f} {t}")
    print(f"[eval] summary -> {out_path}")


if __name__ == "__main__":
    main()
