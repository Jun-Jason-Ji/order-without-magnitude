#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pixel_oracle_courtdyn.py — 像素级 oracle 基线 (plan §〇 A5 / §7.3 C2)。CPU。

问题: 模型在 CourtDyn 上全线 chance / ρ≈0, 到底是"看不清"还是"不会看"?
答法: 写一个只看**模型看到的那几张图 + 题面**的程序 —— 从降采样后的帧里用颜色
阈值找红框 (被问球员) / 蓝框 (选项 B) / 黄圈 (球), 用框高与题面里的身高先验当尺子,
按 dynamics_qa 的定义算速度 / 路程 / 追及时间 / 加减速 / 谁更快, 再用与模型完全
相同的指标打分。它能拿多少, 就是"证据在不在像素里"的硬上界。

公平性: oracle 拿到的输入与模型逐字相同 —— 同一批 jpg、先缩到 max_pixels 对应的
分辨率、同一句题面。GT 轨迹只用来**验证检测是否找对了框** (报检出率与中心误差),
不参与预测。

分辨率臂:
  model   960x540 渲染帧再缩到 597x336 (= Qwen max_pixels 200704 的实际输入)
  render  960x540 (不再缩; 等价于放大 max_pixels 的模型)
  zoom    T19 的 448² 裁剪帧 (只有 accel_vis 家族)

产出 results/courtdyn/pixel_oracle.md + pixel_oracle.json
"""
import json
import math
import os
import re
import statistics as st
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import evaluate as E                                        # noqa: E402
from engine.dynamics_qa import load_seqinfo, load_tracks    # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
FR = os.path.join(ROOT, "data", "courtdyn", "frames")
SEQ = os.path.join(ROOT, "data", "external_validation", "teamtrack",
                   "teamtrack-mot", "teamtrack-mot", "basketball_side",
                   "test", "Q4_side_480-510")
MODEL_W, MODEL_H = 597, 336          # 960x540 -> max_pixels 200704
HP_RE = re.compile(r"is ([\d.]+) m tall")
DUR_RE = re.compile(r"spanning ([\d.]+) seconds")

# 颜色阈值 (RGB)。画框用的是饱和纯色, 球场的粉红/湖蓝不饱和, 阈值收紧就分得开。
def mask_red(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 170) & (g < 80) & (b < 80)


def mask_blue(rgb):                    # 画的是 BGR (255,128,0) -> RGB (0,128,255)
    # 球场远端有一片蓝漆 (RGB≈(40,190,180)) 与蓝框相邻, 松阈值会把框和地面连成一块。
    # 蓝框线芯是 (≈1,127,252): 饱和度远高于地面漆 —— 用 b>235 & g<150 把它切开。
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > 235) & (r < 40) & (g > 105) & (g < 150)


def mask_yellow(rgb):                  # BGR (0,220,255) -> RGB (255,220,0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 170) & (g > 150) & (b < 90)


def load_rgb(path, mode):
    img = cv2.imread(path)
    if img is None:
        return None
    if mode == "model":
        img = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def largest_blob_box(mask, min_px=12):
    """最大的**像框的**连通块的外接框 (x0,y0,x1,y1); 没有则 None。

    球场地面有一块蓝漆与蓝框的颜色几乎一样 (RGB≈(0,128,255)), 单纯取最大连通块
    会抓到那条 210×27 的扁带子。框是细线矩形轮廓, 用形状而不是颜色把它挑出来:
    填充率 (像素数 / 外接框面积) 不能像实心区域那样接近 1, 且不能是扁带
    (宽 ≤ 2.5×高 + 6)。这些约束只用几何常识, 不碰 GT。
    """
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        fill = area / float(max(1, w * h))
        if h < 5 or w > 2.5 * h + 6 or fill > 0.9:
            continue
        cands.append((area, (x, y, x + w, y + h)))
    if not cands:
        return None
    return max(cands)[1]


def box_foot(b):
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def detect(paths, mode, want_blue=False, want_ball=False):
    """每帧: 红框 (+蓝框, +球心)。返回 list of dict, 缺检为 None。"""
    out = []
    for p in paths:
        rgb = load_rgb(p, mode)
        if rgb is None:
            out.append(None)
            continue
        d = {"red": largest_blob_box(mask_red(rgb))}
        if want_blue:
            d["blue"] = largest_blob_box(mask_blue(rgb))
        if want_ball:
            y = largest_blob_box(mask_yellow(rgb), min_px=6)
            d["ball"] = None if y is None else ((y[0] + y[2]) / 2.0, (y[1] + y[3]) / 2.0)
        out.append(d)
    return out


def path_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def predict(item, mode, img_root):
    """按题面家族给出 oracle 预测 (字符串, 与模型输出同格式) 或 None (检测失败)。"""
    cat = item["category"]
    q = item["question"]
    paths = [os.path.join(img_root, x) for x in item["image_ids"]]
    need_blue = cat.endswith("_faster")
    need_ball = cat == "dynamics_time_intercept"
    det = detect(paths, mode, want_blue=need_blue, want_ball=need_ball)
    if any(d is None or d["red"] is None for d in det):
        return None
    reds = [d["red"] for d in det]
    feet = [box_foot(b) for b in reds]
    hmed = st.median(b[3] - b[1] for b in reds)
    m = HP_RE.search(q)
    hp = float(m.group(1)) if m else None
    dur = float(DUR_RE.search(q).group(1))
    mpp = (hp / hmed) if (hp and hmed) else None

    if cat == "dynamics_speed_player":
        return None if mpp is None else f"{path_len(feet) * mpp / dur:.1f}"
    if cat == "dynamics_path_player":
        return None if mpp is None else f"{path_len(feet) * mpp:.1f}"
    if cat == "dynamics_time_intercept":
        ball = det[-1].get("ball")
        if ball is None or mpp is None:
            return None
        v = path_len(feet) * mpp / dur
        return None if v <= 0 else f"{math.dist(feet[-1], ball) * mpp / v:.1f}"
    if cat.startswith("relational_reasoning_dyn_accel"):
        s1, s2 = math.dist(feet[0], feet[1]), math.dist(feet[2], feet[3])
        up = s2 > s1
        # 题面里 (A)/(B) 的语义顺序可能互换 (v1 语言侧最小对)
        a_is_up = "(A) speeding up" in q
        return ("A" if up else "B") if a_is_up else ("B" if up else "A")
    if cat.endswith("_faster"):
        if any(d["blue"] is None for d in det):
            return None
        blue_feet = [box_foot(d["blue"]) for d in det]
        hb = st.median(d["blue"][3] - d["blue"][1] for d in det)
        # 各用各的尺子 (与 GT 定义一致: 每人按自己的框高折算)
        va = path_len(feet) / hmed
        vb = path_len(blue_feet) / hb
        return "A" if va > vb else "B"
    return None


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def detection_check(items, mode, img_root, tracks, scale):
    """检测验证: 红框中心与 GT 框中心的误差 (模型输入像素), 检出率。"""
    errs, hit, n = [], 0, 0
    for it in items[:200]:
        t = it["meta"].get("track", it["meta"].get("track_a"))
        for fid, name in zip(it["meta"]["frames"], it["image_ids"]):
            n += 1
            rgb = load_rgb(os.path.join(img_root, name), mode)
            if rgb is None:
                continue
            b = largest_blob_box(mask_red(rgb))
            g = tracks[t].get(fid)
            if b is None or g is None:
                continue
            hit += 1
            gx, gy = (g[0] + g[2] / 2) * scale, (g[1] + g[3] / 2) * scale
            errs.append(math.dist(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), (gx, gy)))
    return {"frames": n, "detected": hit, "rate": hit / n if n else None,
            "center_err_px_median": st.median(errs) if errs else None,
            "center_err_px_p90": sorted(errs)[int(0.9 * len(errs))] if errs else None}


def score_family(items, preds, grp):
    """与模型同口径: 数值 -> T-MRA(floor_v2) + T=0 + ρ; A/B -> 逐题 + paired。"""
    if grp in E.SCALAR_TMRA_SPEC:
        T, floor = E.SCALAR_TMRA_SPEC[grp]
        xs, ys, tm, t0 = [], [], [], []
        for it, p in zip(items, preds):
            g = it["answer"]
            if p is None:
                tm.append(0.0)
                t0.append(0.0)
                continue
            tm.append(E.t_mra_scalar(p, g, T, floor))
            t0.append(E.t_mra_scalar(p, g, 0.0, floor))
            xs.append(float(p))
            ys.append(float(g))
        return {"n": len(items), "n_detected": len(xs),
                "tmra": 100 * st.mean(tm), "tmra_T0": 100 * st.mean(t0),
                "rho": spearman(xs, ys)}
    ok = [(p is not None and p == it["answer"]) for it, p in zip(items, preds)]
    pairs = {}
    for it, o in zip(items, ok):
        pairs.setdefault(it["pair_id"], []).append(o)
    comp = [v for v in pairs.values() if len(v) == 2]
    return {"n": len(items), "n_detected": sum(p is not None for p in preds),
            "item": 100 * st.mean(ok),
            "paired": 100 * st.mean(all(v) for v in comp) if comp else None,
            "n_pairs": len(comp)}


def main():
    tracks = load_tracks(SEQ)
    info = load_seqinfo(SEQ)
    qa = json.load(open(os.path.join(CD, "qa_dyn_v1.json"), encoding="utf-8"))
    vis = json.load(open(os.path.join(CD, "qa_dyn_v2_paired_accel_vis.json"), encoding="utf-8"))
    zoom = json.load(open(os.path.join(CD, "qa_dyn_v2_paired_accel_vis_zoom.json"), encoding="utf-8"))
    fams = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path"),
            ("dynamics_time_intercept", "timing"),
            ("relational_reasoning_dyn_accel", "accel(v1 语言对)"),
            ("relational_reasoning_dyn_faster", "faster")]
    R = {"schema": "courtdyn-pixel-oracle-v1", "detection": {}, "scores": {}}
    lines = ["# CourtDyn 像素级 oracle (C2)", "",
             "只看模型看到的那几张图 + 题面: 颜色阈值找框 → 框高×题面身高先验当尺子 → "
             "按 dynamics_qa 定义计算。GT 轨迹只用于验证检测。指标与模型同口径。", ""]

    for mode, label in (("model", "model 597×336 (Qwen 实际输入)"), ("render", "render 960×540")):
        scale = (960.0 / info["width"]) * ((MODEL_W / 960.0) if mode == "model" else 1.0)
        R["detection"][mode] = detection_check(
            [i for i in qa if i["category"] == "dynamics_speed_player"], mode, FR, tracks, scale)
        lines += [f"## {label}", "",
                  f"红框检出率 {100*R['detection'][mode]['rate']:.1f}% · 中心误差中位 "
                  f"{R['detection'][mode]['center_err_px_median']:.1f} px (p90 "
                  f"{R['detection'][mode]['center_err_px_p90']:.1f})", "",
                  "| 家族 | n | 检到 | T-MRA | T-MRA(T=0) | ρ | 逐题 | paired |",
                  "|---|---|---|---|---|---|---|---|"]
        for cat, grp in fams:
            items = [i for i in qa if i["category"] == cat]
            preds = [predict(i, mode, FR) for i in items]
            s = score_family(items, preds, grp if grp in E.SCALAR_TMRA_SPEC else "ab")
            R["scores"][f"{mode}|{grp}"] = s
            f = lambda k, nd=1: "—" if s.get(k) is None or (isinstance(s.get(k), float) and math.isnan(s[k])) else f"{s[k]:.{nd}f}"  # noqa: E731
            lines.append(f"| {grp} | {s['n']} | {s['n_detected']} | {f('tmra')} | {f('tmra_T0')} | "
                         f"{f('rho', 2)} | {f('item')} | {f('paired')} |")
        # accel 视觉侧最小对 (全景)
        preds = [predict(i, mode, FR) for i in vis]
        s = score_family(vis, preds, "ab")
        R["scores"][f"{mode}|accel_vis"] = s
        lines.append(f"| accel_vis (T19 视觉对, 全景) | {s['n']} | {s['n_detected']} | — | — | — | "
                     f"{s['item']:.1f} | {s['paired']:.1f} |")
        lines.append("")

    # zoom 臂 (448², 只有 accel_vis)
    zr = os.path.join(FR, "zoom")
    preds = [predict(i, "render", zr) for i in zoom]
    s = score_family(zoom, preds, "ab")
    R["scores"]["zoom|accel_vis"] = s
    lines += ["## zoom 448² (T19 放大臂, accel_vis)", "",
              f"检到 {s['n_detected']}/{s['n']} · 逐题 **{s['item']:.1f}** · paired **{s['paired']:.1f}** "
              f"(chance 25)", ""]

    lines += ["## 读法", "",
              "- 数值家族: oracle 的 ρ 是「证据在像素里」的上界; 模型 ρ≈0 而 oracle ρ 高 = 模型没在用像素。",
              "- oracle 的 T-MRA 也受 4 帧稀疏采样限制 (GT 用的是连续轨迹), 这正是模型面对的同一限制。",
              "- accel: 全景 vs zoom 的 oracle paired 差, 就是分辨率本身造成的上界差; "
              "模型在 zoom 上仍 chance 而 oracle 不是 = 能力问题 (T19 结论的独立验证)。", ""]
    with open(os.path.join(CD, "pixel_oracle.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(CD, "pixel_oracle.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
