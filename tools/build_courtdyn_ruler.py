#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_ruler.py — N2: "能不能用递到手里的尺子" 与 "是不是在图像平面测量" 两个题池。

动机 (docs/courtdyn_results_summary_2026-09-05.md §一 E1b): 俯视片段上 ρ 到 0.45–0.80 (序读得出),
T-MRA 却低于该片段的常数中位数 (尺读不出), 且删掉身高句 (noprior 臂) 预测中位数与 ρ 都不变
=> 题面给的尺度句是惰性的。本脚本把 "尺子" 这一侧做成两个可分辨的臂:

  A. **ruler**  给一把**正确的**尺子: 把身高句换成 "In these frames, one meter on the floor spans
     about K pixels."。K 由球场单应导出 (逐题算 渲染像素路程 / 米路程, 取该片段中位数),
     所以它是**片段级常数**, 不携带任何逐题信息 (不泄漏答案), 模型仍需自己量像素再做一次除法。
     ⚠ 只对**俯视**片段成立: 俯视 K 的 IQR 约 ±1% (Q1_top 27.06, Q4_top 24.88, Q2_top 27.50),
     侧视 K 在 9.47–36.55 之间变化 4 倍, 单一标量尺子在那里没有意义, 脚本会拒绝构造。
     ⚠ 不能用 "红框高 H 像素 = 身高 1.93 m" 那种写法: 俯视外接框根本不是身高 (身高尺在俯视差 2 倍),
     那句话是假的, 模型忽略它反而是对的。

  B. **pxunit** 干脆按**图像平面**提问: 单位从米换成像素, GT 换成渲染像素位移, 身高句删掉
     (不需要尺子)。若 "读的是图像平面位移" 成立, 这应当是模型的**最好**条件 —— 分数跳起来
     就把负结果变成了正的机制结果 ("它在测像素, 只是不会换算")。

两臂共用同一批 item / 同一批帧 / 同一个窗口, 只改题面 (与 GT 的单位), 因此可与 full / noprior 直接对比。
只保留 speed / path 两个数值家族 (timing 的单应真值不可用, 见 plan)。

用法:
  python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant ruler
  python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant pxunit
产物: results/courtdyn/seq_<seq>/qa_dyn_v1_<variant>.json (+ .manifest.json)
"""
import argparse
import io
import json
import math
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import dynamics_qa as DQ                    # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer   # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
TT = os.path.join(ROOT, "data", "external_validation", "teamtrack", "teamtrack-mot", "teamtrack-mot")
PROBE_HITS = os.path.join(CD, "teamtrack_probe_hits.json")
MAIN = "Q4_side_480-510"

RENDER_W = 960          # tools/build_courtdyn_qa.py OUT_W
PRIOR_RE = re.compile(r"Assume a typical player on this court is [0-9.]+\s*m tall\.")
SPEED, PATH = "dynamics_speed_player", "dynamics_path_player"
KEEP = (SPEED, PATH)
# 片段级 K 只有在尺度近似均匀时才诚实; 俯视 IQR/中位 ≈ 1%, 侧视 ≈ 20%+。
MAX_K_SPREAD = 0.05     # (Q3-Q1)/median 上限


def seq_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def tt_split(seq):
    try:
        for fam, split, name in json.load(io.open(PROBE_HITS, encoding="utf-8")):
            if name == seq:
                return os.path.join(TT, fam, split, seq)
    except Exception:
        pass
    return os.path.join(TT, "basketball_top", "train", seq)


def duration_s(it, fps):
    f0, f1 = it["meta"]["window"]
    return (f1 - f0) / fps


def clip_scale(seq, items, tracks, scale):
    """K = 渲染像素 / 米, 逐 path 题算一次后取中位数; 同时报离散度。"""
    ks = []
    plane = CourtPlane.load(seq)
    for it in items:
        if it["category"] != PATH:
            continue
        m = it["meta"]
        bb = tracks[m["track"]]
        f0, f1 = m["window"]
        px = DQ.path_length_px(bb, f0, f1)
        mm = plane.path_length_m(bb, f0, f1)
        if px is None or mm is None or mm < 0.3:      # 站着不动的窗口比值不稳, 不参与
            continue
        ks.append(px * scale / mm)
    if len(ks) < 20:
        raise SystemExit(f"{seq}: 可用于估计 K 的窗口只有 {len(ks)} 个, 拒绝构造")
    ks.sort()
    med = st.median(ks)
    spread = (ks[3 * len(ks) // 4] - ks[len(ks) // 4]) / med
    return med, spread, ks[0], ks[-1], len(ks), plane


def px_answer(it, tracks, scale, fps):
    """该 item 的渲染像素 GT (path: 像素路程; speed: 像素/秒)。"""
    m = it["meta"]
    f0, f1 = m["window"]
    px = DQ.path_length_px(tracks[m["track"]], f0, f1)
    if px is None:
        return None
    px *= scale
    return px if it["category"] == PATH else px / duration_s(it, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", required=True, choices=["ruler", "pxunit"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow_uneven_scale", action="store_true",
                    help="侧视片段上强行构造 ruler 臂 (默认拒绝)")
    args = ap.parse_args()

    sd = seq_dir(args.seq)
    src = os.path.join(sd, "qa_dyn_v1.json")
    out = args.out or os.path.join(sd, f"qa_dyn_v1_{args.variant}.json")
    if os.path.abspath(out) == os.path.abspath(src):
        raise SystemExit("--out 不能与主池相同")

    data = json.load(io.open(src, encoding="utf-8"))
    items = data if isinstance(data, list) else data["qa"]
    info = DQ.load_seqinfo(tt_split(args.seq))
    tracks = DQ.load_tracks(tt_split(args.seq))
    fps = float(info["fps"])
    scale = RENDER_W / float(info["width"])

    K, spread, kmin, kmax, nk, plane = clip_scale(args.seq, items, tracks, scale)
    print(f"[{args.variant}] {args.seq}: K = {K:.2f} 渲染像素/米 "
          f"(n={nk}, IQR/中位 {spread*100:.1f}%, min/max {kmin:.2f}/{kmax:.2f})")
    if args.variant == "ruler" and spread > MAX_K_SPREAD and not args.allow_uneven_scale:
        raise SystemExit(f"K 的离散度 {spread*100:.1f}% > {MAX_K_SPREAD*100:.0f}%: "
                         "该片段的像素/米尺度不均匀 (多半是侧视), 单一标量尺子会是假话, 拒绝构造。"
                         " 确要构造请加 --allow_uneven_scale。")

    kept, dropped, stripped = [], 0, 0
    ruler_sentence = (f"In these frames, one meter on the floor spans about {K:.0f} pixels.")
    for it in items:
        if it["category"] not in KEEP:
            continue
        q = it["question"]
        if not PRIOR_RE.search(q):
            raise SystemExit("题面里找不到身高先验句, 格式可能变了, 拒绝写出")
        it = json.loads(json.dumps(it))          # 深拷贝, 不动源池
        if args.variant == "ruler":
            # 递过去的尺子是单应导出的真尺度, 因此 GT 必须换成同一个世界的 v3 答案。
            # 主池存的是 v1 (身高尺) 答案, 俯视上它差约 2 倍 —— 若继续用 v1 计分,
            # "模型真的用了这把尺子" 反而会被判错, 整个实验就反了。
            r = recompute_answer(plane, it, tracks, fps)
            if r is None:
                dropped += 1
                continue
            it["question"] = PRIOR_RE.sub(ruler_sentence, q)
            it["meta"]["ruler_px_per_m"] = round(K, 3)
            it["meta"]["v1_answer"] = it["answer"]
            it["meta"]["answer_unit"] = "m (v3 homography)" if it["category"] == PATH else "m/s (v3 homography)"
            it["answer"] = r[0]
        else:
            a = px_answer(it, tracks, scale, fps)
            if a is None:
                dropped += 1
                continue
            q2 = PRIOR_RE.sub("", q)
            if it["category"] == PATH:
                q2 = q2.replace("travel along the floor over the whole clip, in meters?",
                                "move across the image over the whole clip, in pixels?")
            else:
                q2 = q2.replace("What is the average speed of the player marked with the red box,"
                                " in meters per second?",
                                "What is the average speed of the player marked with the red box"
                                " across the image, in pixels per second?")
            if "pixels" not in q2:
                raise SystemExit(f"单位替换失败, 题面格式可能变了:\n{q}")
            it["question"] = re.sub(r"\s{2,}", " ", q2).strip()
            it["meta"]["m_answer"] = it["answer"]
            it["meta"]["answer_unit"] = "px" if it["category"] == PATH else "px/s"
            it["meta"]["ruler_px_per_m"] = round(K, 3)
            it["answer"] = f"{a:.1f}"
        stripped += 1
        kept.append(it)

    if not kept:
        raise SystemExit("没有产出任何题目")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)

    man = {
        "schema": "courtdyn-qa-ruler-v1",
        "variant": args.variant,
        "seq": args.seq,
        "source_pool": os.path.relpath(src, ROOT).replace("\\", "/"),
        "render_width": RENDER_W,
        "render_scale": scale,
        "fps": fps,
        "k_px_per_m": {"median": round(K, 3), "iqr_over_median": round(spread, 4),
                       "min": round(kmin, 3), "max": round(kmax, 3), "n_windows": nk},
        "counts": {"kept": len(kept), "dropped_no_px": dropped,
                   "by_family": {c: sum(1 for x in kept if x["category"] == c) for c in KEEP}},
        "question_change": (ruler_sentence if args.variant == "ruler"
                            else "height sentence removed; unit switched to image-plane pixels"),
        "answer_unit": ("meters, recomputed with the court homography (v3); the pool's original"
                        " v1 height-ruler answer is kept in meta.v1_answer" if args.variant == "ruler"
                        else "rendered pixels / pixels per second"),
        "notes": [
            "K 是片段级常数 (逐题算后取中位数), 不携带逐题信息, 因此不泄漏答案。",
            "ruler 臂只在像素/米尺度近似均匀的片段上构造 (IQR/中位 <= 5%), 即俯视片段。",
            "pxunit 臂的 T-MRA 不能用主口径阈值: 等价容差 = 米制阈值 x K, 由 summarize_courtdyn_t28.py 计算。",
            "两臂与 full / noprior 共用同一批 item 与同一批帧, 只改题面 (两臂都另改 GT 口径)。",
            "ruler 臂的 GT 是 v3 (单应) 答案, 与递过去的尺子同一个世界; 与 full 比较时 full 也必须按 v3 重算 "
            "(const_tmra_v3 已在 findings 里), 不能拿 v1 口径的 tmra_official 直接对比。",
        ],
    }
    with io.open(out.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"[{args.variant}] {len(items)} -> {len(kept)} 题 "
          f"(speed {man['counts']['by_family'][SPEED]}, path {man['counts']['by_family'][PATH]}"
          f"{', 丢弃 %d' % dropped if dropped else ''}) -> {out}")


if __name__ == "__main__":
    main()
