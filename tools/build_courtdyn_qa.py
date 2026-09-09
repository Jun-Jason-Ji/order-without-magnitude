#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_qa.py — CourtDyn 动力学评测集构建 (CPU, 无需 GPU)。

从 TeamTrack basketball_side/Q4_side_480-510 的 MOT 连续轨迹程序化生成:

  data/courtdyn/frames/*.jpg                    带高亮框的多帧序列
  results/courtdyn/qa_dyn_v1.json               主评测集 (多帧, image_ids)
  results/courtdyn/qa_dyn_v1_cf_prior.json      反事实尺度先验臂 (GT 已重标)
  results/courtdyn/qa_dyn_v1.manifest.json      来源 sha256 / 参数 / 统计 / 假设

数学与尺度假设见 engine/dynamics_qa.py 的模块 docstring —— 那里写清了身高尺
先验、深度压缩偏置和反事实臂的设计, 论文方法节直接引用。

高亮约定 (画在帧上, 题面里用颜色指称):
  红框 = 被问球员 / 成对题的 A 方   蓝框 = 成对题的 B 方   黄圈 = 球

用法:
  python tools/build_courtdyn_qa.py                    # 默认参数全量构建
  python tools/build_courtdyn_qa.py --max_per_family 60 --dry_run   # 只算不抽帧
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.dynamics_qa import (ACCEL_RATIO, CF_PLAYER_HEIGHT_M,
                                PLAYER_HEIGHT_M, generate,
                                generate_accel_visual, load_seqinfo,
                                rescale_for_prior, windows_disjoint)

SEQ_DIR = os.path.join(ROOT, "data", "external_validation", "teamtrack",
                       "teamtrack-mot", "teamtrack-mot", "basketball_side",
                       "test", "Q4_side_480-510")
FRAME_DIR = os.path.join(ROOT, "data", "courtdyn", "frames")
OUT_DIR = os.path.join(ROOT, "results", "courtdyn")

OUT_W = 960          # 渲染宽度; 模型侧 max_pixels=200704 会再降采样
LINE_W = 5           # 高亮线宽 —— 降到 448² 后仍需可见
COLORS = {"red": (0, 0, 255), "blue": (255, 128, 0), "yellow": (0, 220, 255)}

# ---- 放大 (zoom) 对照臂 -------------------------------------------------
# 全景帧走到模型眼前只剩 597x336, 球员约 26 px 高, 而 accel 题要判的
# "后半段步长 - 前半段步长" 中位数只有 3.75 px (最低十分位 0.76 px) —— 小于
# Qwen 的 patch。这一臂把原始 4K 帧按固定 640² 窗口裁到球员身上再放到 448²,
# 同一 item 的 4 帧用**同一个**裁剪框 (跟着球员裁会把运动本身裁没), 尺寸对全家族
# 固定 (免得"裁得大小"本身泄漏答案)。放大约 4.5x, 那个 3.75 px 变成约 17 px。
CROP_PX = 640        # 原始 4K 像素; 覆盖全部 116 条的"球员+轨迹"外接框
ZOOM_OUT = 448       # 输出边长; 正好等于 max_pixels=200704, 不再被降采样
ZOOM_LINE_W = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"[courtdyn {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =======================================================================
# 抽帧 + 高亮
# =======================================================================
def hl_key(hl):
    """高亮签名 —— 同帧同签名的图只渲染一次。"""
    return "-".join(f"{h['color'][0]}{h['track']}" for h in sorted(
        hl, key=lambda h: (h["color"], h["track"])))


def extract_base_frames(video, need_frames, out_dir, scale):
    """顺序解码一遍 mp4, 把需要的帧降采样后存成 jpg (seek 慢, 顺读一次最省)。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    todo = {f: os.path.join(out_dir, f"base_{f:04d}.jpg") for f in need_frames}
    missing = {f: p for f, p in todo.items() if not os.path.isfile(p)}
    if not missing:
        log(f"底帧已齐 ({len(todo)} 张), 跳过解码")
        return todo
    log(f"解码 {os.path.basename(video)} -> 抽 {len(missing)}/{len(todo)} 张底帧")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1                       # MOT 帧号从 1 起
        if idx in missing:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (OUT_W, int(round(h * OUT_W / w))),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(missing[idx], small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            written += 1
            if written % 25 == 0:
                log(f"  底帧 {written}/{len(missing)}")
    cap.release()
    still = [f for f, p in todo.items() if not os.path.isfile(p)]
    if still:
        raise RuntimeError(f"这些帧没解出来 (视频比 gt 短?): {still[:5]}")
    return todo


def render_overlays(items, tracks, base_paths, scale, out_dir):
    """给每条 item 的每一帧画高亮, 返回 item -> [相对路径]。同签名复用。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    n_new = 0
    for it in items:
        rels = []
        for f in it["window"]["frames"]:
            key = (f, hl_key(it["highlight"]))
            if key not in cache:
                name = f"f{f:04d}_{hl_key(it['highlight'])}.jpg"
                path = os.path.join(out_dir, name)
                if not os.path.isfile(path):
                    img = cv2.imread(base_paths[f])
                    for h in it["highlight"]:
                        box = tracks[h["track"]].get(f)
                        if box is None:
                            continue
                        x, y, w, hh = (v * scale for v in box)
                        c = COLORS[h["color"]]
                        if h["shape"] == "circle":
                            cx, cy = int(x + w / 2), int(y + hh / 2)
                            r = max(int(max(w, hh) * 0.9), 9)
                            cv2.circle(img, (cx, cy), r, c, LINE_W)
                        else:
                            cv2.rectangle(img, (int(x), int(y)),
                                          (int(x + w), int(y + hh)), c, LINE_W)
                    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    n_new += 1
                cache[key] = name
            rels.append(cache[key])
        it["image_ids"] = rels
        it["image_id"] = rels[0]        # 单帧对照臂 / 兼容旧字段
    log(f"高亮帧: {len(cache)} 张唯一 ({n_new} 张新渲染)")
    return items


# =======================================================================
def strip(it):
    """落盘前去掉内部字段, 保留评测需要的 + meta 供事后分析。"""
    keep = {"image_id", "image_ids", "category", "question", "answer",
            "pair_id", "pair_role", "meta"}
    out = {k: v for k, v in it.items() if k in keep}
    out["meta"] = dict(out.get("meta", {}),
                       frames=it["window"]["frames"],
                       window=[it["window"]["f0"], it["window"]["f1"]])
    return out


def extract_base_frames_native(video, need_frames, out_dir):
    """同 extract_base_frames, 但**不降采样** —— zoom 臂要从原始 4K 帧上裁。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    todo = {f: os.path.join(out_dir, f"raw_{f:04d}.jpg") for f in need_frames}
    missing = {f: p for f, p in todo.items() if not os.path.isfile(p)}
    if not missing:
        log(f"原始底帧已齐 ({len(todo)} 张), 跳过解码")
        return todo
    log(f"解码 {os.path.basename(video)} -> 抽 {len(missing)}/{len(todo)} 张 4K 底帧")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1                       # MOT 帧号从 1 起
        if idx in missing:
            cv2.imwrite(missing[idx], frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
            if written % 25 == 0:
                log(f"  4K 底帧 {written}/{len(missing)}")
    cap.release()
    still = [f for f, p in todo.items() if not os.path.isfile(p)]
    if still:
        raise RuntimeError(f"这些帧没解出来: {still[:5]}")
    return todo


def crop_rect(tracks, tid, frames, w_img, h_img):
    """本 item 的固定裁剪框 —— 覆盖球员在这 4 帧里的全部位置, 居中, 贴边则内推。

    **必须对 4 帧用同一个框**: 逐帧跟着球员裁会把球员钉在画面中心, 位移信息
    连同答案一起被裁掉。
    """
    xs, ys = [], []
    for f in frames:
        b = tracks[tid].get(f)
        if b is None:
            continue
        x, y, w, h = b
        xs += [x, x + w]
        ys += [y, y + h]
    if not xs:
        return None
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    x0 = int(round(min(max(cx - CROP_PX / 2.0, 0), w_img - CROP_PX)))
    y0 = int(round(min(max(cy - CROP_PX / 2.0, 0), h_img - CROP_PX)))
    if max(xs) - min(xs) > CROP_PX or max(ys) - min(ys) > CROP_PX:
        return None                     # 轨迹装不进固定框 -> 整对丢弃
    return x0, y0


def render_zoom(items, tracks, raw_paths, out_dir, info):
    """裁剪 + 画框 + 缩到 ZOOM_OUT²。返回 (可用 items, 丢弃的 pair_id 集合)。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    k = ZOOM_OUT / float(CROP_PX)
    dropped, n_new = set(), 0
    for it in items:
        wi = it["meta"]["window_index"] if "meta" in it else None
        tid = it["highlight"][0]["track"]
        frames = it["window"]["frames"]
        rect = crop_rect(tracks, tid, frames, info["width"], info["height"])
        if rect is None:
            dropped.add(it["pair_id"])
            continue
        x0, y0 = rect
        rels = []
        for f in frames:
            name = f"z_w{wi:02d}_t{tid}_f{f:04d}.jpg"
            path = os.path.join(out_dir, name)
            if not os.path.isfile(path):
                img = cv2.imread(raw_paths[f])
                sub = img[y0:y0 + CROP_PX, x0:x0 + CROP_PX].copy()
                b = tracks[tid].get(f)
                if b is not None:
                    x, y, w, h = b
                    p1 = (int(round((x - x0) * k)), int(round((y - y0) * k)))
                    p2 = (int(round((x + w - x0) * k)),
                          int(round((y + h - y0) * k)))
                    sub = cv2.resize(sub, (ZOOM_OUT, ZOOM_OUT),
                                     interpolation=cv2.INTER_AREA)
                    cv2.rectangle(sub, p1, p2, COLORS["red"], ZOOM_LINE_W)
                else:
                    sub = cv2.resize(sub, (ZOOM_OUT, ZOOM_OUT),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, sub, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_new += 1
            rels.append(name)
        it["image_ids"] = rels
        it["image_id"] = rels[0]
        it["meta"]["crop_origin_xy"] = [x0, y0]
        it["meta"]["crop_px"] = CROP_PX
        it["meta"]["render_px"] = ZOOM_OUT
    keep = [it for it in items if it["pair_id"] not in dropped]
    log(f"zoom 帧: {n_new} 张新渲染; 丢弃 {len(dropped)} 对 (轨迹超出固定裁剪框)")
    return keep, dropped


def build_accel_visual_zoom(args, info):
    """accel 视觉侧最小对的**放大对照臂** —— 同样的对、同样的 GT、同样的题面,
    只把画面从全景换成以球员为中心的固定 640² 裁剪 (放到 448²)。

    目的是把"模型做不到"和"画面里根本没有可分辨的证据"分开: 全景臂上要判的
    半程步长差中位数只有 3.75 px, 低于 patch 尺度; 放大后约 17 px。两臂分数
    都在 chance 附近 -> 是能力问题; 放大臂显著更高 -> 全景臂是分辨率下限造成的,
    绝对不能拿它写"VLM 不会动力学"。
    """
    from engine.dynamics_qa import load_tracks

    items, stats = generate_accel_visual(
        args.seq_dir, win_s=args.win_s, hop_s=args.hop_s,
        n_frames=args.n_frames, max_pairs=args.max_per_family // 2)
    log(f"生成 {len(items)} 条 (与全景臂同一批对), 开始裁剪放大")
    tracks = load_tracks(args.seq_dir)
    need = sorted({f for it in items for f in it["window"]["frames"]})
    raw_dir = os.path.join(args.frame_dir, "_raw4k")
    raw_paths = extract_base_frames_native(
        os.path.join(args.seq_dir, "img1.mp4"), need, raw_dir)
    zoom_dir = os.path.join(args.frame_dir, "zoom")
    items, dropped = render_zoom(items, tracks, raw_paths, zoom_dir, info)

    rows = [strip(i) for i in items]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    bad = [p for p, v in by_pair.items() if len(v) != 2]
    if bad:
        raise SystemExit(f"{len(bad)} 个 pair_id 不是 2 条: {bad[:5]}")
    for p, v in by_pair.items():
        if {x["answer"] for x in v} != {"A", "B"}:
            raise SystemExit(f"{p} 的两个成员 GT 没有翻转")
        if len({x["question"] for x in v}) != 1:
            raise SystemExit(f"{p} 的两个成员题面不一致")
    n_a = sum(1 for r in rows if r["answer"] == "A")
    if n_a * 2 != len(rows):
        raise SystemExit(f"GT 不平衡: A={n_a} / {len(rows)}")
    dup = len(rows) - len({(r["image_id"], r["question"]) for r in rows})
    if dup:
        raise SystemExit(f"{dup} 条 (image_id, question) 重复, 拒绝写出")

    os.makedirs(args.out_dir, exist_ok=True)
    name = "qa_dyn_v2_paired_accel_vis_zoom.json"
    with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    log(f"写出 {name}: {len(rows)} 条 / {len(by_pair)} 对 "
        f"(丢弃 {len(dropped)} 对)")

    manifest = {
        "schema": "courtdyn-qa-v2-accel-visual-zoom",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "resolution control for qa_dyn_v2_paired_accel_vis.json — same "
            "pairs, same ground truth, same wording, only the framing differs. "
            "In the full-court arm the player is ~26 px tall in the model's "
            "actual input and the first-half/second-half step-length "
            "difference the question turns on is a median 3.75 px (p10 0.76), "
            "below Qwen's patch size. This arm crops a fixed 640 px window "
            "around the player from the native 4K frame and renders it at "
            "448 px, ~4.5x magnification, so that difference becomes ~17 px. "
            "If both arms sit at chance the limit is the model; if this arm is "
            "clearly higher, the full-court arm was measuring resolution, not "
            "dynamics ability, and must not be reported as the latter."
        ),
        "crop": {
            "crop_px_native": CROP_PX, "render_px": ZOOM_OUT,
            "magnification_vs_full_arm": round(
                (ZOOM_OUT / CROP_PX) / (960.0 / info["width"] * 0.622), 2),
            "rule": "one fixed rectangle per item, shared by all 4 frames "
                    "(a per-frame tracking crop would centre the player and "
                    "delete the motion the question asks about); size is "
                    "constant across the family so apparent scale cannot leak",
        },
        "counts": {"items": len(rows), "pairs": len(by_pair),
                   "answer_A": n_a, "answer_B": len(rows) - n_a,
                   "pairs_dropped_trajectory_too_large": len(dropped)},
        "stats": {k: v for k, v in stats.items() if k.startswith(("n_", "drop_"))},
    }
    mpath = os.path.join(args.out_dir, "qa_dyn_v2_accel_vis_zoom.manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    log(f"写出 manifest -> {mpath}")


def build_accel_visual(args, info):
    """v2 修复: 把 accel 家族从选项互换的语言侧最小对改成视觉侧最小对。

    只写 qa_dyn_v2_paired_accel_vis.json 和它的 manifest —— v1 的 QA / 帧 / T18
    结果一律不动, 老家族仍可作为"有洞的对照"继续引用 (见 engine/dynamics_qa.py
    generate_accel_visual 的 docstring)。
    """
    from engine.dynamics_qa import load_tracks

    items, stats = generate_accel_visual(
        args.seq_dir, win_s=args.win_s, hop_s=args.hop_s,
        n_frames=args.n_frames, max_pairs=args.max_per_family // 2)
    log(f"生成 {len(items)} 条视觉侧 accel QA "
        f"({stats['n_pairs_kept']}/{stats['n_pairs_available']} 对); "
        + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())
                    if k.startswith(("drop_", "n_pairs_a", "n_pairs_k", "n_w"))))
    if not items:
        raise SystemExit("没有生成任何视觉侧 accel 对 —— 检查 ACCEL_RATIO / 窗口")

    if args.dry_run:
        log(f"--- 样例\n    Q: {items[0]['question']}\n"
            f"    A(orig)={items[0]['answer']}  A(twin)={items[1]['answer']}\n"
            f"    windows: {items[0]['window']['frames']} vs "
            f"{items[1]['window']['frames']}")
        return

    tracks = load_tracks(args.seq_dir)
    need = sorted({f for it in items for f in it["window"]["frames"]})
    base_dir = os.path.join(args.frame_dir, "_base")
    base_paths = extract_base_frames(
        os.path.join(args.seq_dir, "img1.mp4"), need, base_dir,
        OUT_W / info["width"])
    items = render_overlays(items, tracks, base_paths,
                            OUT_W / info["width"], args.frame_dir)

    rows = [strip(i) for i in items]

    # ---- 校验: 这个家族的全部价值都在成对结构上, 坏一条就不该写出 ----------
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    bad = [p for p, v in by_pair.items() if len(v) != 2]
    if bad:
        raise SystemExit(f"{len(bad)} 个 pair_id 不是 2 条: {bad[:5]}")
    for p, v in by_pair.items():
        if {x["answer"] for x in v} != {"A", "B"}:
            raise SystemExit(f"{p} 的两个成员 GT 没有翻转: "
                             f"{[x['answer'] for x in v]}")
        if {x["question"] for x in v} != {v[0]["question"]}:
            raise SystemExit(f"{p} 的两个成员题面不一致 —— 视觉侧最小对要求逐字相同")
        wa, wb = ({"f0": x["meta"]["window"][0], "f1": x["meta"]["window"][1]}
                  for x in v)
        if not windows_disjoint(wa, wb):
            raise SystemExit(f"{p} 的两个窗口重叠, 不是不同片段: {wa} {wb}")
        if set(v[0]["meta"]["frames"]) & set(v[1]["meta"]["frames"]):
            raise SystemExit(f"{p} 的两个成员共享帧")
    n_a = sum(1 for r in rows if r["answer"] == "A")
    if n_a * 2 != len(rows):
        raise SystemExit(f"GT 不平衡: A={n_a} / {len(rows)}")
    dup = len(rows) - len({(r["image_id"], r["question"]) for r in rows})
    if dup:
        raise SystemExit(f"{dup} 条 (image_id, question) 重复 —— 打分器按这个键"
                         "对齐预测, 重复会串行, 拒绝写出")

    # ---- 混淆检查: 加速窗口是否系统性地比减速窗口整体更快 ------------------
    import statistics as st
    sp = {a: [r["meta"]["v_window_mean"] for r in rows if r["answer"] == a]
          for a in ("A", "B")}
    conf = {"mean_speed_A_speeding_up": round(st.mean(sp["A"]), 3),
            "mean_speed_B_slowing_down": round(st.mean(sp["B"]), 3),
            "median_speed_A": round(st.median(sp["A"]), 3),
            "median_speed_B": round(st.median(sp["B"]), 3)}
    conf["note"] = ("若两侧整体速度差很大, 模型可能靠'看起来快不快'而不是"
                    "'快慢在变'来作答; 差值小才说明成对结构逼着它看变化。")
    log(f"混淆检查 窗口平均速度 A(加速)={conf['mean_speed_A_speeding_up']} "
        f"vs B(减速)={conf['mean_speed_B_slowing_down']} m/s")

    os.makedirs(args.out_dir, exist_ok=True)
    name = "qa_dyn_v2_paired_accel_vis.json"
    with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    log(f"写出 {name}: {len(rows)} 条 / {len(by_pair)} 对")

    manifest = {
        "schema": "courtdyn-qa-v2-accel-visual",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "supersedes": {
            "family": "relational_reasoning_dyn_accel",
            "file": "qa_dyn_v1_paired_accel.json",
            "why": (
                "the v1 accel family was an option-swap LANGUAGE minimal pair "
                "(both members share the same frames), which defends only "
                "against positional bias, not semantic bias. On the T18 "
                "blank-image arm GRPO scored 47.1 paired without seeing "
                "anything, because a constant 'slowing down' answer satisfies "
                "both members. This v2 family is a VISUAL minimal pair: same "
                "player, two frame-disjoint windows, identical wording, GT "
                "flips — both bias types now score 0 paired, chance is 25%."
            ),
        },
        "source": {
            "sequence": info["name"],
            "gt_sha256": sha256(os.path.join(args.seq_dir, "gt", "gt.txt")),
            "video_sha256": sha256(os.path.join(args.seq_dir, "img1.mp4")),
            "seqinfo": info,
        },
        "params": {
            "win_s": args.win_s, "hop_s": args.hop_s, "n_frames": args.n_frames,
            "max_pairs": args.max_per_family // 2, "render_width": OUT_W,
            "accel_ratio": ACCEL_RATIO,
        },
        "pairing": {
            "rule": "same track, one speeding-up window + one slowing-down "
                    "window, windows must be frame-disjoint, closest in time "
                    "first, each window used at most once in this family",
            "chance_item": 50.0, "chance_paired": 25.0,
        },
        "counts": {"items": len(rows), "pairs": len(by_pair),
                   "answer_A": n_a, "answer_B": len(rows) - n_a},
        "confound_check": conf,
        "stats": {k: v for k, v in stats.items()
                  if k.startswith(("n_", "drop_"))},
        "highlight_convention": {"red": "queried player", "blue": "unused",
                                 "yellow": "unused"},
    }
    mpath = os.path.join(args.out_dir, "qa_dyn_v2_accel_vis.manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    log(f"写出 manifest -> {mpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default=SEQ_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--frame_dir", default=FRAME_DIR)
    ap.add_argument("--win_s", type=float, default=2.0, help="窗口时长 (秒)")
    ap.add_argument("--hop_s", type=float, default=1.0, help="窗口步进 (秒)")
    ap.add_argument("--n_frames", type=int, default=4,
                    help="每题帧数; 8GB 卡上 4 帧是吞吐/信息的折中")
    ap.add_argument("--max_per_family", type=int, default=140,
                    help="每个 QA 家族封顶 (成对家族按对计, 不拆对)")
    ap.add_argument("--dry_run", action="store_true", help="只算 QA, 不抽帧")
    ap.add_argument("--accel_visual", action="store_true",
                    help="只构建 accel 家族的视觉侧最小对 (v2 修复版), 写出 "
                         "qa_dyn_v2_paired_accel_vis.json; v1 的任何产物都不动")
    ap.add_argument("--accel_visual_zoom", action="store_true",
                    help="构建 accel 视觉侧最小对的放大对照臂 (固定 640² 裁剪 -> "
                         "448²), 写 qa_dyn_v2_paired_accel_vis_zoom.json")
    args = ap.parse_args()

    info = load_seqinfo(args.seq_dir)
    log(f"序列 {info['name']}: {info['length']} 帧 @ {info['fps']:.2f}fps "
        f"({info['length']/info['fps']:.1f}s), {info['width']}x{info['height']}")

    if args.accel_visual:
        return build_accel_visual(args, info)
    if args.accel_visual_zoom:
        return build_accel_visual_zoom(args, info)

    items, stats = generate(args.seq_dir, win_s=args.win_s, hop_s=args.hop_s,
                            n_frames=args.n_frames,
                            max_per_family=args.max_per_family)
    log(f"生成 {len(items)} 条 QA; 统计: "
        + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if not items:
        raise SystemExit("没有生成任何 QA —— 检查轨迹与过滤阈值")

    if args.dry_run:
        for cat in sorted({i["category"] for i in items}):
            ex = next(i for i in items if i["category"] == cat)
            log(f"--- {cat} (n={stats.get('n_'+cat)})\n    Q: {ex['question']}"
                f"\n    A: {ex['answer']}")
        return

    from engine.dynamics_qa import load_tracks
    tracks = load_tracks(args.seq_dir)
    need = sorted({f for it in items for f in it["window"]["frames"]})
    base_dir = os.path.join(args.frame_dir, "_base")
    base_paths = extract_base_frames(
        os.path.join(args.seq_dir, "img1.mp4"), need, base_dir, OUT_W / info["width"])
    items = render_overlays(items, tracks, base_paths,
                            OUT_W / info["width"], args.frame_dir)

    cf = rescale_for_prior(items, CF_PLAYER_HEIGHT_M)
    os.makedirs(args.out_dir, exist_ok=True)
    main_rows = [strip(i) for i in items]
    cf_rows = [strip(i) for i in cf]
    # eval/paired_relational_scorer.py 要求池里每条都带 pair_id/pair_role, 且
    # 两个成对家族测的是不同东西 (accel = 语言侧最小对, faster = 视觉侧最小对),
    # 必须分开打分, 所以各切一个子集出来。
    paired = {
        "qa_dyn_v1_paired_accel.json":
            [r for r in main_rows if r["category"].endswith("_dyn_accel")],
        "qa_dyn_v1_paired_faster.json":
            [r for r in main_rows if r["category"].endswith("_dyn_faster")],
    }
    dup = len(main_rows) - len({(r["image_id"], r["question"]) for r in main_rows})
    if dup:
        raise SystemExit(f"{dup} 条 (image_id, question) 重复 —— 打分器按这个键"
                         "对齐预测, 重复会串行, 拒绝写出")
    for name, rows in (("qa_dyn_v1.json", main_rows),
                       ("qa_dyn_v1_cf_prior.json", cf_rows), *paired.items()):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
        log(f"写出 {name}: {len(rows)} 条")

    manifest = {
        "schema": "courtdyn-qa-v1",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "sequence": info["name"],
            "gt_sha256": sha256(os.path.join(args.seq_dir, "gt", "gt.txt")),
            "video_sha256": sha256(os.path.join(args.seq_dir, "img1.mp4")),
            "seqinfo": info,
        },
        "params": {
            "win_s": args.win_s, "hop_s": args.hop_s, "n_frames": args.n_frames,
            "max_per_family": args.max_per_family, "render_width": OUT_W,
        },
        "scale_prior": {
            "player_height_m": PLAYER_HEIGHT_M,
            "counterfactual_height_m": CF_PLAYER_HEIGHT_M,
            "method": "per-track bbox-height ruler (monocular, uncalibrated)",
            "known_biases": [
                "depth compression for camera-ward motion (mitigated by "
                "DEPTH_TOL bbox-height stability filter)",
                "individual height variance ~+-8% enters GT directly",
                "the camera is elevated and oblique (not a true side view), so a "
                "standing person's bbox height is foreshortened; this inflates "
                "m_per_px and therefore the absolute speed/path GT",
            ],
            "claim_resolution": "~10% relative; no centimetre-level claims",
            "scale_invariant_families": [
                "dynamics_time_intercept (distance and speed scale together)",
                "relational_reasoning_dyn_accel (within-track ratio)",
                "relational_reasoning_dyn_faster (between-track ratio)",
            ],
            "planned_upgrade": (
                "the full court is visible with clean line markings — fitting a "
                "ground-plane homography from the court lines would replace the "
                "body-height ruler with true metric coordinates and remove both "
                "the foreshortening and the individual-height bias. Absolute "
                "speed/path numbers should be treated as provisional until then; "
                "the three scale-invariant families above are unaffected."
            ),
        },
        "families": {k: v for k, v in stats.items() if k.startswith("n_")},
        "filters": {k: v for k, v in stats.items() if k.startswith(("drop_", "cap_"))},
        "counts": dict({"main": len(main_rows), "cf_prior": len(cf_rows)},
                       **{k: len(v) for k, v in paired.items()}),
        "highlight_convention": {"red": "queried player / option A",
                                 "blue": "option B", "yellow": "ball"},
    }
    mpath = os.path.join(args.out_dir, "qa_dyn_v1.manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    log(f"写出 manifest -> {mpath}")


if __name__ == "__main__":
    main()
