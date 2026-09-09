#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sample_courtdyn_qc.py — CourtDyn 人工抽检子集 (基准质量抽检, 不是标注真值)。

论文里要挡的质疑是 "你的题全自动生成, 有没有噪声"。做法: 分层随机抽 N 条,
每条渲染成一张 4 帧横条, 附上**自动检查**结果, 由人眼确认三件事:
  Q1 红框标的确实是题目问的那名球员, 且四帧是同一个人 (身份一致);
  Q2 四帧确实覆盖了题目问的那段运动 (窗口正确);
  Q3 没有遮挡 / 出画 / 画错框 等会让题目无法回答的情况。

自动检查 (给人眼做参考, 不替代):
  auto_red_frames  4 帧里颜色阈值能检出红框的帧数 (来自 pixel_oracle 的同一检测器, 该检测器
                   在 597x336 模型输入下检出率 100%, 中心误差中位 0.3 px);
  auto_edge        红框是否贴到画面边缘 (出画风险);
  auto_px_disp     渲染像素位移 (太小 => 题目接近退化, 四帧看不出运动);
  auto_id_jump     相邻帧红框中心跳变的最大值 (像素), 明显大于位移尺度 => 可能跟错人。

分层: 每条片段 × {speed, path} 各抽 --per_cell 条, 固定 seed, 可复现。
产物: results/courtdyn/qc_sample.json (逐条记录 + 自动检查)
      results/courtdyn/qc_strips/<item_id>.jpg (4 帧横条)

用法: python tools/sample_courtdyn_qc.py [--per_cell 3] [--seed 42]
"""
import argparse
import io
import json
import math
import os
import random
import statistics as st
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import dynamics_qa as DQ                                   # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer       # noqa: E402
from tools.pixel_oracle_courtdyn import detect, box_foot               # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_JSON = os.path.join(CD, "qc_sample.json")
STRIP_DIR = os.path.join(CD, "qc_strips")
MAIN = "Q4_side_480-510"
SEQS = [("Q1_top_0-30", "俯视"), ("Q2_top_480-510", "俯视"), ("Q4_top_0-30", "俯视"),
        (MAIN, "侧视 (原片段)"), ("Q1_side_0-30", "侧视"),
        ("Q2_side_300-330", "侧视"), ("Q4_side_570-600", "侧视")]
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
RENDER_W = 960
TILE_W = 330
GAP = 6
ZOOM_FACTOR = 6          # 放大行的裁剪边长 = 框最长边 x 该系数


def seq_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def img_root(seq):
    return os.path.join(ROOT, "data", "courtdyn", "frames" if seq == MAIN else f"frames_{seq}")


def tt_split(seq):
    hits = json.load(io.open(os.path.join(CD, "teamtrack_probe_hits.json"), encoding="utf-8"))
    for fam, split, name in hits:
        if name == seq:
            return os.path.join(ROOT, "data", "external_validation", "teamtrack",
                                "teamtrack-mot", "teamtrack-mot", fam, split, seq)
    raise SystemExit(f"probe_hits 里找不到 {seq}")


def make_strip(paths, out_path, boxes=None, scale=1.0):
    """两行: 上排 4 张全景 (上下文), 下排以红框为中心的放大裁剪 (身份核对)。

    人眼要判断的是"红框标的是不是同一个人", 全景里球员只有十几像素, 根本看不清,
    所以必须给放大行。裁剪用 GT 框 (QC 检查的是渲染/出题是否与题面一致, 不是检测精度)。
    """
    top, zoom = [], []
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            return None
        H, W = im.shape[:2]
        h = int(H * TILE_W / W)
        top.append(cv2.resize(im, (TILE_W, h), interpolation=cv2.INTER_AREA))
        if boxes and i < len(boxes) and boxes[i] is not None:
            x, y, bw, bh = (v * scale for v in boxes[i])
            cx, cy = x + bw / 2.0, y + bh / 2.0
            half = max(bw, bh) * ZOOM_FACTOR / 2.0
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
            sub = im[y0:y1, x0:x1]
            if sub.size:
                zoom.append(cv2.resize(sub, (TILE_W, TILE_W), interpolation=cv2.INTER_CUBIC))
                continue
        zoom.append(cv2.resize(im[0:1, 0:1], (TILE_W, TILE_W)) * 0)

    def row(tiles):
        out = tiles[0]
        for t in tiles[1:]:
            out = cv2.hconcat([out, cv2.copyMakeBorder(t, 0, 0, GAP, 0,
                                                       cv2.BORDER_CONSTANT, value=(12, 13, 11))])
        return out

    r1, r2 = row(top), row(zoom)
    canvas = cv2.vconcat([r1, cv2.copyMakeBorder(r2, GAP, 0, 0, 0,
                                                 cv2.BORDER_CONSTANT, value=(12, 13, 11))])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 84])
    return canvas.shape[1], canvas.shape[0]


def auto_checks(item, seq):
    """颜色阈值检测器上的自动检查; 全部只看渲染帧, 不看 GT。"""
    paths = [os.path.join(img_root(seq), x) for x in item["image_ids"]]
    det = detect(paths, "render")
    reds = [d["red"] if d else None for d in det]
    n_red = sum(1 for b in reds if b is not None)
    edge, disp, jump, hs = False, None, None, []
    got = [b for b in reds if b is not None]
    if got:
        im = cv2.imread(paths[0])
        W, H = (im.shape[1], im.shape[0]) if im is not None else (RENDER_W, 540)
        for b in got:
            if b[0] <= 2 or b[1] <= 2 or b[2] >= W - 3 or b[3] >= H - 3:
                edge = True
            hs.append(b[3] - b[1])
    if len(got) == len(reds) and len(got) >= 2:
        feet = [box_foot(b) for b in got]
        steps = [math.dist(feet[i], feet[i + 1]) for i in range(len(feet) - 1)]
        disp = sum(steps)
        jump = max(steps)
    return {"auto_red_frames": f"{n_red}/{len(reds)}", "auto_edge": edge,
            "auto_px_disp": None if disp is None else round(disp, 1),
            "auto_max_step_px": None if jump is None else round(jump, 1),
            "auto_box_h_median": None if not hs else round(st.median(hs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cell", type=int, default=3, help="每 (片段 × 家族) 抽几条")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = []
    for seq, view in SEQS:
        items = json.load(io.open(os.path.join(seq_dir(seq), "qa_dyn_v1.json"), encoding="utf-8"))
        tt = tt_split(seq)
        info, tracks = DQ.load_seqinfo(tt), DQ.load_tracks(tt)
        fps, scale = float(info["fps"]), RENDER_W / float(info["width"])
        try:
            plane = CourtPlane.load(seq)
        except FileNotFoundError:
            plane = None
        for cat, fam in FAMS:
            pool = [it for it in items if it["category"] == cat]
            for it in rng.sample(pool, min(args.per_cell, len(pool))):
                m = it["meta"]
                iid = f"{seq}__{fam}__w{m['window_index']}__t{m['track']}"
                strip = os.path.join(STRIP_DIR, iid + ".jpg")
                bb = tracks[m["track"]]
                boxes = [bb.get(f) for f in m["frames"]]
                if make_strip([os.path.join(img_root(seq), x) for x in it["image_ids"]],
                              strip, boxes=boxes, scale=scale) is None:
                    print(f"  ! 渲染帧缺失, 跳过 {iid}")
                    continue
                v3 = None
                if plane:
                    r = recompute_answer(plane, it, tracks, fps)
                    v3 = r[0] if r else None
                f0, f1 = m["window"]
                pxd = DQ.path_length_px(tracks[m["track"]], f0, f1)
                rec = {
                    "item_id": iid, "seq": seq, "view": view, "family": fam,
                    "question": it["question"], "answer_v1": it["answer"], "answer_v3": v3,
                    "px_disp_gt": None if pxd is None else round(pxd * scale, 1),
                    "track": m["track"], "window": m["window"], "frames": m["frames"],
                    "image_ids": it["image_ids"],
                    "strip": os.path.relpath(strip, ROOT).replace("\\", "/"),
                }
                rec.update(auto_checks(it, seq))
                out.append(rec)
                print(f"  {iid}: red {rec['auto_red_frames']}, disp {rec['auto_px_disp']} px, "
                      f"edge {rec['auto_edge']}, v1 {rec['answer_v1']} / v3 {rec['answer_v3']}")
    man = {"schema": "courtdyn-qc-v1", "seed": args.seed, "per_cell": args.per_cell,
           "n": len(out), "criteria": {
               "Q1": "红框标的确实是题目问的那名球员, 且四帧是同一个人",
               "Q2": "四帧确实覆盖了题目问的那段运动",
               "Q3": "没有遮挡 / 出画 / 画错框等使题目无法回答的情况"},
           "items": out}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"\n[qc] {len(out)} 条 -> {OUT_JSON}; 横条图 -> {STRIP_DIR}")


if __name__ == "__main__":
    main()
