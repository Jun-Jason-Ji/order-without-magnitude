#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_zoom2.py — CourtDyn 主池 speed / path 的 **2× 缩放干预臂** (paper 2, T27)。

问题: 模型在俯视片段上 speed / path 的秩相关 0.5–0.8 (T21/T26), 首帧×4 归零 (T26) —— 它确实在读多帧
运动; 但它读的是"图像平面上红框动了多少像素", 还是套了题面身高尺 / 透视之后的"世界里动了多少米"?
干预: 同一 item、同一 4 帧、同一题面 (含 "Assume a typical player … m tall")、同一 GT, 只把画面换成
以球员轨迹为中心的 **半幅裁剪** (原始 W/2 × H/2), 再缩到与全景臂**相同**的 960 宽 —— 模型眼前的像素
位移恰好 ×2, 框高也 ×2 (尺子随之 ×2, 会用尺子的读者答案不变), 世界量不变。
  预测中位比 pred(zoom2)/pred(full) ≈ 2 ⇒ 读像素位移, 不归一化;  ≈ 1 ⇒ 用了尺子 (或读的是世界量)。
4 帧用同一个裁剪框 (跟着球员裁会把运动裁没), 轨迹装不进半幅框的 item 丢弃并记录。
顺序解码一遍 mp4, 直接裁出并写 960×540 的成品帧, 不落 4K 底帧 (省 ~1 GB/片段)。

用法: python tools/build_courtdyn_zoom2.py --seq Q2_top_480-510
产物: results/courtdyn/seq_<seq>/qa_dyn_v1_zoom2.json (+ .manifest.json);
      帧在 data/courtdyn/frames_<seq>/zoom2/z2_w{wi}_t{tid}_f{frame}.jpg (image_ids 带 "zoom2/" 前缀,
      run_bench 的 --img_root 仍指 frames_<seq>)。原片段 Q4_side_480-510 的池在 results/courtdyn/,
      帧在 data/courtdyn/frames/。
"""
import argparse
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import dynamics_qa as DQ                       # noqa: E402
from tools.build_courtdyn_qa import OUT_W, LINE_W, COLORS   # noqa: E402
from tools.summarize_courtdyn_t26 import seq_dir_of         # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
MAIN = "Q4_side_480-510"
KEEP = ("dynamics_speed_player", "dynamics_path_player")
ZOOM = 2


def log(msg):
    print(f"[zoom2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def qa_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def frame_dir(seq):
    return os.path.join(ROOT, "data", "courtdyn", "frames" if seq == MAIN else f"frames_{seq}")


def crop_rect(track, frames, W, H):
    cw, ch = W // ZOOM, H // ZOOM
    xs, ys = [], []
    for f in frames:
        b = track.get(f)
        if b is None:
            continue
        x, y, w, h = b
        xs += [x, x + w]
        ys += [y, y + h]
    if not xs:
        return None
    if max(xs) - min(xs) > cw * 0.9 or max(ys) - min(ys) > ch * 0.9:
        return None
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    x0 = int(round(min(max(cx - cw / 2.0, 0), W - cw)))
    y0 = int(round(min(max(cy - ch / 2.0, 0), H - ch)))
    return x0, y0, cw, ch


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    args = ap.parse_args()
    seq = args.seq
    sd = seq_dir_of(seq)
    info = DQ.load_seqinfo(sd)
    tracks = DQ.load_tracks(sd)
    W, H = info["width"], info["height"]
    v1 = json.load(open(os.path.join(qa_dir(seq), "qa_dyn_v1.json"), encoding="utf-8"))
    items = [dict(it) for it in v1 if it["category"] in KEEP]
    out_dir = os.path.join(frame_dir(seq), "zoom2")
    os.makedirs(out_dir, exist_ok=True)
    scale = OUT_W / float(W // ZOOM)          # 裁剪块 -> 960 宽, 与全景臂同一输出宽度
    need = {}                                 # frame -> {(wi, tid): rect}
    kept, dropped = [], []
    for it in items:
        m = it["meta"]
        wi, tid, frames = m["window_index"], m["track"], m["frames"]
        rect = crop_rect(tracks[tid], frames, W, H)
        if rect is None:
            dropped.append({"category": it["category"], "window_index": wi, "track": tid})
            continue
        rels = []
        for f in frames:
            name = f"z2_w{wi:02d}_t{tid}_f{f:04d}.jpg"
            need.setdefault(f, {})[(wi, tid)] = (rect, name)
            rels.append("zoom2/" + name)
        it = dict(it, image_ids=rels, image_id=rels[0],
                  meta=dict(m, zoom2={"crop_origin_xy": [rect[0], rect[1]], "crop_wh": [rect[2], rect[3]],
                                     "render_w": OUT_W, "zoom": ZOOM}))
        kept.append(it)
    todo = {f: {k: v for k, v in d.items() if not os.path.isfile(os.path.join(out_dir, v[1]))} for f, d in need.items()}
    todo = {f: d for f, d in todo.items() if d}
    log(f"{seq}: {len(items)} 条 speed/path, 保留 {len(kept)}, 丢弃 {len(dropped)}; "
        f"需要 {sum(len(d) for d in need.values())} 张成品帧, 其中 {sum(len(d) for d in todo.values())} 张待渲染")
    if todo:
        cap = cv2.VideoCapture(os.path.join(sd, "img1.mp4"))
        if not cap.isOpened():
            raise SystemExit(f"打不开视频: {sd}")
        idx = written = 0
        last = max(todo)
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1                          # MOT 帧号从 1 起
            d = todo.get(idx)
            if not d:
                continue
            for (wi, tid), (rect, name) in d.items():
                x0, y0, cw, ch = rect
                sub = frame[y0:y0 + ch, x0:x0 + cw]
                sub = cv2.resize(sub, (OUT_W, int(round(ch * scale))), interpolation=cv2.INTER_AREA)
                b = tracks[tid].get(idx)
                if b is not None:
                    x, y, w, h = b
                    cv2.rectangle(sub, (int((x - x0) * scale), int((y - y0) * scale)),
                                  (int((x + w - x0) * scale), int((y + h - y0) * scale)), COLORS["red"], LINE_W)
                cv2.imwrite(os.path.join(out_dir, name), sub, [cv2.IMWRITE_JPEG_QUALITY, 88])
                written += 1
            if written % 200 == 0:
                log(f"  已渲染 {written} 张 (帧 {idx}/{last})")
        cap.release()
        missing = [v[1] for d in todo.values() for v in d.values() if not os.path.isfile(os.path.join(out_dir, v[1]))]
        if missing:
            raise SystemExit(f"{len(missing)} 张没渲染出来 (视频比 gt 短?): {missing[:3]}")
        log(f"渲染完成 {written} 张")
    out = os.path.join(qa_dir(seq), "qa_dyn_v1_zoom2.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    with io.open(out.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "courtdyn-qa-v1-zoom2", "seq": seq, "zoom": ZOOM, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "source": "qa_dyn_v1.json (speed/path only; question, answer, meta unchanged)",
                   "crop": f"{W // ZOOM}x{H // ZOOM} native, centred on the 4-frame trajectory bbox, one rect per item, rendered to {OUT_W} wide",
                   "n_source": len(items), "n_kept": len(kept), "dropped": dropped,
                   "prediction": "pixel-displacement reader: pred(zoom2)/pred(full) ~ 2; ruler/world reader: ~ 1"},
                  f, ensure_ascii=False, indent=1)
    log(f"写出 {out} ({len(kept)} 条)")


if __name__ == "__main__":
    main()
