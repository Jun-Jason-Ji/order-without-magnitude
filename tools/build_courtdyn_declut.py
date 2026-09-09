#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_declut.py — zoom2 干预的**去干扰对照臂** (paper 2, T31 / N5-1)。

为什么需要它
------------
T27 的 zoom2 臂把侧视 ρ 从 0.0-0.2 拉到 0.43-0.50 (§一 E2-C)。但那个裁剪同时改了两件事:
  (1) 目标变大   —— 裁剪块被放大到与全景臂相同的 960 宽, 表观尺寸 x2;
  (2) 干扰变少   —— 裁剪块以外的球员/场地内容全部消失。
两者在 zoom2 里不可分离, 所以正文只能报一个复合干预, Limitations 里写明"要拆开需要一条对照臂"。
本脚本就是那条对照臂。

设计 (与 zoom2 严格互补)
------------------------
源片段 3840x2160, OUT_W=960, 于是:
  full   : 整帧 3840x2160  -> scale 0.25 -> 960x540      (表观尺寸 1x, 干扰全在)
  zoom2  : 裁 1920x1080    -> scale 0.50 -> 960x540      (表观尺寸 2x, 干扰已去)
  declut : 裁 1920x1080    -> scale 0.25 -> 480x270,
           居中贴进 960x540 的黑底画布                    (表观尺寸 1x, 干扰已去)  <= 本脚本

**裁剪矩形直接复用 build_courtdyn_zoom2.crop_rect**, 不是重新实现 —— 保证与 zoom2 臂
保留的像素内容逐条完全一致, 丢弃的 item 也完全一致, 否则相减没有意义。
红框线宽用 LINE_W (与 full 臂相同), 且球员在画布上的像素尺寸与 full 臂逐像素相同,
因此 declut 与 full 的唯一差别就是"裁剪块以外的内容没有了"。

于是三条臂可以相减:
    full -> zoom2   = Δ(表观尺寸) + Δ(周边内容)
    full -> declut  =               Δ(周边内容)
    两者相减        = Δ(表观尺寸)

诚实的边界 (必须写进正文, 不要写成"去掉了其他球员")
--------------------------------------------------
黑边抹掉的是**裁剪块以外的一切**, 既包括干扰球员, 也包括场地线与远景 —— 后者正是透视线索。
所以这条臂测的是"周边内容", 不是"其他球员"。真要单独隔离球员, 需要把他们涂掉而保留场地,
那会引入自己的显著伪影, 是另一件事。

用法: python tools/build_courtdyn_declut.py --seq Q4_side_480-510
产物: results/courtdyn/seq_<seq>/qa_dyn_v1_declut.json (+ .manifest.json);
      帧在 data/courtdyn/frames_<seq>/declut/dc_w{wi}_t{tid}_f{frame}.jpg (image_ids 带 "declut/" 前缀)。
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
from engine import dynamics_qa as DQ                              # noqa: E402
from tools.build_courtdyn_qa import OUT_W, LINE_W, COLORS         # noqa: E402
from tools.build_courtdyn_zoom2 import crop_rect, qa_dir, frame_dir, KEEP, ZOOM   # noqa: E402
from tools.summarize_courtdyn_t26 import seq_dir_of               # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
PAD = (0, 0, 0)          # 黑底; 与灰板臂 (grey) 刻意不同色, 免得两条对照臂混淆


def log(msg):
    print(f"[declut {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import cv2
    import numpy as np
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
    out_dir = os.path.join(frame_dir(seq), "declut")
    os.makedirs(out_dir, exist_ok=True)

    scale = OUT_W / float(W)                  # == full 臂的缩放, 不是 zoom2 的
    canvas_w, canvas_h = OUT_W, int(round(H * scale))
    sub_w, sub_h = int(round((W // ZOOM) * scale)), int(round((H // ZOOM) * scale))
    ox, oy = (canvas_w - sub_w) // 2, (canvas_h - sub_h) // 2
    log(f"{seq}: {W}x{H} -> 画布 {canvas_w}x{canvas_h}; 裁剪块 {W//ZOOM}x{H//ZOOM} "
        f"-> {sub_w}x{sub_h} 居中贴于 ({ox},{oy}); scale={scale:.4f} (与 full 臂相同)")

    need = {}
    kept, dropped = [], []
    for it in items:
        m = it["meta"]
        wi, tid, frames = m["window_index"], m["track"], m["frames"]
        rect = crop_rect(tracks[tid], frames, W, H)      # 与 zoom2 逐字相同的矩形
        if rect is None:
            dropped.append({"category": it["category"], "window_index": wi, "track": tid})
            continue
        rels = []
        for f in frames:
            name = f"dc_w{wi:02d}_t{tid}_f{f:04d}.jpg"
            need.setdefault(f, {})[(wi, tid)] = (rect, name)
            rels.append("declut/" + name)
        it = dict(it, image_ids=rels, image_id=rels[0],
                  meta=dict(m, declut={"crop_origin_xy": [rect[0], rect[1]], "crop_wh": [rect[2], rect[3]],
                                       "render_w": sub_w, "canvas_wh": [canvas_w, canvas_h],
                                       "paste_xy": [ox, oy], "apparent_scale": "same as full arm"}))
        kept.append(it)

    todo = {f: {k: v for k, v in d.items() if not os.path.isfile(os.path.join(out_dir, v[1]))}
            for f, d in need.items()}
    todo = {f: d for f, d in todo.items() if d}
    log(f"{seq}: {len(items)} 条 speed/path, 保留 {len(kept)}, 丢弃 {len(dropped)}; "
        f"需要 {sum(len(d) for d in need.values())} 张, 待渲染 {sum(len(d) for d in todo.values())} 张")

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
                sub = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (sub_w, sub_h),
                                 interpolation=cv2.INTER_AREA)
                b = tracks[tid].get(idx)
                if b is not None:
                    x, y, w, h = b
                    cv2.rectangle(sub, (int((x - x0) * scale), int((y - y0) * scale)),
                                  (int((x + w - x0) * scale), int((y + h - y0) * scale)),
                                  COLORS["red"], LINE_W)
                canvas = np.full((canvas_h, canvas_w, 3), PAD, dtype=np.uint8)
                canvas[oy:oy + sub_h, ox:ox + sub_w] = sub
                cv2.imwrite(os.path.join(out_dir, name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
                written += 1
            if written % 200 == 0:
                log(f"  已渲染 {written} 张 (帧 {idx}/{last})")
        cap.release()
        missing = [v[1] for d in todo.values() for v in d.values()
                   if not os.path.isfile(os.path.join(out_dir, v[1]))]
        if missing:
            raise SystemExit(f"{len(missing)} 张没渲染出来 (视频比 gt 短?): {missing[:3]}")
        log(f"渲染完成 {written} 张")

    out = os.path.join(qa_dir(seq), "qa_dyn_v1_declut.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    with io.open(out.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "courtdyn-qa-v1-declut", "seq": seq, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "source": "qa_dyn_v1.json (speed/path only; question, answer, meta unchanged)",
                   "crop": f"{W // ZOOM}x{H // ZOOM} native, identical rect to the zoom2 arm "
                           f"(crop_rect reused verbatim), rendered to {sub_w}x{sub_h} and centred "
                           f"on a {canvas_w}x{canvas_h} black canvas",
                   "apparent_scale": "identical to the full arm (scale = OUT_W / W)",
                   "isolates": "content outside the crop (other players AND court context), "
                               "with apparent size held fixed",
                   "decomposition": "full->zoom2 = d(size)+d(context); full->declut = d(context); "
                                    "difference = d(size)",
                   "caveat": "the black border removes court lines and background as well as "
                             "distracting players; this arm is NOT 'other players removed'",
                   "n_source": len(items), "n_kept": len(kept), "dropped": dropped},
                  f, ensure_ascii=False, indent=1)
    log(f"写出 {out} ({len(kept)} 条)")


if __name__ == "__main__":
    main()
