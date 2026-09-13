#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_hm3.py — N1: 第二场地的 CourtDyn 题池, 建在 Human-M3 篮球序列上。

为什么换掉原计划的 TeamTrack: TeamTrack 篮球那 77 条序列**全部是同一场比赛**
(Q1/Q2/Q4 x side/top 的 30 秒切片), 里面根本没有第二场地。Human-M3 篮球是另一个场地,
而且比 TeamTrack 更强 —— 它自带**真三维世界坐标**与相机标定:

  * GT 不再需要身高尺 (v1) 或球场单应 (v3): 直接用世界坐标算地面位移, 一次消掉论文现在
    最大的一条 Limitation, 也绕开了俯视上 "题面尺子 != 计分尺子" 那道 2 倍歧义。
  * 同一段世界运动**同时**被 3-4 台不同几何的相机拍到 => 可以在**同一批 item**上直接测
    "视角决定序": 世界量完全相同, 只有相机几何变, ρ 若随相机变化, E1a 就不再依赖跨片段比较。

序列 (data/external_validation/human_m3_basketball_test/, 已在盘上, 零下载):
  basketball1/split1  4 相机 x 200 帧 (10 fps, 19.9 s)
  basketball1/split2  4 相机 x 200 帧
  basketball2         3 相机 x 500 帧 (49.9 s)

构造与 CourtDyn 逐字对齐 (2.0 s 窗口 / 4 帧 / 红框标注 / 同一套题面), 只把 GT 换成世界坐标:
  人物地面位置 = 该帧全部关节的 XY 中位数 (对肢体摆动稳健; 与 TeamTrack 版用足点同为 "地面位置" 口径)
  path_m       = 窗口内逐帧位置的累计位移;  speed = path / 时长
  像素位移     = 同一批帧上投影框足点的累计位移 (写进 meta, 供 pixel_reader 直接用)

用法: python tools/build_courtdyn_hm3.py --clip basketball1/split1/camera_0
      python tools/build_courtdyn_hm3.py --all
产物: results/courtdyn/seq_hm3_<clip>/qa_dyn_v1.json (+ manifest), 帧 data/courtdyn/frames_hm3_<clip>/
"""
import argparse
import glob
import io
import json
import math
import os
import statistics as st
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HM3 = os.path.join(ROOT, "data", "external_validation", "human_m3_basketball_test",
                   "annotated", "humanm3", "test")
CD = os.path.join(ROOT, "results", "courtdyn")
FRAMES = os.path.join(ROOT, "data", "courtdyn")

CLIPS = ["basketball1/split1", "basketball1/split2", "basketball2"]
OUT_W = 960                 # 与 CourtDyn 渲染宽度一致
WIN_S, HOP_S, N_FRAMES = 2.0, 1.0, 4
HEIGHT_PRIOR_M = 1.93
MIN_BOX_H_PX = 14           # 渲染后框高下限, 太小人眼与模型都看不出是谁
EDGE_MARGIN = 4
MAX_PER_FAMILY = 140
SPEED, PATH = "dynamics_speed_player", "dynamics_path_player"

Q_SPEED = ("The {n} frames are consecutive samples from one basketball clip spanning {dur} seconds, "
           "in chronological order. Assume a typical player on this court is {hp} m tall. "
           "What is the average speed of the player marked with the red box, in meters per second? "
           "Output only the number, one decimal place.")
Q_PATH = ("The {n} frames are consecutive samples from one basketball clip spanning {dur} seconds, "
          "in chronological order. Assume a typical player on this court is {hp} m tall. "
          "How far does the player marked with the red box travel along the floor over the whole "
          "clip, in meters? Output only the number, one decimal place.")


def clip_dir(clip):
    return os.path.join(HM3, *clip.split("/"))


def load_camera(clip, cam):
    p = os.path.join(clip_dir(clip), "camera_calibration", cam + ".json")
    d = json.load(io.open(p, encoding="utf-8"))
    K = np.array(d["intrinsic"], dtype=np.float64)      # 3x4
    E = np.array(d["extrinsic"], dtype=np.float64)      # 4x4 (world -> camera)
    return K @ E                                        # 3x4 world -> image


def project(P, pts):
    """pts: (n,3) 世界坐标 -> (n,2) 图像坐标; 相机后方的点返回 None。"""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    v = (P @ h.T).T
    if np.any(v[:, 2] <= 1e-6):
        return None
    return v[:, :2] / v[:, 2:3]


def load_poses(clip):
    """frame index -> {person id: (n_joints,3) 世界坐标}。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(clip_dir(clip), "pose_calib", "*.json"))):
        idx = int(os.path.splitext(os.path.basename(p))[0])
        d = json.load(io.open(p, encoding="utf-8"))
        out[idx] = {pid: np.asarray(j, dtype=np.float64) for pid, j in d.items()}
    return out


def image_list(clip, cam):
    d = os.path.join(clip_dir(clip), "images", cam)
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.jpeg")))
    return fs


def ground_xy(joints):
    """人物地面位置 = 全部关节 XY 的中位数 (对摆臂/迈步稳健)。"""
    return (float(np.median(joints[:, 0])), float(np.median(joints[:, 1])))


def box_from_joints(P, joints, W, H, pad=0.12):
    uv = project(P, joints)
    if uv is None:
        return None
    x0, y0 = uv[:, 0].min(), uv[:, 1].min()
    x1, y1 = uv[:, 0].max(), uv[:, 1].max()
    w, h = x1 - x0, y1 - y0
    if h < 1 or w < 1:
        return None
    x0, x1 = x0 - w * pad, x1 + w * pad
    y0, y1 = y0 - h * pad, y1 + h * pad
    if x0 < EDGE_MARGIN or y0 < EDGE_MARGIN or x1 > W - EDGE_MARGIN or y1 > H - EDGE_MARGIN:
        return None                       # 出画 / 贴边的不出题
    return (x0, y0, x1, y1)


def build_clip(clip, cam, poses, fps, limit):
    P = load_camera(clip, cam)
    imgs = image_list(clip, cam)
    frames = sorted(poses)
    if len(imgs) != len(frames):
        print(f"  ! {clip}/{cam}: 图像 {len(imgs)} 帧 vs 标注 {len(frames)} 帧, 跳过")
        return None
    idx_of = {f: i for i, f in enumerate(frames)}
    probe = cv2.imread(imgs[0])
    if probe is None:
        print(f"  ! {clip}/{cam}: 读不出首帧, 跳过")
        return None
    H0, W0 = probe.shape[:2]
    scale = OUT_W / W0
    win = int(round(WIN_S * fps))
    hop = int(round(HOP_S * fps))
    cand = []
    for s in range(0, len(frames) - win, hop):
        f0, f1 = frames[s], frames[s + win]
        span = [frames[s + k] for k in range(win + 1)]
        sample = [span[round(k * win / (N_FRAMES - 1))] for k in range(N_FRAMES)]
        ids = set(poses[f0]).intersection(*[set(poses[f]) for f in span])
        for pid in sorted(ids):
            boxes = {}
            ok = True
            for f in sample:
                b = box_from_joints(P, poses[f][pid], W0, H0)
                if b is None or (b[3] - b[1]) * scale < MIN_BOX_H_PX:
                    ok = False
                    break
                boxes[f] = b
            if not ok:
                continue
            xy = [ground_xy(poses[f][pid]) for f in span]
            path_m = sum(math.dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1))
            feet = [((boxes[f][0] + boxes[f][2]) / 2 * scale, boxes[f][3] * scale) for f in sample]
            px = sum(math.dist(feet[i], feet[i + 1]) for i in range(len(feet) - 1))
            dur = (f1 - f0) / fps
            cand.append({"window_index": s // hop, "track": int(pid), "window": [f0, f1],
                         "frames": sample, "boxes": boxes, "path_m": path_m,
                         "speed_mps": path_m / dur, "dur": dur, "px_disp": px,
                         "box_h_px": st.median((boxes[f][3] - boxes[f][1]) * scale for f in sample)})
    if not cand:
        return None
    # 覆盖动态范围: 按 path 排序后等距抽 limit 条 (与 CourtDyn 的 max_per_family 同义)
    cand.sort(key=lambda c: c["path_m"])
    if len(cand) > limit:
        step = len(cand) / limit
        cand = [cand[int(i * step)] for i in range(limit)]
    return {"cands": cand, "imgs": imgs, "idx_of": idx_of, "scale": scale, "P": P,
            "size": (W0, H0), "fps": fps}


def render(clip, cam, built, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    need = {}
    for c in built["cands"]:
        for f in c["frames"]:
            need.setdefault(f, []).append((c["track"], c["window_index"], c["boxes"][f]))
    scale = built["scale"]
    written = 0
    for f, marks in sorted(need.items()):
        src = built["imgs"][built["idx_of"][f]]
        base = None
        for track, wi, box in marks:
            name = f"f{f:05d}_t{track}_w{wi}.jpg"
            out = os.path.join(frame_dir, name)
            if os.path.isfile(out):
                continue
            if base is None:
                im = cv2.imread(src)
                if im is None:
                    break
                base = cv2.resize(im, (OUT_W, int(round(im.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            img = base.copy()
            x0, y0, x1, y1 = (v * scale for v in box)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            written += 1
    return written


def emit(clip, cam, built, out_dir, frame_rel):
    items = []
    for c in built["cands"]:
        ids = [f"f{f:05d}_t{c['track']}_w{c['window_index']}.jpg" for f in c["frames"]]
        common = {"window_index": c["window_index"], "track": c["track"],
                  "window": c["window"], "frames": c["frames"],
                  "height_prior_m": HEIGHT_PRIOR_M, "fps": built["fps"],
                  "seq": f"hm3_{clip.replace('/', '_')}_{cam}",
                  "gt_source": "human-m3 3d world pose (median joint XY)",
                  "px_disp_render": round(c["px_disp"], 2),
                  "box_h_px_render": round(c["box_h_px"], 2),
                  "path_m": c["path_m"], "speed_mps": c["speed_mps"]}
        dur = f"{c['dur']:.1f}"
        items.append({"category": SPEED, "answer": f"{c['speed_mps']:.1f}",
                      "question": Q_SPEED.format(n=N_FRAMES, dur=dur, hp=HEIGHT_PRIOR_M),
                      "meta": dict(common), "image_ids": ids, "image_id": ids[0]})
        items.append({"category": PATH, "answer": f"{c['path_m']:.1f}",
                      "question": Q_PATH.format(n=N_FRAMES, dur=dur, hp=HEIGHT_PRIOR_M),
                      "meta": dict(common), "image_ids": ids, "image_id": ids[0]})
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, "qa_dyn_v1.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    paths = [c["path_m"] for c in built["cands"]]
    speeds = [c["speed_mps"] for c in built["cands"]]
    man = {"schema": "courtdyn-qa-hm3-v1", "clip": clip, "camera": cam,
           "source": "Human-M3 test split (basketball), 3D world pose + camera calibration",
           "params": {"win_s": WIN_S, "hop_s": HOP_S, "n_frames": N_FRAMES,
                      "render_width": OUT_W, "max_per_family": MAX_PER_FAMILY,
                      "min_box_h_px": MIN_BOX_H_PX, "source_size": built["size"],
                      "fps": built["fps"]},
           "ground_truth": {"space": "true 3D world metres (no height ruler, no homography)",
                            "position": "median XY over all joints of that person in that frame",
                            "path_m": "sum of per-frame position deltas over the window"},
           "counts": {"items": len(items), "windows": len(built["cands"]),
                      SPEED: len(built["cands"]), PATH: len(built["cands"])},
           "gt_spread": {"path_m": {"min": round(min(paths), 2), "median": round(st.median(paths), 2),
                                    "max": round(max(paths), 2)},
                         "speed_mps": {"min": round(min(speeds), 2),
                                       "median": round(st.median(speeds), 2),
                                       "max": round(max(speeds), 2)}},
           "frames_dir": frame_rel,
           "notes": ["同一 (clip, window, track) 在不同相机下是同一段世界运动, GT 逐字相同, 只有像素不同。",
                     "身高句保留只为与 CourtDyn 题面逐字一致; 本池的 GT 是真三维世界量, 不依赖那把尺子。"]}
    with io.open(os.path.join(out_dir, "qa_dyn_v1.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return len(items), man["gt_spread"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None, help="basketball1/split1/camera_0 形式; 省略则用 --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=MAX_PER_FAMILY)
    args = ap.parse_args()

    targets = []
    if args.clip:
        parts = args.clip.split("/")
        targets = [("/".join(parts[:-1]), parts[-1])]
    elif args.all:
        for c in CLIPS:
            cams = sorted(os.listdir(os.path.join(clip_dir(c), "images")))
            targets += [(c, cam) for cam in cams]
    else:
        raise SystemExit("给 --clip 或 --all")

    for clip, cam in targets:
        poses = load_poses(clip)
        if not poses:
            print(f"  ! {clip}: 没有 pose_calib, 跳过")
            continue
        imgs = image_list(clip, cam)
        if len(imgs) < 2:
            print(f"  ! {clip}/{cam}: 没有图像, 跳过")
            continue
        ts = [int(os.path.splitext(os.path.basename(x))[0]) for x in imgs]
        fps = 1e9 / float(np.median(np.diff(ts)))
        built = build_clip(clip, cam, poses, fps, args.limit)
        if not built:
            print(f"  ! {clip}/{cam}: 没有可用窗口, 跳过")
            continue
        tag = f"hm3_{clip.replace('/', '_')}_{cam}"
        out_dir = os.path.join(CD, f"seq_{tag}")
        frame_dir = os.path.join(FRAMES, f"frames_{tag}")
        n_img = render(clip, cam, built, frame_dir)
        n, spread = emit(clip, cam, built, out_dir,
                         os.path.relpath(frame_dir, ROOT).replace("\\", "/"))
        print(f"  {tag}: {n} 题 ({len(built['cands'])} 窗口), 新渲染 {n_img} 帧, "
              f"fps {fps:.1f}, path {spread['path_m']['min']}–{spread['path_m']['max']} m "
              f"(中位 {spread['path_m']['median']})")


if __name__ == "__main__":
    main()
