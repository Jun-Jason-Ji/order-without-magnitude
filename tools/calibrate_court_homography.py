#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""球场线 → 地平面单应 (plan D2)。

CourtDyn v1 的尺度先验是"身高尺": m/px = 1.93 m / 框高中位数。它把透视前缩直接
带进 GT (审计 A8: ρ(GT, 框高) = +0.38), 也是模型可以"读框大小当速度"的泄漏通道。
本工具从**球场线**拟合图像 → 球场平面的单应 H, 让足点先落到米制地面再算位移,
尺度来自 28 × 15 m 的 FIBA 场地而不是任何一个人的身高。

流程 (全自动, 每条序列独立):
  1. 从 img1.mp4 等距抽 N 帧 (原生 4K)。
  2. 青色场地掩膜 → 含图像中心的连通块 → 凸包 → 两条最接近水平的边 = 边线;
     另两条边 = 端线。球员遮挡只会让连通块缩小, 所以跨帧取最靠外的端线当种子。
  3. 用白线像素在种子线 ±BAND px 内重新 RANSAC (亚像素级), 得四角点。
  4. 各帧角点取中位数 (球员遮挡/相机漂移 ≤ 10 px 都被压掉), 解 H。
  5. 验证: 把红色钥匙区映射到球场坐标, 与 FIBA 5.8 × 4.9 m 比 (粗, ±0.2 m);
     输出俯视校正图 (叠 FIBA 线模型) 供肉眼复核 —— 这是决定性的检查。

写出: <out_dir>/H_<seq>.json  (H 以米为单位: [x,y,1]_img → [X_m, Y_m, w])
      <out_dir>/H_<seq>_rectified.png, <out_dir>/H_<seq>_corners.png

球场坐标: X 沿长边, 左端线 X=0, 右端线 X=28; Y 沿短边, 远边线 Y=0, 近边线 Y=15。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COURT_L, COURT_W = 28.0, 15.0          # FIBA, m
KEY_DEPTH, KEY_WIDTH = 5.8, 4.9        # FIBA 2010+ 矩形钥匙区
BAND = 35                              # 白线搜索带宽 (px, 4K)
TOL_WHITE = 1.5                        # 白线 RANSAC 内点阈值 (px)
TOL_HULL = 2.0


def log(msg):
    print(f"[calib {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- masks
def teal_mask(a):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (g > 140) & (b > 120) & (r < 110) & (g > r + 60)


def red_mask(a):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (r > 170) & (g < 110) & (b > 90) & (b < 190)


def white_mask(a):
    mx = a.max(-1).astype(int)
    mn = a.min(-1).astype(int)
    return (mn > 150) & ((mx - mn) < 60)


# ---------------------------------------------------------------- lines
def fit_line(pts):
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return np.array([n[0], n[1], -n @ m])


def ransac_line(pts, tol, iters=600, min_span=100, seed=0):
    rng = np.random.default_rng(seed)
    best_in = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        if np.hypot(*d) < min_span:
            continue
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs((pts - pts[i]) @ n) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None or best_in.sum() < 2:
        return None, 0
    return fit_line(pts[best_in]), int(best_in.sum())


def line_through(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    d = q - p
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return np.array([n[0], n[1], -n @ p])


def dist_to(L, pts):
    return pts @ np.array([L[0], L[1]]) + L[2]


def intersect(L1, L2):
    x = np.cross(L1, L2)
    return x[:2] / x[2]


def hull_points(bpts, step=2.0):
    from scipy.spatial import ConvexHull
    h = ConvexHull(bpts)
    v = bpts[h.vertices]
    out = []
    for i in range(len(v)):
        p, q = v[i], v[(i + 1) % len(v)]
        n = max(int(np.hypot(*(q - p)) / step), 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(p + t * (q - p))
    return np.concatenate(out)


# ---------------------------------------------------------------- seeds
def hull_lines(a, seed_xy):
    """单帧粗线: 含种子点的青色连通块 → 凸包 → 4 条 RANSAC 直线。
    两条最接近水平的 = 边线 (far / near), 另两条 = 端线 (left / right)。
    球员遮挡只会让连通块**缩小** (青色被人体切掉), 不会变大 —— 所以跨帧取
    "最完整" 的凸包 (见 calibrate) 就能拿到没被遮的端线。"""
    m = teal_mask(a)
    lab, _ = ndimage.label(m)
    l = lab[seed_xy[1], seed_xy[0]]
    if l == 0:
        return None
    comp = ndimage.binary_fill_holes(ndimage.binary_closing(lab == l, iterations=3))
    er = ndimage.binary_erosion(comp)
    ys, xs = np.nonzero(comp & ~er)
    b = np.stack([xs, ys], 1).astype(float)
    if len(b) < 500:
        return None
    hp = hull_points(b)
    lines = []
    pts = hp
    for _ in range(4):
        if len(pts) < 20:
            break
        L, n = ransac_line(pts, TOL_HULL, seed=len(lines))
        if L is None:
            break
        lines.append(L)
        pts = pts[np.abs(dist_to(L, pts)) >= TOL_HULL]
    if len(lines) < 4:
        return None
    lines.sort(key=lambda L: abs(L[0]) / (abs(L[1]) + 1e-9))
    far, near = sorted(lines[:2], key=lambda L: -(L[0] * seed_xy[0] + L[2]) / L[1])
    y_mid = ((-(far[0] * seed_xy[0] + far[2]) / far[1]) + (-(near[0] * seed_xy[0] + near[2]) / near[1])) / 2
    left, right = sorted(lines[2:], key=lambda L: -(L[1] * y_mid + L[2]) / L[0])
    return {"far": far, "near": near, "left": left, "right": right}


def refine_with_white(a, seeds):
    w = white_mask(a)
    ys, xs = np.nonzero(w)
    P = np.stack([xs, ys], 1).astype(float)
    if len(P) > 300000:
        P = P[np.random.default_rng(2).choice(len(P), 300000, replace=False)]
    c = {k: intersect(seeds[a_], seeds[b_]) for k, (a_, b_) in
         {"far_left": ("far", "left"), "far_right": ("far", "right"),
          "near_left": ("near", "left"), "near_right": ("near", "right")}.items()}
    ext = {"far": ("far_left", "far_right"), "near": ("near_left", "near_right"),
           "left": ("far_left", "near_left"), "right": ("far_right", "near_right")}
    out, quality = {}, {}
    for name, (k1, k2) in ext.items():
        L0 = seeds[name]
        p, q = c[k1], c[k2]
        lo = np.minimum(p, q) - 40
        hi = np.maximum(p, q) + 40
        sel = (np.abs(dist_to(L0, P)) < BAND) & np.all((P > lo) & (P < hi), axis=1)
        pts = P[sel]
        if len(pts) < 80:
            return None, {name: (0, int(len(pts)))}
        L, n = ransac_line(pts, TOL_WHITE, seed=7)
        if L is None or n < 60:
            return None, {name: (n, int(len(pts)))}
        out[name] = L
        quality[name] = (n, int(len(pts)))
    corners = {"far_left": intersect(out["far"], out["left"]),
               "far_right": intersect(out["far"], out["right"]),
               "near_right": intersect(out["near"], out["right"]),
               "near_left": intersect(out["near"], out["left"])}
    return corners, quality


# ---------------------------------------------------------------- homography
KEYS = ["far_left", "far_right", "near_right", "near_left"]
COURT_CORNERS = np.float32([[0, 0], [COURT_L, 0], [COURT_L, COURT_W], [0, COURT_W]])


def solve_H(corners):
    src = np.float32([corners[k] for k in KEYS])
    return cv2.getPerspectiveTransform(src, COURT_CORNERS)


def img_to_court(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def validate(frames_rgb, H):
    """钥匙区 / 中线 检查, 全在球场坐标里; 红色 / 白色像素跨所有抽样帧合并 (遮挡互补)。"""
    rep = {}
    reds, whites = [], []
    for a in frames_rgb:
        ys, xs = np.nonzero(red_mask(a)); reds.append(np.stack([xs, ys], 1))
        ys, xs = np.nonzero(white_mask(a)); whites.append(np.stack([xs, ys], 1))
    cc = img_to_court(H, np.concatenate(reds))
    inside = (cc[:, 0] > 0.2) & (cc[:, 0] < COURT_L - 0.2) & (cc[:, 1] > 0.2) & (cc[:, 1] < COURT_W - 0.2)
    cc = cc[inside]
    for side, sel in (("left", cc[:, 0] < COURT_L / 2), ("right", cc[:, 0] >= COURT_L / 2)):
        q = cc[sel]
        if len(q) < 3000:
            rep[f"key_{side}"] = None
            continue
        # 用直方图找红色密度台阶 (比分位数稳): 沿 X 和 Y 各投影
        def edge(v, lo, hi, from_low):
            h, e = np.histogram(v, bins=int((hi - lo) / 0.05), range=(lo, hi))
            thr = h.max() * 0.5
            idx = np.nonzero(h > thr)[0]
            return float(e[idx[0]] if from_low else e[idx[-1] + 1])
        if side == "left":
            x_in = edge(q[:, 0], 0, 10, from_low=False)
            depth = x_in
        else:
            x_in = edge(q[:, 0], COURT_L - 10, COURT_L, from_low=True)
            depth = COURT_L - x_in
        y0 = edge(q[:, 1], 2, COURT_W - 2, from_low=True)
        y1 = edge(q[:, 1], 2, COURT_W - 2, from_low=False)
        rep[f"key_{side}"] = {"depth_m": round(depth, 3), "width_m": round(y1 - y0, 3),
                              "y_span": [round(y0, 3), round(y1, 3)],
                              "fiba": [KEY_DEPTH, KEY_WIDTH]}
    rep["note"] = ("key extents from red-pixel density steps (~±0.2 m); the decisive check is visual: "
                   "H_<seq>_rectified.png overlays the FIBA model on the rectified frame")
    return rep


def draw_rectified(a, H, path, S=50):
    Hs = np.diag([S, S, 1.0]) @ H
    top = cv2.warpPerspective(a[..., ::-1].copy(), Hs, (int(COURT_L * S), int(COURT_W * S)))

    def P(x, y):
        return (int(round(x * S)), int(round(y * S)))
    col = (255, 255, 255)
    cv2.line(top, P(COURT_L / 2, 0), P(COURT_L / 2, COURT_W), col, 2)
    cv2.circle(top, P(COURT_L / 2, COURT_W / 2), int(1.8 * S), col, 2)
    for x0, sgn in ((0, 1), (COURT_L, -1)):
        cv2.rectangle(top, P(x0, COURT_W / 2 - KEY_WIDTH / 2), P(x0 + sgn * KEY_DEPTH, COURT_W / 2 + KEY_WIDTH / 2), col, 2)
        cv2.circle(top, P(x0 + sgn * 1.575, COURT_W / 2), int(6.75 * S), col, 1)
    cv2.imwrite(path, top)


def draw_corners(a, corners, path):
    im = a[..., ::-1].copy()
    pts = [tuple(int(round(v)) for v in corners[k]) for k in KEYS]
    for i in range(4):
        cv2.line(im, pts[i], pts[(i + 1) % 4], (0, 255, 255), 4)
    cv2.imwrite(path, cv2.resize(im, (1920, 1080)))


# ---------------------------------------------------------------- main
def sample_frames(video, n):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(5, total - 6, n).round().astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append((int(i) + 1, fr[..., ::-1]))     # MOT frame index 从 1 起; RGB
    cap.release()
    return out


def calibrate(seq_dir, out_dir, n_frames=9, seed_xy=None):
    from engine.dynamics_qa import load_seqinfo
    info = load_seqinfo(seq_dir)
    name = info["name"]
    frames = sample_frames(os.path.join(seq_dir, "img1.mp4"), n_frames)
    if not frames:
        raise SystemExit("抽不到帧")
    h, w = frames[0][1].shape[:2]
    seed_xy = seed_xy or (w // 2, h // 2)
    # ---- 阶段 1: 每帧凸包粗线; 取"最完整"的凸包当种子 ----
    coarse = []
    for fi, a in frames:
        hl = hull_lines(a, seed_xy)
        if hl is None:
            log(f"{name} f{fi}: 凸包失败, 跳过")
            continue
        c = {"far_left": intersect(hl["far"], hl["left"]), "far_right": intersect(hl["far"], hl["right"]),
             "near_right": intersect(hl["near"], hl["right"]), "near_left": intersect(hl["near"], hl["left"])}
        # 合理性: 角点落在画面附近, 端线与边线夹角 > 20°
        ok = all(-0.2 * w < c[k][0] < 1.2 * w and -0.2 * h < c[k][1] < 1.2 * h for k in KEYS)
        sd = np.array([hl["far"][1], -hl["far"][0]])
        for k in ("left", "right"):
            ld = np.array([hl[k][1], -hl[k][0]])
            ang = np.degrees(np.arccos(min(1.0, abs(ld @ sd) / (np.linalg.norm(ld) * np.linalg.norm(sd) + 1e-9))))
            ok = ok and ang > 20
        if not ok:
            log(f"{name} f{fi}: 凸包角点不合理, 跳过")
            continue
        coarse.append((fi, hl, c))
    if not coarse:
        raise SystemExit(f"{name}: 没有一帧能找到球场")
    # 遮挡只会把端线往里推: 左端线取最靠左的帧, 右端线取最靠右的帧; 边线取中位
    left_best = min(coarse, key=lambda t: t[2]["far_left"][0] + t[2]["near_left"][0])
    right_best = max(coarse, key=lambda t: t[2]["far_right"][0] + t[2]["near_right"][0])
    far_med = np.median([t[1]["far"] for t in coarse], 0)
    near_med = np.median([t[1]["near"] for t in coarse], 0)
    seeds = {"far": far_med, "near": near_med, "left": left_best[1]["left"], "right": right_best[1]["right"]}
    log(f"{name}: 种子来自 左=f{left_best[0]} 右=f{right_best[0]}; 粗角点 "
        + " ".join(f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in
                   {"far_left": intersect(seeds["far"], seeds["left"]), "far_right": intersect(seeds["far"], seeds["right"]),
                    "near_right": intersect(seeds["near"], seeds["right"]), "near_left": intersect(seeds["near"], seeds["left"])}.items()))
    # ---- 阶段 2: 每帧白线细化, 跨帧中位 ----
    per = []
    for fi, a in frames:
        c, q = refine_with_white(a, seeds)
        if c is None:
            log(f"{name} f{fi}: 白线细化失败 {q}, 跳过")
            continue
        per.append({"frame": fi, "corners": {k: [float(v) for v in c[k]] for k in KEYS}, "quality": q})
        log(f"{name} f{fi}: " + " ".join(f"{k}=({c[k][0]:.0f},{c[k][1]:.0f})" for k in KEYS))
    if not per:
        raise SystemExit(f"{name}: 白线细化全部失败")
    arr = np.array([[p["corners"][k] for k in KEYS] for p in per])
    med = np.median(arr, 0)
    dev = np.abs(arr - med).max(axis=(1, 2))
    corners = {k: med[i].tolist() for i, k in enumerate(KEYS)}
    H = solve_H(corners)
    a0 = frames[0][1]
    rep = validate([fr for _, fr in frames], H)
    os.makedirs(out_dir, exist_ok=True)
    draw_rectified(a0, H, os.path.join(out_dir, f"H_{name}_rectified.png"))
    draw_corners(a0, corners, os.path.join(out_dir, f"H_{name}_corners.png"))
    result = {
        "schema": "courtdyn-homography-v1",
        "sequence": name, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "court_m": [COURT_L, COURT_W], "frame_size": [w, h],
        "corners_px": corners,
        "H_img2court_m": H.tolist(),
        "per_frame": per,
        "per_frame_max_dev_px": [round(float(d), 1) for d in dev],
        "validation": rep,
        "assumptions": [
            "court is FIBA 28 x 15 m (scale comes from this, not from any player)",
            "players' feet are on the court plane (bbox bottom-centre = foot point)",
            "camera drift over the clip is absorbed by the temporal-median corners (<= max_dev_px)",
        ],
    }
    with open(os.path.join(out_dir, f"H_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log(f"{name}: 角点中位 {json.dumps({k: [round(v) for v in corners[k]] for k in KEYS})}; "
        f"逐帧最大偏差 {result['per_frame_max_dev_px']}")
    log(f"{name}: 验证 {json.dumps(rep)}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, nargs="+")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "courtdyn", "homography"))
    ap.add_argument("--n_frames", type=int, default=9)
    args = ap.parse_args()
    for sd in args.seq_dir:
        calibrate(sd, args.out_dir, args.n_frames)


if __name__ == "__main__":
    main()
