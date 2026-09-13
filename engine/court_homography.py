# -*- coding: utf-8 -*-
"""地平面单应版的动力学 GT (plan D2) —— 与 dynamics_qa 的身高尺版并列, 不替换它。

身高尺 (v1):  m/px = 1.93 / 框高中位数, 位移在**图像平面**量, 再乘 m/px。
单应   (v3):  足点 (框底中点) 先经 H 落到球场平面 (FIBA 28 x 15 m), 位移在**地面**量。
               尺度来自球场, 与任何球员的框高无关 —— 这正是审计 A8 要堵的泄漏通道。

两版共用同一套窗口 / 平滑 / 积分参数 (SMOOTH_HALF, STRIDE), 所以同一条 item 的
v1 与 v3 只差"尺子", 可以逐题对照。

H 文件: results/courtdyn/homography/H_<seq>.json (tools/calibrate_court_homography.py)。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from engine.dynamics_qa import (SMOOTH_HALF, STRIDE, ball_center, point_at)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H_DIR = os.path.join(ROOT, "results", "courtdyn", "homography")


class CourtPlane:
    def __init__(self, H, meta=None):
        self.H = np.asarray(H, dtype=np.float64)
        self.meta = meta or {}

    @classmethod
    def load(cls, seq_name, h_dir=H_DIR):
        p = os.path.join(h_dir, f"H_{seq_name}.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["H_img2court_m"], d)

    def to_court(self, xy):
        """图像像素 (x, y) -> 球场米制 (X, Y)。"""
        x, y = xy
        v = self.H @ np.array([x, y, 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    # ---- 与 dynamics_qa 同构的量 -------------------------------------------
    def foot_series_m(self, box_by_frame, frames):
        """足点序列 (米), 先在图像平面做 ±SMOOTH_HALF 滑动平均再投影 (与 v1 同序)。"""
        raw = {}
        for f, (x, y, w, h) in box_by_frame.items():
            raw[f] = (x + w / 2.0, y + h)
        out = []
        for f in frames:
            win = [raw[g] for g in range(f - SMOOTH_HALF, f + SMOOTH_HALF + 1) if g in raw]
            if not win:
                return None
            px = (sum(p[0] for p in win) / len(win), sum(p[1] for p in win) / len(win))
            out.append(self.to_court(px))
        return out

    def path_length_m(self, box_by_frame, f0, f1):
        frames = list(range(f0, f1 + 1, STRIDE))
        if frames[-1] != f1:
            frames.append(f1)
        pts = self.foot_series_m(box_by_frame, frames)
        if pts is None or len(pts) < 2:
            return None
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def mean_speed(self, box_by_frame, f0, f1, fps):
        d = self.path_length_m(box_by_frame, f0, f1)
        return None if d is None else d / ((f1 - f0) / fps)

    def time_to_ball(self, box_by_frame, ball_box_by_frame, f0, f1, fps):
        """time-to-intercept: 球心的地面投影 (假设球贴地; 球在空中时是近似) 到足点的
        地面距离 / 该窗口平均速度。返回 (t, dist_m, v)。"""
        v = self.mean_speed(box_by_frame, f0, f1, fps)
        pp, bp = point_at(box_by_frame, f1), ball_center(ball_box_by_frame, f1)
        if v is None or v <= 0 or pp is None or bp is None:
            return None
        dist = math.dist(self.to_court(pp), self.to_court(bp))
        return dist / v, dist, v


def recompute_answer(plane, item, tracks, fps):
    """按 item 的 category / meta 用单应重算 GT。返回 (answer_str, detail) 或 None。

    speed / path / timing: 数值; accel / faster / accel_vis: 字母。
    窗口与球员完全沿用 item.meta (frames / window / track...), 只换尺子。
    """
    cat = item["category"]
    m = item["meta"]
    f0, f1 = m["window"]
    if cat == "dynamics_speed_player":
        v = plane.mean_speed(tracks[m["track"]], f0, f1, fps)
        return None if v is None else (f"{v:.1f}", {"speed_mps": v})
    if cat == "dynamics_path_player":
        d = plane.path_length_m(tracks[m["track"]], f0, f1)
        return None if d is None else (f"{d:.1f}", {"path_m": d})
    if cat == "dynamics_time_intercept":
        ball = _ball_track(item, tracks)
        r = plane.time_to_ball(tracks[m["track"]], tracks[ball], f0, f1, fps)
        return None if r is None else (f"{r[0]:.1f}", {"time_s": r[0], "dist_m": r[1], "speed_mps": r[2]})
    if cat in ("relational_reasoning_dyn_accel", "relational_reasoning_dyn_accel_vis"):
        mid = (f0 + f1) // 2
        bb = tracks[m["track"]]
        v1, v2 = plane.mean_speed(bb, f0, mid, fps), plane.mean_speed(bb, mid, f1, fps)
        if v1 is None or v2 is None:
            return None
        up = v2 > v1
        q = item["question"]
        # 选项顺序按题面: "(A) speeding up" 在前 => A = 加速
        a_is_up = "(A) speeding up" in q
        ans = ("A" if up else "B") if a_is_up else ("B" if up else "A")
        return ans, {"v1": v1, "v2": v2, "ratio": max(v1, v2) / max(min(v1, v2), 1e-9)}
    if cat == "relational_reasoning_dyn_faster":
        va = plane.mean_speed(tracks[m["track_a"]], f0, f1, fps)
        vb = plane.mean_speed(tracks[m["track_b"]], f0, f1, fps)
        if va is None or vb is None:
            return None
        return ("A" if va > vb else "B"), {"v_a": va, "v_b": vb, "ratio": max(va, vb) / max(min(va, vb), 1e-9)}
    return None


def _ball_track(item, tracks):
    from engine.dynamics_qa import split_roles
    _, ball = split_roles(tracks)
    return ball
