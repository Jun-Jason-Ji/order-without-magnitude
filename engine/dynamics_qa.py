#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dynamics_qa.py — CourtDyn 定量动力学 QA 生成核心 (下一篇论文, 见
docs/benchmark_roadmap_2026-08-31.md 与 docs/courtdyn_paper_plan_2026-09-01.md)。

数据源: TeamTrack basketball_side / Q4_side_480-510 (MOT gt.txt, 900 帧 @29.97fps,
3840x2160, 10 名球员 + 1 个球, 逐帧连续轨迹) —— 程序化生成 GT, 不需人工标注。

尺度先验 (这是全篇的方法核心, 也是反事实实验的抓手)
--------------------------------------------------
这条序列是**未标定的单目侧视**, 没有单应矩阵。QuantiPhy 用篮筐直径 / 球场线宽
作尺度先验; 在只有 MOT 框的条件下, 等价且更稳的锚是**人体身高尺**:

    m_per_px(track, window) = PLAYER_HEIGHT_M / median(bbox_height_px)

即用被问球员自己的检测框高度当尺子。这带来两个必须写进论文的偏置:

  1. **深度压缩** —— 朝向/远离相机的位移在像面上被压缩。缓解: 只保留窗口内框高
     变化 < DEPTH_TOL 的样本 (深度近似恒定), 见 `depth_stable`。
  2. **身高个体差** —— 统一按 PLAYER_HEIGHT_M 折算, 个体误差 ±8% 直接进 GT。
     因此本基准的可信区间是"±10% 量级的定量能力", 阈值 T 按此放宽 (见 evaluate.py
     的 speed/path/timing 三组), 不做厘米级声明。

反事实 (counterfactual) 实验正是拿这个先验做的: 把提示词里的身高先验从 1.93 m
换成 1.45 m, 速度/路径长度 GT 按同比例缩放 —— 真正在用先验做度量的模型应当跟着
缩放, 只会背答案分布的模型不会。注意 **time-to-intercept 对尺度不变** (距离与
速度同比缩放, 商不变), 所以它的 GT 在反事实臂里保持原值, 是一条免费的对照。

QA 家族 (5 类)
--------------
  A speed        dynamics_speed_player   球员窗口平均速度 (m/s, 数值)
  B path         dynamics_path_player    球员窗口移动路程 (m, 数值)
  C timing       dynamics_time_intercept 按当前速度追到球当前位置需几秒 (s, 数值)
  D relational   relational_reasoning_dyn_accel  后半段比前半段快还是慢 (A/B)
                 —— 语言侧最小对 (选项互换孪生题), 沿用 T16 的 paired 协议。
                 **T18 证明这个协议有洞, 见 generate_accel_visual 的 docstring;
                 保留它只是为了继续当"有洞的对照", 主表请用 F 家族。**
  E relational   relational_reasoning_dyn_faster 两名高亮球员谁更快 (A/B)
                 —— **视觉侧最小对**: 同一对球员、同一问法、不同片段, GT 翻转。
                 这是 build_paired_relational.py 里明确留给 CourtDyn 的那一项。
  F relational   relational_reasoning_dyn_accel_vis  同 D 的问法, 但配成
                 **视觉侧最小对** (同一球员、两个不相交窗口、GT 翻转)。
                 D 的修复版, 由 generate_accel_visual() 单独生成。

本模块只做数学, 不碰视频; 抽帧与画高亮框在 tools/build_courtdyn_qa.py。
"""
from __future__ import annotations

import configparser
import math
import os
import statistics
from collections import defaultdict

# ---- 尺度与物理常数 ----------------------------------------------------
PLAYER_HEIGHT_M = 1.93       # 职业篮球运动员身高中位数, 作为默认尺度先验
CF_PLAYER_HEIGHT_M = 1.45    # 反事实臂里换上的错误先验
BALL_DIAMETER_M = 0.2426     # FIBA 7 号球; 仅用于记录, 球位置走最近球员的尺子

# ---- 采样与过滤参数 ----------------------------------------------------
SMOOTH_HALF = 5      # 足点滑动平均半宽 (帧); 压 bbox 抖动, 否则路程被噪声撑爆
STRIDE = 5           # 路程积分步长 (帧); 与平滑配合, 约 6 Hz 采样
DEPTH_TOL = 0.15     # 窗口内框高相对变化上限 —— 超过认为深度变化大, 丢弃
MIN_BOX_H = 60       # 像素; 低于此判为球而非球员

SPEED_RANGE = (0.4, 9.0)     # m/s, 合理球员速度区间; 越界样本丢弃
PATH_RANGE = (0.8, 30.0)     # m
TIME_RANGE = (0.4, 8.0)      # s
ACCEL_RATIO = 1.35           # D 类: 后/前半段速度比超过它才出题 (GT 无歧义)
FASTER_RATIO = 1.40          # E 类: 两人速度比超过它才出题


# =======================================================================
# MOT 读取
# =======================================================================
def load_seqinfo(seq_dir):
    """读 seqinfo.ini -> dict(fps, width, height, length, name)。"""
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"), encoding="utf-8")
    s = cp["Sequence"]
    return {
        "name": s.get("name"),
        "fps": float(s.get("framerate")),
        "width": int(s.get("imwidth")),
        "height": int(s.get("imheight")),
        "length": int(s.get("seqlength")),
    }


def load_tracks(seq_dir):
    """读 MOT gt.txt -> {track_id: {frame: (x, y, w, h)}} (frame 从 1 起)。"""
    path = os.path.join(seq_dir, "gt", "gt.txt")
    tracks = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame, tid = int(p[0]), int(p[1])
            tracks[tid][frame] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return dict(tracks)


def split_roles(tracks):
    """按框高中位数分出 (球员 id 列表, 球 id)。球在这条序列上是 ~27px 的小框。"""
    med = {t: statistics.median(b[3] for b in d.values()) for t, d in tracks.items()}
    players = sorted(t for t, h in med.items() if h >= MIN_BOX_H)
    balls = [t for t, h in med.items() if h < MIN_BOX_H]
    ball = min(balls, key=lambda t: med[t]) if balls else None
    return players, ball


# =======================================================================
# 几何 / 度量
# =======================================================================
def foot_series(box_by_frame, frames):
    """足点 (框底中点) 序列, 带 ±SMOOTH_HALF 滑动平均。缺帧返回 None。"""
    raw = {}
    for f in box_by_frame:
        x, y, w, h = box_by_frame[f]
        raw[f] = (x + w / 2.0, y + h)
    out = []
    for f in frames:
        win = [raw[g] for g in range(f - SMOOTH_HALF, f + SMOOTH_HALF + 1) if g in raw]
        if not win:
            return None
        out.append((sum(p[0] for p in win) / len(win),
                    sum(p[1] for p in win) / len(win)))
    return out


def median_box_h(box_by_frame, f0, f1):
    hs = [box_by_frame[f][3] for f in range(f0, f1 + 1) if f in box_by_frame]
    return statistics.median(hs) if hs else None


def depth_stable(box_by_frame, f0, f1):
    """窗口内框高相对变化是否在 DEPTH_TOL 内 (深度近似恒定)。"""
    h0 = median_box_h(box_by_frame, f0, min(f0 + SMOOTH_HALF * 2, f1))
    h1 = median_box_h(box_by_frame, max(f1 - SMOOTH_HALF * 2, f0), f1)
    if not h0 or not h1:
        return False
    return abs(h1 / h0 - 1.0) <= DEPTH_TOL


def ruler(box_by_frame, f0, f1, height_m=PLAYER_HEIGHT_M):
    """m/px —— 用该球员窗口内的框高中位数当尺子。"""
    h = median_box_h(box_by_frame, f0, f1)
    return (height_m / h) if h else None


def path_length_px(box_by_frame, f0, f1):
    """足点路程 (像素), 平滑后按 STRIDE 积分。"""
    frames = list(range(f0, f1 + 1, STRIDE))
    if frames[-1] != f1:
        frames.append(f1)
    pts = foot_series(box_by_frame, frames)
    if pts is None or len(pts) < 2:
        return None
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def mean_speed(box_by_frame, f0, f1, fps, height_m=PLAYER_HEIGHT_M):
    """窗口平均速率 (m/s)。返回 None 表示该窗口不可用。"""
    if not depth_stable(box_by_frame, f0, f1):
        return None
    px = path_length_px(box_by_frame, f0, f1)
    mpp = ruler(box_by_frame, f0, f1, height_m)
    if px is None or mpp is None:
        return None
    return px * mpp / ((f1 - f0) / fps)


def path_m(box_by_frame, f0, f1, height_m=PLAYER_HEIGHT_M):
    if not depth_stable(box_by_frame, f0, f1):
        return None
    px = path_length_px(box_by_frame, f0, f1)
    mpp = ruler(box_by_frame, f0, f1, height_m)
    return None if (px is None or mpp is None) else px * mpp


def point_at(box_by_frame, frame):
    if frame not in box_by_frame:
        return None
    x, y, w, h = box_by_frame[frame]
    return (x + w / 2.0, y + h)


def ball_center(ball_box_by_frame, frame):
    if frame not in ball_box_by_frame:
        return None
    x, y, w, h = ball_box_by_frame[frame]
    return (x + w / 2.0, y + h / 2.0)


# =======================================================================
# 题面
# =======================================================================
HL_A, HL_B = "red", "blue"          # 高亮框颜色 (画框在 build_courtdyn_qa.py)

SPEED_Q = ("The {n} frames are consecutive samples from one basketball clip "
           "spanning {dur:.1f} seconds, in chronological order. Assume a typical "
           "player on this court is {hp:.2f} m tall. What is the average speed of "
           "the player marked with the {c} box, in meters per second? "
           "Output only the number, one decimal place.")
PATH_Q = ("The {n} frames are consecutive samples from one basketball clip "
          "spanning {dur:.1f} seconds, in chronological order. Assume a typical "
          "player on this court is {hp:.2f} m tall. How far does the player marked "
          "with the {c} box travel along the floor over the whole clip, in meters? "
          "Output only the number, one decimal place.")
TIME_Q = ("The {n} frames are consecutive samples from one basketball clip "
          "spanning {dur:.1f} seconds, in chronological order. Assume a typical "
          "player on this court is {hp:.2f} m tall. If the player marked with the "
          "{c} box keeps moving at the speed they have in this clip, how many "
          "seconds until they reach the current position of the ball (marked with "
          "the yellow circle)? Output only the number, one decimal place.")
ACCEL_Q = ("The {n} frames are consecutive samples from one basketball clip "
           "spanning {dur:.1f} seconds, in chronological order. Comparing the "
           "second half of the clip with the first half, is the player marked with "
           "the {c} box: ({a}) speeding up ({b}) slowing down? "
           "Output only the single uppercase letter.")
FASTER_Q = ("The {n} frames are consecutive samples from one basketball clip "
            "spanning {dur:.1f} seconds, in chronological order. Which of the two "
            "is moving faster over this clip: (A) the player in the red box "
            "(B) the player in the blue box? Output only the single uppercase letter.")


def _fmt(v):
    return f"{v:.1f}"


# =======================================================================
# 生成
# =======================================================================
class Windows:
    """把序列切成等长窗口, 每个窗口取 n_frames 张等间隔帧。"""

    def __init__(self, length, fps, win_s, n_frames, hop_s):
        self.fps, self.n_frames = fps, n_frames
        self.win = int(round(win_s * fps))
        hop = int(round(hop_s * fps))
        # 首尾各留 SMOOTH_HALF 帧给平滑窗
        lo, hi = 1 + SMOOTH_HALF, length - SMOOTH_HALF
        self.items = []
        f0 = lo
        while f0 + self.win <= hi:
            f1 = f0 + self.win
            step = (f1 - f0) / (n_frames - 1)
            self.items.append({
                "f0": f0, "f1": f1,
                "frames": [int(round(f0 + i * step)) for i in range(n_frames)],
                "dur": (f1 - f0) / fps,
            })
            f0 += hop
        self.dur = self.win / fps


def generate(seq_dir, *, win_s=2.0, hop_s=1.0, n_frames=4,
             height_m=PLAYER_HEIGHT_M, max_per_family=0):
    """返回 (items, stats)。

    每个 item:
      window        {f0, f1, frames, dur}
      highlight     [{"track": tid, "color": "red"|"blue", "shape": "box"}, ...]
                    球用 {"track": ball, "color": "yellow", "shape": "circle"}
      category / question / answer / meta
    scale-变体 (反事实) 由 rescale_for_prior() 生成, 保持 window/highlight 不变。
    """
    info = load_seqinfo(seq_dir)
    tracks = load_tracks(seq_dir)
    players, ball = split_roles(tracks)
    W = Windows(info["length"], info["fps"], win_s, n_frames, hop_s)
    fps = info["fps"]

    items = []
    stats = defaultdict(int)
    stats["n_windows"] = len(W.items)
    stats["n_players"] = len(players)

    def base(w, cat, q, ans, hl, **meta):
        return {
            "window": w, "highlight": hl, "category": cat,
            "question": q, "answer": ans,
            "meta": dict(meta, height_prior_m=height_m,
                         seq=info["name"], fps=fps),
        }

    # ---- A/B/C/D: 单球员 -------------------------------------------------
    for wi, w in enumerate(W.items):
        f0, f1, mid = w["f0"], w["f1"], (w["f0"] + w["f1"]) // 2
        for tid in players:
            bb = tracks[tid]
            v = mean_speed(bb, f0, f1, fps, height_m)
            if v is None:
                stats["drop_unstable"] += 1
                continue
            hl_one = [{"track": tid, "color": HL_A, "shape": "box"}]
            qfmt = dict(n=n_frames, dur=w["dur"], hp=height_m, c=HL_A)

            # A speed
            if SPEED_RANGE[0] <= v <= SPEED_RANGE[1]:
                items.append(base(w, "dynamics_speed_player",
                                  SPEED_Q.format(**qfmt), _fmt(v), hl_one,
                                  window_index=wi, track=tid, speed_mps=v))
            else:
                stats["drop_speed_range"] += 1

            # B path
            d = path_m(bb, f0, f1, height_m)
            if d is not None and PATH_RANGE[0] <= d <= PATH_RANGE[1]:
                items.append(base(w, "dynamics_path_player",
                                  PATH_Q.format(**qfmt), _fmt(d), hl_one,
                                  window_index=wi, track=tid, path_m=d))

            # C time-to-intercept (球位置用该球员的尺子折算)
            if ball is not None:
                pp, bp = point_at(bb, f1), ball_center(tracks[ball], f1)
                mpp = ruler(bb, f0, f1, height_m)
                if pp and bp and mpp and v > 0:      # v == 0: 站着不动, 没有截球时间
                    dist_m = math.dist(pp, bp) * mpp
                    t = dist_m / v
                    if TIME_RANGE[0] <= t <= TIME_RANGE[1]:
                        hl_t = hl_one + [{"track": ball, "color": "yellow",
                                          "shape": "circle"}]
                        items.append(base(w, "dynamics_time_intercept",
                                          TIME_Q.format(**qfmt), _fmt(t), hl_t,
                                          window_index=wi, track=tid,
                                          dist_m=dist_m, speed_mps=v,
                                          scale_invariant=True))

            # D 加/减速 (语言侧最小对: 选项互换孪生题)
            v1 = mean_speed(bb, f0, mid, fps, height_m)
            v2 = mean_speed(bb, mid, f1, fps, height_m)
            if v1 and v2 and max(v1, v2) / min(v1, v2) >= ACCEL_RATIO:
                up = v2 > v1
                pid = f"acc_w{wi:03d}_t{tid}"
                items.append(base(w, "relational_reasoning_dyn_accel",
                                  ACCEL_Q.format(a="A", b="B", **qfmt),
                                  "A" if up else "B", hl_one,
                                  window_index=wi, track=tid, v1=v1, v2=v2,
                                  pair_id=pid, pair_role="orig"))
                items[-1]["pair_id"], items[-1]["pair_role"] = pid, "orig"
                tw = base(w, "relational_reasoning_dyn_accel",
                          ACCEL_Q.format(a="A", b="B", **qfmt)
                          .replace("(A) speeding up (B) slowing down",
                                   "(A) slowing down (B) speeding up"),
                          "B" if up else "A", hl_one,
                          window_index=wi, track=tid, v1=v1, v2=v2,
                          pair_id=pid, pair_role="twin")
                tw["pair_id"], tw["pair_role"] = pid, "twin"
                items.append(tw)

    # ---- E: 视觉侧最小对 (同一对球员, 不同片段, GT 翻转) -------------------
    # 先算每个 (窗口, 球员对) 的胜者, 再在同一对球员的窗口里配翻转的两条。
    cand = defaultdict(list)   # (ta, tb) -> [(wi, winner_is_a)]
    for wi, w in enumerate(W.items):
        f0, f1 = w["f0"], w["f1"]
        sp = {}
        for tid in players:
            v = mean_speed(tracks[tid], f0, f1, fps, height_m)
            if v is not None and SPEED_RANGE[0] <= v <= SPEED_RANGE[1]:
                sp[tid] = v
        ids = sorted(sp)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ta, tb = ids[i], ids[j]
                r = sp[ta] / sp[tb]
                if r >= FASTER_RATIO or r <= 1 / FASTER_RATIO:
                    cand[(ta, tb)].append((wi, r >= FASTER_RATIO))

    n_pairs = 0
    for (ta, tb), lst in sorted(cand.items()):
        pos = [wi for wi, a_fast in lst if a_fast]
        neg = [wi for wi, a_fast in lst if not a_fast]
        for k in range(min(len(pos), len(neg))):
            pid = f"vis_t{ta}_{tb}_{k:02d}"
            for wi, ans, role in ((pos[k], "A", "orig"), (neg[k], "B", "twin")):
                w = W.items[wi]
                it = base(w, "relational_reasoning_dyn_faster",
                          FASTER_Q.format(n=n_frames, dur=w["dur"]), ans,
                          [{"track": ta, "color": HL_A, "shape": "box"},
                           {"track": tb, "color": HL_B, "shape": "box"}],
                          window_index=wi, track_a=ta, track_b=tb,
                          pair_kind="visual")
                it["pair_id"], it["pair_role"] = pid, role
                items.append(it)
            n_pairs += 1
    stats["n_visual_pairs"] = n_pairs

    if max_per_family:
        items = _cap(items, max_per_family, stats)
    for c in {i["category"] for i in items}:
        stats[f"n_{c}"] = sum(1 for i in items if i["category"] == c)
    stats["n_items"] = len(items)
    return items, dict(stats)


def _cap(items, cap, stats):
    """按家族封顶。成对家族按 pair_id 整对保留, 绝不拆对。"""
    import random
    rng = random.Random(42)
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    out = []
    for cat, lst in sorted(by_cat.items()):
        if any("pair_id" in x for x in lst):
            pairs = defaultdict(list)
            for x in lst:
                pairs[x["pair_id"]].append(x)
            keys = sorted(pairs)
            rng.shuffle(keys)
            keep = keys[: max(1, cap // 2)]
            for k in sorted(keep):
                out.extend(sorted(pairs[k], key=lambda z: z["pair_role"] != "orig"))
            stats[f"cap_{cat}"] = f"{len(keys)}->{len(keep)} pairs"
        else:
            rng.shuffle(lst)
            out.extend(lst[:cap])
            stats[f"cap_{cat}"] = f"{len(lst)}->{min(len(lst), cap)}"
    return out


# =======================================================================
# F: 加/减速的视觉侧最小对 (D 家族的修复版)
# =======================================================================
def windows_disjoint(wa, wb):
    """两个窗口是否完全不重叠 (因此不共享任何一帧)。"""
    return wa["f1"] < wb["f0"] or wb["f1"] < wa["f0"]


def generate_accel_visual(seq_dir, *, win_s=2.0, hop_s=1.0, n_frames=4,
                          height_m=PLAYER_HEIGHT_M, max_pairs=70):
    """加/减速家族的**视觉侧**最小对 —— 独立于 generate(), 不影响 v1 产物。

    为什么需要它 (T18 的实测结论, 写进论文的 protocol 节)
    ----------------------------------------------------
    D 家族用的是**选项互换**孪生题: 两个成员画面逐字相同, 只把
    "(A) speeding up (B) slowing down" 换成 "(A) slowing down (B) speeding up",
    GT 随之翻转。这只挡得住**位置偏置**(总答同一个字母), 挡不住**语义偏置**
    (总答 "slowing down"): 语义恒定的模型在两个成员上都会答对, paired 拿满分。

    T18 灰板臂把这个洞照得很清楚 —— 模型看不到任何画面时:
        base  逐题 50.0 / paired  0.0   (位置偏置, 被挡住了)
        GRPO  逐题 47.1 / paired 47.1   (语义偏置, 直接穿过去)
    也就是说 D 家族的 paired 列**不是视觉证据**。

    本函数改成与 E (faster) 同构的视觉侧最小对:
      同一名球员 · 两个**不相交**的窗口 (不共享任何一帧) · 题面逐字相同 ·
      一个窗口在加速 (GT=A), 另一个在减速 (GT=B)。
    这样位置偏置和语义偏置都只能拿 0 分 paired, chance 回到 25%,
    而两个成员之间**唯一**的差别就是画面内容。

    配对策略: 同一球员的 (加速窗口, 减速窗口) 里取时间上最接近且不相交的一组,
    贪心配到不能配为止; 每个窗口在本家族内最多用一次 —— 既保证
    (image_id, question) 唯一 (题面全家族相同, 全靠 image_id 区分), 也让
    成对的两个片段在场上位置/光照上尽量接近, 差别集中在运动本身。

    返回 (items, stats)。items 结构与 generate() 一致, 可直接喂给
    tools/build_courtdyn_qa.py 的 render_overlays。
    """
    info = load_seqinfo(seq_dir)
    tracks = load_tracks(seq_dir)
    players, _ball = split_roles(tracks)
    W = Windows(info["length"], info["fps"], win_s, n_frames, hop_s)
    fps = info["fps"]

    stats = defaultdict(int)
    stats["n_windows"] = len(W.items)
    stats["n_players"] = len(players)

    # 1) 找出每名球员所有"加速/减速判定无歧义"的窗口
    events = defaultdict(list)          # tid -> [(wi, up, v1, v2)]
    for wi, w in enumerate(W.items):
        f0, f1 = w["f0"], w["f1"]
        mid = (f0 + f1) // 2
        for tid in players:
            bb = tracks[tid]
            v1 = mean_speed(bb, f0, mid, fps, height_m)
            v2 = mean_speed(bb, mid, f1, fps, height_m)
            if not v1 or not v2:
                stats["drop_unstable_half"] += 1
                continue
            if max(v1, v2) / min(v1, v2) < ACCEL_RATIO:
                stats["drop_ambiguous"] += 1
                continue
            events[tid].append((wi, v2 > v1, v1, v2))

    # 2) 每名球员内部贪心配对: 时间最近且窗口不相交, 每个窗口只用一次
    pairs = []                          # [(tid, up_event, down_event)]
    for tid in sorted(events):
        ups = [e for e in events[tid] if e[1]]
        downs = [e for e in events[tid] if not e[1]]
        cand = []
        for u in ups:
            for d in downs:
                if windows_disjoint(W.items[u[0]], W.items[d[0]]):
                    cand.append((abs(u[0] - d[0]), u[0], d[0], u, d))
        cand.sort(key=lambda c: (c[0], c[1], c[2]))
        used = set()
        for _gap, wu, wd, u, d in cand:
            if wu in used or wd in used:
                continue
            used.add(wu)
            used.add(wd)
            pairs.append((tid, u, d))
        stats["n_pairs_track_%d" % tid] = sum(1 for p in pairs if p[0] == tid)

    stats["n_pairs_available"] = len(pairs)
    if max_pairs and len(pairs) > max_pairs:
        import random
        # 专用 rng: 绝不与 generate() 的 _cap 共享随机流, 加家族不会扰动老家族
        rng = random.Random("courtdyn|accel_visual|42")
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        pairs = [pairs[i] for i in sorted(idx[:max_pairs])]
    stats["n_pairs_kept"] = len(pairs)

    # 3) 出题 —— 两个成员题面逐字相同, 只有 image_ids 不同
    q = ACCEL_Q.format(n=n_frames, dur=W.dur, c=HL_A, a="A", b="B")
    items = []
    per_track = defaultdict(int)
    for tid, u, d in pairs:
        k = per_track[tid]
        per_track[tid] += 1
        pid = f"accvis_t{tid}_{k:02d}"
        for (wi, _up, va, vb), ans, role in ((u, "A", "orig"), (d, "B", "twin")):
            w = W.items[wi]
            items.append({
                "window": w,
                "highlight": [{"track": tid, "color": HL_A, "shape": "box"}],
                "category": "relational_reasoning_dyn_accel_vis",
                "question": q,
                "answer": ans,
                "pair_id": pid,
                "pair_role": role,
                "meta": {
                    "height_prior_m": height_m, "seq": info["name"], "fps": fps,
                    "window_index": wi, "track": tid,
                    "v_first_half": va, "v_second_half": vb,
                    "v_ratio": (vb / va),
                    "v_window_mean": (va + vb) / 2.0,
                    "pair_kind": "visual",
                    "scale_invariant": True,
                    "supersedes": "relational_reasoning_dyn_accel "
                                  "(option-swap language pair; see T18 blank arm)",
                },
            })
    stats["n_relational_reasoning_dyn_accel_vis"] = len(items)
    stats["n_items"] = len(items)
    return items, dict(stats)


# =======================================================================
# 反事实尺度先验
# =======================================================================
SCALE_SENSITIVE = {"dynamics_speed_player", "dynamics_path_player"}


def rescale_for_prior(items, new_height_m=CF_PLAYER_HEIGHT_M):
    """把身高先验换成 new_height_m, 同步改题面与 GT。

    速度 / 路程 与先验成正比 -> GT 按 new/old 缩放。
    time-to-intercept 距离与速度同比缩放, 商不变 -> GT 保持 (尺度不变对照)。
    A/B 判别题与尺度无关 -> 题面里没有先验, 原样跳过。
    """
    out = []
    for it in items:
        old = it["meta"]["height_prior_m"]
        if f"{old:.2f} m tall" not in it["question"]:
            continue                       # 判别题不含先验, 反事实臂里不出现
        q = it["question"].replace(f"{old:.2f} m tall", f"{new_height_m:.2f} m tall")
        new = dict(it)
        new["question"] = q
        new["meta"] = dict(it["meta"], height_prior_m=new_height_m,
                           original_answer=it["answer"],
                           original_height_prior_m=old)
        if it["category"] in SCALE_SENSITIVE:
            new["answer"] = _fmt(float(it["answer"]) * new_height_m / old)
            new["meta"]["cf_effect"] = "rescaled"
        else:
            new["meta"]["cf_effect"] = "invariant"
        out.append(new)
    return out
