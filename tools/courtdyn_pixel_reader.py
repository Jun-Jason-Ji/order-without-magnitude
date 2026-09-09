#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""courtdyn_pixel_reader.py — 模型读的是"图像平面位移"还是"世界运动"? (paper 2, T27-A, 纯 CPU)

T26 的判决: 俯视片段上 SFT / GRPO 的 speed / path ρ 0.5–0.8, 首帧×4 归零 ⇒ 多帧运动确实被读到了;
而侧视片段上 ρ 在 −0.25 … 0.51 之间摆, 主片段 ≈ 0.16。最省事的统一解释: 模型读的是红框在
图像平面上的位移 (像素), 既不套身高尺也不做透视校正。俯视时图像平面 ≈ 地平面, 像素位移与
世界位移单调, ρ 高; 侧视时透视把两者解耦, ρ 随片段的相机几何摆动。

本脚本对每个 (片段 × 模型 × 帧条件) 的 speed / path 预测算:
  ρ(pred, px)      预测 vs 足点像素路程 (speed 用 px/s)
  ρ(pred, v3)      预测 vs 单应世界量 (GT v3)
  ρ(px, v3)        两个候选解释变量之间的相关 (俯视 ≈ 1, 分不开; 侧视 < 1, 分得开)
  ρ(pred, v3 | px) 控制像素位移后, 预测还剩多少"世界"信息  —— 读世界者 > 0
  ρ(pred, px | v3) 控制世界量后, 预测还剩多少"像素"信息    —— 读像素者 > 0
  ρ(pred, px | h)  控制框高后的像素相关 (排除"框大=近=快"的捷径)
三者的 bootstrap 95% CI (按题重采样, 10 000 次改 2 000 次, 够用)。
像素量与 GT v1 同源 (engine/dynamics_qa.path_length_px), 只是不乘身高尺。

产物: results/courtdyn/pixel_reader.md + pixel_reader.json。缺格显示 —, 不报错。
"""
import io
import json
import math
import os
import random
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import dynamics_qa as DQ                                   # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer      # noqa: E402
from tools.summarize_courtdyn_v2 import spearman, jload, fmt          # noqa: E402
from tools.rescore_courtdyn_homography import partial_spearman        # noqa: E402
from tools.summarize_courtdyn_t26 import seq_dir_of                   # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "pixel_reader.md")
OUT_JSON = os.path.join(CD, "pixel_reader.json")
MAIN = "Q4_side_480-510"
SEQS = [(MAIN, "侧视 (原片段)"), ("Q1_side_0-30", "侧视"), ("Q2_side_300-330", "侧视"), ("Q4_side_570-600", "侧视"),
        ("Q2_top_480-510", "俯视"), ("Q1_top_0-30", "俯视"), ("Q4_top_0-30", "俯视")]
MODELS = [("base", "base"), ("sft", "source-SFT"), ("grpo", "GRPO-v3"), ("qwen25vl7b", "Qwen2.5-VL-7B")]
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
N_BOOT = 2000


def qa_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def parsed_candidates(seq, mode, model):
    if seq == MAIN:
        if mode == "full":
            return ([f"eval_full_{model}_parsed"] if model != "qwen25vl7b" else ["t22_main_full_qwen25vl7b_parsed"])
        return [f"t20_main_{mode}_{model}_parsed", f"t27_{seq}_main_{mode}_{model}_parsed"]
    if mode == "full":
        return [f"t21_{seq}_main_{model}_parsed", f"t26_{seq}_main_full_{model}_parsed", f"t27_{seq}_main_full_{model}_parsed"]
    return [f"t26_{seq}_main_{mode}_{model}_parsed", f"t27_{seq}_main_{mode}_{model}_parsed"]


def find_parsed(seq, mode, model):
    for c in parsed_candidates(seq, mode, model):
        d = os.path.join(CD, c)
        if os.path.isfile(os.path.join(d, "predictions.json")):
            return d
    return None


def key(r):
    return (r["image_id"], r["question"])


def boot3(fn, a, b, c, seed=42, n_boot=N_BOOT):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([a[i] for i in idx], [b[i] for i in idx], [c[i] for i in idx])
        if v == v:
            vals.append(v)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def boot2(a, b, seed=42, n_boot=N_BOOT):
    return boot3(lambda x, y, _z: spearman(x, y), a, b, a, seed, n_boot)


def item_features(seq, tracks, fps):
    """key -> dict(px_path, px_speed, h, v3)。像素量与 v1 同源, 不乘尺子。"""
    v1 = jload(os.path.join(qa_dir(seq), "qa_dyn_v1.json")) or []
    try:
        plane = CourtPlane.load(seq)
    except FileNotFoundError:
        plane = None
    F = {}
    for it in v1:
        fam = dict(FAMS).get(it["category"])
        if not fam:
            continue
        m = it["meta"]
        f0, f1 = m["window"]
        bb = tracks[m["track"]]
        px = DQ.path_length_px(bb, f0, f1)
        if px is None:
            continue
        h = DQ.median_box_h(bb, f0, f1)
        v3 = None
        if plane:
            r = recompute_answer(plane, it, tracks, fps)
            v3 = float(r[0]) if r else None
        F[key(it)] = {"fam": fam, "px": px if fam == "path" else px / ((f1 - f0) / fps), "h": h,
                      "v1": float(it["answer"]), "v3": v3}
    return F


def cell_stats(P, F, fam):
    xs, px, v3, hs = [], [], [], []
    for r in P:
        f = F.get(key(r))
        if not f or f["fam"] != fam or f["v3"] is None:
            continue
        try:
            x = float(r["vlm_answer"])
        except (TypeError, ValueError):
            continue
        xs.append(x); px.append(f["px"]); v3.append(f["v3"]); hs.append(f["h"])
    if len(xs) < 8:
        return {"n": len(xs)}
    const = len(set(xs)) < 2
    e = {"n": len(xs), "n_distinct": len(set(xs)), "rho_pred_px": None if const else spearman(xs, px),
         "rho_pred_v3": None if const else spearman(xs, v3)}
    if not const:
        e["ci_pred_px"] = boot2(xs, px)
        e["ci_pred_v3"] = boot2(xs, v3)
        e["partial_v3_given_px"] = partial_spearman(xs, v3, px)
        e["ci_partial_v3_given_px"] = boot3(partial_spearman, xs, v3, px)
        e["partial_px_given_v3"] = partial_spearman(xs, px, v3)
        e["ci_partial_px_given_v3"] = boot3(partial_spearman, xs, px, v3)
        e["partial_px_given_h"] = partial_spearman(xs, px, hs)
    return e


def log(msg):
    print(f"[pixel-reader {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    R = {"schema": "courtdyn-pixel-reader-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seqs": {}}
    md = [f"# CourtDyn · 像素位移 vs 世界运动: 模型在读哪一个? (T27-A)\n\n生成 {R['generated']} · `tools/courtdyn_pixel_reader.py` · "
          "px = 足点像素路程 (speed 为 px/s), 与 GT v1 同源但不乘身高尺; v3 = 单应世界量; h = 框高中位数。"
          "偏相关在秩上做; [] 为 bootstrap 95% CI (按题重采样 2 000 次)。"
          "读世界者: ρ(pred, v3 | px) > 0; 读像素者: ρ(pred, px | v3) > 0。俯视上 ρ(px, v3) ≈ 1, 两者分不开, 判决靠侧视。\n"]
    md.append("## A. 只看 GT: 像素量与世界量之间的相关 (侧视越低, 判决力越强)\n")
    md.append("| 片段 | 视角 | 家族 | n | ρ(px, v3) | ρ(px, v1) | ρ(px, h) | ρ(v3, h) |")
    md.append("|---|---|---|---|---|---|---|---|")
    feats = {}
    for seq, view in SEQS:
        sd = seq_dir_of(seq)
        if not os.path.isfile(os.path.join(sd, "gt", "gt.txt")) or not os.path.isfile(os.path.join(qa_dir(seq), "qa_dyn_v1.json")):
            log(f"{seq}: 缺 gt 或池, 跳过")
            continue
        info = DQ.load_seqinfo(sd)
        tracks = DQ.load_tracks(sd)
        F = item_features(seq, tracks, info["fps"])
        feats[seq] = F
        R["seqs"][seq] = {"view": view, "gt": {}, "cells": {}}
        for _, fam in FAMS:
            rows = [f for f in F.values() if f["fam"] == fam and f["v3"] is not None]
            if len(rows) < 8:
                md.append(f"| {seq} | {view} | {fam} | {len(rows)} | — | — | — | — |")
                continue
            px = [f["px"] for f in rows]; v3 = [f["v3"] for f in rows]; v1 = [f["v1"] for f in rows]; h = [f["h"] for f in rows]
            g = {"n": len(rows), "rho_px_v3": spearman(px, v3), "rho_px_v1": spearman(px, v1),
                 "rho_px_h": spearman(px, h), "rho_v3_h": spearman(v3, h)}
            R["seqs"][seq]["gt"][fam] = g
            md.append(f"| {seq} | {view} | {fam} | {g['n']} | {fmt(g['rho_px_v3'], 2)} | {fmt(g['rho_px_v1'], 2)} | "
                      f"{fmt(g['rho_px_h'], 2)} | {fmt(g['rho_v3_h'], 2)} |")
    md.append("\n## B. 模型格 · full 帧 (+ 已有的首帧×4 作对照)\n")
    md.append("| 片段 | 视角 | 模型 | 帧 | 家族 | n (distinct) | ρ(pred, px) [CI] | ρ(pred, v3) [CI] | ρ(pred, v3 \\| px) [CI] | ρ(pred, px \\| v3) [CI] | ρ(pred, px \\| h) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for seq, view in SEQS:
        F = feats.get(seq)
        if not F:
            continue
        for m, mlabel in MODELS:
            for mode in ("full", "static4"):
                d = find_parsed(seq, mode, m)
                if not d:
                    continue
                p = jload(os.path.join(d, "predictions.json"))
                P = p if isinstance(p, list) else (p or {}).get("predictions") or []
                for _, fam in FAMS:
                    e = cell_stats(P, F, fam)
                    e["dir"] = os.path.basename(d)
                    R["seqs"][seq]["cells"][f"{m}@{mode}@{fam}"] = e
                    if e.get("rho_pred_px") is None:
                        md.append(f"| {seq} | {view} | {mlabel} | {mode} | {fam} | {e['n']} ({e.get('n_distinct', '—')}) | — | — | — | — | — |")
                        continue

                    def ci(k):
                        c = e.get(k) or (None, None)
                        return f"[{fmt(c[0], 2)},{fmt(c[1], 2)}]"
                    md.append(f"| {seq} | {view} | {mlabel} | {mode} | {fam} | {e['n']} ({e['n_distinct']}) | "
                              f"{fmt(e['rho_pred_px'], 2)} {ci('ci_pred_px')} | {fmt(e['rho_pred_v3'], 2)} {ci('ci_pred_v3')} | "
                              f"{fmt(e['partial_v3_given_px'], 2)} {ci('ci_partial_v3_given_px')} | "
                              f"{fmt(e['partial_px_given_v3'], 2)} {ci('ci_partial_px_given_v3')} | {fmt(e['partial_px_given_h'], 2)} |")
    # C. 侧视判决汇总: 对每个模型, 把 4 条侧视片段的两个偏相关列出来
    md.append("\n## C. 侧视判决 (full 帧): 每模型 × 家族, 4 条侧视片段的偏相关\n")
    md.append("| 模型 | 家族 | ρ(pred, v3 \\| px) 逐片段 | 中位 | ρ(pred, px \\| v3) 逐片段 | 中位 |")
    md.append("|---|---|---|---|---|---|")
    side = [s for s, v in SEQS if v.startswith("侧视")]
    for m, mlabel in MODELS:
        for _, fam in FAMS:
            a, b = [], []
            for s in side:
                e = R["seqs"].get(s, {}).get("cells", {}).get(f"{m}@full@{fam}") or {}
                if e.get("partial_v3_given_px") is not None:
                    a.append(e["partial_v3_given_px"]); b.append(e["partial_px_given_v3"])
            if not a:
                md.append(f"| {mlabel} | {fam} | — | — | — | — |")
                continue
            md.append(f"| {mlabel} | {fam} | {' / '.join(fmt(x, 2) for x in a)} | {fmt(st.median(a), 2)} | "
                      f"{' / '.join(fmt(x, 2) for x in b)} | {fmt(st.median(b), 2)} |")
    md.append("\n读法: 若 ρ(pred, px | v3) 的中位明显 > 0 而 ρ(pred, v3 | px) ≈ 0, 模型读的是图像平面位移, "
              "不做透视校正也不套尺子 —— 这与俯视高 ρ、侧视随片段摆动、T-MRA 从不超过常数中位数三件事同时相容。"
              "反之则模型确实在读世界量 (需要另找 T-MRA 低的原因)。首帧×4 行两列都应 ≈ 0 (无运动信息)。\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
