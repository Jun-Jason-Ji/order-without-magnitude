#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_t28.py — T28 汇总: N2 (ruler / pxunit 两臂) + N3 (target-SFT 臂)。

三节:
A. **尺子臂**  full vs ruler, 全部在 **v3 (单应) 世界**里计分 —— ruler 递过去的尺子是单应导出的,
   所以 full 也必须按 v3 重算, 不能拿主表的 v1 口径 tmra_official 直接比。
   报 ρ、T-MRA(v3)、预测中位数, 以及 ruler − full 的差; 参照行 = v3 常数中位数基线。
   不变 ⇒ 递到手里的尺子也不用; 改善 ⇒ 瓶颈在尺子的提取而不是换算。
B. **像素单位臂**  pxunit 按渲染像素计分, 等价容差 = 米制阈值 × K (floor 同步 × K),
   因此与 A 节的 T-MRA 可比。关键列是 pred(pxunit)/pred(full_m) 的中位比:
   若模型真的换到像素单位作答, 该比应当 ≈ K; 若它只是重复原来的数, 应当 ≈ 1。
C. **target-SFT 臂**  main full / static-4 的 ρ 与 Δρ, 以及 zoom2 预测比 (与 T27 同口径)。

缺格显示 —。产物 results/courtdyn/courtdyn_t28_table.md + courtdyn_t28_findings.json。
"""
import io
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import eval.evaluate as E                                                  # noqa: E402
from engine import dynamics_qa as DQ                                       # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer           # noqa: E402
from tools.summarize_courtdyn_v2 import spearman, jload, fmt               # noqa: E402
from tools import courtdyn_pixel_reader as PR                              # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "courtdyn_t28_table.md")
OUT_JSON = os.path.join(CD, "courtdyn_t28_findings.json")
TOPS = ["Q1_top_0-30", "Q2_top_480-510"]
MODELS = [("sft", "source-SFT"), ("grpo", "GRPO-v3"), ("tsft", "target-SFT-token"),
          ("base", "base 4B"), ("qwen25vl7b", "Qwen2.5-VL-7B"),
          ("cdnative", "CourtDyn-native SFT")]
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
RENDER_W = 960


def ikey(r):
    m = r.get("meta") or {}
    return (r.get("category"), m.get("window_index"), m.get("track"))


def preds(d):
    """parsed 目录 -> {item key: float 预测}。"""
    if not d:
        return {}
    p = jload(os.path.join(d, "predictions.json"))
    P = p if isinstance(p, list) else (p or {}).get("predictions") or []
    out = {}
    for r in P:
        try:
            out[ikey(r)] = float(r["vlm_answer"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def t28_parsed(seq, pool, mode, model):
    """T28 建的格子在 t28_ 前缀下; T32 (公开 7B 的 ruler / pxunit 臂) 在 t32_ 下 —— 两个都找。

    保留各自前缀而不是让 T32 直接写进 t28_, 是为了留住出处: 目录名能看出这一格是哪条队列跑的。
    """
    for pref in ("t28", "t32", "t33"):
        d = os.path.join(CD, f"{pref}_{seq}_{pool}_{mode}_{model}_parsed")
        if os.path.isfile(os.path.join(d, "predictions.json")):
            return d
    return None


def full_parsed(seq, model):
    """full 臂: tsft 只有 T28 才跑, 其余沿用 T21/T26。"""
    return t28_parsed(seq, "main", "full", model) or PR.find_parsed(seq, "full", model)


def tmra(pairs, grp, k=1.0):
    """pairs = [(pred, gt)]; k 为单位换算系数 (px 臂 = K, 米臂 = 1)。"""
    T, floor = E.SCALAR_TMRA_SPEC[grp]
    if not pairs:
        return None
    return 100 * st.mean(E.t_mra_scalar(f"{p:.1f}", f"{g:.1f}", T * k, floor * k) for p, g in pairs)


def seq_ctx(seq):
    """该片段的 GT 上下文: v3 答案、渲染像素答案、K、常数中位数基线。"""
    sd = os.path.join(CD, f"seq_{seq}")
    items = jload(os.path.join(sd, "qa_dyn_v1.json")) or []
    tt = None
    for fam, split, name in (jload(os.path.join(CD, "teamtrack_probe_hits.json")) or []):
        if name == seq:
            tt = os.path.join(ROOT, "data", "external_validation", "teamtrack",
                              "teamtrack-mot", "teamtrack-mot", fam, split, seq)
    info = DQ.load_seqinfo(tt)
    tracks = DQ.load_tracks(tt)
    fps, scale = float(info["fps"]), RENDER_W / float(info["width"])
    plane = CourtPlane.load(seq)
    v3, px, fam_of = {}, {}, {}
    for it in items:
        fam = dict(FAMS).get(it["category"])
        if not fam:
            continue
        k = ikey(it)
        fam_of[k] = fam
        r = recompute_answer(plane, it, tracks, fps)
        if r:
            v3[k] = float(r[0])
        m = it["meta"]
        f0, f1 = m["window"]
        d = DQ.path_length_px(tracks[m["track"]], f0, f1)
        if d is not None:
            d *= scale
            px[k] = d if fam == "path" else d / ((f1 - f0) / fps)
    K = (jload(os.path.join(sd, "qa_dyn_v1_ruler.manifest.json")) or {}) \
        .get("k_px_per_m", {}).get("median")
    const = {}
    for _, fam in FAMS:
        for unit, table in (("v3", v3), ("px", px)):
            g = [v for k, v in table.items() if fam_of.get(k) == fam]
            if not g:
                continue
            med = st.median(g)
            const[(fam, unit)] = tmra([(med, x) for x in g], fam, K if unit == "px" else 1.0)
    return {"v3": v3, "px": px, "fam_of": fam_of, "K": K, "const": const, "n": len(items)}


def cell_row(P, gt, fam_of, fam, k=1.0):
    """一格: ρ / T-MRA / 预测中位数 / n。"""
    pairs, xs, ys = [], [], []
    for key, p in P.items():
        if fam_of.get(key) != fam or key not in gt:
            continue
        pairs.append((p, gt[key]))
        xs.append(p)
        ys.append(gt[key])
    if len(xs) < 5:
        return None
    return {"n": len(xs), "rho": spearman(xs, ys), "tmra": tmra(pairs, fam, k),
            "pred_median": st.median(xs), "gt_median": st.median(ys)}


def ratio_median(A, B, fam_of, fam):
    """同 item 的 pred(A)/pred(B) 中位比 (B > 0)。"""
    rs = [A[k] / B[k] for k in A if k in B and fam_of.get(k) == fam and B[k] > 0]
    return (st.median(rs), len(rs)) if rs else (None, 0)


def g(cell, key, nd=1):
    return "—" if not cell or cell.get(key) is None else fmt(cell[key], nd)


def main():
    R = {"schema": "courtdyn-t28-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seqs": {}}
    md = [f"# CourtDyn T28 · 递一把尺子 / 换成像素单位 / target-SFT 臂\n",
          f"生成 {R['generated']} · `tools/summarize_courtdyn_t28.py`。",
          "A 节全部在 **v3 (单应) 世界**计分 (ruler 递过去的尺子由单应导出, full 同步按 v3 重算);",
          "B 节按渲染像素计分, 等价容差 = 米制阈值 × K, 因此与 A 节可比。每格自己减自己片段的常数中位数行。\n"]

    for seq in TOPS:
        ctx = seq_ctx(seq)
        K, fam_of = ctx["K"], ctx["fam_of"]
        R["seqs"][seq] = {"k_px_per_m": K, "const": {f"{a}_{b}": v for (a, b), v in ctx["const"].items()},
                          "cells": {}}
        md += [f"\n## {seq} · K = {fmt(K, 2)} 渲染像素/米\n",
               "### A. 递一把正确的尺子 (v3 口径)\n",
               "| 模型 | 家族 | full ρ | full T-MRA | ruler ρ | ruler T-MRA | Δ T-MRA | "
               "pred 中位 full → ruler | ruler/full 比 | 常数中位数 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for mk, mlabel in MODELS:
            Pfull, Pruler = preds(full_parsed(seq, mk)), preds(t28_parsed(seq, "ruler", "full", mk))
            for _, fam in FAMS:
                cf = cell_row(Pfull, ctx["v3"], fam_of, fam)
                cr = cell_row(Pruler, ctx["v3"], fam_of, fam)
                d = (None if not (cf and cr) else cr["tmra"] - cf["tmra"])
                rat, _n = ratio_median(Pruler, Pfull, fam_of, fam)
                R["seqs"][seq]["cells"][f"{mk}@ruler@{fam}"] = {"full": cf, "ruler": cr,
                                                               "delta_tmra": d, "ratio": rat}
                md.append(f"| {mlabel} | {fam} | {g(cf,'rho',2)} | {g(cf,'tmra')} | {g(cr,'rho',2)} | "
                          f"{g(cr,'tmra')} | {fmt(d) if d is not None else '—'} | "
                          f"{g(cf,'pred_median')} → {g(cr,'pred_median')} | "
                          f"{fmt(rat,2) if rat else '—'} | {fmt(ctx['const'].get((fam,'v3')))} |")

        md += ["\n### B. 换成图像平面单位 (像素口径)\n",
               "| 模型 | 家族 | pxunit ρ | pxunit T-MRA | pred 中位 (px) | GT 中位 (px) | "
               "pred(pxunit)/pred(full_m) | 期望若换算成功 = K | 常数中位数 (px) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for mk, mlabel in MODELS:
            Pfull, Ppx = preds(full_parsed(seq, mk)), preds(t28_parsed(seq, "pxunit", "full", mk))
            for _, fam in FAMS:
                cp = cell_row(Ppx, ctx["px"], fam_of, fam, k=K or 1.0)
                rat, _n = ratio_median(Ppx, Pfull, fam_of, fam)
                R["seqs"][seq]["cells"][f"{mk}@pxunit@{fam}"] = {"pxunit": cp, "ratio_to_full": rat}
                md.append(f"| {mlabel} | {fam} | {g(cp,'rho',2)} | {g(cp,'tmra')} | "
                          f"{g(cp,'pred_median')} | {g(cp,'gt_median')} | "
                          f"{fmt(rat,2) if rat else '—'} | {fmt(K,1)} | "
                          f"{fmt(ctx['const'].get((fam,'px')))} |")

        md += ["\n### C. target-SFT 臂 (与 T26/T27 同口径)\n",
               "| 模型 | 家族 | full ρ (v1) | static-4 ρ | Δρ | zoom2/full 预测比 |",
               "|---|---|---|---|---|---|"]
        v1 = {}
        for it in (jload(os.path.join(CD, f"seq_{seq}", "qa_dyn_v1.json")) or []):
            if dict(FAMS).get(it["category"]):
                try:
                    v1[ikey(it)] = float(it["answer"])
                except (TypeError, ValueError):
                    pass
        for mk, mlabel in [("tsft", "target-SFT-token")]:
            Pf = preds(full_parsed(seq, mk))
            Ps = preds(t28_parsed(seq, "main", "static4", mk))
            Pz = preds(t28_parsed(seq, "zoom2", "full", mk))
            for _, fam in FAMS:
                cf, cs = cell_row(Pf, v1, fam_of, fam), cell_row(Ps, v1, fam_of, fam)
                dr = None if not (cf and cs) else cf["rho"] - cs["rho"]
                zr, _n = ratio_median(Pz, Pf, fam_of, fam)
                R["seqs"][seq]["cells"][f"{mk}@tsft@{fam}"] = {"full": cf, "static4": cs,
                                                              "delta_rho": dr, "zoom_ratio": zr}
                md.append(f"| {mlabel} | {fam} | {g(cf,'rho',2)} | {g(cs,'rho',2)} | "
                          f"{fmt(dr,2) if dr is not None else '—'} | "
                          f"{fmt(zr,2) if zr else '—'} |")

    md += ["\n读法: **A 节** ruler 与 full 的 Δ T-MRA ≈ 0 且 ruler/full 比 ≈ 1 ⇒ 递到手里的正确尺子也没被使用",
           "(E1b 从\"尺度句惰性\"升级为\"给了也不用\"); Δ 明显为正 ⇒ 瓶颈是尺子的提取, 不是换算。",
           "**B 节** pred(pxunit)/pred(full_m) ≈ K 且 pxunit T-MRA 明显高于米制 ⇒ 模型确实在图像平面测量,",
           "只是不会换算 —— 负结果就变成了正的机制结果; ≈ 1 ⇒ 它只是重复同一个先验数字, 连单位都没换。",
           "**C 节** 与 T26/T27 同口径, 用于判断逐 token 标签监督是否比标量奖励更容易学到归一化。"]

    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print(f"[t28] -> {OUT_MD}")
    print(f"[t28] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
