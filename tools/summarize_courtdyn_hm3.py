#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_hm3.py — T29 (N1) 汇总: 第二场地 + 同一段运动的多相机视角。

三节:
A. **逐相机主表** 每 (序列 x 相机 x 模型 x 家族): ρ(预测, 世界 GT) [CI] / T-MRA / 预测中位 /
   该相机自己的常数中位数基线。GT 是真三维世界量, 不涉及身高尺与单应, 因此**没有 v1/v3 歧义**。
B. **视角判决 (本篇最干净的一张表)** 同一序列的多台相机拍的是**同一段世界运动**, GT 逐字相同。
   逐相机报 ρ(像素位移, 世界位移) 与模型的 ρ。若模型 ρ 随 ρ(px, world) 一起升降, 就是在同一批
   item 上直接证实 "读的是图像平面位移" —— 不再依赖跨片段比较。
C. **首帧x4 对照** 第二场地上 Δρ(full − static-4) 是否复现。

产物 results/courtdyn/courtdyn_hm3_table.md + courtdyn_hm3_findings.json。缺格显示 —。
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

import eval.evaluate as E                                              # noqa: E402
from tools.summarize_courtdyn_v2 import spearman, jload, fmt           # noqa: E402
from tools.summarize_courtdyn_t28 import ikey, preds, tmra             # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "courtdyn_hm3_table.md")
OUT_JSON = os.path.join(CD, "courtdyn_hm3_findings.json")
SEQ_GROUPS = [("basketball1/split1", [f"hm3_basketball1_split1_camera_{c}" for c in range(4)]),
              ("basketball1/split2", [f"hm3_basketball1_split2_camera_{c}" for c in range(4)]),
              ("basketball2", [f"hm3_basketball2_camera_{c}" for c in range(3)])]
MODELS = [("sft", "source-SFT"), ("grpo", "GRPO-v3"), ("base", "base 4B"),
          ("qwen25vl7b", "Qwen2.5-VL-7B")]
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]


def parsed(seq, mode, model):
    """T29 建的格子在 t29_ 前缀下; T30 (N6) 补的首帧x4 格子在 t30_ 下 —— 两个都找, 保留出处。"""
    for pref in ("t29", "t30"):
        d = os.path.join(CD, f"{pref}_{seq}_main_{mode}_{model}_parsed")
        if os.path.isfile(os.path.join(d, "predictions.json")):
            return d
    return None


def seq_ctx(seq):
    """世界 GT + 像素位移 (逐 item), 以及该相机的常数中位数基线。"""
    p = os.path.join(CD, f"seq_{seq}", "qa_dyn_v1.json")
    items = jload(p) or []
    gt, px, fam_of = {}, {}, {}
    for it in items:
        fam = dict(FAMS).get(it["category"])
        if not fam:
            continue
        k = ikey(it)
        fam_of[k] = fam
        try:
            gt[k] = float(it["answer"])
        except (TypeError, ValueError):
            continue
        m = it["meta"]
        d = m.get("px_disp_render")
        if d is not None:
            px[k] = d if fam == "path" else d / (m["window"][1] - m["window"][0]) * m["fps"]
    const = {}
    for _, fam in FAMS:
        g = [v for k, v in gt.items() if fam_of[k] == fam]
        if g:
            const[fam] = tmra([(st.median(g), x) for x in g], fam)
    return {"gt": gt, "px": px, "fam_of": fam_of, "const": const, "n_items": len(items)}


def cell(P, gt, fam_of, fam):
    xs, ys, pairs = [], [], []
    for k, v in P.items():
        if fam_of.get(k) != fam or k not in gt:
            continue
        xs.append(v)
        ys.append(gt[k])
        pairs.append((v, gt[k]))
    if len(xs) < 5:
        return None
    return {"n": len(xs), "rho": spearman(xs, ys), "tmra": tmra(pairs, fam),
            "pred_median": st.median(xs)}


def rho_px_world(ctx, fam):
    xs = [(ctx["px"][k], ctx["gt"][k]) for k in ctx["px"]
          if ctx["fam_of"].get(k) == fam and k in ctx["gt"]]
    return spearman([a for a, _ in xs], [b for _, b in xs]) if len(xs) >= 5 else None


def g(c, key, nd=1):
    return "—" if not c or c.get(key) is None else fmt(c[key], nd)


def main():
    R = {"schema": "courtdyn-hm3-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seqs": {}}
    md = [f"# CourtDyn · 第二场地 (Human-M3) 与多相机视角判决\n",
          f"生成 {R['generated']} · `tools/summarize_courtdyn_hm3.py`。",
          "GT 是 Human-M3 自带的**真三维世界坐标** (关节 XY 中位数的逐帧累计位移), "
          "既不用身高尺也不用球场单应 —— 因此这里**没有 v1/v3 那道 2 倍歧义**。",
          "同一序列的多台相机拍的是**同一段世界运动**, GT 逐字相同, 只有相机几何与像素不同。\n",
          "## A. 逐相机主表\n",
          "| 序列 | 相机 | 模型 | 家族 | ρ(预测, 世界) | T-MRA | 常数中位数 | Δ vs 常数 | 预测中位 | n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ctxs = {}
    for group, seqs in SEQ_GROUPS:
        for seq in seqs:
            if not os.path.isfile(os.path.join(CD, f"seq_{seq}", "qa_dyn_v1.json")):
                continue
            ctxs[seq] = seq_ctx(seq)
            cam = seq.rsplit("_", 1)[-1]
            R["seqs"][seq] = {"const": ctxs[seq]["const"], "cells": {}}
            for mk, mlabel in MODELS:
                P = preds(parsed(seq, "full", mk))
                if not P:
                    continue
                for _, fam in FAMS:
                    c = cell(P, ctxs[seq]["gt"], ctxs[seq]["fam_of"], fam)
                    R["seqs"][seq]["cells"][f"{mk}@full@{fam}"] = c
                    d = None if not c else c["tmra"] - ctxs[seq]["const"].get(fam, 0)
                    md.append(f"| {group} | cam {cam} | {mlabel} | {fam} | {g(c,'rho',2)} | "
                              f"{g(c,'tmra')} | {fmt(ctxs[seq]['const'].get(fam))} | "
                              f"{fmt(d) if d is not None else '—'} | {g(c,'pred_median')} | "
                              f"{c['n'] if c else '—'} |")

    md += ["\n## B. 视角判决 · 同一段世界运动, 只换相机\n",
           "| 序列 | 家族 | 相机 | ρ(像素位移, 世界位移) | source-SFT ρ | GRPO ρ | 世界 GT 中位 |",
           "|---|---|---|---|---|---|---|"]
    for group, seqs in SEQ_GROUPS:
        avail = [s for s in seqs if s in ctxs]
        if len(avail) < 2:
            continue
        for _, fam in FAMS:
            for seq in avail:
                ctx = ctxs[seq]
                cam = seq.rsplit("_", 1)[-1]
                rpw = rho_px_world(ctx, fam)
                cs = cell(preds(parsed(seq, "full", "sft")), ctx["gt"], ctx["fam_of"], fam)
                cg = cell(preds(parsed(seq, "full", "grpo")), ctx["gt"], ctx["fam_of"], fam)
                gts = [v for k, v in ctx["gt"].items() if ctx["fam_of"][k] == fam]
                R["seqs"][seq].setdefault("rho_px_world", {})[fam] = rpw
                md.append(f"| {group} | {fam} | cam {cam} | {fmt(rpw,2) if rpw is not None else '—'} "
                          f"| {g(cs,'rho',2)} | {g(cg,'rho',2)} | "
                          f"{fmt(st.median(gts),2) if gts else '—'} |")

    md += ["\n## C. 首帧×4 对照 (第二场地上的复现)\n",
           "| 序列 | 相机 | 模型 | 家族 | full ρ | static-4 ρ | Δρ |",
           "|---|---|---|---|---|---|---|"]
    for seq in ctxs:
        for mk, mlabel in MODELS:
            Ps, Pf = preds(parsed(seq, "static4", mk)), preds(parsed(seq, "full", mk))
            if not Ps or not Pf:
                continue
            for _, fam in FAMS:
                cf = cell(Pf, ctxs[seq]["gt"], ctxs[seq]["fam_of"], fam)
                cs = cell(Ps, ctxs[seq]["gt"], ctxs[seq]["fam_of"], fam)
                dr = None if not (cf and cs) else cf["rho"] - cs["rho"]
                R["seqs"][seq]["cells"][f"{mk}@static4@{fam}"] = cs
                md.append(f"| {seq.replace('hm3_','')} | {seq.rsplit('_',1)[-1]} | {mlabel} | {fam} | "
                          f"{g(cf,'rho',2)} | {g(cs,'rho',2)} | "
                          f"{fmt(dr,2) if dr is not None else '—'} |")

    md += ["\n读法: **A 节** 与 TeamTrack 主表同口径, 但 GT 是真世界量, 可直接检验 "
           "\"第二场地上结论是否复现\"。**B 节**是本篇最干净的一张表: 世界量固定, 只有视角变 —— "
           "若模型 ρ 随 ρ(像素, 世界) 一起升降, \"读的是图像平面位移\" 就在同一批 item 上被证实; "
           "若模型 ρ 在各相机上一样高, 那它读的就是世界量, 主论点要改。**C 节**检验多帧是否仍被使用。"]

    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print(f"[hm3] -> {OUT_MD}")
    print(f"[hm3] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
