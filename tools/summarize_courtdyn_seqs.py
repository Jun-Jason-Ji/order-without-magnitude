#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D1 逐序列汇总 (T21) —— 5 条片段 (原片段 + 4 条新片段), 每条各自一行, 不合并。

两部分:
  A. **只依赖 GT 的诊断** (不需要模型预测, 随时可跑):
     每条序列: 题数; 身高尺 GT (v1) 与单应 GT (v3) 的一致性 ρ(v1, v3)、v3/v1 中位;
     **ρ(GT, 框高)** 在 v1 与 v3 下各是多少 —— 检验 "靠近相机的球员真的跑得快" 是不是
     Q4_side_480-510 一条片段的巧合 (plan D2 的意外发现); 常数中位数基线 (v1 / v3)。
  B. **模型格** (T21 跑完后): 每序列 × {base, SFT, GRPO} × full:
     speed / path ρ [CI] 对 v1 与对 v3, 偏相关 ρ(预测, v3 | 框高), T-MRA (对 v1, 官方 summary),
     faster paired, accel_vis paired。原片段 (Q4_side_480-510) 的格取 T18 的 eval_full_*。

写出 results/courtdyn/courtdyn_seqs_table.md + courtdyn_seqs_findings.json。
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import dynamics_qa as DQ                                   # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer      # noqa: E402
from tools.summarize_courtdyn_v2 import (spearman, boot_ci, tmra, E, jload, fmt,  # noqa: E402
                                         MODEL_LABEL, NUMERIC)
from tools.rescore_courtdyn_homography import partial_spearman        # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
TT = os.path.join(ROOT, "data", "external_validation", "teamtrack", "teamtrack-mot", "teamtrack-mot")
SEQS = [  # name, seq_dir, qa_dir, {model: parsed dir for main}, {model: parsed dir for vis}
    ("Q4_side_480-510", os.path.join(TT, "basketball_side", "test", "Q4_side_480-510"), CD,
     {m: os.path.join(CD, f"eval_full_{m}_parsed") for m in ("base", "sft", "grpo")},
     {m: os.path.join(CD, f"accel_vis_full_{m}_parsed") for m in ("base", "sft", "grpo")}),
]
for seq, sub in (("Q1_side_0-30", "basketball_side/train"), ("Q2_side_300-330", "basketball_side/train"),
                 ("Q4_side_570-600", "basketball_side/test"), ("Q2_top_480-510", "basketball_top/train")):
    SEQS.append((seq, os.path.join(TT, *sub.split("/"), seq), os.path.join(CD, f"seq_{seq}"),
                 {m: os.path.join(CD, f"t21_{seq}_main_{m}_parsed") for m in ("base", "sft", "grpo")},
                 {m: os.path.join(CD, f"t21_{seq}_vis_{m}_parsed") for m in ("base", "sft", "grpo")}))
VIEW = {"Q4_side_480-510": "侧视 (原片段)", "Q1_side_0-30": "侧视", "Q2_side_300-330": "侧视",
        "Q4_side_570-600": "侧视", "Q2_top_480-510": "**俯视**"}


def log(msg):
    print(f"[seqs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def key(r):
    return (r["image_id"], r["question"])


def preds_of(d):
    p = jload(os.path.join(d, "predictions.json"))
    if not p:
        return None
    return p if isinstance(p, list) else p.get("predictions")


def gt_section(seq, seq_dir, qa_dir):
    info = DQ.load_seqinfo(seq_dir)
    tracks = DQ.load_tracks(seq_dir)
    v1 = jload(os.path.join(qa_dir, "qa_dyn_v1.json")) or []
    try:
        plane = CourtPlane.load(seq)
    except FileNotFoundError:
        plane = None
    out = {"n_items": len(v1), "fps": info["fps"], "families": {}}
    v3_by = {}
    if plane:
        for it in v1:
            r = recompute_answer(plane, it, tracks, info["fps"])
            if r:
                v3_by[key(it)] = r[0]
    out["h_validation"] = plane.meta.get("validation") if plane else None
    out["h_max_dev_px"] = plane.meta.get("per_frame_max_dev_px") if plane else None
    for cat, fam in NUMERIC:
        a, b, h = [], [], []
        for it in v1:
            if it["category"] != cat:
                continue
            a.append(float(it["answer"]))
            h.append(DQ.median_box_h(tracks[it["meta"]["track"]], *it["meta"]["window"]))
            b.append(float(v3_by[key(it)]) if key(it) in v3_by else None)
        T = E.SCALAR_TMRA_SPEC[fam][0]
        d = {"n": len(a), "rho_v1_boxh": spearman(a, h), "ci_v1_boxh": boot_ci(spearman, a, h),
             "median_v1": st.median(a) if a else None,
             "const_tmra_v1": tmra([st.median(a)] * len(a), a, fam, T) if a else None}
        if plane and all(x is not None for x in b) and b:
            ratios = sorted(y / x for x, y in zip(a, b) if x > 0)
            d.update({"rho_v1_v3": spearman(a, b), "ratio_med": st.median(ratios) if ratios else None,
                      "rho_v3_boxh": spearman(b, h), "ci_v3_boxh": boot_ci(spearman, b, h),
                      "median_v3": st.median(b), "const_tmra_v3": tmra([st.median(b)] * len(b), b, fam, T)})
        out["families"][fam] = d
    return out, v1, v3_by, tracks


def model_cells(seq, v1, v3_by, tracks, main_dirs, vis_dirs, qa_dir):
    cells = {}
    for m, d in main_dirs.items():
        P = preds_of(d)
        if not P:
            continue
        summ = jload(os.path.join(d, "summary.json")) or {}
        c = {"n_pred": len(P)}
        for cat, fam in NUMERIC:
            xs, y1, y3, hs = [], [], [], []
            for r in P:
                if r["category"] != cat:
                    continue
                try:
                    x = float(r["vlm_answer"])
                except (TypeError, ValueError):
                    continue
                xs.append(x); y1.append(float(r["answer"]))
                y3.append(float(v3_by[key(r)]) if key(r) in v3_by else None)
                hs.append(DQ.median_box_h(tracks[r["meta"]["track"]], *r["meta"]["window"]))
            e = {"n": len(xs), "rho_v1": spearman(xs, y1) if len(xs) > 2 else None,
                 "ci_v1": boot_ci(spearman, xs, y1) if len(xs) > 2 else None,
                 "tmra_official": (summ.get("per_group", {}).get(fam) or {}).get("tmra")}
            if y3 and all(v is not None for v in y3) and len(xs) > 3:
                e["rho_v3"] = spearman(xs, y3)
                e["partial_v3_boxh"] = partial_spearman(xs, y3, hs) if fam != "timing" else None
            c[fam] = e
        fa = jload(os.path.join(d, "paired_faster_score.json"))
        c["faster_paired"] = 100 * fa["paired_accuracy"] if fa else None
        c["faster_pairs"] = fa.get("n_pairs") if fa else None
        cells[m] = c
    for m, d in vis_dirs.items():
        s = jload(os.path.join(d, "paired_accel_vis_score.json"))
        if s:
            cells.setdefault(m, {})["accel_vis_paired"] = 100 * s["paired_accuracy"]
            cells[m]["accel_vis_pairs"] = s.get("n_pairs")
    return cells


def main():
    R = {"schema": "courtdyn-seqs-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seqs": {}}
    md = [f"# CourtDyn 逐序列汇总 (plan D1, T21)\n\n生成 {R['generated']} · `tools/summarize_courtdyn_seqs.py` · "
          "每条片段一行, 不跨片段合并; 原片段的模型格来自 T18 (eval_full_*), 新片段来自 T21。\n"]
    md.append("## A. 只看 GT: 身高尺 (v1) vs 单应 (v3), 以及 GT 与框高的相关\n")
    md.append("| 序列 | 视角 | 题数 | 家族 | n | ρ(v1,v3) | v3/v1 中位 | ρ(v1, 框高) | **ρ(v3, 框高)** [CI] | 常数中位数 T-MRA v1 / v3 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    for seq, seq_dir, qa_dir, main_dirs, vis_dirs in SEQS:
        if not os.path.isfile(os.path.join(qa_dir, "qa_dyn_v1.json")):
            log(f"{seq}: 没有 qa_dyn_v1.json, 跳过")
            continue
        g, v1, v3_by, tracks = gt_section(seq, seq_dir, qa_dir)
        per[seq] = (v1, v3_by, tracks, main_dirs, vis_dirs, qa_dir)
        R["seqs"][seq] = {"gt": g}
        for fam in ("speed", "path", "timing"):
            d = g["families"][fam]
            md.append(f"| {seq} | {VIEW[seq]} | {g['n_items']} | {fam} | {d['n']} | {fmt(d.get('rho_v1_v3'), 2)} | "
                      f"{fmt(d.get('ratio_med'), 3)} | {fmt(d['rho_v1_boxh'], 2)} | "
                      f"**{fmt(d.get('rho_v3_boxh'), 2)}** [{fmt((d.get('ci_v3_boxh') or (None, None))[0], 2)}, "
                      f"{fmt((d.get('ci_v3_boxh') or (None, None))[1], 2)}] | "
                      f"{fmt(d.get('const_tmra_v1'))} / {fmt(d.get('const_tmra_v3'))} |")
    md.append("\ntiming 的 v3 不可靠 (球不贴地), 只引 speed / path。俯视片段几乎没有透视, 框高 ≈ 常数, "
              "ρ(GT, 框高) 在那里若 ≈ 0 而侧视 > 0, 就说明侧视的相关是相机几何 + 战术位置的耦合, 不是尺子。\n")

    md.append("## B. 模型格 · 每序列 × 模型 × full 帧\n")
    md.append("| 序列 | 模型 | speed ρ v1 [CI] | speed ρ v3 | 偏相关 (v3\\|框高) | speed T-MRA (官方) | path ρ v1 [CI] | path ρ v3 | 偏相关 | path T-MRA | timing ρ v1 | faster paired (n 对) | accel_vis paired (n 对) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for seq, (v1, v3_by, tracks, main_dirs, vis_dirs, qa_dir) in per.items():
        cells = model_cells(seq, v1, v3_by, tracks, main_dirs, vis_dirs, qa_dir)
        R["seqs"][seq]["models"] = cells
        for m in ("base", "sft", "grpo"):
            c = cells.get(m)
            if not c:
                md.append(f"| {seq} | {MODEL_LABEL[m]} | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            def col(fam):
                e = c.get(fam, {})
                ci = e.get("ci_v1") or (None, None)
                rho = f"{fmt(e.get('rho_v1'), 2)} [{fmt(ci[0], 2)},{fmt(ci[1], 2)}]" if e.get("rho_v1") is not None else "—"
                t = e.get("tmra_official")
                return rho, fmt(e.get("rho_v3"), 2), fmt(e.get("partial_v3_boxh"), 2), (fmt(100 * t) if t is not None else "—")
            s_rho, s_v3, s_p, s_t = col("speed")
            p_rho, p_v3, p_p, p_t = col("path")
            t_rho = col("timing")[0]
            fp = c.get("faster_paired"); ap = c.get("accel_vis_paired")
            md.append(f"| {seq} | {MODEL_LABEL[m]} | {s_rho} | {s_v3} | {s_p} | {s_t} | {p_rho} | {p_v3} | {p_p} | {p_t} | {t_rho} | "
                      f"{fmt(fp)} ({c.get('faster_pairs') or '—'}) | {fmt(ap)} ({c.get('accel_vis_pairs') or '—'}) |")
    md.append("\n读法: 每格自己减自己序列的常数中位数 T-MRA (§A 末列); paired chance 25。"
              "论文里报 5 条片段的逐条数字 + 中位, 不报合并池。\n")
    with open(os.path.join(CD, "courtdyn_seqs_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(CD, "courtdyn_seqs_findings.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=float)
    log("写出 courtdyn_seqs_table.md / courtdyn_seqs_findings.json")


if __name__ == "__main__":
    main()
