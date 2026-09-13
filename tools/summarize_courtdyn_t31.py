#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_t31.py — T31 汇总: (A) 裁剪干预的三臂分解; (B) 无先验句臂补齐。

A. 拆开 T27 的复合干预
   full   : 整帧            表观 1x, 周边全在
   zoom2  : 裁剪块 -> 960 宽  表观 2x, 周边已去
   declut : 同一裁剪块 -> 480 宽居中贴黑底   表观 1x, 周边已去
   三臂的 GT 完全相同, zoom2 与 declut 连裁剪矩形都逐条相同 (declut 直接复用 zoom2 的 crop_rect)。
   于是:
       Δ周边 = ρ(declut) − ρ(full)
       Δ尺寸 = ρ(zoom2)  − ρ(declut)
       合计   = ρ(zoom2)  − ρ(full)      (= T27 报的那个数)
   **三臂必须落在同一批 item 上再算 ρ**, 否则相减没有意义 —— 本脚本取三臂的交集。

   ⚠ 诚实边界: declut 的黑边抹掉的是裁剪块以外的**一切** (干扰球员 + 场地线 + 远景),
   所以 Δ周边 测的是"周边内容", 不是"其他球员"。场地线正是透视线索, 不能写成"去掉了干扰球员"。

B. 无先验句臂 (E1b(b))
   删掉题面的 "Assume a typical player on this court is 1.93 m tall."。这是**臂内**比较,
   不受 v1/v3 选择影响; 关键读数是**预测中位数动没动**, ρ 只是参考。
   Q2_top 早先由 T26 跑过, Q1_top / Q4_top 由 T31 补齐 -> 三条俯视片段齐了。

产物 results/courtdyn/courtdyn_t31_table.md + courtdyn_t31_findings.json。缺格显示 —。
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

from engine import dynamics_qa as DQ                                # noqa: E402
from tools.summarize_courtdyn_v2 import spearman, jload, fmt        # noqa: E402
from tools import courtdyn_pixel_reader as PR                       # noqa: E402
from tools.summarize_courtdyn_t26 import seq_dir_of                 # noqa: E402
from tools.summarize_courtdyn_t27 import ikey, preds                # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "courtdyn_t31_table.md")
OUT_JSON = os.path.join(CD, "courtdyn_t31_findings.json")
MAIN = "Q4_side_480-510"
DECLUT_SEQS = [("Q1_top_0-30", "俯视"), ("Q2_top_480-510", "俯视"),
               ("Q4_top_0-30", "俯视"), (MAIN, "侧视 (原片段)")]
NOPRIOR_SEQS = [("Q1_top_0-30", "俯视"), ("Q2_top_480-510", "俯视"), ("Q4_top_0-30", "俯视")]
ARMS = [("sft", "source-SFT"), ("grpo", "GRPO-v3")]
FAMS = PR.FAMS
MIN_N = 8


def find_parsed(seq, pool, model, prefixes=("t31", "t27", "t26", "t20")):
    """同一条臂在不同队列下建的目录前缀不同, 逐个试; 保留前缀是为了留住出处。"""
    for pref in prefixes:
        d = os.path.join(CD, f"{pref}_{seq}_{pool}_full_{model}_parsed")
        if os.path.isfile(os.path.join(d, "predictions.json")):
            return d
    return None


def v3_of(seq):
    """该片段每个 item 的 v3 (单应) 真值。"""
    sd = seq_dir_of(seq)
    if not os.path.isfile(os.path.join(sd, "gt", "gt.txt")):
        return {}
    info, tracks = DQ.load_seqinfo(sd), DQ.load_tracks(sd)
    F = PR.item_features(seq, tracks, info["fps"])
    out = {}
    for it in jload(os.path.join(PR.qa_dir(seq), "qa_dyn_v1.json")) or []:
        f = F.get(PR.key(it))
        if f and f["v3"] is not None:
            out[ikey(it)] = f["v3"]
    return out


def rho(P, ks, gt):
    xs = [P[k] for k in ks]
    return spearman(xs, [gt[k] for k in ks]) if len(set(xs)) > 1 else None


def d(a, b):
    return None if (a is None or b is None) else a - b


def main():
    R = {"schema": "courtdyn-t31-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
         "declut": {}, "noprior": {}}
    md = [f"# CourtDyn T31 · 裁剪干预的三臂分解 + 无先验句臂补齐\n",
          f"生成 {R['generated']} · `tools/summarize_courtdyn_t31.py`。",
          "A 节三臂 (full / zoom2 / declut) 的 GT 完全相同, zoom2 与 declut 的裁剪矩形逐条相同;",
          "ρ 一律对 GT **v3 (单应)**, 且**只在三臂共同的 item 上**计算, 否则相减无意义。\n",
          "## A. 拆开裁剪干预: 表观尺寸 vs 周边内容\n",
          "| 片段 | 视角 | 模型 | 家族 | n (三臂交集) | ρ full | ρ declut | ρ zoom2 | **Δ周边** | **Δ尺寸** | 合计 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    for seq, view in DECLUT_SEQS:
        gt = v3_of(seq)
        if not gt:
            continue
        R["declut"][seq] = {}
        for m, mlabel in ARMS:
            Pf = preds(find_parsed(seq, "main", m, ("t26", "t21", "t20")) or PR.find_parsed(seq, "full", m))
            Pz = preds(find_parsed(seq, "zoom2", m, ("t27",)))
            Pd = preds(find_parsed(seq, "declut", m, ("t31",)))
            for cat, fam in FAMS:
                ks = [k for k in Pf if k[0] == cat and k in Pz and k in Pd and k in gt]
                if len(ks) < MIN_N:
                    md.append(f"| {seq} | {view} | {mlabel} | {fam} | {len(ks)} | — | — | — | — | — | — |")
                    continue
                rf, rd, rz = rho(Pf, ks, gt), rho(Pd, ks, gt), rho(Pz, ks, gt)
                e = {"n": len(ks), "rho_full_v3": rf, "rho_declut_v3": rd, "rho_zoom2_v3": rz,
                     "d_context": d(rd, rf), "d_size": d(rz, rd), "d_total": d(rz, rf)}
                R["declut"][seq][f"{m}@{fam}"] = e
                md.append(f"| {seq} | {view} | {mlabel} | {fam} | {len(ks)} | {fmt(rf,2)} | {fmt(rd,2)} | "
                          f"{fmt(rz,2)} | **{fmt(e['d_context'],2)}** | **{fmt(e['d_size'],2)}** | {fmt(e['d_total'],2)} |")

    md += ["\n## B. 无先验句臂 (E1b(b), 三条俯视片段)\n",
           "删掉题面的身高句。**臂内比较, 不受 v1/v3 选择影响**; 关键读数是预测中位数动没动。\n",
           "| 片段 | 模型 | 家族 | n | ρ full | ρ noprior | Δρ | pred 中位 full → noprior | 比 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for seq, _view in NOPRIOR_SEQS:
        gt = v3_of(seq)
        if not gt:
            continue
        R["noprior"][seq] = {}
        for m, mlabel in ARMS:
            Pf = preds(find_parsed(seq, "main", m, ("t26", "t21", "t20")) or PR.find_parsed(seq, "full", m))
            Pn = preds(find_parsed(seq, "noprior", m, ("t31", "t26", "t20")))
            for cat, fam in FAMS:
                ks = [k for k in Pf if k[0] == cat and k in Pn and k in gt]
                if len(ks) < MIN_N:
                    md.append(f"| {seq} | {mlabel} | {fam} | {len(ks)} | — | — | — | — | — |")
                    continue
                rf, rn = rho(Pf, ks, gt), rho(Pn, ks, gt)
                mf, mn = st.median([Pf[k] for k in ks]), st.median([Pn[k] for k in ks])
                e = {"n": len(ks), "rho_full_v3": rf, "rho_noprior_v3": rn, "d_rho": d(rn, rf),
                     "pred_med_full": mf, "pred_med_noprior": mn,
                     "ratio": (mn / mf) if mf else None}
                R["noprior"][seq][f"{m}@{fam}"] = e
                md.append(f"| {seq} | {mlabel} | {fam} | {len(ks)} | {fmt(rf,2)} | {fmt(rn,2)} | "
                          f"{fmt(e['d_rho'],2)} | {fmt(mf,2)} → {fmt(mn,2)} | "
                          f"{fmt(e['ratio'],2) if e['ratio'] else '—'} |")

    md += ["\n读法:",
           "**A 节** Δ周边 与 Δ尺寸 谁大, 决定 T27 那个 +0.36 该记在谁头上。"
           "若 Δ周边 占大头 ⇒ 侧视的低分主要是**画面里干扰太多 / 目标周边信息太杂**, 不是分辨率;"
           "若 Δ尺寸 占大头 ⇒ 主要是**目标太小看不清**。两者都不小 ⇒ 如实报两条, 复合干预依旧不可整体归因。",
           "⚠ declut 的黑边抹掉的是裁剪块以外的**一切** (干扰球员 + 场地线 + 远景), "
           "所以 Δ周边 是\"周边内容\"的效应, **不能写成\"去掉其他球员\"** —— 场地线正是透视线索。",
           "**B 节** 三条俯视片段上预测中位数若都基本不动 (比 ≈ 1), "
           "E1b(b) \"题面那把尺是惰性的\" 就从一条片段扩到三条; 这是臂内比较, 与拿哪把尺子计分无关。"]

    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print(f"[t31] -> {OUT_MD}")
    print(f"[t31] -> {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
