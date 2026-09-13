#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_t27.py — T27 汇总: (A) 2× 缩放干预的预测比; (B) 侧视片段 Δρ(full − static-4)。

A. 每 (片段 × 模型 × 家族): 同一 item 在 full 与 zoom2 两臂的预测比 r = pred(zoom2)/pred(full)
   (pred(full) > 0 的 item), 报中位、四分位、r > 1.5 的占比、以及两臂各自对 GT v3 的 ρ 与预测中位数。
   像素读者: 中位 r ≈ 2; 尺子/世界读者: ≈ 1。首帧×4 下预测塌到 0 的模型 (Qwen 系) 在这里不受影响 —— 两臂都是真 4 帧。
B. 侧视三条片段 + 原片段: ρ(pred, GT v1) full vs static-4 (与 T26 同口径), Δρ。
末尾顺手重跑 tools/courtdyn_pixel_reader.py (它会自动吸收 t27 的侧视 static-4 与 7B full 格)。
产物 results/courtdyn/courtdyn_t27_table.md + courtdyn_t27_findings.json。缺格显示 —。
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

from engine import dynamics_qa as DQ                                              # noqa: E402
from tools.summarize_courtdyn_v2 import spearman, jload, fmt                     # noqa: E402
from tools import courtdyn_pixel_reader as PR                                     # noqa: E402
from tools.summarize_courtdyn_t26 import seq_dir_of                               # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "courtdyn_t27_table.md")
OUT_JSON = os.path.join(CD, "courtdyn_t27_findings.json")
MAIN = "Q4_side_480-510"
ZOOM_SEQS = [("Q2_top_480-510", "俯视"), ("Q1_top_0-30", "俯视"), ("Q4_top_0-30", "俯视"), (MAIN, "侧视 (原片段)")]
SIDE_SEQS = [(MAIN, "侧视 (原片段)"), ("Q1_side_0-30", "侧视"), ("Q2_side_300-330", "侧视"), ("Q4_side_570-600", "侧视")]
MODELS = PR.MODELS
FAMS = PR.FAMS


def ikey(r):
    m = r.get("meta", {})
    return (r["category"], m.get("window_index"), m.get("track"))


def preds(d):
    if not d:
        return {}
    p = jload(os.path.join(d, "predictions.json"))
    P = p if isinstance(p, list) else (p or {}).get("predictions") or []
    out = {}
    for r in P:
        try:
            out[ikey(r)] = float(r["vlm_answer"])
        except (TypeError, ValueError):
            continue
    return out


def zoom_dir(seq, model):
    d = os.path.join(CD, f"t27_{seq}_zoom2_full_{model}_parsed")
    return d if os.path.isfile(os.path.join(d, "predictions.json")) else None


def q(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    R = {"schema": "courtdyn-t27-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "zoom2": {}, "side_static4": {}}
    md = [f"# CourtDyn T27 · 2× 缩放干预 + 侧视首帧×4\n\n生成 {R['generated']} · `tools/summarize_courtdyn_t27.py`。"
          "A: r = pred(zoom2)/pred(full), 同 item 配对 (pred(full) > 0); 像素读者 r ≈ 2, 尺子/世界读者 r ≈ 1。"
          "GT 在两臂完全相同; ρ 对 GT v3 (单应)。B: 与 T26 同口径的 Δρ(full − static-4), ρ 对 GT v1。\n",
          "## A. 缩放干预: 预测比 r = pred(zoom2) / pred(full)\n",
          "| 片段 | 视角 | 模型 | 家族 | n 对 | r 中位 [Q1, Q3] | r > 1.5 占比 | pred 中位 full → zoom2 | ρ(full, v3) | ρ(zoom2, v3) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for seq, view in ZOOM_SEQS:
        sd = seq_dir_of(seq)
        if not os.path.isfile(os.path.join(sd, "gt", "gt.txt")):
            continue
        info = DQ.load_seqinfo(sd)
        tracks = DQ.load_tracks(sd)
        F = PR.item_features(seq, tracks, info["fps"])
        v3_by = {}
        v1 = jload(os.path.join(PR.qa_dir(seq), "qa_dyn_v1.json")) or []
        for it in v1:
            f = F.get(PR.key(it))
            if f and f["v3"] is not None:
                v3_by[ikey(it)] = f["v3"]
        R["zoom2"][seq] = {}
        for m, mlabel in MODELS:
            Pf = preds(PR.find_parsed(seq, "full", m))
            Pz = preds(zoom_dir(seq, m))
            for cat, fam in FAMS:
                ks = [k for k in Pf if k[0] == cat and k in Pz and k in v3_by]
                if len(ks) < 8:
                    md.append(f"| {seq} | {view} | {mlabel} | {fam} | {len(ks)} | — | — | — | — | — |")
                    continue
                pf = [Pf[k] for k in ks]; pz = [Pz[k] for k in ks]; g = [v3_by[k] for k in ks]
                rs = [Pz[k] / Pf[k] for k in ks if Pf[k] > 0]
                e = {"n": len(ks), "n_ratio": len(rs), "r_median": st.median(rs) if rs else None,
                     "r_q1": q(rs, 0.25), "r_q3": q(rs, 0.75),
                     "share_gt_1p5": (sum(1 for r in rs if r > 1.5) / len(rs)) if rs else None,
                     "pred_med_full": st.median(pf), "pred_med_zoom2": st.median(pz),
                     "rho_full_v3": spearman(pf, g) if len(set(pf)) > 1 else None,
                     "rho_zoom2_v3": spearman(pz, g) if len(set(pz)) > 1 else None}
                R["zoom2"][seq][f"{m}@{fam}"] = e
                md.append(f"| {seq} | {view} | {mlabel} | {fam} | {len(rs)} | {fmt(e['r_median'], 2)} [{fmt(e['r_q1'], 2)}, {fmt(e['r_q3'], 2)}] | "
                          f"{fmt(100 * e['share_gt_1p5']) if e['share_gt_1p5'] is not None else '—'}% | "
                          f"{fmt(e['pred_med_full'], 2)} → {fmt(e['pred_med_zoom2'], 2)} | {fmt(e['rho_full_v3'], 2)} | {fmt(e['rho_zoom2_v3'], 2)} |")
    md.append("\n## B. 侧视片段: ρ(pred, GT v1) full vs 首帧×4\n")
    md.append("| 片段 | 模型 | 家族 | ρ full | ρ static-4 | Δρ | 首帧×4 预测 distinct 数 |")
    md.append("|---|---|---|---|---|---|---|")
    for seq, view in SIDE_SEQS:
        sd = seq_dir_of(seq)
        if not os.path.isfile(os.path.join(sd, "gt", "gt.txt")):
            continue
        info = DQ.load_seqinfo(sd)
        tracks = DQ.load_tracks(sd)
        F = PR.item_features(seq, tracks, info["fps"])
        v1_by = {ikey(it): f["v1"] for it in (jload(os.path.join(PR.qa_dir(seq), "qa_dyn_v1.json")) or [])
                 for f in [F.get(PR.key(it))] if f}
        R["side_static4"][seq] = {}
        for m, mlabel in MODELS:
            Pf = preds(PR.find_parsed(seq, "full", m))
            Ps = preds(PR.find_parsed(seq, "static4", m))
            for cat, fam in FAMS:
                kf = [k for k in Pf if k[0] == cat and k in v1_by]
                ks = [k for k in Ps if k[0] == cat and k in v1_by]
                rf = spearman([Pf[k] for k in kf], [v1_by[k] for k in kf]) if len(kf) > 8 and len({Pf[k] for k in kf}) > 1 else None
                rs = spearman([Ps[k] for k in ks], [v1_by[k] for k in ks]) if len(ks) > 8 and len({Ps[k] for k in ks}) > 1 else None
                nd = len({Ps[k] for k in ks}) if ks else None
                e = {"rho_full": rf, "rho_static4": rs, "delta": (rf - rs) if rf is not None and rs is not None else None, "static4_distinct": nd}
                R["side_static4"][seq][f"{m}@{fam}"] = e
                md.append(f"| {seq} | {mlabel} | {fam} | {fmt(rf, 2)} | {fmt(rs, 2)} | {fmt(e['delta'], 2)} | {nd if nd is not None else '—'} |")
    md.append("\n读法: A 里 r 中位 ≈ 2 且 r > 1.5 占比高 ⇒ 模型报的是图像平面位移 (没有用题面身高尺归一化); "
              "r ≈ 1 ⇒ 用了尺子或读的是世界量。B 里 Δρ 明显 > 0 ⇒ 侧视上也在读多帧; static-4 的 distinct 数很小 ⇒ 预测塌到常数 (变化探测器)。\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        PR.main()
    except Exception as e:
        print(f"[t27-sum] pixel_reader 重跑失败: {e!r}", flush=True)


if __name__ == "__main__":
    main()
