#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_t26.py — T26 汇总: 俯视片段 × {full, static-4, blank, no-prior} × {base, SFT, GRPO, 7B}。

复用 summarize_courtdyn_seqs.py 的 gt_section / model_cells (同一套 ρ / bootstrap CI / 偏相关 / 常数行),
只是把 "模型" 维度扩成 模型@帧条件。每条俯视片段一块, 块内每行一个 (模型, 帧条件), 末尾给
Δρ(full − static-4) —— 这是 T26 的判决量: Δ ≈ 0 ⇒ 读的是站位/间距; Δ 明显 > 0 ⇒ 多帧运动被读到了。
产物 results/courtdyn/courtdyn_t26_table.md + courtdyn_t26_findings.json。缺失的格显示 —, 不报错。
"""
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.summarize_courtdyn_seqs import gt_section, model_cells, jload, fmt, TT  # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
OUT_MD = os.path.join(CD, "courtdyn_t26_table.md")
OUT_JSON = os.path.join(CD, "courtdyn_t26_findings.json")
PROBE_HITS = os.path.join(CD, "teamtrack_probe_hits.json")

SEQS = ["Q2_top_480-510", "Q1_top_0-30", "Q4_top_0-30"]
MODELS = [("base", "base"), ("sft", "source-SFT"), ("grpo", "GRPO-v3"), ("qwen25vl7b", "Qwen2.5-VL-7B (公开)")]
MODES = [("full", "完整 4 帧"), ("static4", "首帧×4"), ("blank", "灰板"), ("noprior", "full · 无先验句")]


def seq_dir_of(seq):
    try:
        for fam, split, name in json.load(open(PROBE_HITS, encoding="utf-8")):
            if name == seq:
                return os.path.join(TT, fam, split, seq)
    except Exception:
        pass
    return os.path.join(TT, "basketball_top", "train", seq)


def parsed_dir(seq, pool, mode, model):
    """T26 的格; 旧俯视片段的 full 格来自 T21 (t21_<seq>_<pool>_<model>_parsed)。"""
    d = os.path.join(CD, f"t26_{seq}_{pool}_{mode}_{model}_parsed")
    if os.path.isdir(d):
        return d
    if mode == "full" and pool in ("main", "vis"):
        return os.path.join(CD, f"t21_{seq}_{pool}_{model}_parsed")
    return d


def log(msg):
    print(f"[t26-sum {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    R = {"schema": "courtdyn-t26-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seqs": {}}
    md = [f"# CourtDyn T26 · 俯视片段的对照臂 + 新俯视片段\n\n生成 {R['generated']} · `tools/summarize_courtdyn_t26.py` · "
          "ρ = Spearman(预测, GT v1) [bootstrap 95% CI]; 偏相关 = ρ(预测, GT v3 | 框高); T-MRA 要减掉该片段自己的常数中位数行; "
          "paired chance 25。full 格来自 T21, 其余来自 T26。\n"]
    for seq in SEQS:
        qa_dir = os.path.join(CD, f"seq_{seq}")
        if not os.path.isfile(os.path.join(qa_dir, "qa_dyn_v1.json")):
            log(f"{seq}: 没有 qa_dyn_v1.json, 跳过")
            md.append(f"## {seq}\n\n(池未建)\n")
            continue
        g, v1, v3_by, tracks = gt_section(seq, seq_dir_of(seq), qa_dir)
        R["seqs"][seq] = {"gt": g, "cells": {}}
        fs, fp = g["families"]["speed"], g["families"]["path"]
        md.append(f"## {seq} · {g['n_items']} 题 · fps {g['fps']}\n")
        md.append(f"GT: speed ρ(v3, 框高) {fmt(fs.get('rho_v3_boxh'), 2)}, path {fmt(fp.get('rho_v3_boxh'), 2)}; "
                  f"常数中位数 T-MRA speed {fmt(fs.get('const_tmra_v1'))} / path {fmt(fp.get('const_tmra_v1'))} (v1); "
                  f"单应 {'有' if g.get('h_validation') else '无 (只报 v1 ρ)'}。\n")
        md.append("| 模型 | 帧条件 | speed ρ [CI] | speed 偏相关 | speed T-MRA | path ρ [CI] | path 偏相关 | path T-MRA | timing ρ | faster paired (n) | accel_vis paired (n) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        main_dirs, vis_dirs = {}, {}
        for m, _ in MODELS:
            for mode, _ in MODES:
                pool = "noprior" if mode == "noprior" else "main"
                main_dirs[f"{m}@{mode}"] = parsed_dir(seq, pool, "full" if mode == "noprior" else mode, m)
                if mode in ("full", "static4"):
                    vis_dirs[f"{m}@{mode}"] = parsed_dir(seq, "vis", mode, m)
        cells = model_cells(seq, v1, v3_by, tracks, main_dirs, vis_dirs, qa_dir)
        R["seqs"][seq]["cells"] = cells
        for m, mlabel in MODELS:
            for mode, mode_label in MODES:
                c = cells.get(f"{m}@{mode}")
                if not c:
                    md.append(f"| {mlabel} | {mode_label} | — | — | — | — | — | — | — | — | — |")
                    continue

                def col(fam):
                    e = c.get(fam, {})
                    ci = e.get("ci_v1") or (None, None)
                    rho = (f"{fmt(e.get('rho_v1'), 2)} [{fmt(ci[0], 2)},{fmt(ci[1], 2)}]"
                           if e.get("rho_v1") is not None else "—")
                    t = e.get("tmra_official")
                    return rho, fmt(e.get("partial_v3_boxh"), 2), (fmt(100 * t) if t is not None else "—")
                s_rho, s_par, s_t = col("speed")
                p_rho, p_par, p_t = col("path")
                t_rho = fmt((c.get("timing") or {}).get("rho_v1"), 2)
                fa = c.get("faster_paired")
                av = c.get("accel_vis_paired")
                md.append(f"| {mlabel} | {mode_label} | {s_rho} | {s_par} | {s_t} | {p_rho} | {p_par} | {p_t} | {t_rho} | "
                          f"{fmt(fa) if fa is not None else '—'} ({c.get('faster_pairs') or '—'}) | "
                          f"{fmt(av) if av is not None else '—'} ({c.get('accel_vis_pairs') or '—'}) |")
        # 判决量
        md.append("\n**Δρ (full − static-4)**: " + "; ".join(
            f"{mlabel} speed {fmt(_delta(cells, m, 'speed'), 2)} / path {fmt(_delta(cells, m, 'path'), 2)}"
            for m, mlabel in MODELS) + "\n")
    md.append("读法: static-4 与 full 的 token 预算相同、运动信息为零。Δρ ≈ 0 ⇒ 俯视上的高 ρ 来自单帧可见的站位/间距 "
              "(仍是像素, 但不是运动); Δρ 明显 > 0 且 static-4 的 CI 含 0 ⇒ 多帧运动被读到了。灰板行是先验地板, "
              "无先验句行检验尺度句是否惰性。三条俯视片段逐条报, 不合并。\n")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def _delta(cells, m, fam):
    a = (cells.get(f"{m}@full") or {}).get(fam, {}).get("rho_v1")
    b = (cells.get(f"{m}@static4") or {}).get(fam, {}).get("rho_v1")
    return None if a is None or b is None else a - b


if __name__ == "__main__":
    main()
