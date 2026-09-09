#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D2: 用球场单应重算 CourtDyn GT (v3), 并把已有的模型预测按 v3 重新打分。

不占卡: 题面 / 图片 / 预测一个都不动, 只换 GT 的尺子。回答三个问题:
  1. 换尺子后 GT 变了多少?  ρ(v1, v3) 与比值分布 —— 身高尺到底歪了多少。
  2. A8 验收: ρ(GT, 框高) 从 +0.38 掉到 ~0 了吗? (v3 的尺度与框高无关)
  3. 模型结论变了吗?  每格 ρ / T-MRA 对 v3 重算; 常数中位数行也重算。
     成对家族: v3 下标签翻转的对 (两条不再一正一反) 作废, 报剩余对的 paired acc。

写出:
  results/courtdyn/qa_dyn_v3_homog.json            (主池 664, answer=v3, meta 带 v1)
  results/courtdyn/qa_dyn_v2_paired_accel_vis_v3.json (accel_vis 116, 同上)
  results/courtdyn/homography_rescore.md / .json
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

from engine import dynamics_qa as DQ                     # noqa: E402
from engine.court_homography import CourtPlane, recompute_answer   # noqa: E402
from tools.summarize_courtdyn_v2 import (registry, spearman, boot_ci, tmra, E,  # noqa: E402
                                         jload, fmt, MODEL_LABEL, ARM_LABEL, NUMERIC)

CD = os.path.join(ROOT, "results", "courtdyn")
SEQ = os.path.join(ROOT, "data", "external_validation", "teamtrack", "teamtrack-mot",
                   "teamtrack-mot", "basketball_side", "test", "Q4_side_480-510")
PAIRED = (("relational_reasoning_dyn_faster", "faster"),
          ("relational_reasoning_dyn_accel_vis", "accel_vis"))


def log(msg):
    print(f"[rescore {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def box_h(tracks, tid, f0, f1):
    return DQ.median_box_h(tracks[tid], f0, f1)


def rebuild(items, plane, tracks, fps):
    out, drop = [], defaultdict(int)
    for it in items:
        r = recompute_answer(plane, it, tracks, fps)
        if r is None:
            drop[it["category"]] += 1
            continue
        ans, det = r
        new = dict(it)
        new["answer"] = ans
        new["meta"] = dict(it["meta"], v1_answer=it["answer"], v3=det,
                           scale_method="court-line homography (FIBA 28x15 m)")
        out.append(new)
    return out, dict(drop)


def key(r):
    return (r["image_id"], r["question"])


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return rk


def partial_spearman(x, y, z):
    """ρ(x, y | z): 秩上做偏相关 —— 控制框高后预测与 GT 还剩多少关系。"""
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
        return num / den if den else float("nan")
    rxy, rxz, ryz = corr(rx, ry), corr(rx, rz), corr(ry, rz)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den else float("nan")


def main():
    info = DQ.load_seqinfo(SEQ)
    fps = info["fps"]
    tracks = DQ.load_tracks(SEQ)
    plane = CourtPlane.load(info["name"])
    log(f"H 载入: {plane.meta.get('sequence')} 验证 {json.dumps(plane.meta.get('validation'))}")

    v1 = jload(os.path.join(CD, "qa_dyn_v1.json"))
    vis = jload(os.path.join(CD, "qa_dyn_v2_paired_accel_vis.json"))
    v3, drop = rebuild(v1, plane, tracks, fps)
    vis3, drop_vis = rebuild(vis, plane, tracks, fps)
    log(f"主池 {len(v1)} -> v3 {len(v3)} (丢 {drop}); accel_vis {len(vis)} -> {len(vis3)} (丢 {drop_vis})")
    with open(os.path.join(CD, "qa_dyn_v3_homog.json"), "w", encoding="utf-8") as f:
        json.dump(v3, f, indent=1, ensure_ascii=False)
    with open(os.path.join(CD, "qa_dyn_v2_paired_accel_vis_v3.json"), "w", encoding="utf-8") as f:
        json.dump(vis3, f, indent=1, ensure_ascii=False)

    R = {"schema": "courtdyn-homography-rescore-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
         "homography": {k: plane.meta.get(k) for k in ("sequence", "validation", "per_frame_max_dev_px", "corners_px")},
         "gt_agreement": {}, "boxheight": {}, "constant_median_v3": {}, "cells": {}, "paired": {}}
    md = [f"# CourtDyn D2: 单应 GT (v3) 重打分\n\n生成 {R['generated']} · `tools/rescore_courtdyn_homography.py` · "
          f"H: `results/courtdyn/homography/H_{info['name']}.json` (验证: {json.dumps(plane.meta.get('validation'))})\n"]

    # ---- 1. GT 一致性 + 2. 框高 ----
    md.append("## 1. 换尺子后 GT 变了多少 (同一条 item, v1 身高尺 vs v3 单应)\n")
    md.append("| 家族 | n | ρ(v1, v3) | v3/v1 中位 [p10, p90] | ρ(v3/v1, 框高) | ρ(v1, 框高) | **ρ(v3, 框高)** (A8 验收: → 0?) |")
    md.append("|---|---|---|---|---|---|---|")
    v3_by = {key(r): r for r in v3}
    for cat, fam in NUMERIC:
        a, b, h = [], [], []
        for r in v1:
            if r["category"] != cat or key(r) not in v3_by:
                continue
            a.append(float(r["answer"])); b.append(float(v3_by[key(r)]["answer"]))
            h.append(box_h(tracks, r["meta"]["track"], *r["meta"]["window"]))
        ratios = sorted(y / x for x, y in zip(a, b) if x > 0)
        rho_ratio_h = spearman([y / x for x, y in zip(a, b) if x > 0], [hh for x, hh in zip(a, h) if x > 0])
        rho_ab = spearman(a, b); rho_ah = spearman(a, h); rho_bh = spearman(b, h)
        ci_bh = boot_ci(spearman, b, h)
        R["gt_agreement"][fam] = {"n": len(a), "rho_v1_v3": rho_ab,
                                  "ratio_med": st.median(ratios), "ratio_p10": ratios[int(0.1 * len(ratios))],
                                  "ratio_p90": ratios[int(0.9 * len(ratios)) - 1]}
        R["boxheight"][fam] = {"rho_v1_boxh": rho_ah, "rho_v3_boxh": rho_bh, "ci_v3": ci_bh, "rho_ratio_v3v1_boxh": rho_ratio_h}
        md.append(f"| {fam} | {len(a)} | {fmt(rho_ab, 2)} | {fmt(st.median(ratios), 3)} "
                  f"[{fmt(ratios[int(0.1*len(ratios))], 2)}, {fmt(ratios[int(0.9*len(ratios))-1], 2)}] | {fmt(rho_ratio_h, 2)} | "
                  f"{fmt(rho_ah, 2)} | **{fmt(rho_bh, 2)}** [{fmt(ci_bh[0], 2)}, {fmt(ci_bh[1], 2)}] |")
    md.append("\ntiming: v3 把球心当作贴地点投到地面, 球在空中 / 手里时投影会偏几米 (仰角小), 所以 v3 的 timing **不可靠**, timing 列只作参考, D2 结论只引 speed / path。  \n**ρ(v3, 框高) 没有归零** ⇒ 框高与真实地面速度的相关是这条片段本身的 (靠近相机的球员真的跑得快), 不是身高尺伪影; A8 改用**偏相关** ρ(预测, GT | 框高) 处理 (§3 末列), 标定消不掉它。D1 多序列可检验它是否只是这条片段的巧合。\n")

    # ---- 3a. 常数中位数 (v3) ----
    md.append("## 2. 常数中位数基线 (v3 GT)\n")
    md.append("| 家族 | GT 中位 (v3) | T-MRA | T=0 |")
    md.append("|---|---|---|---|")
    for cat, fam in NUMERIC:
        ys = [float(r["answer"]) for r in v3 if r["category"] == cat]
        med = st.median(ys)
        T = E.SCALAR_TMRA_SPEC[fam][0]
        t = tmra([med] * len(ys), ys, fam, T); t0 = tmra([med] * len(ys), ys, fam, 0)
        R["constant_median_v3"][fam] = {"median": med, "tmra": t, "tmra_T0": t0, "n": len(ys)}
        md.append(f"| {fam} | {med:.1f} | **{t:.1f}** | {t0:.1f} |")

    # ---- 3b. 每格: 数值家族 ----
    reg = registry()
    md.append("\n## 3. 每格重打分 · ρ / T-MRA 对 v1 → 对 v3 (预测不变, 只换 GT)\n")
    md.append("| 池 | 帧条件 | 模型 | speed ρ v1→v3 | speed T-MRA v1→v3 | path ρ v1→v3 | path T-MRA v1→v3 | timing ρ v1→v3 | timing T-MRA v1→v3 | **偏相关 ρ(预测, v3 given 框高)** speed / path |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(reg, key=lambda k: (k[0] != "main", k[0], ARM_LABEL.get(k[1], k[1]), list(MODEL_LABEL).index(k[2]) if k[2] in MODEL_LABEL else 9))
    for (qa, arm, m) in order:
        if qa not in ("main", "noprior"):
            continue
        preds = jload(os.path.join(reg[(qa, arm, m)], "predictions.json"))
        if not preds:
            continue
        preds = preds if isinstance(preds, list) else preds.get("predictions", [])
        cells = {}
        row = []
        partial = []
        for cat, fam in NUMERIC:
            xs1, ys1, xs3, ys3, hs = [], [], [], [], []
            for r in preds:
                if r["category"] != cat:
                    continue
                # noprior 池的 question 变了, 用 image_id + 原题面找 v3
                k = key(r) if qa == "main" else (r["image_id"], r["meta"].get("original_question", r["question"]))
                t3 = v3_by.get(k)
                if t3 is None:
                    continue
                try:
                    x = float(r["vlm_answer"])
                except (TypeError, ValueError):
                    continue
                xs1.append(x); ys1.append(float(r["answer"])); xs3.append(x); ys3.append(float(t3["answer"]))
                hs.append(box_h(tracks, t3["meta"]["track"], *t3["meta"]["window"]))
            if len(xs1) < 3:
                row += ["—", "—"]
                if fam != "timing":
                    partial.append("—")
                continue
            if fam != "timing":
                pr = partial_spearman(xs3, ys3, hs)
                partial.append(fmt(pr, 2))
            rho1, rho3 = spearman(xs1, ys1), spearman(xs3, ys3)
            ci3 = boot_ci(spearman, xs3, ys3)
            # T-MRA: 官方口径 = 全体 item, 解析失败记 0
            n_all = sum(1 for r in preds if r["category"] == cat)
            T = E.SCALAR_TMRA_SPEC[fam][0]
            t1 = tmra(xs1, ys1, fam, T) * len(xs1) / n_all
            t3v = tmra(xs3, ys3, fam, T) * len(xs3) / n_all
            cells[fam] = {"rho_v1": rho1, "rho_v3": rho3, "ci_v3": ci3, "tmra_v1": t1, "tmra_v3": t3v, "n": len(xs1), "n_all": n_all,
                          "partial_rho_v3_given_boxh": (partial_spearman(xs3, ys3, hs) if fam != "timing" else None)}
            row += [f"{fmt(rho1, 2)}→**{fmt(rho3, 2)}** [{fmt(ci3[0], 2)},{fmt(ci3[1], 2)}]", f"{t1:.1f}→{t3v:.1f}"]
        R["cells"][f"{qa}|{arm}|{m}"] = cells
        md.append(f"| {qa} | {ARM_LABEL.get(arm, arm)} | {MODEL_LABEL.get(m, m)} | " + " | ".join(row) + f" | {' / '.join(partial)} |")

    # ---- 3c. 成对家族 ----
    md.append("\n## 4. 成对家族 · v3 标签 (翻转的对作废)\n")
    md.append("| 家族 | 帧条件 | 模型 | 对数 v1 | v3 仍成对 | 标签与 v1 一致 (条) | paired acc v1 → v3 (仍成对的子集) |")
    md.append("|---|---|---|---|---|---|---|")
    pools = {"faster": (v1, v3), "accel_vis": (vis, vis3)}
    for cat, fam in PAIRED:
        src, new = pools[fam]
        new_by = {key(r): r for r in new}
        pairs = defaultdict(list)
        for r in src:
            if r["category"] == cat:
                pairs[r["pair_id"]].append(r)
        valid = {pid for pid, lst in pairs.items() if len(lst) == 2 and all(key(x) in new_by for x in lst)
                 and new_by[key(lst[0])]["answer"] != new_by[key(lst[1])]["answer"]}
        agree = sum(1 for r in src if r["category"] == cat and key(r) in new_by and new_by[key(r)]["answer"] == r["answer"])
        n_items = sum(1 for r in src if r["category"] == cat)
        R["paired"][fam] = {"pairs_v1": len(pairs), "pairs_v3_valid": len(valid), "label_agree": agree, "n_items": n_items, "cells": {}}
        for (qa, arm, m) in order:
            if (fam == "faster" and qa != "main") or (fam == "accel_vis" and qa != "vis"):
                continue
            preds = jload(os.path.join(reg[(qa, arm, m)], "predictions.json"))
            if not preds:
                continue
            preds = preds if isinstance(preds, list) else preds.get("predictions", [])
            pk = {key(r): (r.get("vlm_answer") or "").strip().upper()[:1] for r in preds if r["category"] == cat}
            ok1 = ok3 = 0
            for pid in valid:
                a, b = pairs[pid]
                if pk.get(key(a)) == a["answer"] and pk.get(key(b)) == b["answer"]:
                    ok1 += 1
                if pk.get(key(a)) == new_by[key(a)]["answer"] and pk.get(key(b)) == new_by[key(b)]["answer"]:
                    ok3 += 1
            if not valid:
                continue
            acc1, acc3 = 100 * ok1 / len(valid), 100 * ok3 / len(valid)
            R["paired"][fam]["cells"][f"{arm}|{m}"] = {"paired_v1": acc1, "paired_v3": acc3, "n_pairs": len(valid)}
            md.append(f"| {fam} | {ARM_LABEL.get(arm, arm)} | {MODEL_LABEL.get(m, m)} | {len(pairs)} | {len(valid)} | "
                      f"{agree}/{n_items} | {acc1:.1f} → **{acc3:.1f}** |")

    md.append("\n读法: 若 §1 的 ρ(v3, 框高) ≈ 0 而 §3 的模型 ρ 仍 ≈ 0, 则 v1 的负结果不是身高尺的伪影; "
              "若某模型 ρ 在 v3 下明显升高, 说明它在 v1 下被框高泄漏拖累 / 抬高。\n")
    with open(os.path.join(CD, "homography_rescore.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(CD, "homography_rescore.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    log("写出 homography_rescore.md / .json")


if __name__ == "__main__":
    main()
