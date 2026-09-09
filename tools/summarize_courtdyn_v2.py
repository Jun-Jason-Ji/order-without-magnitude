#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""summarize_courtdyn_v2.py — CourtDyn 汇总 v2: T18 + T19 + T20 一张表 (plan §7.3 C1/C3/C4)。

相对 v1 (summarize_courtdyn.py) 的改动全部来自方案审计 (plan §〇):
  C1  数值家族每格报: T-MRA (floor_v2) / T-MRA(T=0, 去掉加性 slack) /
      **Spearman ρ + bootstrap 95% CI** / n; 并加**常数中位数预测器**一行 (A1)。
      ρ 尺度不变, 是数值家族的主指标 —— 它绕开身高尺 / 单应问题。
  C3  反事实臂诊断: Δ = T-MRA(cf, GT 重标) − T-MRA(full)。**正值 = 输出与 GT 无关**
      (GT 往模型的常数方向缩, 分数反而涨)。同时报 noprior 臂: 删先验句后
      预测中位数 / ρ 变不变 (A9)。
  C4  每格每家族: Spearman(预测, 框高) —— 模型是不是把框大小当速度在读 (A8)。
  帧对照只报有效的组合 (A2): shuffle 只对 timing / accel; static4 对全部家族;
  single_v2 替代 single。

产出: results/courtdyn/courtdyn_table_v2.md + findings_v2.json
"""
import json
import math
import os
import random
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import evaluate as E                                        # noqa: E402
from engine.dynamics_qa import load_tracks                  # noqa: E402

CD = os.path.join(ROOT, "results", "courtdyn")
SEQ = os.path.join(ROOT, "data", "external_validation", "teamtrack",
                   "teamtrack-mot", "teamtrack-mot", "basketball_side",
                   "test", "Q4_side_480-510")
NUMERIC = (("dynamics_speed_player", "speed"), ("dynamics_path_player", "path"),
           ("dynamics_time_intercept", "timing"))
MODEL_LABEL = {"base": "base", "sft": "source-SFT", "grpo": "GRPO-v3",
               "tsft": "target-SFT-token", "smol": "SmolVLM2-2.2B"}
ARM_LABEL = {"full": "完整 4 帧", "static4": "首帧×4 (static-4)",
             "single": "静帧 (旧, 题面说 4 帧)", "single_v2": "静帧 (single_v2)",
             "shuffle": "打乱帧序", "blank": "灰板", "cf_prior": "反事实先验",
             "noprior": "无先验句"}
ORDER_SENSITIVE = {"timing"}          # shuffle 只对这些数值家族有效 (A2)
N_BOOT = 1000

# ---- 格子注册表: (arm, model) -> parsed dir. T18 / T19 / T20 三套命名统一到这里 ----
def registry():
    reg = {}
    # T18 主池
    for arm in ("full", "single", "shuffle", "blank", "cf_prior"):
        for m in ("base", "sft", "grpo", "tsft"):
            d = os.path.join(CD, f"eval_{arm}_{m}_parsed")
            if os.path.isfile(os.path.join(d, "summary.json")):
                reg[("main", arm, m)] = d
    # T20 主池 / 无先验 / smol
    for qa in ("main", "noprior", "vis"):
        for arm in ("full", "static4", "single_v2", "blank"):
            for m in ("base", "sft", "grpo", "smol"):
                d = os.path.join(CD, f"t20_{qa}_{arm}_{m}_parsed")
                if os.path.isfile(os.path.join(d, "summary.json")):
                    reg[(qa, arm, m)] = d
    # T19 accel 视觉侧最小对 (全景 vis / 放大 zoom)
    for var in ("vis", "zoom"):
        for arm in ("full", "single", "shuffle", "blank"):
            for m in ("base", "sft", "grpo", "tsft"):
                d = os.path.join(CD, f"accel_{var}_{arm}_{m}_parsed")
                if os.path.isfile(os.path.join(d, "summary.json")):
                    reg[(var if var == "zoom" else "vis", arm, m)] = d
    return reg


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def boot_ci(fn, xs, ys, seed=42):
    rng = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([xs[i] for i in idx], [ys[i] for i in idx])
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def tmra(xs, ys, grp, T):
    _, floor = E.SCALAR_TMRA_SPEC[grp]
    return 100 * st.mean(E.t_mra_scalar(f"{x:.1f}", f"{y:.1f}", T, floor)
                         for x, y in zip(xs, ys))


def numeric_pairs(preds, cat):
    xs, ys, hs = [], [], []
    for r in preds:
        if r["category"] != cat:
            continue
        try:
            xs.append(float(r["vlm_answer"]))
            ys.append(float(r["answer"]))
            hs.append(r)
        except (TypeError, ValueError):
            pass
    return xs, ys, hs


def fmt(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    reg = registry()
    qa_main = jload(os.path.join(CD, "qa_dyn_v1.json")) or []
    tracks = load_tracks(SEQ)
    lines = ["# CourtDyn 汇总 v2 (T18 + T19 + T20)", "",
             f"生成 {time.strftime('%Y-%m-%d %H:%M:%S')} · `tools/summarize_courtdyn_v2.py` · "
             "方案审计见 plan §〇。**数值家族主指标 = Spearman ρ** (尺度不变); T-MRA 仅作参照, "
             "且必须与常数中位数行对比。", ""]
    F = {"schema": "courtdyn-findings-v2", "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ================= 数值家族 =================
    const_row = {}
    for cat, grp in NUMERIC:
        gts = [float(r["answer"]) for r in qa_main if r["category"] == cat]
        med = st.median(gts)
        T = E.SCALAR_TMRA_SPEC[grp][0]
        const_row[grp] = {"median": med,
                          "tmra": tmra([med] * len(gts), gts, grp, T),
                          "tmra_T0": tmra([med] * len(gts), gts, grp, 0.0)}
    F["constant_median_baseline"] = const_row

    lines += ["## 数值家族 · 每格: ρ [95% CI] · T-MRA · T-MRA(T=0) · n", "",
              "| 池 | 帧条件 | 模型 | speed ρ | speed T-MRA | speed T=0 | path ρ | path T-MRA | "
              "path T=0 | timing ρ | timing T-MRA | timing T=0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|",
              "| — | **常数中位数** | — | 0 | **" + fmt(const_row["speed"]["tmra"]) + "** | "
              + fmt(const_row["speed"]["tmra_T0"]) + " | 0 | **" + fmt(const_row["path"]["tmra"])
              + "** | " + fmt(const_row["path"]["tmra_T0"]) + " | 0 | **"
              + fmt(const_row["timing"]["tmra"]) + "** | " + fmt(const_row["timing"]["tmra_T0"]) + " |"]
    num = {}
    for (qa, arm, m), d in sorted(reg.items()):
        if qa not in ("main", "noprior"):
            continue
        preds = jload(os.path.join(d, "predictions.json")) or []
        summ = jload(os.path.join(d, "summary.json")) or {}
        row = {}
        cells = []
        for cat, grp in NUMERIC:
            xs, ys, recs = numeric_pairs(preds, cat)
            n_all = sum(1 for r in preds if r["category"] == cat)
            if len(xs) < 5 or not n_all:
                cells += ["—"] * 3
                continue
            rho = spearman(xs, ys)
            lo, hi = boot_ci(spearman, xs, ys)
            # T-MRA 用官方 summary (解析失败计 0, 与 T18 主表一致); T=0 版同口径:
            # 只对可解析的记录打分, 分母仍是全部题数, 失败 = 0。
            _, floor = E.SCALAR_TMRA_SPEC[grp]
            t0 = 100 * sum(E.t_mra_scalar(f"{x:.1f}", f"{y:.1f}", 0.0, floor)
                           for x, y in zip(xs, ys)) / n_all
            off = (summ.get("per_group", {}).get(grp) or {}).get("tmra")
            # C4: 预测 vs 框高
            hs = []
            for r in recs:
                t, f = r["meta"]["track"], r["meta"]["frames"]
                hs.append(st.median(tracks[t][x][3] for x in f if x in tracks[t]))
            row[grp] = {"rho": rho, "rho_ci": [lo, hi],
                        "tmra": None if off is None else 100 * off,
                        "tmra_T0": t0, "n_parsed": len(xs), "n_all": n_all,
                        "pred_median": st.median(xs), "pred_sd": st.pstdev(xs),
                        "rho_pred_vs_boxheight": spearman(xs, hs)}
            rho_s = ("— (常数)" if math.isnan(rho)
                     else f"{rho:+.2f} [{lo:+.2f},{hi:+.2f}]")
            cells += [f"{rho_s} n={len(xs)}/{n_all}", fmt(row[grp]["tmra"]),
                      fmt(row[grp]["tmra_T0"])]
        num[(qa, arm, m)] = row
        lines.append(f"| {qa} | {ARM_LABEL.get(arm, arm)} | {MODEL_LABEL.get(m, m)} | "
                     + " | ".join(cells) + " |")
    F["numeric_cells"] = {f"{qa}|{arm}|{m}": v for (qa, arm, m), v in num.items()}
    lines += ["", "常数中位数 = 对每题都答该家族 GT 中位数 (A1)。任何 T-MRA 列都要减掉它再读。", ""]

    # ---- 帧对照 (ρ 与 T-MRA), 只报有效组合 ----
    def contrast(qa, a, b, key):
        out = {}
        for m in MODEL_LABEL:
            ra, rb = num.get((qa, a, m)), num.get((qa, b, m))
            if not ra or not rb:
                continue
            out[m] = {grp: round(ra[grp][key] - rb[grp][key], 3)
                      for grp in ra if grp in rb
                      and (b != "shuffle" or grp in ORDER_SENSITIVE)}
        return out
    F["frame_contrasts"] = {}
    lines += ["## 帧条件对照 (full − X), ρ 与 T-MRA", "",
              "有效性 (A2): static4 / blank / single_v2 对全部数值家族有效; shuffle 只报 timing。"
              " `single` (旧) 不报 (A3)。", ""]
    for other in ("static4", "single_v2", "shuffle", "blank"):
        for key in ("rho", "tmra"):
            c = contrast("main", "full", other, key)
            if c:
                F["frame_contrasts"][f"full_minus_{other}_{key}"] = c
                lines += [f"### full − {other} ({key})", "",
                          "| 模型 | speed | path | timing |", "|---|---|---|---|"]
                for m, d in c.items():
                    lines.append(f"| {MODEL_LABEL[m]} | " + " | ".join(
                        fmt(d.get(g), 2 if key == "rho" else 1) for g in ("speed", "path", "timing")) + " |")
                lines.append("")

    # ---- C3: 反事实与无先验 ----
    F["prior_diagnostics"] = {}
    lines += ["## 先验诊断 (C3 / A9)", "",
              "Δcf = T-MRA(反事实, GT×0.751) − T-MRA(full)。**正值 = 输出与 GT 无关** "
              "(GT 往模型的常数方向缩, 分数反涨)。noprior: 删掉身高句, 预测中位数与 ρ 应当**不变**"
              "若先验句是惰性的。", "",
              "| 模型 | 家族 | full 预测中位 | cf 预测中位 | 实测比 (应 0.751) | Δcf T-MRA | "
              "noprior 预测中位 | noprior ρ | full ρ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in ("base", "sft", "grpo"):
        full = num.get(("main", "full", m))
        if not full:
            continue
        cfd = reg.get(("main", "cf_prior", m))
        cfp = jload(os.path.join(cfd, "predictions.json")) if cfd else None
        nop = num.get(("noprior", "full", m))
        for cat, grp in NUMERIC:
            rec = {"full_pred_median": full[grp]["pred_median"], "full_rho": full[grp]["rho"]}
            cfm = ratio = dcf = None
            if cfp:
                xs, ys, _ = numeric_pairs(cfp, cat)
                if len(xs) > 5:
                    cfm = st.median(xs)
                    ratio = cfm / full[grp]["pred_median"] if full[grp]["pred_median"] else None
                    dcf = tmra(xs, ys, grp, E.SCALAR_TMRA_SPEC[grp][0]) - full[grp]["tmra"]
            rec.update(cf_pred_median=cfm, observed_ratio=ratio, delta_cf_tmra=dcf,
                       expected_ratio=1.0 if grp == "timing" else 0.7513)
            if nop and grp in nop:
                rec.update(noprior_pred_median=nop[grp]["pred_median"], noprior_rho=nop[grp]["rho"])
            F["prior_diagnostics"][f"{m}|{grp}"] = rec
            lines.append(f"| {MODEL_LABEL[m]} | {grp} | {fmt(rec['full_pred_median'])} | {fmt(cfm)} | "
                         f"{fmt(ratio, 3)} | {fmt(dcf)} | {fmt(rec.get('noprior_pred_median'))} | "
                         f"{fmt(rec.get('noprior_rho'), 2)} | {fmt(rec['full_rho'], 2)} |")
    lines.append("")

    # ---- C4: 预测 vs 框高 ----
    lines += ["## 模型在读框大小吗 (C4 / A8) · Spearman(预测, 框高), main/full", "",
              "GT 本身与框高 ρ≈+0.38 (身高尺前缩)。模型的这一列若明显高于其 ρ(预测, GT), "
              "说明它更像在读框大小而不是在测运动。", "",
              "| 模型 | speed ρ(pred,框高) | speed ρ(pred,GT) | path ρ(pred,框高) | path ρ(pred,GT) |",
              "|---|---|---|---|---|"]
    F["pred_vs_boxheight"] = {}
    for m in MODEL_LABEL:
        r = num.get(("main", "full", m))
        if not r:
            continue
        F["pred_vs_boxheight"][m] = {g: r[g]["rho_pred_vs_boxheight"] for g in r}
        lines.append(f"| {MODEL_LABEL[m]} | " + " | ".join(
            fmt(v, 2) for v in (r.get("speed", {}).get("rho_pred_vs_boxheight"),
                                r.get("speed", {}).get("rho"),
                                r.get("path", {}).get("rho_pred_vs_boxheight"),
                                r.get("path", {}).get("rho"))) + " |")
    lines.append("")

    # ================= 成对家族 =================
    lines += ["## 成对家族 · paired acc (chance 25) [95% CI] · 逐题 acc · n 对", "",
              "faster = 视觉侧最小对 (v1); accel_vis = 视觉侧最小对 (T19 重建); accel(v1) 有洞, 不报。", "",
              "| 家族 | 池/画面 | 帧条件 | 模型 | **paired** | 逐题 | same-letter | n 对 |",
              "|---|---|---|---|---|---|---|---|"]
    F["paired"] = {}
    for (qa, arm, m), d in sorted(reg.items()):
        for fam, fn in (("faster", "paired_faster_score.json"),
                        ("accel_vis", "paired_accel_vis_score.json")):
            s = jload(os.path.join(d, fn))
            if not s or s.get("paired_accuracy") is None:
                continue
            n = s["n_pairs"]
            p = s["paired_accuracy"]
            # 二项 bootstrap CI
            rng = random.Random(42)
            k = round(p * n)
            vals = sorted(sum(1 for _ in range(n) if rng.random() < k / n) / n for _ in range(N_BOOT))
            lo, hi = vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]
            F["paired"][f"{fam}|{qa}|{arm}|{m}"] = {
                "paired": p, "ci": [lo, hi], "item": s["item_accuracy"],
                "same_letter": s["same_letter_rate"], "n_pairs": n}
            lines.append(f"| {fam} | {qa} | {ARM_LABEL.get(arm, arm)} | {MODEL_LABEL.get(m, m)} | "
                         f"**{100*p:.1f}** [{100*lo:.0f},{100*hi:.0f}] | {100*s['item_accuracy']:.1f} | "
                         f"{100*s['same_letter_rate']:.1f} | {n} |")
    lines.append("")

    os.makedirs(CD, exist_ok=True)
    with open(os.path.join(CD, "courtdyn_table_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(CD, "findings_v2.json"), "w", encoding="utf-8") as f:
        json.dump(F, f, indent=1, ensure_ascii=False, default=float)
    print(f"[summary-v2] {len(reg)} 格 -> courtdyn_table_v2.md + findings_v2.json")


if __name__ == "__main__":
    main()
