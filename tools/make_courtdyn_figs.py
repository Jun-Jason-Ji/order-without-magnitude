#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_courtdyn_figs.py — 论文 2 的图 (N4)。只读冻结产物, 零重算, 缺格自动略过。

fig1_viewpoint_rho   7 条 TeamTrack 片段 x 4 模型的 ρ 点图, 俯视/侧视分面;
                     full 实心、首帧x4 空心, 同一格用细线连起来 => 一张图说完 E1a。
fig2_constant_vs_gt  T-MRA − 该片段常数中位数 vs 常数中位数本身 (v1 与 v3 两个口径并排)。
                     若点沿横轴下滑 => "能不能赢常数" 主要跟该片段 GT 分布的宽窄走, 不是模型能力。
fig3_multiview       Human-M3: 同一段世界运动在多相机下的 ρ(像素, 世界); 模型 ρ 落盘后自动叠上去。
fig4_zoom_ratio      T27 的 zoom2 预测比分布 (箱线 + 参考线 1.0 / 2.0); 需 courtdyn_t27_findings.json。
fig5_ruler_and_units T28 递尺子臂与换单位臂并排: 左 ruler/full 预测比 (参考线 1.0),
                     右 pred(px)/pred(m) 对数轴 vs K≈27 参考带; 需 courtdyn_t28_findings.json。

用法: python tools/make_courtdyn_figs.py [--out paper2/figs]
"""
import argparse
import io
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CD = os.path.join(ROOT, "results", "courtdyn")
MAIN = "Q4_side_480-510"
FAMS = ["speed", "path"]
# 论文用色: 视角用形状区分, 模型用颜色; 全部色盲安全, 灰度可辨
CLR = {"sft": "#1b6ca8", "grpo": "#c8471f", "base": "#8a8f8c", "qwen25vl7b": "#5c9e31",
       "tsft": "#7b4ea3", "cdnative": "#b08015"}
LBL = {"sft": "source-SFT", "grpo": "GRPO-v3", "base": "base 4B",
       "qwen25vl7b": "Qwen2.5-VL-7B", "tsft": "target-SFT", "cdnative": "CD-native"}
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .7, "xtick.major.width": .7,
                     "ytick.major.width": .7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160,
                     "font.family": ["DejaVu Sans"]})


def jload(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ---------------------------------------------------------------- fig 1
def fig_viewpoint(out):
    seqs = jload(os.path.join(CD, "courtdyn_seqs_findings.json"))
    t26 = jload(os.path.join(CD, "courtdyn_t26_findings.json"))
    if not seqs or not t26:
        return print("  ! fig1: 缺 findings, 略过")
    order = [("Q1_top_0-30", "top"), ("Q2_top_480-510", "top"), ("Q4_top_0-30", "top"),
             (MAIN, "side"), ("Q1_side_0-30", "side"),
             ("Q2_side_300-330", "side"), ("Q4_side_570-600", "side")]
    tops = [s for s, v in order if v == "top"]
    sides = [s for s, v in order if v == "side"]

    def rho(seq, model, fam, mode):
        c = ((t26.get("seqs", {}).get(seq, {}).get("cells") or {}).get(f"{model}@{mode}") or {})
        v = (c.get(fam) or {}).get("rho_v1")
        if v is None and mode == "full":
            m = (seqs.get("seqs", {}).get(seq, {}).get("models") or {}).get(model) or {}
            v = (m.get(fam) or {}).get("rho_v1")
        return v

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharey=True,
                             gridspec_kw={"width_ratios": [len(tops), len(sides)]})
    for r, fam in enumerate(FAMS):
        for c, (grp, title) in enumerate([(tops, "Overhead cameras"), (sides, "Side cameras")]):
            ax = axes[r][c]
            for i, seq in enumerate(grp):
                for j, m in enumerate(["sft", "grpo", "base", "qwen25vl7b"]):
                    x = i + (j - 1.5) * .16
                    rf, rs = rho(seq, m, fam, "full"), rho(seq, m, fam, "static4")
                    if rf is None:
                        continue
                    if rs is not None:
                        ax.plot([x, x], [rs, rf], color=CLR[m], lw=.8, alpha=.55, zorder=1)
                        ax.plot(x, rs, "o", ms=3.6, mfc="white", mec=CLR[m], mew=.9, zorder=2)
                    ax.plot(x, rf, "o", ms=4.2, color=CLR[m], zorder=3)
            ax.axhline(0, color="#b9bfbb", lw=.7, ls=(0, (3, 3)), zorder=0)
            ax.set_xticks(range(len(grp)))
            # 两条 Q4_side 只差窗口, 截掉区间会撞名 -> 保留起始秒
            ax.set_xticklabels([s.replace("_0-30", "").replace("-510", "").replace("-330", "")
                                .replace("-600", "") for s in grp],
                               rotation=30, ha="right", fontsize=7)
            ax.set_xlim(-.6, len(grp) - .4)
            if c == 0:
                ax.set_ylabel(f"Spearman rho, {fam}")
            if r == 0:
                ax.set_title(title, fontsize=9, pad=6)
    axes[0][0].set_ylim(-.42, .92)
    handles = [Line2D([], [], marker="o", ls="", color=CLR[m], label=LBL[m]) for m in
               ["sft", "grpo", "base", "qwen25vl7b"]]
    handles += [Line2D([], [], marker="o", ls="", color="#444", label="full (4 frames)"),
                Line2D([], [], marker="o", ls="", mfc="white", mec="#444", label="first frame x4")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, -.06), fontsize=7.5)
    fig.suptitle("Rank agreement varies with viewpoint and temporal content (v1 ground truth)", fontsize=10, y=.99)
    fig.tight_layout()
    save(fig, out, "fig1_viewpoint_rho")


# ---------------------------------------------------------------- fig 2
def fig_constant(out, refresh_tally=False):
    import eval.evaluate as E
    from engine import dynamics_qa as DQ
    from engine.court_homography import CourtPlane, recompute_answer
    from tools.summarize_courtdyn_t28 import ikey, preds, tmra
    from tools import courtdyn_pixel_reader as PR

    seqs = [("Q1_top_0-30", "top"), ("Q2_top_480-510", "top"), ("Q4_top_0-30", "top"),
            (MAIN, "side"), ("Q1_side_0-30", "side"),
            ("Q2_side_300-330", "side"), ("Q4_side_570-600", "side")]
    hits = jload(os.path.join(CD, "teamtrack_probe_hits.json")) or []
    pts = {"v1": [], "v3": []}
    detail = {"v1": [], "v3": []}
    for seq, view in seqs:
        sd = CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")
        items = jload(os.path.join(sd, "qa_dyn_v1.json")) or []
        tt = next((os.path.join(ROOT, "data", "external_validation", "teamtrack",
                                "teamtrack-mot", "teamtrack-mot", f, s, n)
                   for f, s, n in hits if n == seq), None)
        if not tt:
            continue
        info, tracks = DQ.load_seqinfo(tt), DQ.load_tracks(tt)
        fps = float(info["fps"])
        plane = CourtPlane.load(seq)
        gt = {"v1": {}, "v3": {}}
        fam_of = {}
        for it in items:
            fam = dict(PR.FAMS).get(it["category"])
            if not fam:
                continue
            k = ikey(it)
            fam_of[k] = fam
            try:
                gt["v1"][k] = float(it["answer"])
            except (TypeError, ValueError):
                pass
            r = recompute_answer(plane, it, tracks, fps)
            if r:
                gt["v3"][k] = float(r[0])
        for unit in ("v1", "v3"):
            const = {}
            for fam in FAMS:
                g = [v for k, v in gt[unit].items() if fam_of[k] == fam]
                if g:
                    const[fam] = tmra([(st.median(g), x) for x in g], fam)
            for mk, _ in PR.MODELS:
                d = PR.find_parsed(seq, "full", mk)
                if not d:
                    continue
                P = preds(d)
                for fam in FAMS:
                    pr = [(p, gt[unit][k]) for k, p in P.items()
                          if fam_of.get(k) == fam and k in gt[unit]]
                    if len(pr) < 5 or fam not in const:
                        continue
                    dy = tmra(pr, fam) - const[fam]
                    pts[unit].append((const[fam], dy, mk, view))
                    detail[unit].append((const[fam], dy, mk, view, seq, fam))
    if not pts["v1"]:
        return print("  ! fig2: 没有可用格, 略过")

    # 明细落盘: 正文的 "N 格里 M 格高于常数基线" 必须从这里数, 不能凭旧文件。
    # (2026-09-08: 旧的 _const_tally.json 是 9/6 的遗留, 只有 50 格 —— 那时公开 7B 还差 3 条片段;
    #  当时写它的分支后来被删掉, 于是图和正文各说各话。现在图与明细由同一次计算写出。)
    tally = {u: [(seq_of, view, fam, mk, round(dy, 1), round(x, 1))
                 for (x, dy, mk, view, seq_of, fam) in rows]
             for u, rows in detail.items()}
    if refresh_tally:
        with io.open(os.path.join(CD, "_const_tally.json"), "w", encoding="utf-8") as f:
            json.dump(tally, f, ensure_ascii=False)
    else:
        frozen_tally = jload(os.path.join(CD, "_const_tally.json"))
        if frozen_tally != {u: [list(row) for row in rows] for u, rows in tally.items()}:
            raise ValueError("Recomputed constant tally differs from the frozen result; review before refreshing")
    for u in ("v1", "v3"):
        n_above = sum(1 for r in tally[u] if r[4] > 0)
        print(f"  fig2 tally {u}: {len(tally[u])} 格, {n_above} 格高于常数基线")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, unit, title in zip(axes, ("v1", "v3"),
                               ("v1 (box-height heuristic)", "v3 (court homography)")):
        for x, y, mk, view in pts[unit]:
            ax.plot(x, y, "o" if view == "top" else "^", ms=4.6, color=CLR[mk],
                    alpha=.85, mew=0)
        ax.axhline(0, color="#333", lw=.8)
        ax.set_xlabel("test-label median constant T-MRA of that clip")
        ax.set_title(f"{title} - {len(pts[unit])} cells", fontsize=9)
        ax.grid(axis="y", color="#e6eae8", lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("model T-MRA - constant baseline")
    handles = [Line2D([], [], marker="o", ls="", color=CLR[m], label=LBL[m])
               for m in ["sft", "grpo", "base", "qwen25vl7b"]]
    handles += [Line2D([], [], marker="o", ls="", color="#555", label="overhead"),
                Line2D([], [], marker="^", ls="", color="#555", label="side")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, -.13), fontsize=7.5)
    fig.suptitle("Model advantage over a test-distribution constant depends on the scoring convention",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig2_constant_vs_gt")


# ---------------------------------------------------------------- fig 3
def fig_multiview(out):
    d = jload(os.path.join(CD, "courtdyn_hm3_findings.json"))
    if not d:
        return print("  ! fig3: 缺 hm3 findings, 略过")
    groups = [("basketball1/split1", "hm3_basketball1_split1_camera_"),
              ("basketball1/split2", "hm3_basketball1_split2_camera_"),
              ("basketball2", "hm3_basketball2_camera_")]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.4, 2.9), sharey=True)
    any_model = False
    for ax, (title, pre) in zip(axes, groups):
        cams = sorted(k for k in d["seqs"] if k.startswith(pre))
        xs = list(range(len(cams)))
        for fam, mark in zip(FAMS, ("s", "D")):
            ys = [(d["seqs"][c].get("rho_px_world") or {}).get(fam) for c in cams]
            if any(v is not None for v in ys):
                ax.plot(xs, ys, mark + "-", color="#444", lw=1.1, ms=4.2,
                        label=f"rho(pixel, world), {fam}" if title.endswith("split1") else None)
        for mk in ("sft", "grpo"):
            for fam, mark in zip(FAMS, ("o", "v")):
                ys = [((d["seqs"][c].get("cells") or {}).get(f"{mk}@full@{fam}") or {}).get("rho")
                      for c in cams]
                if any(v is not None for v in ys):
                    any_model = True
                    ax.plot(xs, ys, mark + "--", color=CLR[mk], lw=1, ms=4,
                            label=f"{LBL[mk]}, {fam}" if title.endswith("split1") else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([c.rsplit("_", 1)[-1] for c in cams], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="#b9bfbb", lw=.7, ls=(0, (3, 3)))
        ax.set_xlabel("camera")
    axes[0].set_ylabel("Spearman rho")
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.22), fontsize=7.5)
    fig.suptitle("Human-M3: pixel-world coupling and model rank agreement by camera",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig3_multiview" + ("" if any_model else "_gt_only"))


# ---------------------------------------------------------------- fig 4
def fig_zoom(out):
    """zoom2 是两层嵌套: clip -> "model@fam" -> 该格。每个 model@fam 组内是 4 条片段各一个中位比。

    注意两处曾经踩过的坑:
      1. 早先这里只迭代了一层 (拿 clip 字典去 .get 比值), rows 永远为空, 于是每次都静默
         打印 "zoom2 里没有比值, 略过" —— 图 4 因此从来没被生成过。
      2. 汇总脚本写的字段是 r_median / r_q1 / r_q3, 不是 ratio_median。
    base 4B 不画: 它的 speed 臂可解析比值只有 n=7..11, 中位数 12.0 纯属小样本噪声 (见 §一 E2-B)。
    """
    d = jload(os.path.join(CD, "courtdyn_t27_findings.json"))
    if not d or not d.get("zoom2"):
        return print("  ! fig4: T27 还没汇总, 略过 (队列跑完后重跑本脚本)")
    z = d["zoom2"]
    clips = list(z)
    groups = [(m, f) for m in ("sft", "grpo", "qwen25vl7b") for f in FAMS]
    rows = []
    for m, f in groups:
        pts = []
        for c in clips:
            v = (z.get(c) or {}).get(f"{m}@{f}") or {}
            if v.get("r_median") is not None and (v.get("n_ratio") or 0) >= 30:
                pts.append((c, v["r_median"]))
        if pts:
            rows.append((m, f, pts))
    if not rows:
        return print("  ! fig4: zoom2 里没有可用比值 (检查 r_median / n_ratio 字段), 略过")

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for i, (m, f, pts) in enumerate(rows):
        ys = [y for _, y in pts]
        med = st.median(ys)
        ax.hlines(med, i - .28, i + .28, color=CLR[m], lw=2.2, zorder=3)
        for c, y in pts:
            side = "side" in c
            ax.plot(i + (0.13 if side else -0.13), y, marker="s" if side else "o",
                    ms=4.6, mfc="none" if side else CLR[m], mec=CLR[m], mew=1.1,
                    ls="none", zorder=4)
    for y, ls, clr, lbl in ((1.0, "--", "#555", "1.0  unchanged prediction reference"),
                            (2.0, ":", "#c8471f", "2.0  doubled prediction reference")):
        ax.axhline(y, color=clr, lw=.9, ls=ls, zorder=1)
        ax.text(len(rows) - .45, y, "  " + lbl, va="bottom", ha="right", fontsize=7.2, color=clr)
    # 红线 12: 公开 7B 的 1.00 是恒定输出 1.5 造成的退化值, 不能读成 "用了尺子"。
    # 图上必须标出来, 否则读者会照着判据把它读反。
    q = [i for i, (m, _, _) in enumerate(rows) if m == "qwen25vl7b"]
    if q:
        ax.axvspan(min(q) - .45, max(q) + .45, color="#999", alpha=.10, zorder=0)
        ax.annotate("A ratio of 1 alone does not establish scale use;\nconstant outputs can also produce it",
                    xy=((min(q) + max(q)) / 2, 0.52), ha="center", va="center",
                    fontsize=6.6, color="#555")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{LBL[m]}\n{f}" for m, f, _ in rows], fontsize=7.2)
    ax.set_ylabel("median paired pred(zoom2) / pred(full)")
    ax.set_ylim(0.4, 2.25)
    ax.set_title("2x zoom: pixel displacement doubles, world quantity unchanged\n"
                 "clip-level ratios for the tested models", fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="#444", mec="#444", ms=4.6,
                              label="top-view clip"),
                       Line2D([], [], marker="s", ls="none", mfc="none", mec="#444", ms=4.6,
                              label="side-view clip"),
                       Line2D([], [], color="#444", lw=2.2, label="median over clips")],
              fontsize=7, frameon=False, loc="upper left", ncol=3, columnspacing=1.1)
    fig.tight_layout()
    save(fig, out, "fig4_zoom_ratio")


def fig_ruler(out):
    """fig5 (T28): clip-level scale cue and pixel-unit question variants.

    左: 同题 ruler/full 预测比的中位数, 参考线 1.0 仅表示中位比为 1。
    右: pred(像素口径)/pred(米口径), 对数轴; 若模型真的在做单位换算, 该比值应落在 K≈27 那条带上,
        实测却挤在 1 附近 —— 两者差一个数量级, 这是本篇最干净的一格。
    base 4B 不画: ruler 臂比值 1.88/12.00 来自 n_ratio 仅个位数的可解析预测, 且它 ρ 变负 (见 §一 E1b-T28)。
    """
    d = jload(os.path.join(CD, "courtdyn_t28_findings.json"))
    if not d or not d.get("seqs"):
        return print("  ! fig5: T28 还没汇总, 略过")
    seqs = d["seqs"]
    clips = list(seqs)
    groups = [(m, f) for m in ("sft", "grpo", "tsft", "cdnative") for f in FAMS]
    Ks = [seqs[c].get("k_px_per_m") for c in clips if seqs[c].get("k_px_per_m")]
    if not Ks:
        return print("  ! fig5: 缺 k_px_per_m, 略过")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, (arm, getter, title, ylab) in zip(axes, (
            ("ruler", lambda cell: cell.get("ratio"),
             "(a) provide a clip-level scale cue", "median paired pred(ruler) / pred(full)"),
            ("pxunit", lambda cell: cell.get("ratio_to_full"),
             "(b) ask in pixels instead of metres", "median paired pred(pixels) / pred(metres)"))):
        drawn = 0
        for i, (m, f) in enumerate(groups):
            ys = []
            for c in clips:
                cell = (seqs[c].get("cells") or {}).get(f"{m}@{arm}@{f}") or {}
                v = getter(cell)
                if v is not None:
                    ys.append(v)
            if not ys:
                continue
            drawn += 1
            ax.hlines(st.median(ys), i - .28, i + .28, color=CLR[m], lw=2.2, zorder=3)
            for j, y in enumerate(ys):
                ax.plot(i + (j - (len(ys) - 1) / 2) * .22, y, marker="o", ms=4.6,
                        mfc=CLR[m], mec=CLR[m], ls="none", zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f for m, f in groups], fontsize=7)
        for group_index, model in enumerate(("sft", "grpo", "tsft", "cdnative")):
            ax.text(group_index * 2 + .5, -.17, LBL[model],
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        ax.set_title(title, fontsize=8.8)
        ax.set_ylabel(ylab, fontsize=7.8)
        ax.axhline(1.0, color="#555", lw=.9, ls="--", zorder=1)
        if arm == "ruler":
            ax.set_ylim(.6, 1.5)
            ax.text(len(groups) - .5, 1.0, "  1.0  paired ratio reference",
                    ha="right", va="bottom", fontsize=7, color="#555")
        else:
            ax.set_yscale("log")
            ax.set_ylim(.5, 60)
            ax.axhspan(min(Ks), max(Ks), color="#c8471f", alpha=.13, zorder=0)
            ax.text(len(groups) - .5, sum(Ks) / len(Ks), "  K = 27  rendered-coordinate reference",
                    ha="right", va="center", fontsize=7, color="#c8471f")
            ax.set_yticks([1, 2, 5, 10, 27, 50])
            ax.set_yticklabels(["1", "2", "5", "10", "27", "50"], fontsize=7)
        if not drawn:
            return print(f"  ! fig5: {arm} 臂没有可用比值, 略过")
    fig.suptitle("Prediction changes under an explicit scale cue and an answer-unit swap",
                 fontsize=9.5, y=1.005)
    fig.text(.5, .01, "Each dot represents one overhead clip; horizontal marks show the median over clips",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .02, 1, 1))
    save(fig, out, "fig5_ruler_and_units")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper2", "figs"))
    ap.add_argument("--refresh-constant-tally", action="store_true",
                    help="Explicitly regenerate _const_tally.json; default verifies and preserves it")
    args = ap.parse_args()
    print(f"[figs] -> {args.out}")
    for fn in (fig_viewpoint, fig_constant, fig_multiview, fig_zoom, fig_ruler):
        try:
            if fn == fig_constant:
                fn(args.out, refresh_tally=args.refresh_constant_tally)
            else:
                fn(args.out)
        except Exception as e:
            print(f"  ! {fn.__name__} 失败: {e!r}")


if __name__ == "__main__":
    main()
