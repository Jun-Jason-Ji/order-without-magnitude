#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure: unit response against rank agreement, every adapter cell, every backbone.

(a) metre-only adapters: x = metre Spearman rho, y = paired cm/m ratio R (log).
(b) mixed-unit adapters: x = rho of the centimetre answers vs the reference, y = R.
Exact conversion is R = 100.  Degenerate cells (<= 2 distinct answers in the arm that
defines the point) are drawn hollow; cells whose rho is undefined are omitted from the
scatter and counted in the caption instead.

Sources (all frozen result files, no new numbers):
  Qwen3.5-4B   paper2/manuscript_v2/tables/seed_variability.tex (3 pools x 3 seeds, Q2)
               results/courtdyn/m1_mixunit (via tools/make_paper2_v2_tables.m1_block)
  Qwen2.5-VL-3B / SmolVLM2-2.2B  results/courtdyn/backbone2_r2/verdict.json (round 2)
  InternVL3-2B results/courtdyn/backbone3_internvl/verdict.json
Writes paper2/manuscript_v2/figs/fig_units_vs_rank.pdf (+ .png) and a JSON of the points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_paper2_v2_tables as T                                 # noqa: E402

CD = ROOT / "results" / "courtdyn"
MS = ROOT / "paper2" / "manuscript_v2"
OUT = MS / "figs"

# categorical slots 1-4 of the validated default palette, fixed order; marker = second channel
STYLE = {"Qwen3.5-4B": ("#2a78d6", "o"), "Qwen2.5-VL-3B": ("#eb6834", "s"),
         "InternVL3-2B": ("#1baf7a", "D"), "SmolVLM2-2.2B": ("#eda100", "^"),
         "Pixtral-12B": ("#e87ba4", "v"), "Idefics3-8B": ("#8a63d2", "P"),
         "Gemma3-12B": ("#3b3b3b", "X")}
INK, MUTED = "#1f1f1e", "#6f6e69"


def seed_points():
    pts = []
    for line in (MS / "tables" / "seed_variability.tex").read_text(encoding="utf-8").splitlines():
        cols = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        if len(cols) != 10 or not cols[2][:1].isdigit():
            continue
        for fam, (i_rho, i_r) in (("speed", (2, 4)), ("path", (6, 8))):
            pts.append(dict(backbone="Qwen3.5-4B", family=fam, x=float(cols[i_rho]),
                            R=float(cols[i_r]), degenerate=False))
    return pts


def verdict_points(path, key, arm):
    rep = json.loads(path.read_text(encoding="utf-8"))["backbones"][key]
    pts = []
    for cell, c in rep["cells"].items():
        a = c[arm]
        x = a["rho_m"] if arm == "v3_E" else a["posthoc_rho_cm"]
        deg = (a["distinct_m"] <= 2 or a["distinct_cm"] <= 2) if arm == "v3_E" else a["distinct_cm"] <= 2
        pts.append(dict(cell=cell, family=cell.split("/")[1], x=x, R=a["R"], degenerate=deg))
    return pts


def main() -> int:
    names = {"qwen25vl3b": "Qwen2.5-VL-3B", "smol": "SmolVLM2-2.2B"}
    left = seed_points()
    right = []
    for key, name in names.items():
        for arm, dest in (("v3_E", left), ("mix_E", right)):
            dest += [dict(p, backbone=name) for p in verdict_points(CD / "backbone2_r2" / "verdict.json", key, arm)]
    for arm, dest in (("v3_E", left), ("mix_E", right)):
        dest += [dict(p, backbone="InternVL3-2B")
                 for p in verdict_points(CD / "backbone3_internvl" / "verdict.json", "internvl3_2b", arm)]
    # A100 backbones: every backbone present in the round-4 verdict and in STYLE
    r4 = CD / "backbone4_a100" / "verdict.json"
    a100_names = {"pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B", "gemma3_12b": "Gemma3-12B"}
    if r4.is_file():
        for key in json.loads(r4.read_text(encoding="utf-8"))["backbones"]:
            if a100_names.get(key) in STYLE:
                for arm, dest in (("v3_E", left), ("mix_E", right)):
                    dest += [dict(p, backbone=a100_names[key]) for p in verdict_points(r4, key, arm)]
    # SmolVLM2 failed the preregistered stopping rule (degenerate at the 4-epoch cap):
    # its cells are descriptive only, so every one is drawn hollow
    for p in left + right:
        if p["backbone"] == "SmolVLM2-2.2B":
            p["degenerate"] = True
    for cell, seeds in T.m1_block()["cells"].items():
        for seed, c in seeds.items():
            right.append(dict(backbone="Qwen3.5-4B", cell=f"{cell}/s{seed}", family=cell.split("/")[1],
                              x=c["rho_cm"], R=c["R"], degenerate=c["degenerate"]))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    omitted = {}
    for ax, pts, xlabel, title in (
            (axes[0], left, r"Metre answers: Spearman $\rho$ with reference", "(a) Metre-only adapters"),
            (axes[1], right, r"Centimetre answers: Spearman $\rho_{\mathrm{cm}}$ with reference",
             "(b) Mixed-unit adapters")):
        ax.axhspan(10, 1000, color="#eef4fc", zorder=0, lw=0)
        ax.axhline(100, color=MUTED, lw=1, ls="--", zorder=1)
        ax.axhline(1, color=MUTED, lw=1, ls=":", zorder=1)
        for name, (col, mk) in STYLE.items():
            sel = [p for p in pts if p["backbone"] == name and p["x"] is not None and p["R"]]
            omitted[(title, name)] = sum(1 for p in pts if p["backbone"] == name and p["x"] is None)
            for deg in (False, True):
                q = [p for p in sel if p["degenerate"] == deg]
                if q:
                    ax.scatter([p["x"] for p in q], [p["R"] for p in q], marker=mk, s=34, zorder=3,
                               facecolors="none" if deg else col, edgecolors=col if deg else "white",
                               linewidths=1.2 if deg else 0.8,
                               label=(name + (" (degenerate at cap)" if name.startswith("Smol") else ""))
                               if (ax is axes[0] and (not deg or name.startswith("Smol"))) else None)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 600)
        ax.set_xlim(-0.35, 0.9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ax is axes[0]:           # shared y axis: label the reference lines once
            ax.text(-0.33, 108, "exact conversion ($R=100$)", ha="left", va="bottom", fontsize=7, color=MUTED)
            ax.text(-0.33, 1.07, "unit ignored ($R=1$)", ha="left", va="bottom", fontsize=7, color=MUTED)
    axes[0].set_ylabel("cm/m answer ratio $R$ (log)")
    # every backbone appears in (a) filled; add a hollow key for the degenerate convention
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(axes[0].scatter([], [], marker="o", s=34, facecolors="none", edgecolors=MUTED))
    labels.append("degenerate (≤2 distinct answers)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_units_vs_rank.pdf")
    fig.savefig(OUT / "fig_units_vs_rank.png", dpi=300)
    (OUT / "fig_units_vs_rank_points.json").write_text(json.dumps(
        {"metre_only": left, "mixed_unit": right,
         "omitted_rho_undefined": {f"{t}|{n}": k for (t, n), k in omitted.items() if k}}, indent=1),
        encoding="utf-8")
    print("points:", len(left), len(right), "omitted (rho undefined):",
          {f"{t[:3]}|{n}": k for (t, n), k in omitted.items() if k})
    return 0


if __name__ == "__main__":
    sys.exit(main())
