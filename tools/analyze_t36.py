#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Describe the T36 zero-shot backbone sweep (a100/exp_backbones.py, run by tools/a100_t36.py).

No decision rule exists for T36 (results/courtdyn/backbone4_a100/PREREGISTRATION.json, item (f)):
this is a descriptive table. Definitions are those of tools/analyze_backbone_replication.py:
  distinct   number of distinct parsed answers in an arm; <= 2 = DEGENERATE (kept separate)
  rho        Spearman(parsed answers, reference), None if < 2 distinct or < 3 parsed
  R_cm       median itemwise cm/m answer ratio (paired by item identity; reference ratio 100)
  R_px       median itemwise px/m answer ratio (reference ratio = ruler scale, reported)
  static4    metre rho with the first frame repeated x4, on items parsed in both conditions
A cell {model x clip x family} is LIVE for the unit reading only if both the m and cm arms have
>= 3 distinct answers.

Usage: python tools/analyze_t36.py [--src <.../results/a100/backbones>]
Writes results/courtdyn/t36/summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100.analyze import paired_ratio, read_cell                     # noqa: E402
from analyze_backbone_replication import distinct, fam, rho, static_drop  # noqa: E402

OUT = ROOT / "results" / "courtdyn" / "t36"
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
FAMILIES = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]


def latest_src():
    imports = sorted((ROOT / "results" / "a100_import").glob("*/extracted/results/a100/backbones"))
    if not imports:
        raise SystemExit("no imported A100 T36 results under results/a100_import/")
    return imports[-1]


def parse_rate(rows):
    return sum(r["prediction"] is not None for r in rows) / len(rows) if rows else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def arm_stats(rows):
    return {"n": len(rows), "parse": parse_rate(rows), "distinct": distinct(rows), "rho": rho(rows),
            "median_pred": median([r["prediction"] for r in rows]),
            "median_ref": median([r["reference_value"] for r in rows])}


def analyse(src):
    models = sorted({d.name.split("_Q")[0] for d in src.iterdir() if d.is_dir() and "_Q" in d.name})
    out = {"source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
           "models": {}}
    for model in models:
        entry = {"cells": [], "arithmetic": None}
        for seq in SEQS:
            full = read_cell(src / f"{model}_{seq}_full")
            static = read_cell(src / f"{model}_{seq}_static4")
            if full is None:
                entry["cells"].append({"seq": seq, "missing": True})
                continue
            if full.get("arithmetic"):
                entry["arithmetic"] = full["arithmetic"]
            for family, short in FAMILIES:
                arms = {k: fam(full["arms"].get(k, []), family) for k in ("m_height", "cm_height", "px_height")}
                s = {k: arm_stats(v) for k, v in arms.items()}
                rcm = paired_ratio(arms["cm_height"], arms["m_height"])
                rpx = paired_ratio(arms["px_height"], arms["m_height"])
                cell = {"seq": seq, "family": short, "arms": s,
                        "R_cm": rcm["median"] if rcm else None, "n_pairs_cm": rcm["n"] if rcm else 0,
                        "R_px": rpx["median"] if rpx else None,
                        "R_px_reference": rpx.get("reference_median") if rpx else None,
                        "degenerate_m": s["m_height"]["distinct"] <= 2,
                        "live_unit": s["m_height"]["distinct"] >= 3 and s["cm_height"]["distinct"] >= 3}
                if static is not None:
                    cell["static4"] = static_drop(full, static, family)
                entry["cells"].append(cell)
        # A model that emits no parseable answer in any arm or arithmetic item is a harness failure,
        # not a zero-shot reading (qwen35-4b-bf16: the generic bf16 loader in a100/eval_controls.py
        # does not pass enable_thinking=False, so Qwen3.5 thinks until the 128-token cap).
        cells = [c for c in entry["cells"] if not c.get("missing")]
        entry["invalid"] = bool(cells) and all(
            a["parse"] == 0 for c in cells for a in c["arms"].values()) and             (entry["arithmetic"] or {}).get("correct", 0) == 0
        out["models"][model] = entry
    return out


def fmt(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def markdown(res):
    lines = ["# T36 zero-shot backbone sweep - descriptive summary", "",
             f"Source: `{res['source']}`. No decision rule (backbone-4 prereg item (f)); "
             "degenerate arms (<= 2 distinct answers) are marked, not dropped. "
             "R_cm reference = 100; R_px reference shown per cell.", "",
             "| model | clip | fam | parse m/cm/px | distinct m/cm/px | rho m | rho cm | median m (ref) | R_cm | R_px (ref) | rho m static4 (full) | live |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, e in res["models"].items():
        for c in e["cells"]:
            if c.get("missing"):
                lines.append(f"| {model} | {c['seq']} | – | missing | | | | | | | | |")
                continue
            a = c["arms"]
            st = c.get("static4") or {}
            lines.append(
                f"| {model} | {c['seq'].split('_')[0]} | {c['family']} | "
                + "/".join(fmt(a[k]['parse']) for k in ('m_height', 'cm_height', 'px_height')) + " | "
                + "/".join(str(a[k]['distinct']) for k in ('m_height', 'cm_height', 'px_height')) + " | "
                f"{fmt(a['m_height']['rho'])} | {fmt(a['cm_height']['rho'])} | "
                f"{fmt(a['m_height']['median_pred'])} ({fmt(a['m_height']['median_ref'])}) | "
                f"{fmt(c['R_cm'], 1)} | {fmt(c['R_px'], 1)} ({fmt(c['R_px_reference'], 1)}) | "
                f"{fmt(st.get('rho_static4'))} ({fmt(st.get('rho_full'))}) | "
                f"{'INVALID' if e['invalid'] else 'yes' if c['live_unit'] else 'DEGEN'} |")
    bad = [m for m, e in res["models"].items() if e["invalid"]]
    if bad:
        lines += ["", f"INVALID (harness, not a model result): {', '.join(bad)} -- no parseable answer in any "
                  "arm or arithmetic item. For qwen35-4b-bf16 the generic bf16 loader "
                  "(a100/eval_controls.build_infer) does not pass enable_thinking=False, so Qwen3.5 "
                  "emits its thinking trace and is cut at 128 new tokens; the nf4 cell goes through the "
                  "frozen eval.run_bench loader, which disables thinking."]
    lines += ["", "## Arithmetic (12 conversion items, Q2 full cell)", "", "| model | correct |", "|---|---|"]
    for model, e in res["models"].items():
        a = e["arithmetic"]
        lines.append(f"| {model} | {a['correct']}/{a['n']} |" if a else f"| {model} | – |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path)
    args = ap.parse_args()
    src = (args.src or latest_src()).resolve()
    res = analyse(src)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    md = markdown(res)
    (OUT / "summary.md").write_text(md, encoding="utf8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
