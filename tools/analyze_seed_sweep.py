#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Seed variability on Q2 (the editor's "single random seed" point).

Reads, for each training pool (native, v1, v3) and seed (42 published; 43, 44 from
the local sweep in results/a100/seeds), the Q2 full and first-frame cells, and
reports per family: metre Spearman rho, metre T-MRA (all items, invalid = 0),
paired cm/m and px/m ratios, the first-frame drop in rho, and distinct answers.

For the v1-versus-v3 label contrast it states, per metric and family, whether the
two three-seed ranges overlap.  Non-overlap is the only case in which the paper
may describe a label difference as exceeding seed variation; an overlapping
difference is reported as within seed variation, whatever its single-seed sign.

Writes results/a100/seeds/seed_analysis.json and
paper2/manuscript/tables/seed_variability.tex (generated; do not edit by hand).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100.eval_controls import scalar_score, spearman      # noqa: E402

SEQ = "Q2_top_480-510"
PUBLISHED = ROOT / "results" / "courtdyn" / "revision_execution_20260912"
SWEEP = ROOT / "results" / "a100" / "seeds"
PUB_LABEL = {"native": "native_original", "v1": "eventholdout_v1_s42",
             "v3": "eventholdout_v3_s42"}
POOLS, SEEDS = ["native", "v1", "v3"], [42, 43, 44]
FAMILIES = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
TABLE = ROOT / "paper2" / "manuscript" / "tables" / "seed_variability.tex"


def cell_dir(pool, seed, mode):
    if seed == 42:
        return PUBLISHED / f"{PUB_LABEL[pool]}_{SEQ}_{mode}"
    return SWEEP / f"{pool}_s{seed}_{SEQ}_{mode}"


def load(directory, arm):
    path = directory / f"{arm}.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["category"], r["track"], tuple(r["window"])): r for r in rows}


def rho_of(cell, keys):
    parsed = [k for k in keys if cell[k]["prediction"] is not None]
    if len({cell[k]["prediction"] for k in parsed}) < 2:
        return None
    return spearman([cell[k]["prediction"] for k in parsed],
                    [cell[k]["reference_value"] for k in parsed])


def stats(pool, seed, family):
    full = cell_dir(pool, seed, "full")
    m, cm, px = load(full, "m_height"), load(full, "cm_height"), load(full, "px_height")
    s4 = load(cell_dir(pool, seed, "static4"), "m_height")
    keys = [k for k in m if k[0] == family]
    tmra = sum(scalar_score(m[k]["prediction"], m[k]["reference_value"],
                            m[k]["score_tolerance"], m[k]["score_floor"])
               if m[k]["prediction"] is not None else 0.0 for k in keys) / len(keys)

    def ratio(other):
        r = [other[k]["prediction"] / m[k]["prediction"] for k in keys
             if other[k]["prediction"] is not None and m[k]["prediction"] not in (None, 0)
             and m[k]["prediction"] > 0]
        return st.median(r) if r else None

    rho = rho_of(m, keys)
    control = rho_of(s4, keys) if s4 else None
    return {"rho": rho, "tmra": tmra, "cm_over_m": ratio(cm), "px_over_m": ratio(px),
            "control_rho": control,
            "delta_rho": (rho - control) if (rho is not None and control is not None) else None,
            "static4_present": s4 is not None,
            "distinct_metre": len({m[k]["prediction"] for k in keys
                                   if m[k]["prediction"] is not None})}


def main() -> int:
    cells = {}
    for pool in POOLS:
        for seed in SEEDS:
            for family, short in FAMILIES:
                cells[f"{pool}/s{seed}/{short}"] = stats(pool, seed, family)
    summary = {}
    for pool in POOLS:
        for _, short in FAMILIES:
            vals = [cells[f"{pool}/s{s}/{short}"] for s in SEEDS]
            summary[f"{pool}/{short}"] = {
                k: {"min": min(v[k] for v in vals), "max": max(v[k] for v in vals),
                    "mean": st.mean(v[k] for v in vals)}
                for k in ("rho", "tmra", "cm_over_m", "px_over_m")}
    contrast = {}
    for _, short in FAMILIES:
        for metric in ("rho", "tmra"):
            a, b = summary[f"v1/{short}"][metric], summary[f"v3/{short}"][metric]
            overlap = a["max"] >= b["min"] and b["max"] >= a["min"]
            contrast[f"{short}/{metric}"] = {
                "v1": [a["min"], a["max"]], "v3": [b["min"], b["max"]],
                "mean_diff_v3_minus_v1": b["mean"] - a["mean"],
                "ranges_overlap": overlap,
                "reading": ("within seed variation" if overlap else
                            ("v3 above v1 on every seed" if b["min"] > a["max"]
                             else "v3 below v1 on every seed"))}
    deltas = [c["delta_rho"] for c in cells.values() if c["delta_rho"] is not None]
    report = {"seq": SEQ, "cells": cells, "summary": summary, "label_contrast": contrast,
              "ratio_ranges": {
                  "cm_over_m": [min(c["cm_over_m"] for c in cells.values()),
                                max(c["cm_over_m"] for c in cells.values())],
                  "px_over_m": [min(c["px_over_m"] for c in cells.values()),
                                max(c["px_over_m"] for c in cells.values())],
                  "rho": [min(c["rho"] for c in cells.values()),
                          max(c["rho"] for c in cells.values())]},
              "first_frame": {"defined_cells": len(deltas),
                              "all_positive": all(d > 0 for d in deltas),
                              "delta_rho_range": [min(deltas), max(deltas)]}}
    (SWEEP / "seed_analysis.json").write_text(json.dumps(report, indent=1) + "\n",
                                              encoding="utf-8")
    TABLE.write_text(render(cells), encoding="utf-8")
    for k, v in contrast.items():
        print(f"{k:12s} v1 {v['v1'][0]:.3f}-{v['v1'][1]:.3f}  v3 {v['v3'][0]:.3f}-{v['v3'][1]:.3f}"
              f"  -> {v['reading']}")
    print("ratio ranges:", {k: [round(x, 2) for x in v] for k, v in report["ratio_ranges"].items()})
    print("first frame:", report["first_frame"])
    return 0


def render(cells) -> str:
    f = lambda x, p=2: "---" if x is None else f"{x:.{p}f}"
    name = {"native": "Original", "v1": "v1 labels", "v3": "v3 labels"}
    rows = []
    for pool in POOLS:
        for seed in SEEDS:
            s, p = cells[f"{pool}/s{seed}/speed"], cells[f"{pool}/s{seed}/path"]
            rows.append(f"{name[pool] if seed == 42 else ''} & {seed} & "
                        f"{f(s['rho'])} & {f(s['tmra'], 1)} & {f(s['cm_over_m'])} & "
                        f"{f(s['delta_rho'])} & {f(p['rho'])} & {f(p['tmra'], 1)} & "
                        f"{f(p['cm_over_m'])} & {f(p['delta_rho'])} " + r"\\")
        if pool != POOLS[-1]:
            rows.append(r"\addlinespace")
    caption = (r"\caption{Seed variability on the event-disjoint Q2 clip. Seed 42 is the "
               r"published checkpoint; seeds 43 and 44 were retrained with identical "
               r"hyperparameters. $\rho$ and T-MRA use metre prompts against exact v3 "
               r"references; cm/m is the median paired prediction ratio (exact factor "
               r"100); $\Delta\rho$ is full minus first-frame control, --- where the "
               r"control is undefined (constant predictions) or was not run.}")
    return "\n".join([
        "% Generated by tools/analyze_seed_sweep.py -- do not edit by hand.",
        r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        caption, r"\label{tab:seeds}", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{4}{c}{speed} & \multicolumn{4}{c}{path} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"Pool & Seed & $\rho$ & T-MRA & cm/m & $\Delta\rho$ & $\rho$ & T-MRA & cm/m & $\Delta\rho$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


if __name__ == "__main__":
    sys.exit(main())
