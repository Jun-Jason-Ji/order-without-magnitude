#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate the SoccerNet second-venue cells (tools/run_soccer_venue.py).

EXPLORATORY.  No decision rule was recorded for the soccer venue before it was run
(unlike M1 and the second backbone), so this script reports statistics and forces
no verdict; nothing here may be written up as "replicates" / "fails to replicate"
without saying that the criterion is post hoc.

Per adapter x match x family, from a100.analyze.cell_report (the same code, and the
same exact constant-oracle, as every CourtDyn cell):
  metre arm      T-MRA, constant-oracle T-MRA and the margin, Spearman rho,
                 distinct answers, degenerate flag (<= 2 distinct)
  unit arms      paired cm/m ratio (exact conversion = 100) and px/m ratio
                 (a pixel-reading model would give the match's K px/m)
  first frame    rho(full) - rho(first frame x4) on the common items

Per-item predictions stay in the private (SoccerNet NDA) directory; only these
aggregates are written into the repository:
  results/courtdyn/soccer_venue/summary.json and summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100.analyze import cell_report, read_cell  # noqa: E402

PRIVATE_EVAL = Path(r"E:\datasets\SoccerNet\derived_private\courtdyn_venue\eval")
OUT = ROOT / "results" / "courtdyn" / "soccer_venue"
ADAPTERS = ["qwen35_v3", "qwen35_mixunit", "smol_v3", "smol_mixunit"]


def main() -> int:
    receipt = json.loads((OUT / "build_receipt.json").read_text(encoding="utf-8"))
    seqs = {s["name"]: s for s in receipt["sequences"]}
    report = {"status": "exploratory: no preregistered decision rule for this venue",
              "protocol": receipt["protocol"], "sequences": seqs, "cells": {}}
    rows = []
    for adapter in ADAPTERS:
        for seq in seqs:
            full = read_cell(PRIVATE_EVAL / f"{adapter}_{seq}_full")
            static = read_cell(PRIVATE_EVAL / f"{adapter}_{seq}_static4")
            if full is None:
                report["cells"][f"{adapter}/{seq}"] = None
                continue
            cell = cell_report(full, static)
            cell.pop("dir", None)
            cell["config"].pop("model_path", None)
            report["cells"][f"{adapter}/{seq}"] = cell
            for family, block in cell["families"].items():
                m = block.get("m_height", {})
                cm = block.get("ratio_cm_height_over_m") or {}
                px = block.get("ratio_px_height_over_m") or {}
                s4 = block.get("static4") or {}
                rows.append((adapter, seq, family, m, cm.get("median"), px.get("median"),
                             seqs[seq]["K_px_per_m"], s4.get("delta_rho"),
                             block.get("cm_height", {}).get("n_distinct")))
    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                      encoding="utf-8")

    f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    lines = ["# SoccerNet second venue - aggregates (EXPLORATORY, no preregistered rule)", "",
             "Zero-shot transfer: no adapter saw soccer in training. Exact cm/m conversion = 100; "
             "a pixel-reading model would give px/m = K. `*` = degenerate (<= 2 distinct answers).", "",
             "| adapter | match | family | metre T-MRA | constant | margin | rho | distinct "
             "| cm/m | px/m (K) | delta rho first-frame |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for adapter, seq, family, m, cm, px, k, drho, cm_distinct in rows:
        star = "*" if m.get("degenerate") else ""
        cm_star = "*" if cm_distinct is not None and cm_distinct <= 2 else ""
        lines.append(f"| {adapter} | {seq} | {family} | {f(m.get('tmra'), 1)} | "
                     f"{f(m.get('constant_tmra'), 1)} | {f(m.get('tmra_minus_constant'), 1)} | "
                     f"{f(m.get('rho'))}{star} | {m.get('n_distinct')} | {f(cm, 1)}{cm_star} | "
                     f"{f(px)} ({k:.1f}) | {f(drho)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
