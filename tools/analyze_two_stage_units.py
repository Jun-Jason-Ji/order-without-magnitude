#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score the two-stage unit experiment against its preregistered predictions.

Reads results/courtdyn/two_stage_units/<arm>/summary.json (written by
run_two_stage_units.py) and PREREGISTRATION.json, and decides every prediction
clause mechanically, per checkpoint and family, with the thresholds exactly as
written before stage 2 ran:

  P1  px, every checkpoint:   |ratio/27.498 - 1| <= 0.05  and  ceiling - two_stage <= 5
  P2  cm, three adapters:     |ratio/100 - 1|    >  0.05  and  ceiling - two_stage >= 20
  P3  cm, unadapted backbone: |ratio/100 - 1|    <= 0.05  and  ceiling - two_stage <= 5
      (the T-MRA clause is flagged uninformative: amendment 2, recorded before
       stage 2, found the backbone's stage-1 metre answers degenerate)

A clause that fails is reported as failed.  Nothing here is tuned to the data.

Writes
  results/courtdyn/two_stage_units/verdicts.json
  paper2/manuscript/tables/two_stage.tex      (generated; do not edit by hand)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "courtdyn" / "two_stage_units"
TABLE = ROOT / "paper2" / "manuscript" / "tables" / "two_stage.tex"
ARMS = ["base", "native", "v1", "v3"]
LABEL = {"base": "Base (no adapter)", "native": "CourtDyn-native",
         "v1": "Event-holdout v1", "v3": "Event-holdout v3"}
FAMILIES = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]


def clause(name, passed, detail):
    return {"clause": name, "passed": bool(passed), "detail": detail}


def verdicts(summaries):
    out = []
    for arm, s in summaries.items():
        for family, short in FAMILIES:
            for target in ("px", "cm"):
                c = s["cells"][f"{target}/{family}"]
                ratio = c["two_stage_median_ratio"]
                need = c["required_ratio"]
                gap = c["exact_conversion"]["tmra"] - c["two_stage"]["tmra"]
                rel = abs(ratio / need - 1)
                cell = {"arm": arm, "family": short, "target": target,
                        "ratio": ratio, "required": need, "ceiling_gap": gap}
                if target == "px":
                    cell["prediction"] = "P1"
                    cell["clauses"] = [
                        clause("ratio within 5% of K", rel <= 0.05, f"{ratio:.3f} vs {need}"),
                        clause("T-MRA within 5 of ceiling", gap <= 5, f"gap {gap:.1f}")]
                elif arm == "base":
                    cell["prediction"] = "P3"
                    cell["clauses"] = [
                        clause("ratio within 5% of 100", rel <= 0.05, f"{ratio:.3f}"),
                        clause("T-MRA within 5 of ceiling", gap <= 5,
                               f"gap {gap:.1f}; uninformative (degenerate stage 1)")]
                    cell["clauses"][1]["uninformative"] = True
                else:
                    cell["prediction"] = "P2"
                    cell["clauses"] = [
                        clause("ratio far from 100 (not within 5%)", rel > 0.05, f"{ratio:.3f}"),
                        clause("T-MRA at least 20 below ceiling", gap >= 20, f"gap {gap:.1f}")]
                cell["passed"] = all(x["passed"] for x in cell["clauses"])
                out.append(cell)
    tally = {}
    for p in ("P1", "P2", "P3"):
        cells = [c for c in out if c["prediction"] == p]
        tally[p] = {"cells": len(cells), "passed": sum(c["passed"] for c in cells),
                    "failed": [f"{c['arm']}/{c['family']}" for c in cells if not c["passed"]]}
    return out, tally


def render_table(summaries) -> str:
    rows = []
    for arm in ARMS:
        if arm not in summaries:
            continue
        s = summaries[arm]["cells"]
        for target, unit in (("px", "px"), ("cm", "cm")):
            parts = []
            for family, _ in FAMILIES:
                c = s[f"{target}/{family}"]
                parts.append(f"{c['direct']['tmra']:.1f} & {c['two_stage']['tmra']:.1f} & "
                             f"{c['exact_conversion']['tmra']:.1f}")
            ratio = "/".join(f"{s[f'{target}/{family}']['two_stage_median_ratio']:.1f}"
                             for family, _ in FAMILIES)
            name = LABEL[arm] if target == "px" else ""
            rows.append(f"{name} & {unit} & " + " & ".join(parts) + f" & {ratio} " + r"\\")
        if arm != ARMS[-1]:
            rows.append(r"\addlinespace")
    caption = (
        r"\caption{Two-stage unit answering, Q2. \emph{Direct}: the model is asked for "
        r"the target unit from the image. \emph{Two-stage}: its own metre answer from "
        r"the image is passed back verbatim in a text-only turn with the conversion "
        r"factor. \emph{Ceiling}: that metre answer converted exactly. All three are "
        r"T-MRA against the target arm's own references (rendered pixels; floor motion "
        r"$\times100$). Ratio is the median two-stage answer over the metre answer "
        r"(required: px $27.50$, cm $100$). The unadapted backbone's metre answers are "
        r"degenerate, so its row tests only the conversion step. Predictions were "
        r"recorded before the second stage ran (\ref{app:two_stage}).}")
    return "\n".join([
        "% Generated by tools/analyze_two_stage_units.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        caption,
        r"\label{tab:two_stage}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{speed T-MRA} & \multicolumn{3}{c}{path T-MRA} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Checkpoint & Unit & direct & two-stage & ceiling & direct & two-stage & ceiling "
        r"& ratio \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""])


def main() -> int:
    summaries = {}
    for arm in ARMS:
        path = RUNS / arm / "summary.json"
        if path.is_file():
            summaries[arm] = json.loads(path.read_text(encoding="utf-8"))
    if len(summaries) != len(ARMS):
        print(f"missing arms: {sorted(set(ARMS) - set(summaries))}")
        return 1
    prereg = json.loads((RUNS / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    cells, tally = verdicts(summaries)
    report = {"preregistration_sha256": prereg["predictions_sha256"],
              "preregistered_at_utc": prereg["written_at_utc"],
              "tally": tally, "cells": cells}
    (RUNS / "verdicts.json").write_text(json.dumps(report, indent=1) + "\n",
                                        encoding="utf-8")
    TABLE.write_text(render_table(summaries), encoding="utf-8")
    for p, t in tally.items():
        print(f"{p}: {t['passed']}/{t['cells']} cells pass"
              + (f"; FAILED {t['failed']}" if t["failed"] else ""))
    for c in cells:
        if not c["passed"]:
            print("  failed:", c["arm"], c["family"], c["target"],
                  [f"{x['clause']}: {x['detail']}" for x in c["clauses"] if not x["passed"]])
    print(f"-> {RUNS / 'verdicts.json'}\n-> {TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
