#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate the A100 sweeps into manuscript-ready tables.

Reads the per-cell jsonl written by a100/eval_controls.py and produces:

  $OUT/analysis/seeds.json        per-seed cells plus across-seed spread
  $OUT/analysis/backbones.json    per-model cells with degeneracy flags
  $OUT/analysis/summary.md        a readable digest
  paper2/manuscript/tables/seed_variability.tex
  paper2/manuscript/tables/backbone_units.tex

Two rules are enforced here rather than left to the writing stage, because
both were learned the hard way on this project:

1. A unit ratio near 1 is only evidence of "no conversion" when the model's
   answers vary. Cells with two or fewer distinct answers are reported in a
   separate block and never contribute to a quoted range.
2. A label difference is only reported against the seed spread of the same
   quantity. The seed table prints both, side by side, and the digest states
   plainly when the difference is inside the spread.

CPU only; it loads no model.

Usage
  python -m a100.analyze                 # both sweeps, whatever exists
  python -m a100.analyze --which seeds
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                                   # noqa: E402
from a100.eval_controls import scalar_score, spearman, _median  # noqa: E402

FAMILIES = ["dynamics_speed_player", "dynamics_path_player"]
SHORT = {"dynamics_speed_player": "speed", "dynamics_path_player": "path"}


# --------------------------------------------------------------------- io
def read_cell(directory):
    directory = Path(directory)
    config_path = directory / "run_config.json"
    if not (directory / "summary.json").is_file() or not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf8"))
    arms = {}
    for arm in config["arms"]:
        path = directory / (arm + ".jsonl")
        if not path.is_file():
            continue
        arms[arm] = [json.loads(line) for line in
                     path.read_text(encoding="utf8").splitlines() if line.strip()]
    arithmetic = None
    apath = directory / "arithmetic_results.json"
    if apath.is_file():
        rows = json.loads(apath.read_text(encoding="utf8"))
        arithmetic = dict(n=len(rows), correct=sum(r["correct"] for r in rows))
    return dict(dir=str(directory), config=config, arms=arms, arithmetic=arithmetic)


def item_key(row):
    window = row.get("window")
    return (row["category"], row.get("track"),
            tuple(window) if isinstance(window, list) else window)


# ---------------------------------------------------------------- metrics
def best_constant(rows):
    """Best single answer given the test labels: the constant-oracle baseline.

    This is deliberately an oracle -- it is chosen with knowledge of the test
    distribution and is not a deployable baseline. It exists so that a high
    T-MRA on a narrow answer distribution cannot be read as competence.
    """
    if not rows:
        return None, None
    # T-MRA as a function of a constant answer c is piecewise constant, and it
    # can only change where some item's tolerance band opens or closes:
    #     c = ref_i +/- (T_i + (1-theta) * max(|ref_i|, floor_i))
    # Enumerating those breakpoints and testing the midpoint of every interval
    # finds the exact maximum. Searching only the reference values instead
    # under-reports the oracle (69.21 rather than 69.64 on Q2 speed) and would
    # make every "beats the constant" margin here incomparable with the
    # published 2026-09-12 numbers.
    breaks = set()
    for r in rows:
        ref, tol = r["reference_value"], r["score_tolerance"]
        denominator = max(abs(ref), r["score_floor"])
        for i in range(10):
            width = tol + (1 - (0.5 + 0.05 * i)) * denominator
            breaks.add(ref - width)
            breaks.add(ref + width)
    ordered = sorted(breaks)
    candidates = [ordered[0], ordered[-1]]
    candidates += [(a + b) / 2.0 for a, b in zip(ordered, ordered[1:])]
    best, best_score = None, -1.0
    for candidate in candidates:
        if candidate < 0:
            continue
        score = sum(scalar_score(candidate, r["reference_value"],
                                 r["score_tolerance"], r["score_floor"])
                    for r in rows) / len(rows)
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def family_stats(rows):
    valid = [r for r in rows if r["prediction"] is not None]
    preds = [r["prediction"] for r in valid]
    refs = [r["reference_value"] for r in valid]
    scores = [scalar_score(r["prediction"], r["reference_value"],
                           r["score_tolerance"], r["score_floor"]) for r in valid]
    constant, constant_score = best_constant(rows)
    return dict(
        n=len(rows), parsed=len(valid), n_distinct=len(set(preds)),
        degenerate=len(set(preds)) <= 2,
        tmra=sum(scores) / len(rows) if rows else None,
        constant_tmra=constant_score, constant_value=constant,
        tmra_minus_constant=((sum(scores) / len(rows)) - constant_score)
        if rows and constant_score is not None else None,
        rho=spearman(preds, refs) if len(valid) >= 3 else None,
        prediction_median=_median(preds), reference_median=_median(refs))


def paired_ratio(numerator_rows, denominator_rows):
    """Median itemwise ratio, plus the median itemwise reference ratio.

    Pairing is by item identity, not by position, so a cell that stopped early
    or dropped an item cannot silently shift the pairing.
    """
    den = {item_key(r): r for r in denominator_rows}
    ratios, references = [], []
    for row in numerator_rows:
        other = den.get(item_key(row))
        if other is None:
            continue
        if row["prediction"] is None or other["prediction"] is None:
            continue
        if other["prediction"] <= 0:
            continue
        ratios.append(row["prediction"] / other["prediction"])
        if other["reference_value"]:
            references.append(row["reference_value"] / other["reference_value"])
    if not ratios:
        return None
    return dict(n=len(ratios), median=_median(ratios),
                q1=_median(sorted(ratios)[:len(ratios) // 2]),
                q3=_median(sorted(ratios)[(len(ratios) + 1) // 2:]),
                reference_median=_median(references) if references else None)


def cell_report(cell, static_cell=None):
    out = {"dir": cell["dir"], "config": {
        k: cell["config"].get(k) for k in
        ("label", "seq", "frame_mode", "precision", "model_path", "adapter")}}
    per_family = {}
    for family in FAMILIES:
        block = {}
        for arm, rows in cell["arms"].items():
            fam_rows = [r for r in rows if r["category"] == family]
            if fam_rows:
                block[arm] = family_stats(fam_rows)
        if "m_height" in cell["arms"]:
            base = [r for r in cell["arms"]["m_height"] if r["category"] == family]
            for arm in ("cm_height", "px_height", "cm_noheight", "px_noheight"):
                if arm in cell["arms"]:
                    rows = [r for r in cell["arms"][arm] if r["category"] == family]
                    ratio = paired_ratio(rows, base)
                    if ratio:
                        block[f"ratio_{arm}_over_m"] = ratio
            if static_cell and "m_height" in static_cell["arms"]:
                srows = [r for r in static_cell["arms"]["m_height"]
                         if r["category"] == family]
                keys = {item_key(r) for r in srows}
                common_full = [r for r in base if item_key(r) in keys
                               and r["prediction"] is not None]
                common_keys = {item_key(r) for r in common_full}
                common_static = [r for r in srows if item_key(r) in common_keys
                                 and r["prediction"] is not None]
                shared = ({item_key(r) for r in common_full}
                          & {item_key(r) for r in common_static})
                full_map = {item_key(r): r for r in common_full}
                static_map = {item_key(r): r for r in common_static}
                ordered = sorted(shared)
                rho_full = spearman([full_map[k]["prediction"] for k in ordered],
                                    [full_map[k]["reference_value"] for k in ordered])
                rho_static = spearman([static_map[k]["prediction"] for k in ordered],
                                      [static_map[k]["reference_value"] for k in ordered])
                block["static4"] = dict(
                    n_common=len(ordered), rho_full=rho_full, rho_static4=rho_static,
                    delta_rho=(rho_full - rho_static)
                    if (rho_full is not None and rho_static is not None) else None,
                    n_distinct_static4=len({static_map[k]["prediction"] for k in ordered}),
                    tmra_static4=family_stats(srows)["tmra"])
        per_family[SHORT[family]] = block
    out["families"] = per_family
    out["arithmetic"] = cell["arithmetic"]
    return out


# ----------------------------------------------------------------- sweeps
def collect(root, pattern="*"):
    cells = {}
    if not Path(root).is_dir():
        return cells
    for directory in sorted(Path(root).glob(pattern)):
        cell = read_cell(directory)
        if cell:
            cells[directory.name] = cell
    return cells


def analyse_seeds():
    cells = collect(C.OUT / "seeds")
    reports, index = {}, {}
    for name, cell in cells.items():
        if cell["config"]["frame_mode"] != "full":
            continue
        static_name = name.replace("_full", "_static4")
        reports[name] = cell_report(cell, cells.get(static_name))
        label, seq = cell["config"]["label"], cell["config"]["seq"]
        pool, seed = label.rsplit("_s", 1)
        index.setdefault((pool, seq), {})[int(seed)] = name

    spread = []
    for (pool, seq), by_seed in sorted(index.items()):
        for family in ("speed", "path"):
            for metric, getter in (
                    ("rho", lambda b: (b.get("m_height") or {}).get("rho")),
                    ("tmra", lambda b: (b.get("m_height") or {}).get("tmra")),
                    ("cm_over_m", lambda b: (b.get("ratio_cm_height_over_m") or {}).get("median")),
                    ("delta_rho_static4", lambda b: (b.get("static4") or {}).get("delta_rho"))):
                values = {}
                for seed, name in sorted(by_seed.items()):
                    block = reports[name]["families"].get(family) or {}
                    value = getter(block)
                    if value is not None:
                        values[seed] = value
                if len(values) < 2:
                    continue
                series = list(values.values())
                spread.append(dict(
                    pool=pool, seq=seq, family=family, metric=metric,
                    per_seed=values, mean=st.mean(series),
                    sd=st.pstdev(series) if len(series) > 1 else 0.0,
                    min=min(series), max=max(series),
                    range=max(series) - min(series)))

    verdicts = []
    for seq in {row["seq"] for row in spread}:
        for family in ("speed", "path"):
            for metric in ("rho", "tmra", "cm_over_m"):
                rows = {r["pool"]: r for r in spread
                        if r["seq"] == seq and r["family"] == family
                        and r["metric"] == metric}
                if not {"v1", "v3"} <= set(rows):
                    continue
                difference = rows["v3"]["mean"] - rows["v1"]["mean"]
                widest = max(rows["v1"]["range"], rows["v3"]["range"])
                verdicts.append(dict(
                    seq=seq, family=family, metric=metric,
                    v1_mean=rows["v1"]["mean"], v3_mean=rows["v3"]["mean"],
                    difference=difference, widest_seed_range=widest,
                    exceeds_seed_spread=abs(difference) > widest,
                    reading=("the v3-v1 difference is larger than the widest "
                             "seed range" if abs(difference) > widest else
                             "the v3-v1 difference is inside the seed range and "
                             "must not be reported as a label effect")))
    return dict(schema="a100-seed-analysis-v1", cells=reports,
                spread=spread, label_verdicts=verdicts)


def analyse_backbones():
    cells = collect(C.OUT / "backbones")
    reports = {}
    for name, cell in cells.items():
        if cell["config"]["frame_mode"] != "full":
            continue
        reports[name] = cell_report(cell, cells.get(name.replace("_full", "_static4")))
    usable, degenerate = [], []
    for name, report in sorted(reports.items()):
        for family, block in report["families"].items():
            row = dict(cell=name, label=report["config"]["label"],
                       seq=report["config"]["seq"],
                       precision=report["config"]["precision"], family=family,
                       rho=(block.get("m_height") or {}).get("rho"),
                       tmra=(block.get("m_height") or {}).get("tmra"),
                       tmra_minus_constant=(block.get("m_height") or {}).get("tmra_minus_constant"),
                       n_distinct=(block.get("m_height") or {}).get("n_distinct"),
                       cm_over_m=(block.get("ratio_cm_height_over_m") or {}).get("median"),
                       px_over_m=(block.get("ratio_px_height_over_m") or {}).get("median"),
                       px_reference=(block.get("ratio_px_height_over_m") or {}).get("reference_median"),
                       delta_rho_static4=(block.get("static4") or {}).get("delta_rho"),
                       arithmetic=report["arithmetic"])
            row["degenerate"] = bool((block.get("m_height") or {}).get("degenerate"))
            (degenerate if row["degenerate"] else usable).append(row)
    quotable = {}
    for key in ("cm_over_m", "px_over_m", "rho"):
        values = [r[key] for r in usable if r.get(key) is not None]
        if values:
            quotable[key] = dict(n=len(values), min=min(values), max=max(values))
    return dict(schema="a100-backbone-analysis-v1", cells=reports,
                usable_rows=usable, degenerate_rows=degenerate,
                quotable_ranges=quotable,
                note=("quotable_ranges excludes degenerate cells (<=2 distinct "
                      "answers); a ratio near 1 from a constant-output model is "
                      "not evidence that conversion was declined"))


# ----------------------------------------------------------------- output
def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def seeds_tex(analysis):
    lines = ["% Generated by a100/analyze.py -- do not edit by hand.",
             "\\begin{tabular}{lllrrrr}", "\\toprule",
             "Pool & Event & Family & Seeds & Mean & Range & Metric \\\\",
             "\\midrule"]
    for row in analysis["spread"]:
        if row["metric"] != "rho":
            continue
        lines.append("%s & %s & %s & %d & %s & %s & $\\rho$ \\\\" % (
            row["pool"], row["seq"].split("_")[0], row["family"],
            len(row["per_seed"]), fmt(row["mean"]), fmt(row["range"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def backbones_tex(analysis):
    lines = ["% Generated by a100/analyze.py -- do not edit by hand.",
             "\\begin{tabular}{lllrrrr}", "\\toprule",
             "Model & Event & Family & $\\rho$ & cm/m & px/m & $\\Delta\\rho$ \\\\",
             "\\midrule"]
    for row in analysis["usable_rows"]:
        lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            row["label"].replace("_", "\\_"), row["seq"].split("_")[0],
            row["family"], fmt(row["rho"], 2), fmt(row["cm_over_m"], 2),
            fmt(row["px_over_m"], 2), fmt(row["delta_rho_static4"], 2)))
    if analysis["degenerate_rows"]:
        lines += ["\\midrule",
                  "\\multicolumn{7}{l}{\\emph{Degenerate cells "
                  "($\\leq 2$ distinct answers); excluded from every quoted "
                  "range:}} \\\\"]
        for row in analysis["degenerate_rows"]:
            lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
                row["label"].replace("_", "\\_"), row["seq"].split("_")[0],
                row["family"], fmt(row["rho"], 2), fmt(row["cm_over_m"], 2),
                fmt(row["px_over_m"], 2), fmt(row["delta_rho_static4"], 2)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def digest(seeds, backbones):
    out = ["# A100 sweep digest", ""]
    if seeds and seeds["spread"]:
        out += ["## T35 seeds", ""]
        for row in seeds["label_verdicts"]:
            out.append("- %s / %s / %s: v1 %.3f vs v3 %.3f (diff %+.3f), "
                       "widest seed range %.3f -- %s"
                       % (row["seq"], row["family"], row["metric"],
                          row["v1_mean"], row["v3_mean"], row["difference"],
                          row["widest_seed_range"], row["reading"]))
        out.append("")
    if backbones and backbones["usable_rows"]:
        out += ["## T36 backbones", ""]
        q = backbones["quotable_ranges"]
        for key, block in q.items():
            out.append("- %s across %d non-degenerate cells: %.3f to %.3f"
                       % (key, block["n"], block["min"], block["max"]))
        out.append("- degenerate cells excluded: %d"
                   % len(backbones["degenerate_rows"]))
        out.append("")
    if len(out) == 2:
        out.append("No completed cells found yet.")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["both", "seeds", "backbones"], default="both")
    ap.add_argument("--tables-dir",
                    default=str(C.ROOT / "paper2" / "manuscript" / "tables"))
    args = ap.parse_args()

    out_dir = C.OUT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = Path(args.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    seeds = backbones = None
    if args.which in ("both", "seeds"):
        seeds = analyse_seeds()
        (out_dir / "seeds.json").write_text(
            json.dumps(seeds, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
        if seeds["spread"]:
            (tables / "seed_variability.tex").write_text(seeds_tex(seeds),
                                                         encoding="utf8")
    if args.which in ("both", "backbones"):
        backbones = analyse_backbones()
        (out_dir / "backbones.json").write_text(
            json.dumps(backbones, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
        if backbones["usable_rows"]:
            (tables / "backbone_units.tex").write_text(backbones_tex(backbones),
                                                       encoding="utf8")
    text = digest(seeds, backbones)
    (out_dir / "summary.md").write_text(text, encoding="utf8")
    print(text)
    print("-> " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
