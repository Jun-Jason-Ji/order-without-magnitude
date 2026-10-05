#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the new result tables of the paper-2 rewrite (paper2/manuscript_v2/tables/).

Every number is recomputed from frozen per-item outputs with the same helpers as the
analysis scripts, and cross-checked against their saved JSON where one exists.

  m1_mixunit.tex   mixed-unit supervision, three seeds: cm/m ratio R (degenerate cells
                   starred), rho of the centimetre answers, cm median vs training prior,
                   plus the Q2 metre cost against the metre-only adapter of the same seed.
  soccer_venue.tex second sport, zero-shot (exploratory): metre-only and mixed-unit
                   Qwen3.5-4B adapters per match.
  backbones.tex    second backbones: round-1 (one epoch) and round-2 (dev-selected epoch)
                   verdicts; round-2 cells print PENDING until its verdict file exists.
Also writes results/courtdyn/paper2_v2_numbers.json (all values used in the text).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100.analyze import family_stats, paired_ratio, read_cell  # noqa: E402
from a100.eval_controls import spearman                          # noqa: E402
from analyze_unit_probe import classify, is_diagnostic, mcnemar_exact, wilson  # noqa: E402

CD = ROOT / "results" / "courtdyn"
TABLES = ROOT / "paper2" / "manuscript_v2" / "tables"
SEQS = [("Q1_top_0-30", "Q1"), ("Q2_top_480-510", "Q2")]
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
SEEDS = (42, 43, 44)


def fam_rows(cell, arm, family):
    return [r for r in cell["arms"][arm] if r["category"] == family]


def rho_or_none(rows):
    v = [r for r in rows if r["prediction"] is not None]
    if len({r["prediction"] for r in v}) < 2:
        return None
    return spearman([r["prediction"] for r in v], [r["reference_value"] for r in v])


def mix_dir(seed, seq):
    return CD / "m1_mixunit" / "eval" / f"mixunit{'' if seed == 42 else f'_s{seed}'}_{seq}_full"


def v3_dir(seed, seq):
    if seed == 42:
        return CD / "revision_execution_20260912" / f"eventholdout_v3_s42_{seq}_full"
    return ROOT / "results" / "a100" / "seeds" / f"v3_s{seed}_{seq}_full"


# ------------------------------------------------------------------ M1
def m1_block():
    verdict = json.loads((CD / "m1_mixunit" / "verdict.json").read_text(encoding="utf-8"))
    prior = verdict["post_hoc_diagnostics"]
    out = {"cells": {}, "cost_q2": {}, "family_claims": verdict["amended"]["family_claims"],
           "per_seed_verdicts": {k: v["verdict"] for k, v in verdict["amended"]["per_seed"].items()}}
    for seq, tag in SEQS:
        for family, short in FAMS:
            row = {}
            for seed in SEEDS:
                cell = read_cell(mix_dir(seed, seq))
                m, cm = fam_rows(cell, "m_height", family), fam_rows(cell, "cm_height", family)
                ratio = paired_ratio(cm, m)
                distinct = len({r["prediction"] for r in cm if r["prediction"] is not None})
                amended = verdict["amended"]["per_seed"][f"s{seed}/E"]["cells"][f"{seq}/{short}"]
                assert abs(amended["R"] - ratio["median"]) < 1e-9 and amended["distinct_cm"] == distinct
                row[seed] = {"R": ratio["median"], "distinct_cm": distinct, "degenerate": distinct <= 2,
                             "rho_cm": rho_or_none(cm),
                             "cm_median": st.median(r["prediction"] for r in cm if r["prediction"] is not None),
                             "prior": prior[f"s{seed}/E/{seq}/{short}"]["cm_training_prior_median"]}
            out["cells"][f"{tag}/{short}"] = row
    for family, short in FAMS:
        for seed in SEEDS:
            mix = family_stats(fam_rows(read_cell(mix_dir(seed, "Q2_top_480-510")), "m_height", family))
            v3 = family_stats(fam_rows(read_cell(v3_dir(seed, "Q2_top_480-510")), "m_height", family))
            out["cost_q2"][f"{short}/s{seed}"] = {"mix_tmra": mix["tmra"], "v3_tmra": v3["tmra"],
                                                  "mix_rho": mix["rho"], "v3_rho": v3["rho"]}
    return out


def m1_probe():
    """Mixed-unit adapter on the 246 diagnostic text items, against base and v3."""
    arms = {}
    for arm in ("base", "v3", "mixunit"):
        rows = [json.loads(l) for l in (CD / "unit_probe_v2_runs" / arm / "results.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        arms[arm] = {r["item_id"]: classify(r)[0] == "correct" for r in rows if is_diagnostic(r["meta"])}
    n = len(arms["base"])
    out = {"n": n}
    for arm, v in arms.items():
        k = sum(v.values())
        lo, hi = wilson(k, n)
        out[arm] = {"k": k, "acc": k / n, "lo": lo, "hi": hi}
    for a, b in (("mixunit", "base"), ("mixunit", "v3")):
        only_a = sum(arms[a][i] and not arms[b][i] for i in arms[b])
        only_b = sum(arms[b][i] and not arms[a][i] for i in arms[b])
        out[f"{a}_vs_{b}_p"] = mcnemar_exact(only_a, only_b)
    return out


def m1_tex(m1, probe):
    f = lambda x, d=2: "---" if x is None else f"{x:.{d}f}"
    lines = ["% Generated by tools/make_paper2_v2_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Mixed-unit supervision (half the training targets in centimetres), explicit "
             r"prompts, three seeds. $R$: median paired cm/m prediction ratio (exact conversion $=100$); "
             r"$^{*}$: degenerate centimetre arm ($\le 2$ distinct answers), excluded from every verdict. "
             r"$\rho_{\mathrm{cm}}$: Spearman correlation of the centimetre answers with the reference "
             r"(--- if undefined). Median: centimetre answers' median against the training-pool median "
             r"of centimetre targets for that family.}",
             r"\label{tab:m1}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{$R$} & \multicolumn{3}{c}{$\rho_{\mathrm{cm}}$} & Median / prior \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Family & Event & s42 & s43 & s44 & s42 & s43 & s44 & (s42--44) \\", r"\midrule"]
    for short in ("path", "speed"):
        for tag in ("Q1", "Q2"):
            c = m1["cells"][f"{tag}/{short}"]
            rs = [f"{c[s]['R']:.0f}{'$^{*}$' if c[s]['degenerate'] else ''}" for s in SEEDS]
            rh = [f(c[s]["rho_cm"]) for s in SEEDS]
            meds = "/".join(f"{c[s]['cm_median']:.0f}" for s in SEEDS)
            lines.append(f"{short} & {tag} & " + " & ".join(rs) + " & " + " & ".join(rh)
                         + f" & {meds} / {c[42]['prior']:.0f} \\\\")
        if short == "path":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ soccer
def soccer_block():
    s = json.loads((CD / "soccer_venue" / "summary.json").read_text(encoding="utf-8"))
    boot = json.loads((CD / "bootstrap_ci.json").read_text(encoding="utf-8"))["soccer"]
    out = {}
    for adapter in ("qwen35_v3", "qwen35_mixunit", "smol_v3", "smol_mixunit"):
        for seq, meta in s["sequences"].items():
            cell = s["cells"][f"{adapter}/{seq}"]
            for short in ("speed", "path"):
                b = cell["families"][short]
                m = b["m_height"]
                out[f"{adapter}/{seq}/{short}"] = {
                    "rho": m["rho"], "distinct": m["n_distinct"], "degenerate": m["degenerate"],
                    "tmra": m["tmra"], "margin": m["tmra_minus_constant"],
                    "cm_over_m": (b.get("ratio_cm_height_over_m") or {}).get("median"),
                    "cm_distinct": b.get("cm_height", {}).get("n_distinct"),
                    "px_over_m": (b.get("ratio_px_height_over_m") or {}).get("median"),
                    "K": meta["K_px_per_m"], "delta_rho": (b.get("static4") or {}).get("delta_rho"),
                    "camera_motion_px": meta["camera_motion_px_median"],
                    "ci": boot.get(f"{seq}/{short}") if adapter == "qwen35_v3" else None}
    return out


def soccer_tex(sc):
    f = lambda x, d=2: "---" if x is None else f"{x:.{d}f}"
    lines = ["% Generated by tools/make_paper2_v2_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Second sport, zero-shot (SoccerNet-GSR, three matches; exploratory: no decision "
             r"rule was recorded in advance). Metre-only (v3) and mixed-unit Qwen3.5-4B adapters, trained "
             r"on basketball only. $\rho$ with 95\% window-clustered bootstrap interval; $\Delta\rho$: full "
             r"minus first-frame$\times4$; margin: metre T-MRA minus the best test-label constant; cm/m "
             r"exact $=100$; px/m should equal the match's $K$ if pixels were read. $^{*}$: degenerate "
             r"centimetre arm.}",
             r"\label{tab:soccer}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"Adapter & Match ($K$) & Family & $\rho$ & $\Delta\rho$ & Margin & cm/m & px/m \\",
             r"\midrule"]
    for adapter, label in (("qwen35_v3", "metre-only"), ("qwen35_mixunit", "mixed-unit")):
        for seq in ("SNGS-034", "SNGS-056", "SNGS-096"):
            for short in ("speed", "path"):
                c = sc[f"{adapter}/{seq}/{short}"]
                rho = f(c["rho"])
                if c["ci"] and c["ci"]["rho"]["window"]["lo"] is not None:
                    w = c["ci"]["rho"]["window"]
                    rho += f" [{w['lo']:.2f}, {w['hi']:.2f}]"
                cm = f(c["cm_over_m"], 1) + ("$^{*}$" if (c["cm_distinct"] or 3) <= 2 else "")
                lines.append(f"{label} & {seq} ({c['K']:.1f}) & {short} & {rho} & {f(c['delta_rho'])} & "
                             f"{f(c['margin'], 1)} & {cm} & {f(c['px_over_m'])} \\\\")
        if adapter == "qwen35_v3":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ backbones
def backbone_block():
    r1 = json.loads((CD / "backbone2" / "verdict.json").read_text(encoding="utf-8"))["backbones"]
    r2_path = CD / "backbone2_r2" / "verdict.json"
    r2 = json.loads(r2_path.read_text(encoding="utf-8"))["backbones"] if r2_path.is_file() else None
    out = {}
    for key in ("smol", "qwen25vl3b"):
        a = r1[key]
        stop = CD / "backbone2_r2" / key / "stopping.json"
        out[key] = {"round1": {k: a[k] for k in ("R1", "R1_live", "R2_E", "R2_E_live", "R2_T",
                                                  "R2_T_live", "R4", "R4_defined")}
                    | {"R3": a["R3"]["verdict"], "probe": a["R3"]["accuracy"],
                       "p_v3": a["R3"]["v3"]["mcnemar_p"], "p_mix": a["R3"]["mixunit"]["mcnemar_p"]},
                    "round2": (r2 or {}).get(key),
                    "stopping": json.loads(stop.read_text(encoding="utf-8")) if stop.is_file() else None}
    # third backbone: InternVL3-2B, run only under the round-2 protocol (no one-epoch round)
    r3_path = CD / "backbone3_internvl" / "verdict.json"
    if r3_path.is_file():
        stop = CD / "backbone3_internvl" / "internvl3_2b" / "stopping.json"
        out["internvl3_2b"] = {"round1": None,
                               "round2": json.loads(r3_path.read_text(encoding="utf-8"))["backbones"]["internvl3_2b"],
                               "stopping": json.loads(stop.read_text(encoding="utf-8"))}
    # A100 backbones (round 4 of the analyzer): every backbone in its verdict, in its order
    r4_path = CD / "backbone4_a100" / "verdict.json"
    if r4_path.is_file():
        for key, rep in json.loads(r4_path.read_text(encoding="utf-8"))["backbones"].items():
            stop = CD / "backbone4_a100" / key / "stopping.json"
            out[key] = {"round1": None, "round2": rep,
                        "stopping": json.loads(stop.read_text(encoding="utf-8"))}
    return out


def backbones_tex(bb):
    names = {"smol": "SmolVLM2-2.2B", "qwen25vl3b": "Qwen2.5-VL-3B", "internvl3_2b": "InternVL3-2B",
             "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B", "gemma3_12b": "Gemma3-12B"}
    lines = ["% Generated by tools/make_paper2_v2_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Other backbones against rules recorded before training (R1: order read, units "
             r"not converted; R2: mixed-unit training converts, explicit/in-template; R3: text arithmetic; "
             r"R4: first-frame drop). LIMITED: fewer than three of four cells non-degenerate, i.e.\ "
             r"untestable. Round 1 uses the Qwen3.5-4B recipe (one epoch); round 2 selects the epoch on a "
             r"development clip; InternVL3-2B, Pixtral-12B, Idefics3-8B and Gemma3-12B were run under the "
             r"round-2 protocol only, with their own unadapted-backbone cells. $^\dagger$degenerate at the four-epoch cap (descriptive only). "
             r"In parentheses: non-degenerate cells out of four.}",
             r"\label{tab:backbones}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llllllll}", r"\toprule",
             r"Backbone & Round & Epoch & R1 & R2-E & R2-T & R3 & R4 \\", r"\midrule"]
    for key, d in bb.items():
        a = d["round1"]
        if a is None:
            r2, stop = d["round2"], d["stopping"]
            lines.append(f"{names[key]} & 2 & {stop['k_star']} & {r2['R1']} ({r2['R1_live']}) & "
                         f"{r2['R2_E']} ({r2['R2_E_live']}) & {r2['R2_T']} ({r2['R2_T_live']}) & "
                         f"{r2['R3']['verdict']} & {r2['R4']} ({r2['R4_defined']}) \\\\")
            continue
        lines.append(f"{names[key]} & 1 & 1 & {a['R1']} ({a['R1_live']}) & {a['R2_E']} ({a['R2_E_live']}) & "
                     f"{a['R2_T']} ({a['R2_T_live']}) & {a['R3']} & {a['R4']} ({a['R4_defined']}) \\\\")
        r2, stop = d["round2"], d["stopping"]
        if r2 and stop:
            lines.append(f" & 2 & {stop['k_star']}{'$^\\dagger$' if stop['degenerate_at_cap'] else ''} & "
                         f"{r2['R1']} ({r2['R1_live']}) & {r2['R2_E']} ({r2['R2_E_live']}) & "
                         f"{r2['R2_T']} ({r2['R2_T_live']}) & {r2['R3']['verdict']} & "
                         f"{r2['R4']} ({r2['R4_defined']}) \\\\")
        else:
            lines.append(r" & 2 & \textbf{PENDING} & \multicolumn{5}{l}{\textbf{PENDING}} \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    # the preregistration's label is LIMITED; the paper prints the plain word
    return ("\n".join(lines) + "\n").replace("LIMITED", "untestable")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    m1, probe = m1_block(), m1_probe()
    sc, bb = soccer_block(), backbone_block()
    (TABLES / "m1_mixunit.tex").write_text(m1_tex(m1, probe), encoding="utf-8")
    (TABLES / "soccer_venue.tex").write_text(soccer_tex(sc), encoding="utf-8")
    (TABLES / "backbones.tex").write_text(backbones_tex(bb), encoding="utf-8")
    from make_paper2_extra_tables import build as build_extra
    extra = build_extra()
    numbers = {"m1": m1, "m1_probe": probe, "soccer": sc, "backbones": bb, "extra": extra}
    # short captions + notes under the table (idempotent; see tools/trim_table_captions.py)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools" / "trim_table_captions.py")], check=True, stdout=subprocess.DEVNULL)
    (CD / "paper2_v2_numbers.json").write_text(json.dumps(numbers, indent=1, default=str) + "\n",
                                               encoding="utf-8")
    print(m1_tex(m1, probe)); print(soccer_tex(sc)); print(backbones_tex(bb))
    print(json.dumps({"probe": probe, "cost_q2": m1["cost_q2"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
