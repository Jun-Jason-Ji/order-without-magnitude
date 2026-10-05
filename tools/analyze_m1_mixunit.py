#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decide the M1 mixed-unit control against its preregistered rule.

Reads the cells written by tools/run_m1_mixunit.py and the published v3 seed-42
cells, computes the paired cm/m ratio R (median of itemwise ratios over items
where both answers parse and the metre answer is positive) per cell, and applies
the rule recorded in results/courtdyn/m1_mixunit/PREREGISTRATION.json:

  S  >= 3 of 4 cells R >= 10      N  >= 3 of 4 cells R <= 3      I  otherwise

to E (explicit-coordinate arms) and T (in-template arms), then maps (E, T) to the
interpretation fixed in advance.  Also reports, without verdicts: metre rho and
T-MRA against v3 s42, px/m ratios, first-frame drops, and text-probe accuracy.
If the seed sweep has produced v3 seeds 43/44, their explicit R values are listed
as a seed-noise reference.

Seeds 43/44 and the degeneracy amendment (PREREGISTRATION.json "amendments",
written after seed 42 was seen, before seeds 43/44 were trained) are applied under
"amended": a cell whose centimetre arm has <= 2 distinct parsed answers is
DEGENERATE and never counts toward S or N; per seed and wording S / N need >= 3 of
4 non-degenerate cells, else "limited" (fewer than 3 non-degenerate) or I; per
family, "supervision explains it" needs non-degenerate R >= 10 in >= 2 of 3 seeds
on both events.  The top-level E / T are the seed-42 rule exactly as first
preregistered and are kept only for the record: they count degenerate cells.

"post_hoc_diagnostics" is not preregistered and carries no verdict: per cell it
asks whether the centimetre answers convert the model's own metre answers (share
of itemwise ratios in [50, 200], rho of cm vs metre and vs reference) or sit at
the centimetre training-pool median.  A large R with the cm median at the prior
and the metre median far from it is the prior, not a conversion.

Writes results/courtdyn/m1_mixunit/verdict.json and prints a summary.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "courtdyn" / "m1_mixunit"
PUBLISHED = ROOT / "results" / "courtdyn" / "revision_execution_20260912"
SEEDS = ROOT / "results" / "a100" / "seeds"
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
FAMILIES = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]


def load(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["category"], r["track"], tuple(r["window"])): r for r in rows}


def paired_ratio(num, den, family):
    ratios = [num[k]["prediction"] / den[k]["prediction"] for k in den
              if k[0] == family and k in num
              and num[k]["prediction"] is not None
              and den[k]["prediction"] not in (None, 0) and den[k]["prediction"] > 0]
    return (st.median(ratios), len(ratios)) if ratios else (None, 0)


def cell_stats(directory: Path, m_arm, other_arm, family):
    from a100.eval_controls import scalar_score, spearman
    m = load(directory / f"{m_arm}.jsonl")
    o = load(directory / f"{other_arm}.jsonl")
    r, n = paired_ratio(o, m, family)
    rows = [v for k, v in m.items() if k[0] == family]
    tmra = sum(scalar_score(v["prediction"], v["reference_value"], v["score_tolerance"],
                            v["score_floor"]) if v["prediction"] is not None else 0
               for v in rows) / len(rows)
    parsed = [v for v in rows if v["prediction"] is not None]
    rho = spearman([v["prediction"] for v in parsed], [v["reference_value"] for v in parsed]) \
        if len({v["prediction"] for v in parsed}) > 1 else None
    return {"R": r, "n_pairs": n, "metre_tmra": tmra, "metre_rho": rho,
            "distinct_metre": len({v["prediction"] for v in parsed})}


SEED_LIST = (42, 43, 44)
WORDINGS = {"E": ("full", "m_height", "cm_height"),
            "T": ("full_intemplate", "legacy_m", "legacy_cm")}


def distinct_answers(directory: Path, arm, family):
    return len({v["prediction"] for k, v in load(directory / f"{arm}.jsonl").items()
                if k[0] == family and v["prediction"] is not None})


def mix_dir(seed, seq, sub):
    return OUT / "eval" / f"mixunit{'' if seed == 42 else f'_s{seed}'}_{seq}_{sub}"


def amended_verdict(cells):
    """cells: list of 4 (R, degenerate) or None; the amendment's per-seed rule."""
    if any(c is None or c[0] is None for c in cells):
        return "incomplete"
    live = [r for r, degenerate in cells if not degenerate]
    if sum(r >= 10 for r in live) >= 3:
        return "S"
    if sum(r <= 3 for r in live) >= 3:
        return "N"
    return "limited" if len(live) < 3 else "I"


def training_prior_medians():
    """Median target per (family, unit) in the mixed-unit training pool."""
    items = json.loads((OUT / "mixunit_v3_pool.json").read_text(encoding="utf-8"))
    groups = {}
    for it in items:
        unit = "cm" if "centimet" in it["question"] else "m"
        groups.setdefault((it["category"], unit), []).append(float(it["answer"]))
    return {f"{cat}/{unit}": st.median(v) for (cat, unit), v in groups.items()}


def conversion_diagnostic(directory: Path, m_arm, cm_arm, family, prior):
    from a100.eval_controls import spearman
    m, c = load(directory / f"{m_arm}.jsonl"), load(directory / f"{cm_arm}.jsonl")
    keys = [k for k in c if k[0] == family and k in m and c[k]["prediction"] is not None
            and m[k]["prediction"] not in (None, 0) and m[k]["prediction"] > 0]
    cm = [c[k]["prediction"] for k in keys]
    mm = [m[k]["prediction"] for k in keys]
    ref = [c[k]["reference_value"] for k in keys]
    rho = lambda a, b: spearman(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else None
    return {"n": len(keys),
            "share_ratio_in_50_200": sum(50 <= a / b <= 200 for a, b in zip(cm, mm)) / len(keys),
            "rho_cm_vs_metre": rho(cm, mm), "rho_cm_vs_reference": rho(cm, ref),
            "cm_median": st.median(cm), "metre_median_x100": st.median(mm) * 100,
            "cm_training_prior_median": prior}


def amended_report():
    prior = training_prior_medians()
    seeds, diagnostics = {}, {}
    for seed in SEED_LIST:
        for wording, (sub, m_arm, cm_arm) in WORDINGS.items():
            cells = {}
            for seq in SEQS:
                d = mix_dir(seed, seq, sub)
                for family, short in FAMILIES:
                    if not (d / "summary.json").is_file():
                        cells[f"{seq}/{short}"] = None
                        continue
                    r, n = paired_ratio(load(d / f"{cm_arm}.jsonl"),
                                        load(d / f"{m_arm}.jsonl"), family)
                    k = distinct_answers(d, cm_arm, family)
                    cells[f"{seq}/{short}"] = {"R": r, "n_pairs": n, "distinct_cm": k,
                                               "degenerate": k <= 2}
                    diagnostics[f"s{seed}/{wording}/{seq}/{short}"] = conversion_diagnostic(
                        d, m_arm, cm_arm, family, prior[f"{family}/cm"])
            seeds[f"s{seed}/{wording}"] = {
                "cells": cells,
                "verdict": amended_verdict([None if c is None else (c["R"], c["degenerate"])
                                            for c in cells.values()])}
    families = {}
    for wording in WORDINGS:
        for _, short in FAMILIES:
            supporting = []
            for seed in SEED_LIST:
                cs = [seeds[f"s{seed}/{wording}"]["cells"][f"{seq}/{short}"] for seq in SEQS]
                if all(c is not None and not c["degenerate"] and c["R"] is not None
                       and c["R"] >= 10 for c in cs):
                    supporting.append(seed)
            families[f"{wording}/{short}"] = {
                "seeds_supporting_on_both_events": supporting,
                "supervision_explains_it": len(supporting) >= 2}
    return {"rule": "PREREGISTRATION.json amendments[0]", "per_seed": seeds,
            "family_claims": families}, diagnostics


def verdict(values):
    vals = [v for v in values if v is not None]
    if len(vals) != 4:
        return "incomplete"
    if sum(v >= 10 for v in vals) >= 3:
        return "S"
    if sum(v <= 3 for v in vals) >= 3:
        return "N"
    return "I"


def interpret(e, t):
    if "incomplete" in (e, t):
        return "incomplete: not all cells present"
    if e == "S":
        return ("Unit non-response is a consequence of single-unit supervision; "
                "the paper's claim must be reframed around it.")
    if e == "N" and t == "S":
        return ("Mixed supervision teaches the unit only in its trained wording; the "
                "explicit-arm failure is a failure to transfer across question templates.")
    if e == "N" and t == "N":
        return ("Metre-only supervision does not explain the centimetre failure; the "
                "confound is ruled out for the metric-unit contrast.")
    return "Intermediate outcome: reported as a distribution; no reframing claimed."


def main() -> int:
    prereg = json.loads((OUT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    report = {"preregistered_at_utc": prereg["written_at_utc"],
              "decision_rule_sha256": prereg["decision_rule_sha256"], "cells": {}}
    explicit, intemplate, baseline_t = [], [], []
    for seq in SEQS:
        mix_full = OUT / "eval" / f"mixunit_{seq}_full"
        mix_t = OUT / "eval" / f"mixunit_{seq}_full_intemplate"
        v3_t = OUT / "eval" / f"v3s42_{seq}_full_intemplate"
        pub = PUBLISHED / f"eventholdout_v3_s42_{seq}_full"
        for family, short in FAMILIES:
            key = f"{seq}/{short}"
            c = {}
            if (mix_full / "summary.json").is_file():
                c["mix_explicit"] = cell_stats(mix_full, "m_height", "cm_height", family)
                c["mix_explicit_px_over_m"] = paired_ratio(
                    load(mix_full / "px_height.jsonl"), load(mix_full / "m_height.jsonl"),
                    family)[0]
            c["v3s42_explicit"] = cell_stats(pub, "m_height", "cm_height", family)
            if (mix_t / "summary.json").is_file():
                c["mix_intemplate"] = cell_stats(mix_t, "legacy_m", "legacy_cm", family)
            if (v3_t / "summary.json").is_file():
                c["v3s42_intemplate"] = cell_stats(v3_t, "legacy_m", "legacy_cm", family)
            for s in (43, 44):
                d = SEEDS / f"v3_s{s}_{seq}_full"
                if (d / "summary.json").is_file():
                    c[f"v3s{s}_explicit"] = cell_stats(d, "m_height", "cm_height", family)
            report["cells"][key] = c
            explicit.append((c.get("mix_explicit") or {}).get("R"))
            intemplate.append((c.get("mix_intemplate") or {}).get("R"))
            baseline_t.append((c.get("v3s42_intemplate") or {}).get("R"))
    e, t = verdict(explicit), verdict(intemplate)
    report.update({"E": e, "T": t, "T_baseline_v3s42": verdict(baseline_t),
                   "interpretation": interpret(e, t),
                   "E_T_note": ("seed 42, rule as first preregistered; counts degenerate "
                                "cells. Cite 'amended' and 'post_hoc_diagnostics' instead."),
                   "R_explicit_mix": explicit, "R_intemplate_mix": intemplate,
                   "R_intemplate_v3s42": baseline_t})
    report["amended"], report["post_hoc_diagnostics"] = amended_report()
    probe = ROOT / "results" / "courtdyn" / "unit_probe_v2_runs" / "mixunit" / "summary.json"
    if probe.is_file():
        report["text_probe_all_items"] = json.loads(probe.read_text(encoding="utf-8"))
    (OUT / "verdict.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    fmt = lambda xs: ", ".join("--" if x is None else f"{x:.2f}" for x in xs)
    print(f"E (explicit)    R = [{fmt(explicit)}] -> {e}")
    print(f"T (in-template) R = [{fmt(intemplate)}] -> {t}   "
          f"(v3 s42 in-template: [{fmt(baseline_t)}])")
    print("interpretation (seed 42, as first preregistered):", report["interpretation"])
    print("amended, per seed and wording (R, * = degenerate cm arm):")
    for name, s in report["amended"]["per_seed"].items():
        cells = ", ".join("--" if c is None else f"{c['R']:.0f}{'*' if c['degenerate'] else ''}"
                          for c in s["cells"].values())
        print(f"  {name}: [{cells}] -> {s['verdict']}")
    for name, f in report["amended"]["family_claims"].items():
        print(f"  family {name}: supported={f['supervision_explains_it']} "
              f"(seeds {f['seeds_supporting_on_both_events']})")
    print("post hoc (no verdict): cm rho vs reference, cm median vs metre x100 and prior:")
    fmt_rho = lambda x: "--" if x is None else f"{x:.2f}"
    for name, dg in report["post_hoc_diagnostics"].items():
        if name.split("/")[1] == "E":
            print(f"  {name}: rho {fmt_rho(dg['rho_cm_vs_reference'])}, cm median "
                  f"{dg['cm_median']:.0f} (metre x100 {dg['metre_median_x100']:.0f}, "
                  f"prior {dg['cm_training_prior_median']:.0f})")
    print(f"-> {OUT / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
