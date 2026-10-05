#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decide the second-backbone replication against its preregistered rules R1-R4.

Rules (results/courtdyn/backbone2/PREREGISTRATION.json, written before any training
or evaluation on either backbone), applied per backbone:

  Cells {Q1_top_0-30, Q2_top_480-510} x {speed, path}.  DEGENERATE = the arm in
  question has <= 2 distinct parsed answers; degenerate cells never count, and with
  fewer than 3 of 4 non-degenerate cells the verdict is LIMITED.
  R1 metre-only adapter, explicit arms: >= 3 of 4 non-degenerate cells have metre
     rho >= 0.30 AND paired cm/m ratio R <= 3.  Fails if >= 3 cells have R >= 10.
  R2 mixed-unit adapter: R >= 10 in >= 3 of 4 non-degenerate cells, separately for
     explicit (E) and in-template (T).  Fails if >= 3 cells have R <= 3.
  R3 arithmetic, 246 diagnostic items: metre-only adapter below the bare backbone
     (exact McNemar p < 0.01) and mixed-unit adapter not (p >= 0.01 or above).
  R4 first-frame repetition lowers metre rho in >= 3 of 4 defined cells, metre-only.
  Anything else is intermediate.

Reading of "the arm in question", fixed here and printed with the result: R1 involves
both the metre arm (rho) and the centimetre arm (R), so an R1 cell is degenerate if
EITHER has <= 2 distinct answers; R2 uses the centimetre arm (as in the M1
amendment); R4 needs the metre arm non-degenerate in the full-frame cell and rho
defined in both conditions.

Writes results/courtdyn/backbone2/verdict.json and verdict.md.

--round 2 (results/courtdyn/backbone2_r2/PREREGISTRATION.json): the v3 / mixunit cells and
probes are the round-2 adapters at the preregistered k* (stopping.json); the base-model
cells and base probe are reused from round 1 (same weights, same inputs).  R1, R3, R4 and
the degeneracy reading are unchanged; R2 is AMENDED: a live cell counts toward
"replicates" only if R >= 10 AND Spearman rho(centimetre answers, reference) >= 0.30.
Writes results/courtdyn/backbone2_r2/verdict.json and verdict.md; round 1 is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100.analyze import item_key, paired_ratio, read_cell  # noqa: E402
from a100.eval_controls import spearman                      # noqa: E402
from analyze_unit_probe import classify, is_diagnostic, mcnemar_exact  # noqa: E402

ROUND1 = ROOT / "results" / "courtdyn" / "backbone2"
ROUND2 = ROOT / "results" / "courtdyn" / "backbone2_r2"
ROUND3 = ROOT / "results" / "courtdyn" / "backbone3_internvl"
ROUND4 = ROOT / "results" / "courtdyn" / "backbone4_a100"   # A100: Pixtral-12B, Idefics3-8B, Gemma3-12B
ROUND5 = ROOT / "results" / "courtdyn" / "backbone4_ext"    # A100 extension: seed 43, fixed schedule
ROUND6 = ROOT / "results" / "courtdyn" / "backbone_seeds3"  # seeds 43/44 of every trained backbone
SEED = None          # round 6: the seed being judged
# round 6: base cells/probe live with each backbone's seed-42 record
BASE_OF = {"qwen25vl3b": ROUND1, "smol": ROUND1, "internvl3_2b": ROUND3,
           "pixtral_12b": ROUND4, "idefics3_8b": ROUND4, "gemma3_12b": ROUND4}
BASE_ROOT = ROUND1  # where the unadapted-backbone cells live (round 3 ran its own)
OUT = ROUND1        # directory of the adapter results being judged (set in main)
BACKBONES = ["smol", "qwen25vl3b"]
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
FAMILIES = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]
WORDING = {"E": ("full", "m_height", "cm_height"),
           "T": ("full_intemplate", "legacy_m", "legacy_cm")}


def adir(backbone):
    """Directory holding a backbone's adapter cells/probes for the round being judged."""
    if SEED is None:
        return OUT / backbone
    # the A100 runner writes s<seed>/<key>, the local runner <key>/s<seed>
    a100 = OUT / f"s{SEED}" / backbone
    return a100 if a100.is_dir() else OUT / backbone / f"s{SEED}"


def bdir(backbone):
    """Directory holding the unadapted-backbone cells/probe."""
    return (BASE_OF[backbone] if SEED is not None else BASE_ROOT) / backbone


def fam(rows, family):
    return [r for r in rows if r["category"] == family]


def distinct(rows):
    return len({r["prediction"] for r in rows if r["prediction"] is not None})


def rho(rows):
    v = [r for r in rows if r["prediction"] is not None]
    if len({r["prediction"] for r in v}) < 2 or len(v) < 3:
        return None
    return spearman([r["prediction"] for r in v], [r["reference_value"] for r in v])


def unit_cell(cell, m_arm, cm_arm, family):
    m, cm = fam(cell["arms"][m_arm], family), fam(cell["arms"][cm_arm], family)
    ratio = paired_ratio(cm, m)
    return {"rho_m": rho(m), "distinct_m": distinct(m), "distinct_cm": distinct(cm),
            "R": ratio["median"] if ratio else None, "n_pairs": ratio["n"] if ratio else 0,
            # POST HOC, not part of R1-R4: does the cm answer track the reference at all?
            # R2 (unlike R1) has no tracking condition, so a large R from two unit priors
            # (e.g. metre ~1.8 constant, cm ~170 constant) passes it without any reading.
            "posthoc_rho_cm": rho(cm)}


def rule_counts(cells, degenerate, passes, fails):
    live = [c for c in cells if not degenerate(c)]
    if len(live) < 3:
        return "LIMITED", len(live)
    if sum(passes(c) for c in live) >= 3:
        return "replicates", len(live)
    if sum(fails(c) for c in live) >= 3:
        return "fails", len(live)
    return "intermediate", len(live)


def static_drop(full, static, family):
    f = {item_key(r): r for r in fam(full["arms"]["m_height"], family)
         if r["prediction"] is not None}
    s = {item_key(r): r for r in fam(static["arms"]["m_height"], family)
         if r["prediction"] is not None}
    keys = sorted(set(f) & set(s))
    rf, rs = rho([f[k] for k in keys]), rho([s[k] for k in keys])
    return {"n_common": len(keys), "rho_full": rf, "rho_static4": rs,
            "distinct_full": distinct(f.values()), "distinct_static4": distinct(s.values()),
            "delta": None if rf is None or rs is None else rf - rs}


def probe(backbone):
    arms = {}
    for arm in ("base", "v3", "mixunit"):
        path = (bdir(backbone) if arm == "base" else adir(backbone)) / "unit_probe" / arm / "results.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["class"], _ = classify(r)
        arms[arm] = {r["item_id"]: r["class"] == "correct" for r in rows
                     if is_diagnostic(r["meta"])}
    out = {"n_diagnostic": len(arms["base"]),
           "accuracy": {a: sum(v.values()) / len(v) for a, v in arms.items()}}
    for arm in ("v3", "mixunit"):
        only_base = sum(arms["base"][k] and not arms[arm][k] for k in arms["base"])
        only_arm = sum(arms[arm][k] and not arms["base"][k] for k in arms["base"])
        out[arm] = {"only_base_correct": only_base, "only_arm_correct": only_arm,
                    "mcnemar_p": mcnemar_exact(only_base, only_arm),
                    "below_base": only_base > only_arm}
    v3_below = out["v3"]["below_base"] and out["v3"]["mcnemar_p"] < 0.01
    mix_not = out["mixunit"]["mcnemar_p"] >= 0.01 or not out["mixunit"]["below_base"]
    out["verdict"] = "replicates" if (v3_below and mix_not) else (
        "fails" if not v3_below else "intermediate")
    return out


def analyse(backbone, round_no=1):
    ev = adir(backbone) / "eval"
    rep = {"cells": {}}
    r1, r2 = [], {"E": [], "T": []}
    r4 = []
    for seq in SEQS:
        cells = {k: read_cell(ev / f"{label}_{seq}_{sub}") for k, label, sub in
                 [("v3", "v3", "full"), ("mixunit", "mixunit", "full"),
                  ("v3T", "v3", "full_intemplate"), ("mixunitT", "mixunit", "full_intemplate"),
                  ("v3S", "v3", "static4")]}
        cells["base"] = read_cell(bdir(backbone) / "eval" / f"base_{seq}_full")
        cells = {k: v for k, v in cells.items() if v}
        for family, short in FAMILIES:
            key = f"{seq}/{short}"
            c = {"v3_E": unit_cell(cells["v3"], "m_height", "cm_height", family),
                 "mix_E": unit_cell(cells["mixunit"], "m_height", "cm_height", family),
                 "v3_T": unit_cell(cells["v3T"], "legacy_m", "legacy_cm", family),
                 "mix_T": unit_cell(cells["mixunitT"], "legacy_m", "legacy_cm", family),
                 "v3_static4": static_drop(cells["v3"], cells["v3S"], family)}
            if "base" in cells:
                c["base_E"] = unit_cell(cells["base"], "m_height", "cm_height", family)
            rep["cells"][key] = c
            r1.append(c["v3_E"])
            r2["E"].append(c["mix_E"])
            r2["T"].append(c["mix_T"])
            r4.append(c["v3_static4"])
    rep["R1"], rep["R1_live"] = rule_counts(
        r1, lambda c: c["distinct_m"] <= 2 or c["distinct_cm"] <= 2,
        lambda c: c["rho_m"] is not None and c["rho_m"] >= 0.30 and c["R"] is not None and c["R"] <= 3,
        lambda c: c["R"] is not None and c["R"] >= 10)
    if round_no == 1:
        r2_pass = lambda c: c["R"] is not None and c["R"] >= 10
    else:   # round-2 amendment: the centimetre answers must also track the reference
        r2_pass = lambda c: (c["R"] is not None and c["R"] >= 10
                             and c["posthoc_rho_cm"] is not None and c["posthoc_rho_cm"] >= 0.30)
    for w in ("E", "T"):
        rep[f"R2_{w}"], rep[f"R2_{w}_live"] = rule_counts(
            r2[w], lambda c: c["distinct_cm"] <= 2, r2_pass,
            lambda c: c["R"] is not None and c["R"] <= 3)
    defined = [c for c in r4 if c["distinct_full"] > 2 and c["delta"] is not None]
    rep["R4_defined"] = len(defined)
    rep["R4"] = ("LIMITED" if len(defined) < 3 else
                 "replicates" if sum(c["delta"] > 0 for c in defined) >= 3 else "intermediate")
    rep["R3"] = probe(backbone)
    return rep


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2, 3, 4, 5, 6), default=1)
    ap.add_argument("--seed", type=int, choices=(43, 44), help="round 6 only")
    args = ap.parse_args()
    round_no = args.round
    global BACKBONES, BASE_ROOT, SEED
    OUT = {1: ROUND1, 2: ROUND2, 3: ROUND3, 4: ROUND4, 5: ROUND5, 6: ROUND6}[round_no]
    if round_no == 6:   # three-seed record (backbone_seeds3, ed6aacc): one seed at a time, judged alone
        if args.seed is None:
            ap.error("--round 6 needs --seed")
        SEED = args.seed
        BACKBONES = [b for b in BASE_OF
                     if (adir(b) / "unit_probe" / "mixunit" / "results.jsonl").is_file()
                     and (adir(b) / "stopping.json").is_file()]
    if round_no == 3:   # third backbone: InternVL3-2B, round-2 protocol, fresh base cells
        BACKBONES, BASE_ROOT = ["internvl3_2b"], ROUND3
    if round_no == 4:   # A100 backbones: round-2 protocol, own base cells; only those with a
        # recorded stopping decision and a finished probe are judged (others still running)
        order = ["pixtral_12b", "idefics3_8b", "gemma3_12b"]
        BACKBONES = [b for b in order if (OUT / b / "stopping.json").is_file()
                     and (OUT / b / "unit_probe" / "mixunit" / "results.jsonl").is_file()]
        BASE_ROOT = ROUND4
    if round_no == 5:   # seed-43 replication: fixed schedule (seed-42 k*), base cells from round 4
        BACKBONES = [b for b in ["gemma3_12b", "pixtral_12b"]
                     if (OUT / b / "unit_probe" / "mixunit" / "results.jsonl").is_file()]
        BASE_ROOT = ROUND4
    prereg = json.loads((OUT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(prereg["rules"].encode("utf-8")).hexdigest() == prereg["rules_sha256"]
    report = {"preregistered_at_utc": prereg["written_at_utc"],
              "rules_sha256": prereg["rules_sha256"],
              "degeneracy_reading": ("R1: metre OR centimetre arm <= 2 distinct; R2: centimetre "
                                     "arm; R4: metre arm (full) and rho defined in both"),
              "backbones": {b: analyse(b, round_no) for b in BACKBONES}}
    if round_no >= 2:
        report["round1_rules_sha256"] = prereg.get("round1_rules_sha256")
        report["r2_amendment"] = ("R2 live cell counts toward replicates only if R >= 10 AND "
                                  "rho(cm answers, reference) >= 0.30")
        for b, rep in report["backbones"].items():
            stop = json.loads((adir(b) / "stopping.json").read_text(encoding="utf-8"))
            rep["k_star"], rep["degenerate_at_cap"] = stop["k_star"], stop["degenerate_at_cap"]
    VOUT = OUT / f"s{SEED}" if SEED is not None else OUT
    VOUT.mkdir(parents=True, exist_ok=True)
    if SEED is not None:
        report["seed"] = SEED
    (VOUT / "verdict.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    title = {1: "", 2: ", round 2 (4-epoch schedule, R2 amended)",
             3: ", third backbone InternVL3-2B (round-2 protocol, R2 amended)",
             4: ", A100 backbones (round-2 protocol, R2 amended)",
             5: ", A100 extension: seed 43 (fixed schedule = seed-42 k*, R2 amended)",
             6: f", three-seed record: seed {SEED} (fixed schedule = seed-42 k*, R2 amended)"}[round_no]
    md = [f"# Second-backbone replication (preregistered R1-R4){title}", "",
          f"Rules sha256 `{prereg['rules_sha256'][:16]}...`, recorded "
          f"{prereg['written_at_utc'][:16]} UTC before any run. Degeneracy: {report['degeneracy_reading']}.", ""]
    for b, rep in report["backbones"].items():
        stop_line = [] if round_no == 1 else [
            f"Schedule fixed to the seed-42 selection (dev rule not re-run): k = {rep['k_star']}", ""] \
            if round_no in (5, 6) else [
            f"Stopping rule (dev set, decided before any test cell): k* = {rep['k_star']}"
            + (", DEGENERATE AT THE 4-EPOCH CAP (test cells descriptive only)"
               if rep["degenerate_at_cap"] else ", passes"), ""]
        md += [f"## {b}", ""] + stop_line + [
               f"R1 {rep['R1']} ({rep['R1_live']}/4 live) · R2-E {rep['R2_E']} ({rep['R2_E_live']}/4) · "
               f"R2-T {rep['R2_T']} ({rep['R2_T_live']}/4) · R3 {rep['R3']['verdict']} · "
               f"R4 {rep['R4']} ({rep['R4_defined']}/4 defined)", "",
               "| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | "
               "mix R (T) | v3 delta rho first-frame | base rho_m / distinct |",
               "|---|---|---|---|---|---|---|---|---|"]
        for key, c in rep["cells"].items():
            base = c.get("base_E") or {}
            md.append(f"| {key} | {f(c['v3_E']['rho_m'])} | {c['v3_E']['distinct_m']}/{c['v3_E']['distinct_cm']} | "
                      f"{f(c['v3_E']['R'])} | {f(c['mix_E']['R'], 1)} | {c['mix_E']['distinct_cm']} | "
                      f"{f(c['mix_T']['R'], 1)} | {f(c['v3_static4']['delta'])} | "
                      f"{f(base.get('rho_m'))} / {base.get('distinct_m')} |")
        md += ["", "Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs "
               "reference, E / T: " + "; ".join(
                   f"{k} {f(c['mix_E']['posthoc_rho_cm'])} / {f(c['mix_T']['posthoc_rho_cm'])}"
                   for k, c in rep["cells"].items())]
        p = rep["R3"]
        md += ["", f"Text probe, {p['n_diagnostic']} diagnostic items: base {p['accuracy']['base']:.1%}, "
               f"v3 {p['accuracy']['v3']:.1%} (p = {p['v3']['mcnemar_p']:.3g}), "
               f"mixunit {p['accuracy']['mixunit']:.1%} (p = {p['mixunit']['mcnemar_p']:.3g})", ""]
    (VOUT / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if round_no in (5, 6):   # three-seed aggregation, which applies the recorded title rule (ed6aacc)
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "tools" / "aggregate_backbone_seeds.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
