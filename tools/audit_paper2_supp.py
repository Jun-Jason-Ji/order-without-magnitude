#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Number audit for paper2/manuscript_v2/supplementary.tex (companion of audit_paper2_v2.py).

Every row of the supplement tables that has a frozen result file behind it is regenerated from
that file and must appear verbatim in the source:

  tab:matched_label_units, tab:matched_label_temporal, tab:matched_label_main,
  tab:explicit_unit_controls   <- results/courtdyn/revision_execution_20260912/
                                  analysis_order_compatible/analysis.json
  tab:paired_reference_scores  <- .../paired_reference_scores_complete/analysis.json

plus prose numbers derived from the same files, and hygiene gates: no text left over from the
automated "\\ref -> the main text" replacement, no counter resets that duplicate S-numbers, no
stale single-seed / single-backbone claims.  It certifies agreement with the result files, not
that the results are right.  Exit 1 on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper2" / "manuscript_v2" / "supplementary.tex"
RUN = ROOT / "results" / "courtdyn" / "revision_execution_20260912"
FAM = {"speed": "dynamics_speed_player", "path": "dynamics_path_player"}
SEQ = {"Q1": "Q1_top_0-30", "Q2": "Q2_top_480-510"}
LABEL = {"v1": "eventholdout_v1_s42", "v3": "eventholdout_v3_s42"}


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", tex)
    checks, failures = [], []

    def want(label, text):
        checks.append(label)
        if text not in flat:
            failures.append(f"{label}: expected '{text}'")

    def gate(label, ok, detail=""):
        checks.append(label)
        if not ok:
            failures.append(f"{label}: {detail}")

    a = json.loads((RUN / "analysis_order_compatible" / "analysis.json").read_text(encoding="utf-8"))
    gate("analysis status PASS", a["status"] == "PASS", a["status"])
    runs = {(r["label"], r["seq"], r["frame_mode"]): r for r in a["runs"]}

    def fam(label, seq, arm, family, frames="full"):
        return runs[(label, seq, frames)]["cells"][arm]["families"][FAM[family]]

    def pair(label, seq, num, family):
        for p in runs[(label, seq, "full")]["within_run_pairing"]:
            if p["numerator"] == num and p["denominator"] == "m_height" and p["family"] == FAM[family]:
                return p["ratio"]["median"]
        raise KeyError((label, seq, num, family))

    # ---- tab:matched_label_units: cm T/rho, px T/rho, R_px/m
    for tr in ("v1", "v3"):
        for ev in ("Q1", "Q2"):
            for fm in ("speed", "path"):
                cm, px = fam(LABEL[tr], SEQ[ev], "cm_height", fm), fam(LABEL[tr], SEQ[ev], "px_height", fm)
                want(f"units {tr}/{ev}/{fm}",
                     f"{tr} / {ev} & {fm} & {cm['tmra_all_items']:.2f}/{cm['spearman']:.3f} & "
                     f"{px['tmra_all_items']:.2f}/{px['spearman']:.3f} & "
                     f"{pair(LABEL[tr], SEQ[ev], 'px_height', fm):.2f} \\\\")

    # ---- tab:matched_label_temporal: N, T-MRA full/first, rho full/first
    for t in a["temporal_controls"]:
        tr = "v1" if "v1" in t["label"] else "v3"
        ev = "Q1" if t["seq"].startswith("Q1") else "Q2"
        fm = "speed" if "speed" in t["family"] else "path"
        full_t, first_t = t["paired_all_observed_tmra"]
        full_r, first_r = t["paired_common_parsed_spearman"]
        first_r = "u" if first_r is None else f"{first_r:.3f}"
        want(f"temporal {tr}/{ev}/{fm}",
             f"{tr} / {ev} & {fm} & {t['n_common_parsed']} & {full_t:.2f} & {first_t:.2f} & "
             f"{full_r:.3f} & {first_r} \\\\")

    # ---- tab:matched_label_main: M/O, rho, R(cm/m)
    for tr in ("v1", "v3"):
        for ev in ("Q1", "Q2"):
            parts = []
            for fm in ("speed", "path"):
                m = fam(LABEL[tr], SEQ[ev], "m_height", fm)
                o = m["constant_diagnostics"]["values"]["test_label_oracle"]["tmra"]
                parts.append(f"{m['tmra_all_items']:.2f}/{o:.2f} & {m['spearman']:.3f} & "
                             f"{pair(LABEL[tr], SEQ[ev], 'cm_height', fm):.2f}")
            want(f"main {tr}/{ev}", f"{tr} / {ev} & " + " & ".join(parts) + " \\\\")

    # ---- tab:explicit_unit_controls: native, Q2, both height conditions
    nat = "native_original"
    for h, hname in (("height", "yes"), ("noheight", "no")):
        for unit in ("m", "cm", "px"):
            cells = []
            for fm in ("speed", "path"):
                c = fam(nat, SEQ["Q2"], f"{unit}_{h}", fm)
                if unit == "m":
                    r = "---"
                else:
                    r = None
                    for p in runs[(nat, SEQ["Q2"], "full")]["within_run_pairing"]:
                        if (p["numerator"], p["denominator"], p["family"]) == (f"{unit}_{h}", f"m_{h}", FAM[fm]):
                            r = f"{p['ratio']['median']:.2f}"
                    r = r or "?"
                cells.append(f"{c['tmra_all_items']:.1f} & {c['spearman']:.3f} & {r}")
            want(f"explicit {unit}/{h}", f"{unit} & {hname} & " + " & ".join(cells) + " \\\\")

    # ---- tab:paired_reference_scores: exact v1 / exact v3, T-MRA + MAE
    b = json.loads((RUN / "paired_reference_scores_complete" / "analysis.json").read_text(encoding="utf-8"))
    gate("paired-reference status PASS", b["status"] == "PASS", b["status"])
    pr = {(c["model"], c["seq"], c["frame_mode"], c["family"], c["reference_basis"]): c
          for c in b["primary_exact"]["cells"]}
    for tr in ("v1", "v3"):
        for ev in ("Q1", "Q2"):
            for fm in ("speed", "path"):
                c1 = pr[(LABEL[tr], SEQ[ev], "full", fm, "exact_v1")]
                c3 = pr[(LABEL[tr], SEQ[ev], "full", fm, "exact_v3")]
                want(f"dualref {tr}/{ev}/{fm}",
                     f"{tr} / {ev} & {fm} & {c1['tmra_all_items']:.2f} & {c1['mae_parsed']:.3f} & "
                     f"{c3['tmra_all_items']:.2f} & {c3['mae_parsed']:.3f} \\\\")

    # ---- prose derived from the same files
    dT = [fam(LABEL["v3"], SEQ[ev], "m_height", fm)["tmra_all_items"]
          - fam(LABEL["v1"], SEQ[ev], "m_height", fm)["tmra_all_items"]
          for ev in ("Q1", "Q2") for fm in ("speed", "path")]
    want("v3-v1 T-MRA deltas", f"Q1 speed by ${-dT[0]:.2f}$ points, and Q2 speed/path by "
                               f"${-dT[2]:.2f}$/${-dT[3]:.2f}$; Q1 path increases by ${dT[1]:.2f}$")
    drops = [t["paired_all_observed_tmra"][0] - t["paired_all_observed_tmra"][1] for t in a["temporal_controls"]]
    want("first-frame T-MRA drop range", f"${min(drops):.2f}$--${max(drops):.2f}$ points")
    arith = {tr: runs[(LABEL[tr], SEQ["Q2"], "full")]["arithmetic"]["n_correct"] for tr in ("v1", "v3")}
    want("12-probe v1", f"v1 training gives ${arith['v1']}/12$ correct")
    want("12-probe v3", f"v3 training gives ${arith['v3']}/12$")

    # ---- seed-sweep qualification (from the generated main-text seed table)
    rows = {}
    pool = None
    for line in (TEX.parent / "tables" / "seed_variability.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        if len(cells) == 10 and cells[1].isdigit():
            pool = cells[0] or pool
            rows[(pool, int(cells[1]))] = cells
    v1 = {s: rows[("v1 labels", s)] for s in (42, 43, 44)}
    v3 = {s: rows[("v3 labels", s)] for s in (42, 43, 44)}
    gate("v3 path rho > v1 in every seed", all(float(v3[s][6]) > float(v1[s][6]) for s in v1), "")
    gate("v3 speed T-MRA < v1 in every seed", all(float(v3[s][3]) < float(v1[s][3]) for s in v1), "")
    v1_path_t = [float(v1[s][7]) for s in v1]
    want("v1 path T-MRA seed span", f"${min(v1_path_t):.1f}$--${max(v1_path_t):.1f}$ across seeds")

    # ---- provenance of the revision adapters
    for d, k in (("backbone2_r2/smol", 4), ("backbone2_r2/qwen25vl3b", 4), ("backbone3_internvl/internvl3_2b", 2)):
        st = json.loads((ROOT / "results" / "courtdyn" / d / "stopping.json").read_text(encoding="utf-8"))
        gate(f"k* {d}", st["k_star"] == k, str(st["k_star"]))
    gate("smol degenerate at cap", json.loads((ROOT / "results" / "courtdyn" / "backbone2_r2" / "smol"
                                               / "stopping.json").read_text(encoding="utf-8"))["degenerate_at_cap"], "")
    sys.path.insert(0, str(ROOT / "tools"))
    from mixunit_pool_stats import load_stats      # live pool when present, shipped aggregate otherwise
    ps_ = load_stats()
    want("mixed-unit pool split", f"{ps_['n_cm']} of its {ps_['n_items']:,} items")

    # ---- hygiene
    left = [m.group(0) for m in re.finditer(r"(Equation~the main text|\$R\$ is the main text|"
                                            r"define the main text|\(the main text\)|"
                                            r"^the main text retains)", tex, re.M)]
    gate("no '\\ref -> the main text' leftovers", not left, "; ".join(left))
    gate("no S-number counter resets", "\\setcounter{table}{0}" not in tex
         and "\\setcounter{figure}{0}" not in tex, "setcounter found")
    gate("no stale single-seed claim", "A single seed per label condition does not measure" not in tex, "")
    gate("no 'without deletion'", "without deletion" not in tex, "")

    print(f"{len(checks) - len(failures)}/{len(checks)} checks pass")
    for msg in failures:
        print("  FAIL", msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
