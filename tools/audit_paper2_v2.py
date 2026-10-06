#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check the numbers of paper2/manuscript_v2/main.tex against the frozen results.

Each expected value is recomputed from the saved analysis outputs, formatted as the
text prints it, and must appear in the LaTeX source; a missing string is a failure,
because a silently unmatched check is how wrong numbers survive. Two gates besides:
the source must contain no \\pending{...} (the round-2 backbone text), and the abstract
must stay within 250 words.

It certifies that the text agrees with the result files, not that the results are
right or the wording is fair.  Exit code 0 only if everything passes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper2" / "manuscript_v2" / "main.tex"
CD = ROOT / "results" / "courtdyn"


def j(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(x, d=2):
    return f"{x:.{d}f}"


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    # per-cell numbers of the other backbones now live in the supplement (S5); the audit checks the
    # manuscript package (main + supplement), while the PENDING and abstract gates stay on main.tex
    supp = (TEX.parent / "supplementary.tex").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", tex + " " + supp)
    checks, failures = [], []

    def want(label, text):
        checks.append(label)
        if text not in flat:
            failures.append(f"{label}: expected '{text}' in the text")

    def gate(label, ok, detail=""):
        checks.append(label)
        if not ok:
            failures.append(f"{label}: {detail}")

    # ---------------------------------------------------------------- gates
    pend = re.findall(r"\\pending\{", tex.split("\\newcommand{\\pending}", 1)[1].split("\n", 1)[1])
    gate("no PENDING text", not pend, f"{len(pend)} \\pending{{}} block(s) remain")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    words = len(re.sub(r"\$[^$]*\$", "X", abstract).split())
    gate("abstract <= 250 words", words <= 250, f"{words} words")

    # ---------------------------------------------------------------- seeds
    seeds = j(ROOT / "results" / "a100" / "seeds" / "seed_analysis.json")
    rr, ff = seeds["ratio_ranges"], seeds["first_frame"]
    want("seed rho range", f"${f(rr['rho'][0])}$--${f(rr['rho'][1])}$")
    want("seed cm/m range", f"${f(rr['cm_over_m'][0])}$--${f(rr['cm_over_m'][1])}$")
    want("seed px/m range", f"${f(rr['px_over_m'][0])}$--${f(rr['px_over_m'][1])}$")
    want("first-frame defined cells", f"all {ff['defined_cells']} defined cells")
    want("first-frame delta range", f"${f(ff['delta_rho_range'][0])}$--${f(ff['delta_rho_range'][1])}$")

    # ---------------------------------------------------------------- bootstrap
    boot = j(CD / "bootstrap_ci.json")
    nat = boot["native"]
    for fam in ("speed", "path"):
        w = nat[fam]["metre_rho"]["window"]
        want(f"native {fam} rho CI", f"[${f(w['lo'])}$, ${f(w['hi'])}$]")
        w = nat[fam]["metre_margin"]["window"]
        want(f"native {fam} margin CI", f"${f(w['point'], 1)}$ [${f(w['lo'], 1)}$, ${f(w['hi'], 1)}$]")
        w = nat[fam]["pixel_margin"]["window"]
        want(f"native {fam} pixel margin CI",
             f"${f(-w['point'], 1)}$ [${f(-w['hi'], 1)}$, ${f(-w['lo'], 1)}$]")
    lowest = min(v["rho"]["window"]["lo"] for v in boot["seeds"].values())
    want("lowest seed rho lower limit", f"is ${f(lowest)}$")
    ts, tp = nat["speed"]["metre_margin"]["track"], nat["path"]["metre_margin"]["track"]
    want("track-clustered margin CIs", f"[${f(ts['lo'], 1)}$, ${f(ts['hi'], 1)}$] and [${f(tp['lo'], 1)}$, ${f(tp['hi'], 1)}$] points")
    gate("track-clustered margins exclude zero", ts["lo"] > 0 and tp["lo"] > 0, f"{ts['lo']}, {tp['lo']}")
    gate("track clusters = 10", ts["clusters"] == 10, str(ts["clusters"]))

    # ---------------------------------------------------------------- per-item conversion ratios (post hoc)
    ir = j(CD / "m1_mixunit" / "item_ratios.json")
    sp, sq = ir["summary_path"], ir["summary_speed"]
    pct = lambda x: f"{round(100 * x):d}"
    want("item ratios: path within x2", f"${pct(sp['share_within_factor2_range'][0])}$--${pct(sp['share_within_factor2_range'][1])}\\%$ of path items")
    want("item ratios: path within x1.25", f"factor of $1.25$ in ${pct(sp['share_within_25pct_range'][0])}$--${pct(sp['share_within_25pct_range'][1])}\\%$")
    want("item ratios: speed within x1.25", f"speed the share within a factor of $1.25$ is at most ${pct(sq['share_within_25pct_range'][1])}\\%$")

    # ---------------------------------------------------------------- M1
    nums = j(CD / "paper2_v2_numbers.json")
    probe = nums["m1_probe"]
    mix = probe["mixunit"]
    want("mixunit probe acc", f"${100 * mix['acc']:.1f}\\%$ [${100 * mix['lo']:.1f}$, ${100 * mix['hi']:.1f}$]")
    want("base probe acc", f"(${100 * probe['base']['acc']:.1f}\\%$, $p={probe['mixunit_vs_base_p']:.2f}$)")
    p = probe["mixunit_vs_v3_p"]
    mant, exp = f"{p:.1e}".split("e")
    want("mixunit vs v3 p", f"(${100 * probe['v3']['acc']:.1f}\\%$, $p={mant}\\times10^{{{int(exp)}}}$)")
    cost = nums["m1"]["cost_q2"]
    sp = [cost[f"speed/s{s}"]["v3_tmra"] - cost[f"speed/s{s}"]["mix_tmra"] for s in (42, 43, 44)]
    want("speed metre cost range", f"${f(min(sp), 1)}$--${f(max(sp), 1)}$ points")
    drho = [cost[f"{fa}/s{s}"]["v3_rho"] - cost[f"{fa}/s{s}"]["mix_rho"]
            for fa in ("speed", "path") for s in (42, 43, 44)]
    gate("rho cost all positive", all(d > 0 for d in drho), str(drho))
    want("rho cost range", f"${f(min(drho))}$--${f(max(drho))}$")
    pth = [cost[f"path/s{s}"]["mix_tmra"] - cost[f"path/s{s}"]["v3_tmra"] for s in (42, 43, 44)]
    neg = sorted(x for x in pth if x < 0)
    pos = [x for x in pth if x > 0]
    want("path cost values", f"${f(neg[-1], 1)}$ and ${f(neg[0], 1)}$ in two seeds and ${'+' + f(pos[0], 1)}$")
    cells = {k: {int(s): v for s, v in c.items()} for k, c in nums["m1"]["cells"].items()}  # JSON keys are strings
    path_R = sorted({round(cells[f"{e}/path"][s]["R"]) for e in ("Q1", "Q2") for s in (42, 43, 44)})
    gate("path R values", path_R == [76, 100], str(path_R))
    rc = [cells[f"{e}/path"][s]["rho_cm"] for e in ("Q1", "Q2") for s in (42, 43, 44)]
    want("path rho_cm range", f"$\\rho_{{\\mathrm{{cm}}}}={f(min(rc))}$--${f(max(rc))}$")
    meds = [cells[f"{e}/speed"][s]["cm_median"] for e in ("Q1", "Q2") for s in (42, 43, 44)]
    gate("speed cm medians 5x110 + 1x100", sorted(meds) == [100, 110, 110, 110, 110, 110], str(meds))
    prior = cells["Q1/speed"][42]["prior"]
    gate("speed cm prior 110", prior == 110, str(prior))
    verdicts = nums["m1"]["family_claims"]
    gate("M1 path supported, speed not",
         verdicts["E/path"]["supervision_explains_it"] and not verdicts["E/speed"]["supervision_explains_it"]
         and verdicts["T/path"]["supervision_explains_it"] and not verdicts["T/speed"]["supervision_explains_it"],
         str(verdicts))

    # ---------------------------------------------------------------- soccer
    sc = nums["soccer"]
    v3 = [v for k, v in sc.items() if k.startswith("qwen35_v3/")]
    want("soccer rho range", f"$\\rho={f(min(c['rho'] for c in v3))}$--${f(max(c['rho'] for c in v3))}$")
    want("soccer delta rho range", f"${f(min(c['delta_rho'] for c in v3))}$--${f(max(c['delta_rho'] for c in v3))}$")
    want("soccer cm range", f"${f(min(c['cm_over_m'] for c in v3), 1)}$--${f(max(c['cm_over_m'] for c in v3), 1)}$")
    want("soccer px range", f"${f(min(c['px_over_m'] for c in v3))}$--${f(max(c['px_over_m'] for c in v3))}$")
    want("soccer K range", f"$K={f(min(c['K'] for c in v3), 1)}$--${f(max(c['K'] for c in v3), 1)}$")
    want("soccer margin range", f"${f(min(c['margin'] for c in v3), 1)}$ to ${'+' + f(max(c['margin'] for c in v3), 1)}$")
    gate("soccer rho CIs above zero",
         all(c["ci"]["rho"]["window"]["lo"] > 0 for c in v3), "")
    gate("soccer mix rho lower in all six",
         all(sc[k.replace("qwen35_v3", "qwen35_mixunit")]["rho"] < v["rho"]
             for k, v in sc.items() if k.startswith("qwen35_v3/")), "")

    # ---------------------------------------------------------------- backbones (round 1)
    bb = nums["backbones"]
    q = bb["qwen25vl3b"]["round1"]
    want("qwen25 probe", f"${100 * q['probe']['base']:.1f}\\%$ to ${100 * q['probe']['v3']:.1f}\\%$")
    want("qwen25 mix probe", f"${100 * q['probe']['mixunit']:.1f}\\%$ (${'p=' + f(q['p_mix'], 4)}$)")
    want("smol base probe", f"${100 * bb['smol']['round1']['probe']['base']:.1f}\\%$")

    # ---------------------------------------------------------------- backbones (round 2)
    r2 = j(CD / "backbone2_r2" / "verdict.json")["backbones"]
    for key in ("smol", "qwen25vl3b"):
        gate(f"{key} round 2 is in the table", bb[key]["round2"] is not None, "run make_paper2_v2_tables.py")
    q2 = r2["qwen25vl3b"]
    q2c = q2["cells"]
    rho_m = [c["v3_E"]["rho_m"] for c in q2c.values()]
    want("qwen25 r2 metre rho", f"${f(min(rho_m))}$ to ${f(max(rho_m))}$")
    ratios = [c["v3_E"]["R"] for c in q2c.values()]
    want("qwen25 r2 v3 R", f"$R={f(min(ratios))}$--${f(max(ratios))}$")
    gate("qwen25 r2 Q1 path rho reads 0.295",
         f(q2c["Q1_top_0-30/path"]["v3_E"]["rho_m"], 3) == "0.295", "")
    for cell in ("Q1_top_0-30/path", "Q2_top_480-510/path"):
        s4 = q2c[cell]["v3_static4"]
        d = 3 if cell.startswith("Q1") else 2   # Q1 path is 0.295: printed to 3 places (R1 threshold)
        want(f"qwen25 r2 first-frame {cell}", f"${f(s4['rho_full'], d)}$ to ${f(s4['rho_static4'])}$")
    want("qwen25 r2 probe", f"${100 * q2['R3']['accuracy']['base']:.1f}\\%$ to "
                            f"${100 * q2['R3']['accuracy']['v3']:.1f}\\%$")
    want("qwen25 r2 mix probe", f"${100 * q2['R3']['accuracy']['mixunit']:.1f}\\%$ "
                                f"($p={f(q2['R3']['mixunit']['mcnemar_p'])}$)")
    gate("qwen25 r2 verdicts as written",
         (q2["R1"], q2["R2_E"], q2["R2_T"], q2["R3"]["verdict"], q2["R4"])
         == ("intermediate", "intermediate", "intermediate", "replicates", "LIMITED"), "")
    sm = r2["smol"]
    want("smol r2 metre rho", "$0.06$--$0.19$")
    gate("smol r2 rho range", (f(min(c["v3_E"]["rho_m"] for c in sm["cells"].values())),
                               f(max(c["v3_E"]["rho_m"] for c in sm["cells"].values()))) == ("0.06", "0.19"), "")
    gate("smol r2 degenerate at cap", sm["degenerate_at_cap"] and sm["k_star"] == 4, "")
    gate("smol r2 first-frame lowers in one cell",
         sum(c["v3_static4"]["delta"] > 0 for c in sm["cells"].values()) == 1, "")

    # ---------------------------------------------------------------- third backbone (InternVL3-2B)
    iv = j(CD / "backbone3_internvl" / "verdict.json")["backbones"]["internvl3_2b"]
    ic = iv["cells"]
    gate("internvl verdicts as written",
         (iv["R1"], iv["R2_E"], iv["R2_T"], iv["R3"]["verdict"], iv["R4"])
         == ("intermediate", "LIMITED", "replicates", "fails", "replicates"), "")
    gate("internvl k*=2", j(CD / "backbone3_internvl" / "internvl3_2b" / "stopping.json")["k_star"] == 2, "")
    irho = [c["v3_E"]["rho_m"] for c in ic.values()]
    want("internvl metre rho", f"$\\rho$ is ${f(min(irho))}$--${f(max(irho))}$")
    idel = [c["v3_static4"]["delta"] for c in ic.values()]
    gate("internvl first-frame lowers all four", all(d > 0 for d in idel), "")
    want("internvl first-frame range", f"$\\Delta\\rho={f(min(idel))}$--${f(max(idel))}$")
    want("internvl path R", f"$R={f(ic['Q1_top_0-30/path']['v3_E']['R'])}$ and "
                            f"${f(ic['Q2_top_480-510/path']['v3_E']['R'])}$")
    want("internvl speed R", f"$R={f(ic['Q1_top_0-30/speed']['v3_E']['R'])}$ and "
                             f"${f(ic['Q2_top_480-510/speed']['v3_E']['R'])}$")
    pR = [ic[k][w]["R"] for k in ("Q1_top_0-30/path", "Q2_top_480-510/path") for w in ("mix_E", "mix_T")]
    pr = [ic[k][w]["posthoc_rho_cm"] for k in ("Q1_top_0-30/path", "Q2_top_480-510/path") for w in ("mix_E", "mix_T")]
    want("internvl mix path R/rho", f"$R={min(pR):.0f}$--${max(pR):.0f}$, $\\rho_{{\\mathrm{{cm}}}}={f(min(pr))}$--${f(max(pr))}$")
    want("internvl Q2 speed T rho", f"$\\rho_{{\\mathrm{{cm}}}}={f(ic['Q2_top_480-510/speed']['mix_T']['posthoc_rho_cm'])}$")
    acc = iv["R3"]["accuracy"]
    want("internvl probe", f"${100 * acc['base']:.1f}\\%$ to ${100 * acc['v3']:.1f}\\%$ "
                           f"($p={iv['R3']['v3']['mcnemar_p']:.2g}$)")
    want("internvl mix probe", f"${100 * acc['mixunit']:.1f}\\%$ ($p={iv['R3']['mixunit']['mcnemar_p']:.2g}$)")
    # ---------------------------------------------------------------- A100: Pixtral-12B
    px = j(CD / "backbone4_a100" / "verdict.json")["backbones"]["pixtral_12b"]
    pc = px["cells"]
    gate("pixtral verdicts as written",
         (px["R1"], px["R2_E"], px["R2_T"], px["R3"]["verdict"], px["R4"])
         == ("replicates", "intermediate", "LIMITED", "replicates", "LIMITED"), "")
    gate("pixtral k*=2", j(CD / "backbone4_a100" / "pixtral_12b" / "stopping.json")["k_star"] == 2, "")
    prho = [c["v3_E"]["rho_m"] for c in pc.values()]
    want("pixtral metre rho", f"$\\rho={f(min(prho))}$--${f(max(prho))}$")
    pR = [c["v3_E"]["R"] for c in pc.values()]
    want("pixtral v3 R", f"$R={f(min(pR))}$--${f(max(pR))}$")
    want("pixtral first-frame speed", f"$\\Delta\\rho={f(pc['Q1_top_0-30/speed']['v3_static4']['delta'])}$ and "
                                      f"${f(pc['Q2_top_480-510/speed']['v3_static4']['delta'])}$")
    gate("pixtral path first-frame collapses to one value",
         all(pc[k]["v3_static4"]["distinct_static4"] == 1 and pc[k]["v3_static4"]["delta"] is None
             for k in ("Q1_top_0-30/path", "Q2_top_480-510/path")), "")
    mixR = sorted({round(c["mix_E"]["R"]) for c in pc.values()})
    gate("pixtral mix ratios 100 and 143", mixR == [100, 143], str(mixR))
    live_pass = [c["mix_E"]["posthoc_rho_cm"] for c in pc.values()
                 if c["mix_E"]["distinct_cm"] > 2 and c["mix_E"]["posthoc_rho_cm"] >= 0.30]
    gate("pixtral 2 of 3 live explicit cells track", len(live_pass) == 2 and px["R2_E_live"] == 3, "")
    want("pixtral tracking rhos", f"$\\rho_{{\\mathrm{{cm}}}}={f(min(live_pass))}$ and ${f(max(live_pass))}$")
    pa = px["R3"]["accuracy"]
    want("pixtral probe", f"${100 * pa['base']:.1f}\\%$ to ${100 * pa['v3']:.1f}\\%$")
    want("pixtral probe p", "$p=4.3\\times10^{-6}$")
    gate("pixtral probe p value", f"{px['R3']['v3']['mcnemar_p']:.1e}" == "4.3e-06", "")
    want("pixtral mix probe", f"${100 * pa['mixunit']:.1f}\\%$ ($p=3.1\\times10^{{-4}}$)")
    gate("pixtral mix p value", f"{px['R3']['mixunit']['mcnemar_p']:.1e}" == "3.1e-04", "")

    # ---------------------------------------------------------------- M1-fine
    mf = j(CD / "m1_fine" / "verdict.json")
    gate("m1-fine verdict NOT SUPPORTED", mf["verdict"] == "NOT SUPPORTED", mf["verdict"])
    mc = mf["cells"]
    sp = [mc[k]["fine"] for k in ("Q1_top_0-30/speed", "Q2_top_480-510/speed")]
    pa_ = [mc[k]["fine"] for k in ("Q1_top_0-30/path", "Q2_top_480-510/path")]
    gate("m1-fine speed medians 100", all(c["cm_median"] == 100.0 for c in sp), "")
    gate("m1-fine speed distinct 2-3", sorted(c["distinct_cm"] for c in sp) == [2, 3], "")
    want("m1-fine speed rho", f"$\\rho_{{\\mathrm{{cm}}}}={f(min(c['rho_cm'] for c in sp))}$ and "
                              f"${f(max(c['rho_cm'] for c in sp))}$")
    want("m1-fine path R", f"$R={min(c['R'] for c in pa_):.0f}$--${max(c['R'] for c in pa_):.0f}$")
    want("m1-fine path rho", f"$\\rho_{{\\mathrm{{cm}}}}={f(min(c['rho_cm'] for c in pa_))}$--"
                             f"${f(max(c['rho_cm'] for c in pa_))}$")

    # "no metre-only path cell on either [Qwen2.5-VL-3B, InternVL3-2B] converts (R <= 1.20)"
    path_R = [c["v3_E"]["R"] for k, c in r2["qwen25vl3b"]["cells"].items() if k.endswith("path")]
    path_R += [c["v3_E"]["R"] for k, c in ic.items() if k.endswith("path")]
    gate("qwen25/internvl path R bound is the max", f(max(path_R)) == "1.20", f"max path R = {max(path_R)}")
    want("qwen25/internvl path R bound", r"$R\le1.20$")

    # ---------------------------------------------------------------- A100: Idefics3-8B
    idf = j(CD / "backbone4_a100" / "verdict.json")["backbones"]["idefics3_8b"]
    dc = idf["cells"]
    gate("idefics3 verdicts as written",
         (idf["R1"], idf["R2_E"], idf["R2_T"], idf["R3"]["verdict"], idf["R4"])
         == ("LIMITED", "intermediate", "intermediate", "intermediate", "LIMITED"), "")
    gate("idefics3 k*=2", j(CD / "backbone4_a100" / "idefics3_8b" / "stopping.json")["k_star"] == 2, "")
    gate("idefics3 metre arm 2 distinct in 3 of 4 cells",
         sorted(c["v3_E"]["distinct_m"] for c in dc.values()) == [2, 2, 2, 3], "")
    mixR = [c[w]["R"] for c in dc.values() for w in ("mix_E", "mix_T")]
    want("idefics3 mix R", f"$R={min(mixR):.0f}$--${max(mixR):.0f}$")
    mrho = [c[w]["posthoc_rho_cm"] for c in dc.values() for w in ("mix_E", "mix_T")]
    gate("idefics3 mix rho_cm <= 0.27", f(max(mrho)) == "0.27", f(max(mrho)))
    ia = idf["R3"]["accuracy"]
    want("idefics3 probe", rf"${100 * ia['base']:.1f}\%$ to ${100 * ia['v3']:.1f}\%$")
    gate("idefics3 probe p", f"{idf['R3']['v3']['mcnemar_p']:.1e}" == "9.0e-08", "")
    want("idefics3 probe p text", r"$p=9.0\times10^{-8}$")
    want("idefics3 mix probe", rf"${100 * ia['mixunit']:.1f}\%$")
    gate("idefics3 mix p", f"{idf['R3']['mixunit']['mcnemar_p']:.1e}" == "1.5e-04", "")
    want("idefics3 mix p text", r"$p=1.5\times10^{-4}$")

    # ---------------------------------------------------------------- A100: Gemma3-12B
    gm = j(CD / "backbone4_a100" / "verdict.json")["backbones"]["gemma3_12b"]
    gc = gm["cells"]
    gate("gemma verdicts as written",
         (gm["R1"], gm["R2_E"], gm["R2_T"], gm["R3"]["verdict"], gm["R4"])
         == ("fails", "LIMITED", "intermediate", "fails", "replicates"), "")
    gate("gemma k*=2", j(CD / "backbone4_a100" / "gemma3_12b" / "stopping.json")["k_star"] == 2, "")
    grho = [c["v3_E"]["rho_m"] for c in gc.values()]
    want("gemma metre rho", rf"($\rho={f(min(grho))}$--${f(max(grho))}$)")
    gd = [c["v3_static4"]["delta"] for c in gc.values()]
    want("gemma first-frame", rf"$\Delta\rho={f(min(gd))}$--${f(max(gd))}$")
    want("gemma v3 R", f"$R={f(gc['Q1_top_0-30/speed']['v3_E']['R'])}$ and "
                       f"${f(gc['Q1_top_0-30/path']['v3_E']['R'])}$ on Q1, "
                       f"${f(gc['Q2_top_480-510/speed']['v3_E']['R'])}$ and "
                       f"${f(gc['Q2_top_480-510/path']['v3_E']['R'])}$ on Q2")
    gate("gemma converts in 3 of 4 cells", sum(c["v3_E"]["R"] >= 10 for c in gc.values()) == 3, "")
    gcm = [c["v3_E"]["posthoc_rho_cm"] for c in gc.values()]
    want("gemma v3 rho_cm", rf"$\rho_{{\mathrm{{cm}}}}={f(min(gcm))}$--${f(max(gcm))}$")
    gate("gemma cm rho below metre rho in every cell",
         all(c["v3_E"]["posthoc_rho_cm"] < c["v3_E"]["rho_m"] for c in gc.values()), "")
    ga = gm["R3"]["accuracy"]
    want("gemma probe", rf"${100 * ga['base']:.1f}\%$ unadapted, ${100 * ga['v3']:.1f}\%$ after metre-only "
                        f"training ($p={gm['R3']['v3']['mcnemar_p']:.2g}$)")
    want("gemma mix probe", rf"${100 * ga['mixunit']:.1f}\%$ after mixed-unit training "
                            f"($p={gm['R3']['mixunit']['mcnemar_p']:.2g}$)")
    gate("gemma mix speed 2 distinct (explicit)",
         all(gc[k]["mix_E"]["distinct_cm"] == 2 for k in ("Q1_top_0-30/speed", "Q2_top_480-510/speed")), "")
    gp = sorted(round(gc[k]["mix_E"]["R"]) for k in ("Q1_top_0-30/path", "Q2_top_480-510/path"))
    want("gemma mix path R", f"$R={gp[0]}$ and ${gp[1]}$")

    # ---------------------------------------------------------------- post hoc: prefix-greedy account (unitdyn_prop2_posthoc)
    pp = j(CD / "unitdyn_posthoc" / "prop2_conditional_mode.json")
    sp = [r for r in pp if r["family"] == "speed"]
    m1 = [r for r in sp if r["adapter"].startswith("M1 s")]
    fine = [r for r in sp if r["adapter"] == "M1-fine s42"]
    gate("prefix-greedy value 110 (M1) / 100 (M1-fine)",
         all(r["global_prefix_greedy"] == 110 for r in m1) and all(r["global_prefix_greedy"] == 100 for r in fine), "")
    gate("cm target modes 90 / 79", all(r["global_mode_cm_target"] == 90 for r in m1) and all(r["global_mode_cm_target"] == 79 for r in fine), "")
    gm = [r["match"]["global_prefix"] for r in m1]
    want("prefix match M1", rf"${round(100 * min(gm))}$--${round(100 * max(gm))}\%$")
    gf = [r["match"]["global_prefix"] for r in fine]
    want("prefix match M1-fine", rf"${round(100 * min(gf))}$--${round(100 * max(gf))}\%$")
    want("conditional prefix max", rf"at most ${round(100 * max(r['match']['cond_prefix'] for r in m1))}\%$")
    pa2 = [r for r in pp if r["family"] == "path" and r["adapter"].startswith("M1 s")]
    gate("path prefix-greedy 140", all(r["global_prefix_greedy"] == 140 for r in pa2), "")
    # the pool itself is withheld from the public release; its aggregate properties are shipped as
    # m1_mixunit/pool_stats.json (tools/mixunit_pool_stats.py) and recomputed live when the pool exists
    from mixunit_pool_stats import load_stats
    share = {fam: load_stats()["families"][fam]["cm_first_digit_majority_share"] for fam in ("speed", "path")}
    want("first-digit shares", rf"(${round(100 * share['path'])}\%$ share the majority first digit, against ${round(100 * share['speed'])}\%$ for speed)")


    # ---------------------------------------------------------------- revision: three-seed table, RTC, probe, T36
    agg = j(CD / "backbone_seeds3" / "aggregate.json")["backbones"]
    ps = lambda b, r: agg[b][r]["per_seed"]
    gate("pixtral R1 replicates at seeds 42 and 43", ps("pixtral_12b", "R1")["42"] == "replicates" and ps("pixtral_12b", "R1")["43"] == "replicates", "")
    gate("internvl R1 int at 42, rep at 43", ps("internvl3_2b", "R1")["42"] == "intermediate" and ps("internvl3_2b", "R1")["43"] == "replicates", "")
    gate("gemma R1 fails at 42, int at 43", ps("gemma3_12b", "R1")["42"] == "fails" and ps("gemma3_12b", "R1")["43"] == "intermediate", "")
    gate("R4 replicates on internvl+gemma at 42/43, pixtral at 43",
         all(ps(b, "R4")[sd] == "replicates" for b in ("internvl3_2b", "gemma3_12b") for sd in ("42", "43")) and ps("pixtral_12b", "R4")["43"] == "replicates", "")
    g43 = j(CD / "backbone4_ext" / "verdict.json")["backbones"]["gemma3_12b"]["cells"]
    want("gemma s43 path R", f"($R={f(g43['Q1_top_0-30/path']['v3_E']['R'], 1)}$ and ${f(g43['Q2_top_480-510/path']['v3_E']['R'], 1)}$)")
    gate("gemma s43 speed cells do not convert", all(g43[k]["v3_E"]["R"] < 10 for k in ("Q1_top_0-30/speed", "Q2_top_480-510/speed")), "")
    gate("seeds table input", "\\input{tables/backbone_seeds}" in tex, "")
    # RTC (results/courtdyn/rtc, rtc_rep, backbone4_ext/rtc)
    r42 = j(CD / "rtc" / "verdict.json")["hypotheses"]
    rep_ = j(CD / "rtc_rep" / "verdict.json")["groups"]
    ext_ = j(CD / "backbone4_ext" / "rtc" / "verdict.json")["backbones"]
    h = lambda d, k: d["hypotheses"][k] if "hypotheses" in d else d[k]
    gate("RTC H1 8/5/4", (r42["H1_unseen_units_rtc"][1], rep_["seed43"]["hypotheses"]["H1_unseen_units_rtc"][1],
                          rep_["seed44"]["hypotheses"]["H1_unseen_units_rtc"][1]) == (8, 5, 4), "")
    gate("RTC H6 v3 6/6/6", (r42["H6_format_control_v3s42"][1], rep_["seed43"]["hypotheses"]["H6_format_control_v3s43"][1],
                             rep_["seed44"]["hypotheses"]["H6_format_control_v3s44"][1]) == (6, 6, 6), "")
    gate("RTC H6 M1 6/5/6", (r42["H6_format_control_m1s42"][1], rep_["seed43"]["hypotheses"]["H6_format_control_m1s43"][1],
                             rep_["seed44"]["hypotheses"]["H6_format_control_m1s44"][1]) == (6, 5, 6), "")
    gate("RTC H3 fixed x3", all(x["H3_speed_cm_rtc"][0] == "fixed" for x in (r42, rep_["seed43"]["hypotheses"], rep_["seed44"]["hypotheses"])), "")
    gate("RTC H4 holds at one seed of three", [x["H4_metre_noninferiority"][0] for x in (r42, rep_["seed43"]["hypotheses"], rep_["seed44"]["hypotheses"])].count("holds") == 1, "")
    # RTC other backbones: the per-cell unseen-unit ratios of the generated table back the main-text sentence
    rows = {}
    for line in (TEX.parent / "tables" / "rtc_supp.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) == 8 and cells[0] in ("InternVL3-2B", "Pixtral-12B", "Gemma3-12B"):
            rows.setdefault(cells[0], []).extend(float(re.match(r"[-\d.]+", c).group()) for c in cells[2:6])
    conv = lambda r: 1 / 1.5 <= r <= 1.5
    iv, px, gm = rows["InternVL3-2B"], rows["Pixtral-12B"], rows["Gemma3-12B"]
    gate("RTC InternVL3 unseen ratios 0.10-0.30", len(iv) == 8 and min(iv) >= 0.10 and max(iv) <= 0.30, str(iv))
    want("RTC InternVL3 text", "returns $0.10$--$0.30$ of the exact factor in all eight unseen-unit cells")
    gate("RTC Pixtral factor in 6 cells", sum(map(conv, px)) == 6, str(px))
    gate("RTC Gemma 0.72-1.00 in 6 cells, 0.28 in 2", sum(map(conv, gm)) == 6 and
         all(0.72 <= r <= 1.0 for r in gm if conv(r)) and [r for r in gm if not conv(r)] == [0.28, 0.28], str(gm))
    want("RTC Gemma text", "returns $0.72$--$1.00$ of the factor in six cells and $0.28$ in two")
    gate("causal wording scoped", "causes the failure" not in flat and "ties the two units" not in flat
         and "ties the units" not in flat and "converts its own reading" not in flat, "")
    gate("RTC other backbones H1 0/1/3", (rep_["internvl3_2b"]["hypotheses"]["H1_unseen_units_rtc"][1], ext_["pixtral_12b"]["hypotheses"]["H1_unseen_units_rtc"][1],
                                          ext_["gemma3_12b"]["hypotheses"]["H1_unseen_units_rtc"][1]) == (0, 1, 3), "")
    gate("RTC format control 0 on the three metre-only adapters, 5 on pixtral mix",
         (rep_["internvl3_2b"]["hypotheses"]["H6_format_control_ivlv3"][1], ext_["pixtral_12b"]["hypotheses"]["H6_format_control_pixtral_12b_v3"][1],
          ext_["gemma3_12b"]["hypotheses"]["H6_format_control_gemma3_12b_v3"][1], ext_["pixtral_12b"]["hypotheses"]["H6_format_control_pixtral_12b_mix"][1]) == (0, 0, 0, 5), "")
    pxc = ext_["pixtral_12b"]["cells"]
    gate("pixtral RTC metre path readings 2 distinct", all(pxc[k]["distinct"] == 2 for k in pxc if k.startswith("rtc_pixtral_12b|rtc|") and k.endswith("|m|path")), "")
    want("RTC text", "converts six of the eight unseen-unit cells at every seed")
    want("RTC text 2", "eight of eight at seed 42 but five and four at seeds 43 and 44")
    want("RTC text 3", "zero, one and three cells of eight")
    # representation probe stability (post hoc, 10 fold seeds)
    st = j(CD / "repr_probe" / "stability.json")["table"]
    gate("probe H1 10/10 everywhere", all(st[m][f"{fam}/H1"].get("True", 0) == 10 for m in st for fam in ("speed", "path")), "")
    gate("probe H3 v3 10/10, native 10 and 9, base/mixunit 0",
         st["v3"]["speed/H3"].get("True", 0) == 10 and st["v3"]["path/H3"].get("True", 0) == 10
         and st["native"]["path/H3"].get("True", 0) == 10 and st["native"]["speed/H3"].get("True", 0) == 9
         and all(st[m][f"{fam}/H3"].get("True", 0) == 0 for m in ("base", "mixunit") for fam in ("speed", "path")), "")
    # nested re-analysis (post hoc): the main text states that H1 holds and H3 drops in all four models
    nested = j(CD / "repr_probe" / "nested_reanalysis.json")["models"]
    cells = [nested[m][fam] for m in ("base", "native", "v3", "mixunit") for fam in ("speed", "path")]
    gate("nested probe: 8 cells with 200 permutations", len(cells) == 8 and all(c["perms"] == 200 for c in cells), "")
    gate("nested probe: H1 holds in every cell", all(c["H1_holds"] for c in cells), "")
    gate("nested probe: motion-dependent in every cell", all(c["H3"]["motion_dependent"] for c in cells), "")
    drops = [c["H3"]["rho_full_q2_oof"] - c["H3"]["rho_static4_oof"] for c in cells]
    want("nested probe: drop range", f"drops by ${min(drops):.2f}$--${max(drops):.2f}$ under first-frame repetition")
    import repr_probe_nested_table as NT
    gate("repr_probe_nested.tex regenerated", NT.build() == (TEX.parent / "tables" / "repr_probe_nested.tex").read_text(encoding="utf-8"), "")
    gate("probe contrast claim removed", "appears only after metre-only fine-tuning" not in flat
         and "keep it (0 of 10)" not in flat, "")
    # T36 (descriptive)
    t36 = j(CD / "t36" / "summary.json")["models"]
    live = lambda m: [c for c in t36[m]["cells"] if not c.get("missing") and c["live_unit"]]
    iv = [c["R_cm"] for c in live("internvl3-8b")]; qw = [c["R_cm"] for c in live("qwen25vl-7b")]
    want("T36 R_cm ranges", f"$R_{{\\mathrm{{cm}}}}={min(iv):.0f}$--${max(iv):.0f}$ and ${min(qw):.0f}$--${max(qw):.0f}$")
    gate("T36 internvl3-8b all four cells live", len(live("internvl3-8b")) == 4, "")
    rho_max = max(c["arms"]["m_height"]["rho"] for m in ("internvl3-8b", "qwen25vl-7b") for c in live(m))
    want("T36 rho bound", f"metre $\\rho\\le{rho_max:.2f}$")
    gate("T36 bf16 cell invalid", t36["qwen35-4b-bf16"]["invalid"] is True, "")
    # ---------------------------------------------------------------- three-seed labels vs text
    # Cross-backbone wording must follow results/courtdyn/backbone_seeds3/aggregate.json: the macro
    # file is regenerated from it, hand-written final labels match it, and claims the labels have
    # overturned are absent.
    agg = j(CD / "backbone_seeds3" / "aggregate.json")["backbones"]
    lab = lambda k, r: agg[k][r]["label"]
    sys.path.insert(0, str(ROOT / "tools"))
    import make_paper2_extra_tables as XT
    gen = XT.seed_labels_tex(XT.seeds_block())
    on_disk = (TEX.parent / "tables" / "seed_labels.tex").read_text(encoding="utf-8")
    gate("seed_labels.tex regenerated from aggregate.json", gen == on_disk,
         "run tools/make_paper2_extra_tables.py")
    gate("seed_labels.tex input", "\\input{tables/seed_labels}" in tex, "")
    used = sorted(set(re.findall(r"\\seedlabel\{(\w+)\}\{(\w+)\}", tex)))
    still = [f"{k}/{r}" for k, r in used if k in agg and lab(k, r) == "pending"]
    gate("no PENDING three-seed label in the text", not still, "waiting for: " + ", ".join(still))
    for k in ("internvl3_2b", "gemma3_12b"):
        gate(f"{k} R4 holds 3/3 (text says so)", lab(k, "R4") == "holds at all three seeds", lab(k, "R4"))
        gate(f"{k} R3 fails across seeds (text says so)", lab(k, "R3") == "fails across seeds", lab(k, "R3"))
    want("R4 3/3 text", "holds at all three seeds on InternVL3-2B and Gemma3-12B")
    want("R3 fails text", "R3 fails on InternVL3-2B and Gemma3-12B")
    g44 = j(CD / "backbone_seeds3" / "s44" / "verdict.json")["backbones"]["gemma3_12b"]["cells"]
    g44_max = max(c["v3_E"]["R"] for c in g44.values())
    gate("Gemma s44: no metre-only cell converts (explicit)", g44_max < 10, f"max R {g44_max:.2f}")
    want("Gemma s44 max R (supplement)", f"$R\\le{g44_max:.2f}$")
    g43 = j(CD / "backbone4_ext" / "verdict.json")["backbones"]["gemma3_12b"]
    gate("Gemma arithmetic never below base (s43, s44)",
         not g43["R3"]["v3"]["below_base"]
         and not j(CD / "backbone_seeds3" / "s44" / "verdict.json")["backbones"]["gemma3_12b"]["R3"]["v3"]["below_base"], "")
    if lab("gemma3_12b", "R1") != "fails across seeds":
        gate("no Gemma 'counterexample' wording", "counterexample" not in flat, "Gemma R1 is " + lab("gemma3_12b", "R1"))
        gate("no 'but not on Gemma' wording", "but not on Gemma" not in flat, "")
    gate("abstract Pixtral claim allowed by its R1 label",
         lab("pixtral_12b", "R1") in ("holds at all three seeds", "holds at two of three seeds", "pending")
         or "also on Pixtral-12B" not in abstract, lab("pixtral_12b", "R1"))
    # Limitations hard-codes "Idefics3-8B is untestable on the test clips": only while every judged
    # seed has R1 LIMITED.
    idf = [v for v in agg["idefics3_8b"]["R1"]["per_seed"].values() if v is not None]
    if "Idefics3-8B is untestable on the test clips" in flat:
        gate("Idefics3 'untestable' wording matches R1 at every judged seed",
             all(v == "LIMITED" for v in idf), f"R1 per seed: {idf}")
    # SmolVLM2 "degenerate at the cap at all three seeds": the dev rule must fail at seeds 42, 43, 44.
    import run_backbone_round2 as R2
    smol_dev = [CD / "backbone2_r2" / "smol" / "dev" / "v3_epoch4" / "summary.json",
                CD / "backbone_seeds3" / "s43" / "smol" / "dev" / "v3_epoch4" / "summary.json",
                CD / "backbone_seeds3" / "s44" / "smol" / "dev" / "v3_epoch4" / "summary.json"]
    outcomes = [R2.dev_passes(p) for p in smol_dev if p.is_file()]
    distinct = [f["n_distinct"] for _, fams in outcomes for f in fams.values()]
    gate("SmolVLM2 dev rule fails at all three seeds", len(outcomes) == 3 and not any(ok for ok, _ in outcomes),
         f"{len(outcomes)} seeds judged, passes={[ok for ok, _ in outcomes]}")
    gate("SmolVLM2 distinct range one to four", distinct and min(distinct) == 1 and max(distinct) == 4, str(distinct))
    want("SmolVLM2 three-seed wording", "never met this criterion at any of three seeds")
    want("SmolVLM2 limitation wording", "degenerate at the\n  four-epoch cap at all three seeds".replace("\n  ", " "))
    # Idefics3 seed 43 (supplement): live cells, rho range, R bound, arithmetic drop
    i43 = j(CD / "backbone_seeds3" / "s43" / "verdict.json")["backbones"]["idefics3_8b"]
    cells43 = [c["v3_E"] for c in i43["cells"].values()]
    gate("Idefics3 s43 R1/R3/R4 replicate", (i43["R1"], i43["R3"]["verdict"], i43["R4"]) == ("replicates",) * 3,
         f"{i43['R1']}, {i43['R3']['verdict']}, {i43['R4']}")
    i44 = j(CD / "backbone_seeds3" / "s44" / "verdict.json")["backbones"].get("idefics3_8b")
    if i44:
        gate("Idefics3 s44 R1/R4 intermediate, R3 replicates (supplement sentence)",
             (i44["R1"], i44["R4"], i44["R3"]["verdict"]) == ("intermediate", "intermediate", "replicates"),
             f"{i44['R1']}, {i44['R4']}, {i44['R3']['verdict']}")
        gate("Idefics3 labels: R1/R4 not established, R3 2/3",
             (lab("idefics3_8b", "R1"), lab("idefics3_8b", "R4"), lab("idefics3_8b", "R3"))
             == ("not established", "not established", "holds at two of three seeds"), "")
        want("Idefics3 s44 wording", "At seed 44 the cells are live again but R1 and R4 are intermediate")
    want("Idefics3 s43 distinct range", f"{min(c['distinct_m'] for c in cells43)}--{max(c['distinct_m'] for c in cells43)} distinct metre answers")
    want("Idefics3 s43 rho range", f"$\\rho={min(c['rho_m'] for c in cells43):.2f}$--${max(c['rho_m'] for c in cells43):.2f}$")
    want("Idefics3 s43 R bound", f"$R\\le{max(c['R'] for c in cells43):.2f}$")
    a43 = i43["R3"]["accuracy"]
    want("Idefics3 s43 arithmetic", f"from ${a43['base'] * 100:.1f}\\%$ to ${a43['v3'] * 100:.1f}\\%$")
    if lab("gemma3_12b", "R1") == "seed-dependent":
        want("abstract Gemma seed-dependent", "on Gemma3-12B it depends on the seed")
    print(f"{len(checks) - len(failures)}/{len(checks)} checks pass")
    for msg in failures:
        print("  FAIL", msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
