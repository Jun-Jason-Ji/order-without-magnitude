#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply the preregistered RTC rules (results/courtdyn/rtc/PREREGISTRATION.json) mechanically.

Reads the eval cells written by tools/run_rtc.py (+ the published plain m cells of the comparators),
re-parses RTC-format answers from raw_answer with the preregistered regex, and writes
results/courtdyn/rtc/verdict.json, verdict.md and fig_rtc_units.{pdf,png}.  Cells not yet run are
reported as "pending"; a hypothesis with any pending input is "pending", never decided early.

  python tools/analyze_rtc.py [--root results/courtdyn/rtc]   (--root only for tests)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CD = ROOT / "results" / "courtdyn"
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
SPEED, PATH = "dynamics_speed_player", "dynamics_path_player"
FAM = {SPEED: "speed", PATH: "path"}
RTC_RE = re.compile(r"^\s*(?P<r>[+-]?\d+(?:\.\d+)?)\s*(?P<ru>m/s|m)\s*->\s*"
                    r"(?P<v>[+-]?\d+(?:\.\d+)?)\s*(?P<vu>[A-Za-z/]+)\s*\.?\s*$")
NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
UNSEEN = ["U1", "U2"]
EXPECTED_UNIT = {("U1", SPEED): "km/h", ("U1", PATH): "ft", ("U2", SPEED): "ft/s", ("U2", PATH): "mm",
                 ("cm", SPEED): "cm/s", ("cm", PATH): "cm", ("m", SPEED): "m/s", ("m", PATH): "m",
                 ("pxK", SPEED): "px/s", ("pxK", PATH): "px"}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx <= 0 or syy <= 0 else sxy / math.sqrt(sxx * syy)


def parse_row(row, rtc):
    raw = (row.get("raw_answer") or "").strip()
    if not rtc:
        return (float(raw) if NUMBER.fullmatch(raw) and math.isfinite(float(raw)) else None), None, None
    m = RTC_RE.fullmatch(raw)
    if not m:
        return None, None, None
    return float(m["v"]), float(m["r"]), m["vu"]


def key(r):
    return (r["category"], r["track"], tuple(r["window"]))


def load(path, rtc):
    if not path or not path.is_file():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            v, reading, vu = parse_row(r, rtc)
            out[key(r)] = dict(r, v=v, reading=reading, vu=vu)
    return out


def factor(arm, fam_rows):
    if arm == "pxK":
        ks = [r["reference_value"] / r["metre_reference"] for r in fam_rows if r.get("metre_reference")]
        return st.median(ks)
    return {"m": 1.0, "cm": 100.0}.get(arm) or float(fam_rows[0]["unit_factor"])


def cell_stats(unit_rows, metre_rows, arm, family, rtc):
    rows = [r for r in unit_rows.values() if r["category"] == family]
    parsed = [r for r in rows if r["v"] is not None]
    preds = [r["v"] for r in parsed]
    rho = spearman(preds, [r["reference_value"] for r in parsed]) if len(parsed) >= 3 else None
    f = factor(arm, rows) if arm != "m" else 1.0
    ratios = []
    if metre_rows is not None:
        for r in parsed:
            m = metre_rows.get(key(r))
            if m and m["v"] is not None and m["v"] > 0:
                ratios.append(r["v"] / m["v"])
    R = st.median(ratios) if ratios else None
    out = dict(n=len(rows), parse_rate=len(parsed) / len(rows) if rows else 0.0,
               distinct=len(set(preds)), degenerate=len(set(preds)) <= 2, rho=rho,
               R=R, n_pairs=len(ratios), factor=f, R_over_f=(R / f) if R else None)
    base_ok = bool(out["parse_rate"] >= 0.5 and not out["degenerate"] and R is not None
                   and rho is not None and rho >= 0.30)
    # ERRATA.md item 2 (amendment recorded before any training or evaluation): the conversion
    # band is [f/1.5, 1.5 f]; the band as first registered, [0.5 f, 2 f], admits a non-converting
    # model whose unit answers sit ~2x its metre answers when f ~ 3 (km/h, ft, ft/s).  Both are
    # reported; verdicts use the amended band.
    out["converts_as_registered"] = bool(base_ok and 0.5 * f <= R <= 2 * f)
    out["converts"] = bool(base_ok and f / 1.5 <= R <= 1.5 * f)
    if rtc:                                   # descriptive: the stated reading and the arithmetic
        rd = [r for r in parsed if r["reading"] is not None]
        out["reading_rho_vs_metre_ref"] = spearman([r["reading"] for r in rd],
                                                   [r.get("metre_reference", r["reference_value"]) for r in rd]) \
            if len(rd) >= 3 else None
        internal = [r["v"] / r["reading"] for r in rd if r["reading"] > 0]
        out["median_value_over_reading"] = st.median(internal) if internal else None
        out["unit_token_ok"] = (sum(r["vu"] == EXPECTED_UNIT[(arm, family)] for r in rd) / len(rd)) if rd else None
    return out


def rule_count(cells, need, fail_max):
    if any(c is None for c in cells):
        return "pending", None
    k = sum(c["converts"] for c in cells)
    return ("holds" if k >= need else "fails" if k <= fail_max else "intermediate"), k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(CD / "rtc"))
    args = ap.parse_args()
    out_dir = Path(args.root)
    ev = out_dir / "eval"
    pre = json.loads((CD / "rtc" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pre["rules"].encode("utf-8")).hexdigest() == pre["rules_sha256"]

    def cell_dir(label, seq, mode, arms, fmt):
        return ev / f"{label}_{seq}_{mode}_{'-'.join(arms)}_{fmt}"

    published_m = {"v3s42": lambda s: CD / "revision_execution_20260912" / f"eventholdout_v3_s42_{s}_full" / "m_height.jsonl",
                   "m1s42": lambda s: CD / "m1_mixunit" / "eval" / f"mixunit_{s}_full" / "m_height.jsonl",
                   "m1fines42": lambda s: next(iter(sorted((CD / "m1_fine" / "eval").glob(f"*{s}*full*/m_height.jsonl"))), None)
                   if (CD / "m1_fine" / "eval").is_dir() else None}
    models = {"rtc_s42": ["rtc"], "v3s42": ["plain", "rtc"], "m1s42": ["plain", "rtc"], "m1fines42": ["plain"]}
    res = {}
    for label, fmts in models.items():
        for fmt in fmts:
            for seq in SEQS:
                if fmt == "rtc":
                    d = cell_dir(label, seq, "full", ["m", "cm", "U1", "U2", "pxK"], "rtc")
                    metre = load(d / "rtc_m.jsonl", True)
                    arms = ["m", "cm", "U1", "U2", "pxK"]
                else:
                    d = cell_dir(label, seq, "full", ["U1", "U2", "pxK"], "plain")
                    metre = load(published_m[label](seq), False)
                    arms = ["U1", "U2", "pxK"]
                for arm in arms:
                    rows = load(d / f"{fmt}_{arm}.jsonl", fmt == "rtc")
                    for family in (SPEED, PATH):
                        k = f"{label}|{fmt}|{seq}|{arm}|{FAM[family]}"
                        complete = rows is not None and (d / "summary.json").is_file() and metre is not None
                        res[k] = cell_stats(rows, metre, arm, family, fmt == "rtc") if complete else None
    # first-frame x4 for RTC
    ff = {}
    for seq in SEQS:
        full = load(cell_dir("rtc_s42", seq, "full", ["m", "cm", "U1", "U2", "pxK"], "rtc") / "rtc_m.jsonl", True)
        s4d = cell_dir("rtc_s42", seq, "static4", ["m"], "rtc")
        s4 = load(s4d / "rtc_m.jsonl", True) if (s4d / "summary.json").is_file() else None
        for family in (SPEED, PATH):
            k = f"{seq}|{FAM[family]}"
            if full is None or s4 is None:
                ff[k] = None; continue
            common = [x for x in full if x in s4 and x[0] == family and full[x]["v"] is not None and s4[x]["v"] is not None]
            rf = spearman([full[x]["v"] for x in common], [full[x]["reference_value"] for x in common])
            rs = spearman([s4[x]["v"] for x in common], [s4[x]["reference_value"] for x in common])
            ff[k] = dict(rho_full=rf, rho_static4=rs, delta=None if rf is None or rs is None else rf - rs,
                         distinct_full=len({full[x]["v"] for x in common}))

    def cells(label, fmt, arms, fams=(SPEED, PATH)):
        return [res.get(f"{label}|{fmt}|{s}|{a}|{FAM[f]}") for s in SEQS for a in arms for f in fams]

    H = {}
    H["H1_unseen_units_rtc"] = rule_count(cells("rtc_s42", "rtc", UNSEEN), 6, 2)
    H["H2_pixels_given_scale_rtc"] = rule_count(cells("rtc_s42", "rtc", ["pxK"]), 3, 1)
    sp = cells("rtc_s42", "rtc", ["cm"], (SPEED,))
    H["H3_speed_cm_rtc"] = ("pending", None) if None in sp else \
        ({2: "fixed", 1: "partial", 0: "not fixed"}[sum(c["converts"] for c in sp)], sum(c["converts"] for c in sp))
    # H4 non-inferiority vs published v3 plain metre rho, Q2 cells
    h4 = []
    v3m = load(published_m["v3s42"]("Q2_top_480-510"), False)
    for family in (SPEED, PATH):
        c = res.get(f"rtc_s42|rtc|Q2_top_480-510|m|{FAM[family]}")
        if c is None or v3m is None:
            h4.append(None); continue
        rows = [r for r in v3m.values() if r["category"] == family and r["v"] is not None]
        v3rho = spearman([r["v"] for r in rows], [r["reference_value"] for r in rows])
        h4.append(dict(family=FAM[family], rtc_rho=c["rho"], v3_rho=v3rho,
                       ok=c["rho"] is not None and v3rho is not None and c["rho"] >= v3rho - 0.10))
    H["H4_metre_noninferiority"] = ("pending", None) if None in h4 else \
        ("holds" if all(x["ok"] for x in h4) else "fails", h4)
    defined = [v for v in ff.values() if v and v["delta"] is not None and v["distinct_full"] > 2]
    H["H5_motion_dependence"] = ("pending", None) if None in ff.values() else \
        ("LIMITED" if len(defined) < 3 else "holds" if sum(v["delta"] > 0 for v in defined) >= 3 else "fails",
         len(defined))
    for label in ("v3s42", "m1s42"):
        H[f"H6_format_control_{label}"] = rule_count(cells(label, "rtc", UNSEEN), 6, 2)
        H[f"plain_unseen_{label}"] = rule_count(cells(label, "plain", UNSEEN), 6, 2)
    if any(res.get(f"m1fines42|plain|{s}|U1|speed") for s in SEQS):
        H["plain_unseen_m1fines42"] = rule_count(cells("m1fines42", "plain", UNSEEN), 6, 2)
    probe = CD / "unit_probe_v2_runs" / "rtc_s42" / "summary.json"
    report = dict(rules_sha256=pre["rules_sha256"], hypotheses=H, cells=res, first_frame=ff,
                  text_probe=json.loads(probe.read_text(encoding="utf-8")) if probe.is_file() else None)
    (out_dir / "verdict.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    f2 = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    md = ["# RTC (read-then-convert) - preregistered verdicts", "",
          f"Rules sha256 `{pre['rules_sha256'][:16]}...`. Pending = inputs not yet run.", ""]
    for h, (v, k) in H.items():
        md.append(f"- **{h}**: {v}" + (f" ({k})" if isinstance(k, int) else ""))
    md += ["", "| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in res.items():
        lab, fmt, seq, arm, fam = k.split("|")
        if c is None:
            md.append(f"| {lab} | {fmt} | {seq} | {arm} | {fam} | pending |  |  |  |  |  |  |"); continue
        md.append(f"| {lab} | {fmt} | {seq} | {arm} | {fam} | {c['parse_rate']:.2f} | {c['distinct']} | "
                  f"{f2(c['rho'])} | {f2(c['R'], 3)} | {f2(c['factor'], 3)} | {f2(c['R_over_f'])} | "
                  f"{'yes' if c['converts'] else 'no'} |")
    (out_dir / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:4 + len(H)]))
    make_figure(res, out_dir)
    return 0


def make_figure(res, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = [("rtc_s42", "rtc", "RTC s42", "#2a78d6", "o"), ("m1s42", "plain", "M1 s42 (plain)", "#eb6834", "s"),
              ("v3s42", "plain", "v3 s42 (plain)", "#1baf7a", "D"), ("m1s42", "rtc", "M1 s42 + RTC prompt", "#eda100", "^"),
              ("v3s42", "rtc", "v3 s42 + RTC prompt", "#e87ba4", "v")]
    xs = [(s, a, f) for a in ("U1", "U2", "pxK") for f in ("speed", "path") for s in SEQS]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axhspan(-1, 1, color="#eef4fc", lw=0, zorder=0)
    ax.axhline(0, color="#6f6e69", lw=1, ls="--")
    for gi, (lab, fmt, name, col, mk) in enumerate(groups):
        pts = []
        for i, (s, a, f) in enumerate(xs):
            c = res.get(f"{lab}|{fmt}|{s}|{a}|{f}")
            if c and c["R_over_f"] and c["R_over_f"] > 0:
                pts.append((i + (gi - 2) * 0.12, math.log2(c["R_over_f"]), c["converts"]))
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=28, label=name,
                       facecolors=[col if p[2] else "none" for p in pts], edgecolors=col, linewidths=1.1)
    ax.set_xticks(range(len(xs)))
    names = {"U1": {"speed": "km/h", "path": "ft"}, "U2": {"speed": "ft/s", "path": "mm"}, "pxK": {"speed": "px/s", "path": "px"}}
    ax.set_xticklabels([f"{names[a][f]}\n{s.split('_')[0]}" for s, a, f in xs], fontsize=6.5)
    ax.set_ylabel(r"$\log_2(R/\mathrm{exact})$")
    ax.set_title("Unseen units: filled = converts under the preregistered cell rule", loc="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rtc_units.pdf"); fig.savefig(out_dir / "fig_rtc_units.png", dpi=200)


if __name__ == "__main__":
    sys.exit(main())
