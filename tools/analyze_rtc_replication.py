#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply the preregistered RTC rules to the replications (seeds 43/44, InternVL3-2B).

Rules: results/courtdyn/rtc/PREREGISTRATION.json, ERRATA items 2 (conversion band) and 3
(pre-run specification: comparators, metre cells, H4 reference, summary wording).  Cell
statistics, parsing and rule counting are imported unchanged from tools/analyze_rtc.py.
Each seed / backbone is judged on its own; nothing is pooled.  Summary wording (ERRATA 3d):
"replicates across seeds" only if H1 holds at seeds 42, 43 and 44; "replicates on
InternVL3-2B" only if H1 holds there.  Pending inputs give "pending", never an early verdict.

Writes results/courtdyn/rtc_rep/verdict.json and verdict.md (the md also restates the
seed-42 verdicts from results/courtdyn/rtc/verdict.json, so it is the combined RTC summary).

  python tools/analyze_rtc_replication.py [--root DIR]   (--root only for tests)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_rtc as A                                          # noqa: E402  (read-only reuse)

CD = ROOT / "results" / "courtdyn"
ARMS = ["m", "cm", "U1", "U2", "pxK"]
PLAIN = ["U1", "U2", "pxK"]


def groups(ev: Path):
    """label -> (rtc label, {comparator: metre-cell getter(seq)}, H4 reference comparator)."""
    out = {}
    for s in (43, 44):
        v3q1 = ev / f"v3s{s}_Q1_top_0-30_full_m_plain" / "plain_m.jsonl"
        out[f"seed{s}"] = (f"rtc_s{s}", {
            f"v3s{s}": (lambda seq, s=s, v3q1=v3q1: v3q1 if seq.startswith("Q1")
                        else CD.parent / "a100" / "seeds" / f"v3_s{s}_{seq}_full" / "m_height.jsonl"),
            f"m1s{s}": (lambda seq, s=s: CD / "m1_mixunit" / "eval" / f"mixunit_s{s}_{seq}_full" / "m_height.jsonl"),
        }, f"v3s{s}")
    ivl = CD / "backbone3_internvl" / "internvl3_2b" / "eval"
    out["internvl3_2b"] = ("rtc_ivl", {
        "ivlv3": lambda seq: ivl / f"v3_{seq}_full" / "m_height.jsonl",
        "ivlmix": lambda seq: ivl / f"mixunit_{seq}_full" / "m_height.jsonl",
    }, "ivlv3")
    return out


def judge(ev: Path, rtc_label, comparators, h4_ref):
    def cell_dir(label, seq, mode, arms, fmt):
        return ev / f"{label}_{seq}_{mode}_{'-'.join(arms)}_{fmt}"

    def metre_plain(label, seq):
        p = comparators[label](seq)
        return A.load(p, False) if p is not None else None

    res = {}
    models = {rtc_label: ["rtc"], **{c: ["plain", "rtc"] for c in comparators}}
    for label, fmts in models.items():
        for fmt in fmts:
            for seq in A.SEQS:
                if fmt == "rtc":
                    d = cell_dir(label, seq, "full", ARMS, "rtc")
                    metre, arms = A.load(d / "rtc_m.jsonl", True), ARMS
                else:
                    d = cell_dir(label, seq, "full", PLAIN, "plain")
                    metre, arms = metre_plain(label, seq), PLAIN
                for arm in arms:
                    rows = A.load(d / f"{fmt}_{arm}.jsonl", fmt == "rtc")
                    for family in (A.SPEED, A.PATH):
                        k = f"{label}|{fmt}|{seq}|{arm}|{A.FAM[family]}"
                        complete = rows is not None and (d / "summary.json").is_file() and metre is not None
                        res[k] = A.cell_stats(rows, metre, arm, family, fmt == "rtc") if complete else None
    ff = {}
    for seq in A.SEQS:
        full = A.load(cell_dir(rtc_label, seq, "full", ARMS, "rtc") / "rtc_m.jsonl", True)
        s4d = cell_dir(rtc_label, seq, "static4", ["m"], "rtc")
        s4 = A.load(s4d / "rtc_m.jsonl", True) if (s4d / "summary.json").is_file() else None
        for family in (A.SPEED, A.PATH):
            k = f"{seq}|{A.FAM[family]}"
            if full is None or s4 is None:
                ff[k] = None
                continue
            common = [x for x in full if x in s4 and x[0] == family
                      and full[x]["v"] is not None and s4[x]["v"] is not None]
            rf = A.spearman([full[x]["v"] for x in common], [full[x]["reference_value"] for x in common])
            rs = A.spearman([s4[x]["v"] for x in common], [s4[x]["reference_value"] for x in common])
            ff[k] = dict(rho_full=rf, rho_static4=rs, delta=None if rf is None or rs is None else rf - rs,
                         distinct_full=len({full[x]["v"] for x in common}))

    def cells(label, fmt, arms, fams=(A.SPEED, A.PATH)):
        return [res.get(f"{label}|{fmt}|{s}|{a}|{A.FAM[f]}") for s in A.SEQS for a in arms for f in fams]

    H = {"H1_unseen_units_rtc": A.rule_count(cells(rtc_label, "rtc", A.UNSEEN), 6, 2),
         "H2_pixels_given_scale_rtc": A.rule_count(cells(rtc_label, "rtc", ["pxK"]), 3, 1)}
    sp = cells(rtc_label, "rtc", ["cm"], (A.SPEED,))
    H["H3_speed_cm_rtc"] = ("pending", None) if None in sp else \
        ({2: "fixed", 1: "partial", 0: "not fixed"}[sum(c["converts"] for c in sp)],
         sum(c["converts"] for c in sp))
    h4 = []
    ref = metre_plain(h4_ref, "Q2_top_480-510")
    for family in (A.SPEED, A.PATH):
        c = res.get(f"{rtc_label}|rtc|Q2_top_480-510|m|{A.FAM[family]}")
        if c is None or ref is None:
            h4.append(None)
            continue
        rows = [r for r in ref.values() if r["category"] == family and r["v"] is not None]
        rrho = A.spearman([r["v"] for r in rows], [r["reference_value"] for r in rows])
        h4.append(dict(family=A.FAM[family], rtc_rho=c["rho"], ref_rho=rrho, reference=h4_ref,
                       ok=c["rho"] is not None and rrho is not None and c["rho"] >= rrho - 0.10))
    H["H4_metre_noninferiority"] = ("pending", None) if None in h4 else \
        ("holds" if all(x["ok"] for x in h4) else "fails", h4)
    defined = [v for v in ff.values() if v and v["delta"] is not None and v["distinct_full"] > 2]
    H["H5_motion_dependence"] = ("pending", None) if None in ff.values() else \
        ("LIMITED" if len(defined) < 3 else "holds" if sum(v["delta"] > 0 for v in defined) >= 3
         else "fails", len(defined))
    for label in comparators:
        H[f"H6_format_control_{label}"] = A.rule_count(cells(label, "rtc", A.UNSEEN), 6, 2)
        H[f"plain_unseen_{label}"] = A.rule_count(cells(label, "plain", A.UNSEEN), 6, 2)
    return dict(hypotheses=H, cells=res, first_frame=ff)


def status(h):
    return h[0] if isinstance(h, (list, tuple)) else h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(CD / "rtc_rep"))
    ap.add_argument("--seed42", default=str(CD / "rtc" / "verdict.json"))
    args = ap.parse_args()
    out_dir = Path(args.root)
    ev = out_dir / "eval"
    pre = json.loads((CD / "rtc" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pre["rules"].encode("utf-8")).hexdigest() == pre["rules_sha256"]
    rep = {name: judge(ev, *g) for name, g in groups(ev).items()}
    for name, (rtc_label, _, _) in groups(ev).items():
        pr = out_dir / "unit_probe" / rtc_label / "summary.json"
        rep[name]["text_probe"] = json.loads(pr.read_text(encoding="utf-8")) if pr.is_file() else None
    s42 = Path(args.seed42)
    s42h = json.loads(s42.read_text(encoding="utf-8"))["hypotheses"] if s42.is_file() else {}
    seed_h1 = {42: status(s42h.get("H1_unseen_units_rtc", "pending")),
               43: status(rep["seed43"]["hypotheses"]["H1_unseen_units_rtc"]),
               44: status(rep["seed44"]["hypotheses"]["H1_unseen_units_rtc"])}
    ivl_h1 = status(rep["internvl3_2b"]["hypotheses"]["H1_unseen_units_rtc"])
    if "pending" in seed_h1.values():
        across = "pending"
    else:
        n = sum(v == "holds" for v in seed_h1.values())
        across = "replicates across seeds" if n == 3 else f"H1 holds in {n} of 3 seeds"
    summary = {"H1_by_seed": seed_h1, "across_seeds": across,
               "internvl3_2b": ("pending" if ivl_h1 == "pending" else
                                "replicates on InternVL3-2B" if ivl_h1 == "holds" else f"H1 {ivl_h1} on InternVL3-2B")}
    report = dict(rules_sha256=pre["rules_sha256"], errata_items=[2, 3], summary=summary, groups=rep)
    (out_dir / "verdict.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    f2 = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    md = ["# RTC - combined verdicts (seed 42 + preregistered replications)", "",
          f"Rules sha256 `{pre['rules_sha256'][:16]}...`; ERRATA 2 (band [f/1.5, 1.5f]) and 3 "
          "(replication specification). Pending = not yet run.", "",
          f"- **Across seeds (42/43/44)**: {across}  (H1 by seed: {seed_h1})",
          f"- **InternVL3-2B**: {summary['internvl3_2b']}", "", "## Seed 42 (from rtc/verdict.json)", ""]
    md += [f"- {h}: {status(v)}" for h, v in s42h.items()] or ["- pending"]
    for name, g in rep.items():
        md += ["", f"## {name}", ""]
        for h, v in g["hypotheses"].items():
            k = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else None
            md.append(f"- **{h}**: {status(v)}" + (f" ({k})" if isinstance(k, int) else ""))
        md += ["", "| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in g["cells"].items():
            lab, fmt, seq, arm, fam = k.split("|")
            if c is None:
                md.append(f"| {lab} | {fmt} | {seq} | {arm} | {fam} | pending |  |  |  |  |  |  |  |")
                continue
            md.append(f"| {lab} | {fmt} | {seq} | {arm} | {fam} | {c['parse_rate']:.2f} | {c['distinct']} | "
                      f"{f2(c['rho'])} | {f2(c['R'], 3)} | {f2(c['factor'], 3)} | {f2(c['R_over_f'])} | "
                      f"{'yes' if c['converts'] else 'no'} | {'yes' if c.get('converts_as_registered') else 'no'} |")
    (out_dir / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
