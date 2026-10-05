#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply the preregistered RTC rules (H1-H6) to the A100 extension: Pixtral-12B and Gemma3-12B.

Rules: results/courtdyn/rtc/PREREGISTRATION.json + ERRATA 1-3, as applied by
results/courtdyn/backbone4_ext/PREREGISTRATION.json (group 2).  The per-backbone judgement is
tools/analyze_rtc_replication.judge, imported unchanged (cell statistics, parsing, the [f/1.5, 1.5f]
band and rule counting come from tools/analyze_rtc.py through it).  Comparators' plain metre cells
are their published round-4 explicit metre cells; the H4 reference is the round-4 v3 s42 adapter.
Each backbone is judged on its own; pending inputs give "pending".

Writes results/courtdyn/backbone4_ext/rtc/verdict.json and verdict.md.
  python tools/analyze_rtc_a100.py [--root DIR]   (--root only for tests)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_rtc_replication as AR                             # noqa: E402  (read-only reuse)

CD = ROOT / "results" / "courtdyn"
B4 = CD / "backbone4_a100"
KEYS = ["pixtral_12b", "gemma3_12b"]


def groups(b4: Path = B4):
    out = {}
    for key in KEYS:
        ev = b4 / key / "eval"
        out[key] = (f"rtc_{key}", {
            f"{key}_v3": (lambda seq, ev=ev: ev / f"v3_{seq}_full" / "m_height.jsonl"),
            f"{key}_mix": (lambda seq, ev=ev: ev / f"mixunit_{seq}_full" / "m_height.jsonl"),
        }, f"{key}_v3")
    return out


def status(h):
    return h[0] if isinstance(h, (list, tuple)) else h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(CD / "backbone4_ext" / "rtc"))
    ap.add_argument("--b4", default=str(B4))
    args = ap.parse_args()
    out_dir = Path(args.root)
    ev = out_dir / "eval"
    pre = json.loads((CD / "rtc" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pre["rules"].encode("utf-8")).hexdigest() == pre["rules_sha256"]
    ext = json.loads((CD / "backbone4_ext" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(ext["rules"].encode("utf-8")).hexdigest() == ext["rules_sha256"]
    rep = {key: AR.judge(ev, *g) for key, g in groups(Path(args.b4)).items()}
    for key, (rtc_label, _, _) in groups(Path(args.b4)).items():
        pr = out_dir / "unit_probe" / rtc_label / "summary.json"
        rep[key]["text_probe"] = json.loads(pr.read_text(encoding="utf-8")) if pr.is_file() else None
    report = dict(rtc_rules_sha256=pre["rules_sha256"], ext_rules_sha256=ext["rules_sha256"],
                  errata_items=[1, 2, 3], backbones=rep)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    f2 = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    md = ["# RTC on the A100 backbones (Pixtral-12B, Gemma3-12B) - preregistered verdicts", "",
          f"RTC rules `{pre['rules_sha256'][:16]}...` + ERRATA 1-3 (band [f/1.5, 1.5f]); extension "
          f"prereg `{ext['rules_sha256'][:16]}...`. Pending = not yet run."]
    for key, g in rep.items():
        md += ["", f"## {key}", ""]
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
    print("\n".join(l for l in md if l.startswith(("- **", "## "))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
