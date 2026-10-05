#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate R1-R4 over seeds 42/43/44 per trained backbone, as recorded in
results/courtdyn/backbone_seeds3/PREREGISTRATION.json (commit ed6aacc), then apply the title rule.

Per backbone and rule, from the per-seed verdicts (cells never pooled across seeds):
  "holds at all three seeds"      replicates at 3 of 3 seeds
  "holds at two of three seeds"   replicates at 2 and fails at none
  "fails across seeds"            fails at >= 2 seeds and replicates at none
  "seed-dependent"                replicates at >= 1 seed and fails at >= 1 seed
  "not established"               anything else
  "pending"                       fewer than three seeds judged so far
SmolVLM2-2.2B is degenerate at the 4-epoch cap: its labels are computed but marked descriptive only.

Per-seed sources: seed 42 = the backbone's own record (backbone2_r2 / backbone3_internvl /
backbone4_a100); seed 43 = backbone4_ext for Gemma3-12B and Pixtral-12B, backbone_seeds3/s43 otherwise;
seed 44 = backbone_seeds3/s44. Writes results/courtdyn/backbone_seeds3/aggregate.{json,md} and runs
tools/decide_paper2_title.py --apply (which decides nothing until Pixtral has three judged seeds).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CD = ROOT / "results" / "courtdyn"
OUT = CD / "backbone_seeds3"
RULES = ["R1", "R2_E", "R2_T", "R3", "R4"]
NAMES = {"qwen25vl3b": "Qwen2.5-VL-3B", "internvl3_2b": "InternVL3-2B", "smol": "SmolVLM2-2.2B",
         "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B", "gemma3_12b": "Gemma3-12B"}
SEED42 = {"qwen25vl3b": "backbone2_r2", "smol": "backbone2_r2", "internvl3_2b": "backbone3_internvl",
          "pixtral_12b": "backbone4_a100", "idefics3_8b": "backbone4_a100", "gemma3_12b": "backbone4_a100"}


def source(backbone, seed):
    if seed == 42:
        return CD / SEED42[backbone] / "verdict.json"
    if seed == 43 and backbone in ("gemma3_12b", "pixtral_12b"):
        return CD / "backbone4_ext" / "verdict.json"
    return OUT / f"s{seed}" / "verdict.json"


def verdict(backbone, seed):
    path = source(backbone, seed)
    if not path.is_file():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))["backbones"].get(backbone)
    if rep is None:
        return None
    return {r: (rep[r]["verdict"] if r == "R3" else rep[r]) for r in RULES}


def label(per_seed):
    vals = [v for v in per_seed.values() if v is not None]
    if len(vals) < 3:
        return "pending"
    rep, fail = vals.count("replicates"), vals.count("fails")
    if rep == 3:
        return "holds at all three seeds"
    if rep == 2 and fail == 0:
        return "holds at two of three seeds"
    if fail >= 2 and rep == 0:
        return "fails across seeds"
    if rep >= 1 and fail >= 1:
        return "seed-dependent"
    return "not established"


def main() -> int:
    prereg = json.loads((OUT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(prereg["rules"].encode("utf-8")).hexdigest() == prereg["rules_sha256"]
    agg = {"rules_sha256": prereg["rules_sha256"], "backbones": {}}
    for b in NAMES:
        per = {s: verdict(b, s) for s in (42, 43, 44)}
        entry = {}
        for r in RULES:
            ps = {s: (v or {}).get(r) for s, v in per.items()}
            entry[r] = {"per_seed": ps, "n_seeds_judged": sum(v is not None for v in per.values()),
                        "label": label(ps)}
        entry["descriptive_only"] = b == "smol"
        agg["backbones"][b] = entry
    (OUT / "aggregate.json").write_text(json.dumps(agg, indent=1) + "\n", encoding="utf-8")
    md = ["# Three-seed aggregation (seeds 42/43/44)", "",
          f"Rules sha256 `{prereg['rules_sha256'][:16]}...` (backbone_seeds3). Per-seed verdicts; "
          "cells never pooled across seeds. SmolVLM2-2.2B is descriptive only (degenerate at the cap).", "",
          "| backbone | rule | seed 42 | seed 43 | seed 44 | label |", "|---|---|---|---|---|---|"]
    for b, e in agg["backbones"].items():
        for r in RULES:
            ps = e[r]["per_seed"]
            md.append(f"| {NAMES[b]}{' (descriptive)' if e['descriptive_only'] else ''} | {r} | "
                      + " | ".join(ps[s] or "-" for s in (42, 43, 44)) + f" | {e[r]['label']} |")
    (OUT / "aggregate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return subprocess.run([sys.executable, str(ROOT / "tools" / "decide_paper2_title.py"), "--apply"]).returncode


if __name__ == "__main__":
    sys.exit(main())
