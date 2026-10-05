#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate trained adapters on the SoccerNet second venue (reviewer item 4).

Pools and frames come from tools/build_soccer_venue.py (private: SoccerNet NDA).
Each adapter is evaluated on the three matches with the explicit-coordinate
metre, centimetre and pixel arms and the first-frame control, through the same
a100.eval_controls.run_cell as every CourtDyn cell.  The frame root is pointed at
the private directory with COURTDYN_DATA; per-item predictions stay private, and
tools/analyze_soccer_venue.py writes only aggregates into the repository.

Adapters (zero-shot transfer: none was trained on soccer)
  qwen35 v3       metre-only, the published event-holdout v3 seed-42 adapter
  qwen35 mixunit  the M1 mixed-unit adapter, seed 42
  smol v3/mixunit the SmolVLM2 pair, when tools/run_backbone_replication.py has
                  produced them (skipped, not failed, if absent)

Usage
  python tools/run_soccer_venue.py --dry-run
  python tools/run_soccer_venue.py --run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(r"E:\datasets\SoccerNet\derived_private\courtdyn_venue")
# must precede the a100 import: a100.config reads COURTDYN_DATA at import time
os.environ["COURTDYN_DATA"] = str(PRIVATE / "data")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100 import config as C                            # noqa: E402
from a100.runner import Runner, Stage, python_cmd       # noqa: E402

QWEN35 = os.environ.get("QWEN35_4B") or (
    r"E:\models\hf\hub\models--Qwen--Qwen3.5-4B\snapshots"
    r"\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
SMOL = r"E:\models\SmolVLM2-2.2B-Instruct"
ADAPTERS = [
    ("qwen35_v3", QWEN35, ROOT / "models" / "courtdyn-revision-v3-eventholdout-s42"),
    ("qwen35_mixunit", QWEN35, ROOT / "models" / "courtdyn-mixunit-v3-s42"),
    ("smol_v3", SMOL, ROOT / "models" / "courtdyn-smol-v3-s42"),
    ("smol_mixunit", SMOL, ROOT / "models" / "courtdyn-smol-mixunit-s42"),
]
EVAL_OUT = PRIVATE / "eval"
REPLICATION = ROOT / "tools" / "run_backbone_replication.py"


def sequences():
    receipt = json.loads((ROOT / "results" / "courtdyn" / "soccer_venue" /
                          "build_receipt.json").read_text(encoding="utf-8"))
    return [s["name"] for s in receipt["sequences"]]


def stage(label, model, adapter, seq, arms, mode):
    output = EVAL_OUT / f"{label}_{seq}_{mode}"
    cmd = python_cmd(REPLICATION, "eval-cell", "--model", model, "--label", label,
                     "--adapter", adapter, "--seq", seq, "--arms", *arms,
                     "--frame-mode", mode, "--output", output,
                     "--inputs", PRIVATE / "inputs")
    return Stage(name=f"soccer_{label}_{seq}_{mode}", cmd=cmd,
                 done_file=output / "summary.json",
                 log_name=f"soccer_{label}_{seq}_{mode}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors",
                        PRIVATE / "inputs" / "manifest.json"],
                 note=f"{len(arms) * 280} items")


def build():
    stages = []
    for label, model, adapter in ADAPTERS:
        for seq in sequences():
            stages.append(stage(label, model, adapter, seq, C.ARMS_THREE, "full"))
            stages.append(stage(label, model, adapter, seq, ["m_height"], "static4"))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    runner = Runner("soccer-venue", EVAL_OUT / "state.json", EVAL_OUT / "logs")
    stages = build()
    runner.log(f"{len(stages)} stages; frames from {os.environ['COURTDYN_DATA']}")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
