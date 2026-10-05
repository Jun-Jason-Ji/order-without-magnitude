#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Third-backbone replication: InternVL3-2B under the round-2 protocol.

Rules: results/courtdyn/backbone3_internvl/PREREGISTRATION.json (written before any run).
This reuses tools/run_backbone_round2.py stage builders unchanged; only the output
directory, the backbone registry entry and the training entry point differ:

  * outputs go to results/courtdyn/backbone3_internvl/<key>/ (same layout as round 2);
  * training runs through train/train_qlora_internvl.py, which injects the InternVL family
    into the byte-frozen trainer and otherwise behaves exactly like train_qlora_epochs.py;
  * every evaluation and probe runs through tools/internvl_launch.py, which pins the InternVL
    image policy to one 448x448 tile per frame (see that file for why);
  * there is no round-1 run for this model, so the bare backbone's two explicit test cells
    and the 392-item text probe are run here first (they need no adapter).
Order: base cells -> dev-set stopping rule (v3, epochs 1..4) -> mixed-unit to k* -> test cells.
Everything is resumable: rerun after an interruption.

Usage:  python tools/run_backbone_internvl.py --run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100.runner import Runner, Stage, python_cmd        # noqa: E402
import run_backbone_replication as R1                    # noqa: E402
import run_backbone_round2 as R2                         # noqa: E402

KEY = "internvl3_2b"
MODEL = (r"E:\models\hf\hub\models--OpenGVLab--InternVL3-2B-hf\snapshots"
         r"\cb57a075cb75a2e6d1b668b128d48bb00ae321d2")
OUT = ROOT / "results" / "courtdyn" / "backbone3_internvl"

LAUNCHER = ROOT / "tools" / "internvl_launch.py"


def launched_cmd(script, *args):
    """python_cmd, with evaluation/probe scripts entered through the InternVL launcher."""
    if Path(script).name in ("run_backbone_replication.py", "run_unit_probe.py"):
        return python_cmd(LAUNCHER, script, *args)
    return python_cmd(script, *args)


# Point the round-2 builders at this run. R2.DEV_INPUTS stays the round-2 dev set.
R1.BACKBONES[KEY] = MODEL
R2.OUT = OUT
R2.WRAPPER = ROOT / "train" / "train_qlora_internvl.py"
R2.python_cmd = launched_cmd


def base_stages():
    stages = []
    ev = OUT / KEY / "eval"
    for seq in R1.SEQS:
        out = ev / f"base_{seq}_full"
        stages.append(Stage(
            name=f"eval_{KEY}_{out.name}",
            cmd=launched_cmd(ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                           "--model", MODEL, "--label", "base", "--seq", seq,
                           "--arms", "m_height", "cm_height", "px_height",
                           "--frame-mode", "full", "--output", out),
            done_file=out / "summary.json", log_name=f"eval_{KEY}_{out.name}.log",
            needs=[Path(MODEL) / "config.json"]))
    runs = OUT / KEY / "unit_probe"
    stages.append(Stage(
        name=f"probe_{KEY}_base",
        cmd=launched_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", "base",
                       "--adapter", "none", "--model-path", MODEL, "--out", runs),
        done_file=runs / "base" / "summary.json", log_name=f"probe_{KEY}_base.log",
        needs=[Path(MODEL) / "config.json"]))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    if not (OUT / "PREREGISTRATION.json").is_file():
        raise SystemExit("no PREREGISTRATION.json: record the rules before running")
    if not (R2.DEV_INPUTS / "manifest.json").is_file():
        raise SystemExit("no dev inputs: run tools/build_backbone_dev.py first")
    runner = Runner("backbone3-internvl", OUT / "state.json", OUT / "logs", require_idle_gpu=True)
    if args.dry_run:
        return runner.run(base_stages() + [R2.train_stage(KEY, "v3", 1)]
                          + R2.test_stages(KEY, 4), dry_run=True)
    if runner.run(base_stages()) != 0:
        runner.log(f"{KEY}: some base stages failed; see state.json")
    k = R2.choose_k(KEY, runner)
    if runner.run([R2.train_stage(KEY, "mixunit", k)]) != 0:
        raise SystemExit(f"{KEY}: mixed-unit training failed")
    if runner.run(R2.test_stages(KEY, k)) != 0:
        runner.log(f"{KEY}: some test stages failed; see state.json")
    runner.log("third backbone complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
