#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep T36: the same unit controls on public backbones from several families.

Why this sweep exists
---------------------
IMAVIS-D-26-04792 was desk-rejected partly for "one 4B backbone". Every adapter
in the manuscript is a LoRA on the same Qwen3.5-4B, so nothing published so far
separates "fine-tuned VLMs answer with order but not magnitude" from "this
backbone does". The decisive point is that the unit controls need no training:
any instruction-following VLM can be asked the same 280 questions in metres,
centimetres and pixels, and its ratios read off directly.

What it runs
------------
Each model gets, per clip: the three fixed-prior unit arms at full frames, the
metre arm with the first frame repeated, and the 12 text-only conversion
probes. That covers both halves of the claim -- unit response and multi-frame
dependence -- for every backbone.

Two cells are controls rather than new models:
  qwen35-4b        the replication backbone under NF4, the protocol every
                   published cell used
  qwen35-4b-bf16   the same weights in bf16, so the paper can say how much of
                   any public-model gap is quantisation

Reading the output honestly
---------------------------
a100/eval_controls.py flags a cell `degenerate` when a model emits two or fewer
distinct answers. A centimetre/metre ratio of 1.00 from a model that always
answers "1.5" is not evidence that it declined to convert; it is evidence that
it produced nothing. analyze.py keeps those rows separate, and they must stay
separate in the paper.

Usage
  python -m a100.exp_backbones --dry-run
  python -m a100.exp_backbones --run --models qwen35-4b qwen25vl-7b
  python -m a100.exp_backbones --run                 # the whole sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                 # noqa: E402
from a100.runner import Runner, Stage, python_cmd   # noqa: E402

OUT = C.OUT / "backbones"


def cell_stage(model, seq, arms, frame_mode, arithmetic=False):
    spec = C.BACKBONES[model]
    resolved = C.backbone_path(model)
    target = str(resolved) if resolved else spec["hf"]
    output = OUT / f"{model}_{seq}_{frame_mode}"
    cmd = python_cmd("-m", "a100.eval_controls", "--run",
                     "--model-path", target, "--precision", spec["precision"],
                     "--seq", seq, "--arms", *arms, "--frame-mode", frame_mode,
                     "--label", model, "--output", output, "--resume")
    if arithmetic:
        cmd.append("--arithmetic")
    needs = [C.image_root(seq)]
    if resolved:
        needs.append(Path(resolved) / "config.json")
    return Stage(
        name=f"{model}_{seq}_{frame_mode}",
        cmd=cmd, done_file=output / "summary.json",
        log_name=f"{model}_{seq}_{frame_mode}.log", needs=needs,
        stall_minutes=60,
        note=f"{spec['precision']}, {spec['role']}, "
             f"{len(arms) * C.ITEMS_PER_ARM} items")


def build(models, seqs, arms, static4=True, arithmetic=True):
    stages = []
    for model in models:
        for index, seq in enumerate(seqs):
            stages.append(cell_stage(model, seq, arms, "full",
                                     arithmetic=arithmetic and index == 0))
            if static4:
                stages.append(cell_stage(model, seq, ["m_height"], "static4"))
    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=C.BACKBONE_SWEEP,
                    choices=sorted(C.BACKBONES))
    ap.add_argument("--seqs", nargs="+", default=C.EVAL_SEQS)
    ap.add_argument("--six", action="store_true",
                    help="run the full unit x height factorial instead of the "
                         "three fixed-prior arms (doubles the cost)")
    ap.add_argument("--no-static4", action="store_true")
    ap.add_argument("--no-arithmetic", action="store_true")
    ap.add_argument("--require-idle-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run, or --dry-run to inspect the plan")

    arms = C.ARMS_SIX if args.six else C.ARMS_THREE
    stages = build(args.models, args.seqs, arms,
                   static4=not args.no_static4,
                   arithmetic=not args.no_arithmetic)
    runner = Runner("t36-backbones", OUT / "state.json", OUT / "logs",
                    require_idle_gpu=args.require_idle_gpu)
    unresolved = [m for m in args.models if C.backbone_path(m) is None]
    if unresolved:
        runner.log("will be pulled from the Hugging Face hub (needs network "
                   "and disk): " + ", ".join(unresolved))
    runner.log(f"{len(stages)} cells over {len(args.models)} models, arms={arms}")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
