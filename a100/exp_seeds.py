#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep T35: the same adapters at three seeds, so the label contrast can be
read against training noise instead of assumed to exceed it.

Why this sweep exists
---------------------
IMAVIS-D-26-04792 was desk-rejected partly for "a single random seed". The
manuscript's v1-versus-v3 comparison rests on two checkpoints trained once
each; nothing in it says whether a 0.06 difference in Spearman rho is a label
effect or the spread you get from rerunning the same recipe. This sweep
retrains every pool at seeds 43 and 44 with byte-identical hyperparameters and
re-evaluates all three seeds through one runner.

What it produces
----------------
  $OUT/seeds/<pool>_s<seed>_<seq>_<mode>/   per-cell predictions and summary
  $OUT/seeds/state.json                     resumable stage state
  a100/analyze.py turns those into the seed-variability table.

Seed 42 is included deliberately. Its adapters already exist and are not
retrained, but re-evaluating them here reproduces the published 8 GB numbers on
the A100; a mismatch means the port changed something and must be resolved
before any new seed is believed.

Usage
  python -m a100.exp_seeds --dry-run
  python -m a100.exp_seeds --run                       # everything
  python -m a100.exp_seeds --run --only-train          # adapters first
  python -m a100.exp_seeds --run --pools v1 v3 --seeds 43 44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                 # noqa: E402
from a100.runner import Runner, Stage, python_cmd   # noqa: E402

OUT = C.OUT / "seeds"


def train_stage(pool, seed, backbone):
    adapter = C.adapter_dir(pool, seed)
    qa_json = C.ROOT / C.TRAIN_POOLS[pool]
    args = C.TRAIN_ARGS
    cmd = python_cmd(
        C.ROOT / "train" / "train_qlora.py",
        "--model_path", backbone,
        "--qa_json", qa_json,
        "--img_root", C.DATA,
        "--output_dir", adapter,
        "--num_samples", args["num_samples"],
        "--allow_base_init",
        "--budget_slice", args["budget_slice"],
        "--num_train_epochs", args["num_train_epochs"],
        "--max_steps", args["max_steps"],
        "--seed", seed,
        "--lr", args["lr"],
        "--grad_accum", args["grad_accum"],
        "--lora_r", args["lora_r"],
        "--max_pixels", args["max_pixels"],
        "--save_steps", args["save_steps"],
    )

    def verify():
        receipt = adapter / "training_completion.json"
        if not receipt.is_file():
            return False
        d = json.loads(receipt.read_text(encoding="utf8"))
        return bool(d.get("completed")
                    and d.get("seed") == seed
                    and d.get("selected_samples") == C.EXPECTED_ITEMS[pool]
                    and d.get("global_step") == C.EXPECTED_STEPS[pool]
                    and (adapter / "adapter_model.safetensors").is_file())

    return Stage(
        name=f"train_{C.adapter_label(pool, seed)}",
        cmd=cmd, verify=verify, log_name=f"train_{pool}_s{seed}.log",
        needs=[qa_json], stall_minutes=90, max_restarts=2,
        note=f"{C.EXPECTED_ITEMS[pool]} items, {C.EXPECTED_STEPS[pool]} steps, seed {seed}")


def eval_stage(pool, seed, seq, arms, frame_mode, backbone, arithmetic=False):
    adapter = C.adapter_dir(pool, seed)
    label = C.adapter_label(pool, seed)
    output = OUT / f"{label}_{seq}_{frame_mode}"
    cmd = python_cmd("-m", "a100.eval_controls", "--run",
                     "--model-path", backbone, "--precision", "nf4",
                     "--adapter-dir", adapter, "--seq", seq,
                     "--arms", *arms, "--frame-mode", frame_mode,
                     "--label", label, "--output", output, "--resume")
    if arithmetic:
        cmd.append("--arithmetic")
    return Stage(
        name=f"eval_{label}_{seq}_{frame_mode}",
        cmd=cmd, done_file=output / "summary.json",
        log_name=f"eval_{label}_{seq}_{frame_mode}.log",
        needs=[adapter / "adapter_model.safetensors", C.image_root(seq)],
        note=f"{len(arms) * C.ITEMS_PER_ARM} items"
             + (" + 12 arithmetic probes" if arithmetic else ""))


def build(pools, seeds, seqs, only_train=False, skip_train=False):
    backbone = C.backbone_path("qwen35-4b")
    if backbone is None:
        raise SystemExit(
            "base weights not found. Set QWEN35_4B or COURTDYN_MODELS; see "
            "a100/README.md. Multi-seed replication must use the same backbone "
            "every published cell used.")
    stages = []
    if not skip_train:
        for pool in pools:
            for seed in seeds:
                if seed == 42:
                    continue            # frozen adapters ship with the bundle
                stages.append(train_stage(pool, seed, backbone))
    if only_train:
        return stages
    for pool in pools:
        for seed in seeds:
            for seq in seqs:
                stages.append(eval_stage(pool, seed, seq, C.ARMS_THREE, "full",
                                         backbone,
                                         arithmetic=(seq == seqs[0])))
                stages.append(eval_stage(pool, seed, seq, ["m_height"], "static4",
                                         backbone))
    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=["v1", "v3", "native"],
                    choices=sorted(C.TRAIN_POOLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=C.TRAIN_SEEDS)
    ap.add_argument("--seqs", nargs="+", default=C.EVAL_SEQS)
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--require-idle-gpu", action="store_true",
                    help="wait for an unoccupied GPU before each stage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run, or --dry-run to inspect the plan")

    stages = build(args.pools, args.seeds, args.seqs,
                   only_train=args.only_train, skip_train=args.skip_train)
    items = sum(C.ITEMS_PER_ARM * (3 if "full" in s.name else 1)
                for s in stages if s.name.startswith("eval_"))
    runner = Runner("t35-seeds", OUT / "state.json", OUT / "logs",
                    require_idle_gpu=args.require_idle_gpu)
    runner.log(f"pools={args.pools} seeds={args.seeds} seqs={args.seqs}")
    runner.log(f"{sum(1 for s in stages if s.name.startswith('train_'))} trainings, "
               f"{sum(1 for s in stages if s.name.startswith('eval_'))} evaluation cells, "
               f"{items} visual items")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
