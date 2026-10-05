#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second-backbone replication of the paper's core claims (reviewer item 3).

Every trained adapter so far is a LoRA on one Qwen3.5-4B, so "single-unit
fine-tuning makes a VLM ignore the requested unit" is, until this runs, a
one-model claim.  This queue repeats the three findings on another backbone
with the identical recipe:

  1. the dissociation: after metre-only training on the event-holdout v3 pool,
     does the model rank motion (rho > 0) while ignoring the unit (cm/m ~ 1)?
  2. M1: after training on the mixed-unit pool, does the cm/m ratio move to ~100?
  3. the text probe: does metre-only training degrade unit arithmetic relative
     to the bare backbone, and does mixed-unit training avoid it?

Backbones
  smol        SmolVLM2-2.2B-Instruct (Idefics3 / SmolLM2): a different family.
              Local, no download.  Image policy is the frozen family policy
              (384 px longest edge, no tiling, ~83k px/frame against ~200k for
              Qwen) -- a budget difference that is reported, not tuned away.
  qwen25vl3b  Qwen2.5-VL-3B-Instruct at a pinned revision: a different Qwen
              generation with the same max_pixels budget as the main backbone.

Recipe: a100.config.TRAIN_ARGS unchanged (lr 1e-4, LoRA r16, 1 epoch,
grad_accum 16, seed 42), same pools, same prompts, same scoring, same strict
parser, nf4 loading.  Stages run through a100.runner and resume after any
interruption; --require-idle-gpu waits for a card no other process holds.

Usage
  python tools/run_backbone_replication.py --backbone smol --dry-run
  python tools/run_backbone_replication.py --backbone smol --run --require-idle-gpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100 import config as C                            # noqa: E402
from a100.runner import Runner, Stage, python_cmd       # noqa: E402

BACKBONES = {
    "smol": r"E:\models\SmolVLM2-2.2B-Instruct",
    "qwen25vl3b": (r"E:\models\hf\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots"
                   r"\66285546d2b821cf421d4f5eb2576359d3770cd3"),
}
POOLS = {
    "v3": ROOT / "results" / "courtdyn" / "revision_controls_20260912" / "training"
          / "event_holdout_v3.json",
    "mixunit": ROOT / "results" / "courtdyn" / "m1_mixunit" / "mixunit_v3_pool.json",
}
INTEMPLATE = ROOT / "results" / "courtdyn" / "m1_mixunit" / "intemplate"
SEQS = ["Q2_top_480-510", "Q1_top_0-30"]
SEED, EXPECTED_ITEMS, EXPECTED_STEPS = 42, 1120, 70


def out_root(key):
    return ROOT / "results" / "courtdyn" / "backbone2" / key


def adapter_dir(key, pool):
    return ROOT / "models" / f"courtdyn-{key}-{pool}-s{SEED}"


def train_stage(key, pool):
    a = C.TRAIN_ARGS
    adapter = adapter_dir(key, pool)
    cmd = python_cmd(
        ROOT / "train" / "train_qlora.py",
        "--model_path", BACKBONES[key], "--qa_json", POOLS[pool], "--img_root", C.DATA,
        "--output_dir", adapter, "--num_samples", a["num_samples"],
        "--allow_base_init", "--budget_slice", a["budget_slice"],
        "--num_train_epochs", a["num_train_epochs"], "--max_steps", a["max_steps"],
        "--seed", SEED, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", a["save_steps"])

    def verify():
        receipt = adapter / "training_completion.json"
        if not receipt.is_file():
            return False
        d = json.loads(receipt.read_text(encoding="utf-8"))
        return bool(d.get("completed") and d.get("seed") == SEED
                    and d.get("selected_samples") == EXPECTED_ITEMS
                    and d.get("global_step") == EXPECTED_STEPS
                    and (adapter / "adapter_model.safetensors").is_file())

    return Stage(name=f"train_{key}_{pool}", cmd=cmd, verify=verify,
                 log_name=f"train_{key}_{pool}.log",
                 needs=[POOLS[pool], Path(BACKBONES[key]) / "config.json"],
                 stall_minutes=90, max_restarts=2,
                 note=f"{EXPECTED_ITEMS} items, {EXPECTED_STEPS} steps, seed {SEED}")


def eval_stage(key, label, adapter, seq, arms, frame_mode, inputs=None):
    output = out_root(key) / "eval" / (f"{label}_{seq}_{frame_mode}"
                                       + ("_intemplate" if inputs else ""))
    cmd = python_cmd(Path(__file__).resolve(), "eval-cell", "--model", BACKBONES[key],
                     "--label", label, "--seq", seq, "--arms", *arms,
                     "--frame-mode", frame_mode, "--output", output)
    needs = [Path(BACKBONES[key]) / "config.json"]
    if adapter:
        cmd += ["--adapter", adapter]
        needs.append(Path(adapter) / "adapter_model.safetensors")
    if inputs:
        cmd += ["--inputs", str(inputs)]
    return Stage(name=f"eval_{key}_{output.name}", cmd=cmd,
                 done_file=output / "summary.json",
                 log_name=f"eval_{key}_{output.name}.log", needs=needs,
                 note=f"{len(arms) * 280} items")


def probe_stage(key, label, adapter):
    runs = out_root(key) / "unit_probe"
    cmd = python_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", label,
                     "--adapter", str(adapter) if adapter else "none",
                     "--model-path", BACKBONES[key], "--out", runs)
    needs = [Path(BACKBONES[key]) / "config.json"]
    if adapter:
        needs.append(Path(adapter) / "adapter_model.safetensors")
    return Stage(name=f"probe_{key}_{label}", cmd=cmd,
                 done_file=runs / label / "summary.json",
                 log_name=f"probe_{key}_{label}.log", needs=needs, note="392 text items")


def build(key):
    stages = []
    # the bare backbone: visual unit arms and the text probe
    for seq in SEQS:
        stages.append(eval_stage(key, "base", None, seq, C.ARMS_THREE, "full"))
    stages.append(probe_stage(key, "base", None))
    for pool in ("v3", "mixunit"):
        stages.append(train_stage(key, pool))
    for pool in ("v3", "mixunit"):
        adapter = adapter_dir(key, pool)
        for seq in SEQS:
            stages.append(eval_stage(key, pool, adapter, seq, C.ARMS_THREE, "full"))
            stages.append(eval_stage(key, pool, adapter, seq, ["m_height"], "static4"))
            stages.append(eval_stage(key, pool, adapter, seq, ["legacy_m", "legacy_cm"],
                                     "full", inputs=INTEMPLATE))
        stages.append(probe_stage(key, pool, adapter))
    return stages


def eval_cell(argv):
    ap = argparse.ArgumentParser(prog="run_backbone_replication.py eval-cell")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--frame-mode", default="full")
    ap.add_argument("--output", required=True)
    ap.add_argument("--inputs")
    a = ap.parse_args(argv)
    from a100.eval_controls import run_cell
    summary = run_cell(a.model, "nf4", a.seq, a.arms, Path(a.output), adapter=a.adapter,
                       frame_mode=a.frame_mode, resume=True, inputs=a.inputs,
                       label=a.label)
    print(json.dumps({"status": "COMPLETE", "output": a.output, "summary": summary},
                     indent=1, ensure_ascii=False)[:2000])
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "eval-cell":
        return eval_cell(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-idle-gpu", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    stages = build(args.backbone)
    root = out_root(args.backbone)
    runner = Runner(f"backbone2-{args.backbone}", root / "state.json", root / "logs",
                    require_idle_gpu=args.require_idle_gpu)
    runner.log(f"backbone {args.backbone}: {BACKBONES[args.backbone]}; {len(stages)} stages")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
