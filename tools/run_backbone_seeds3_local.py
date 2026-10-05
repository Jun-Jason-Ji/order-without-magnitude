#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-seed record, LOCAL part: seeds 43 and 44 for InternVL3-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B.

Rules: results/courtdyn/backbone_seeds3/PREREGISTRATION.json (commit ed6aacc, written before any
stage here was built or run).  Each backbone stays on this machine (its seed-42 machine).
Per backbone (internvl3_2b -> qwen25vl3b -> smol) and seed (43, 44):
  0. stopping.json records k FIXED to the seed-42 selection (the dev stopping rule is NOT re-run).
  1. Train the metre-only (v3) and mixed-unit adapters with --seed <s> on the seed-42 4-epoch
     cosine schedule, stopped at step 70k (train_qlora_epochs.py / train_qlora_internvl.py
     --stop_at_step 70k --save_steps 70), exactly the round-2 recipe.  Difference, recorded here:
     the seed-42 v3 adapter reached step 70k in k resumed segments (one per dev-selection epoch);
     here it is trained to 70k in one run, as the seed-42 mixed-unit adapter was.
  2. The v3 step-70k checkpoint answers the 280-item dev set (legacy_m, Q1_side_0-30),
     DESCRIPTIVELY: nothing depends on it.
  3. Both adapters: the round-2 test cells (explicit m/cm/px, first-frame x4, in-template m/cm on
     Q2_top_480-510 and Q1_top_0-30) and the 392-item text probe.  Base cells are NOT rerun.
Outputs: results/courtdyn/backbone_seeds3/<key>/s<seed>/{stopping.json,dev,eval,unit_probe};
adapters: models/courtdyn-<key>-{v3,mixunit}-r2ep4-s<seed>/checkpoint-<70k>.
Every GPU stage goes through gpuq (batch "seeds3-local").  Resumable: rerun after interruption.

Usage:  python tools/run_backbone_seeds3_local.py --dry-run | --run [--backbones ...] [--seeds ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import config as C                             # noqa: E402
from a100.runner import Runner, Stage, python_cmd        # noqa: E402
import run_backbone_replication as R1                    # noqa: E402
import run_backbone_round2 as R2                         # noqa: E402  (constants only; not mutated)

OUT = ROOT / "results" / "courtdyn" / "backbone_seeds3"
PREREG = OUT / "PREREGISTRATION.json"
IVL_MODEL = (r"E:\models\hf\hub\models--OpenGVLab--InternVL3-2B-hf\snapshots"
             r"\cb57a075cb75a2e6d1b668b128d48bb00ae321d2")
IVL_LAUNCHER = ROOT / "tools" / "internvl_launch.py"
MODELS = {"internvl3_2b": IVL_MODEL, **R1.BACKBONES}
WRAPPERS = {"internvl3_2b": ROOT / "train" / "train_qlora_internvl.py",
            "qwen25vl3b": R2.WRAPPER, "smol": R2.WRAPPER}
# k fixed to the seed-42 selection (preregistration); checked against the seed-42 stopping.json
K_FIXED = {"internvl3_2b": 2, "qwen25vl3b": 4, "smol": 4}
SEED42_STOPPING = {"internvl3_2b": ROOT / "results" / "courtdyn" / "backbone3_internvl" / "internvl3_2b",
                   "qwen25vl3b": ROOT / "results" / "courtdyn" / "backbone2_r2" / "qwen25vl3b",
                   "smol": ROOT / "results" / "courtdyn" / "backbone2_r2" / "smol"}
ORDER = ["internvl3_2b", "qwen25vl3b", "smol"]
SEEDS = [43, 44]
SPE = R2.STEPS_PER_EPOCH
# Checkpoint interval (ERRATA 5, 2026-10-02): 10 steps instead of SPE (70) for the remaining training
# stages, so that a hibernation / power loss on the laptop costs minutes, not hours.  Checkpoint
# frequency only; schedule, stop step (SPE*k) and the evaluated checkpoint are unchanged.
SAVE_STEPS = 10


def cmd_for(key, script, *args):
    """Evaluation/probe scripts of InternVL enter through its image-policy launcher (as seed 42)."""
    if key == "internvl3_2b" and Path(script).name in ("run_backbone_replication.py", "run_unit_probe.py"):
        return python_cmd(IVL_LAUNCHER, script, *args)
    return python_cmd(script, *args)


def adapter_root(key, pool, seed):
    return ROOT / "models" / f"courtdyn-{key}-{pool}-r2ep4-s{seed}"


def checkpoint(key, pool, seed):
    return adapter_root(key, pool, seed) / f"checkpoint-{SPE * K_FIXED[key]}"


def out_dir(key, seed):
    return OUT / key / f"s{seed}"


def train_stage(key, pool, seed):
    a, k = C.TRAIN_ARGS, K_FIXED[key]
    ckpt = checkpoint(key, pool, seed)
    cmd = python_cmd(
        WRAPPERS[key], "--model_path", MODELS[key], "--qa_json", R1.POOLS[pool],
        "--img_root", C.DATA, "--output_dir", adapter_root(key, pool, seed),
        "--num_samples", a["num_samples"], "--allow_base_init",
        "--budget_slice", a["budget_slice"], "--num_train_epochs", R2.MAX_EPOCHS,
        "--max_steps", -1, "--seed", seed, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", SAVE_STEPS, "--stop_at_step", SPE * k)
    return Stage(name=f"train_{key}_{pool}_s{seed}", cmd=cmd,
                 verify=lambda: (ckpt / "adapter_model.safetensors").is_file(),
                 log_name=f"train_{key}_{pool}_s{seed}.log",
                 needs=[R1.POOLS[pool], Path(MODELS[key]) / "config.json"],
                 stall_minutes=90, max_restarts=2,
                 note=f"seed {seed}; 4-epoch schedule, stop at step {SPE * k} (k fixed = seed-42 k*)")


def eval_stage(key, seed, label, adapter, seq, arms, mode, output, inputs=None):
    cmd = cmd_for(key, ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                  "--model", MODELS[key], "--label", label, "--adapter", adapter,
                  "--seq", seq, "--arms", *arms, "--frame-mode", mode, "--output", output)
    if inputs:
        cmd += ["--inputs", str(inputs)]
    return Stage(name=f"eval_{key}_s{seed}_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{key}_s{seed}_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"])


def dev_stage(key, seed):
    k = K_FIXED[key]
    return eval_stage(key, seed, f"v3_epoch{k}", checkpoint(key, "v3", seed), R2.DEV_SEQ, ["legacy_m"],
                      "full", out_dir(key, seed) / "dev" / f"v3_epoch{k}", R2.DEV_INPUTS)


def test_stages(key, seed):
    stages, base = [], out_dir(key, seed)
    for label in ("v3", "mixunit"):
        ckpt, ev = checkpoint(key, label, seed), base / "eval"
        for seq in R1.SEQS:
            stages.append(eval_stage(key, seed, label, ckpt, seq, ["m_height", "cm_height", "px_height"],
                                     "full", ev / f"{label}_{seq}_full"))
            stages.append(eval_stage(key, seed, label, ckpt, seq, ["m_height"], "static4",
                                     ev / f"{label}_{seq}_static4"))
            stages.append(eval_stage(key, seed, label, ckpt, seq, ["legacy_m", "legacy_cm"], "full",
                                     ev / f"{label}_{seq}_full_intemplate", R1.INTEMPLATE))
        runs = base / "unit_probe"
        stages.append(Stage(
            name=f"probe_{key}_s{seed}_{label}",
            cmd=cmd_for(key, ROOT / "tools" / "run_unit_probe.py", "--label", label,
                        "--adapter", str(ckpt), "--model-path", MODELS[key], "--out", runs),
            done_file=runs / label / "summary.json", log_name=f"probe_{key}_s{seed}_{label}.log",
            needs=[ckpt / "adapter_model.safetensors"]))
    return stages


def stages_for(key, seed):
    return ([train_stage(key, "v3", seed), train_stage(key, "mixunit", seed), dev_stage(key, seed)]
            + test_stages(key, seed))


def record_schedule(key, seed):
    """stopping.json for this seed: k fixed to the seed-42 decision, written before any stage."""
    s42 = json.loads((SEED42_STOPPING[key] / "stopping.json").read_text(encoding="utf-8"))
    if s42["k_star"] != K_FIXED[key]:
        raise SystemExit(f"{key}: seed-42 k* {s42['k_star']} != fixed k {K_FIXED[key]}")
    path = out_dir(key, seed) / "stopping.json"
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "backbone": key, "seed": seed, "k_star": K_FIXED[key],
        "degenerate_at_cap": bool(s42.get("degenerate_at_cap")),
        "schedule": "fixed to seed-42 selection (dev stopping rule not re-run)",
        "seed42_stopping": str((SEED42_STOPPING[key] / "stopping.json").relative_to(ROOT)),
        "prereg_rules_sha256": json.loads(PREREG.read_text(encoding="utf-8"))["rules_sha256"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_cells_evaluated_before_this": False}, indent=1) + "\n", encoding="utf-8")


def check_inputs():
    missing = [str(p) for p in [PREREG, R2.DEV_INPUTS / "manifest.json", R1.INTEMPLATE, IVL_LAUNCHER,
                                *R1.POOLS.values(), *WRAPPERS.values(),
                                *(Path(m) / "config.json" for m in MODELS.values()),
                                *(d / "stopping.json" for d in SEED42_STOPPING.values())]
               if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backbones", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, choices=SEEDS)
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    check_inputs()
    runner = Runner("seeds3-local", OUT / "local_state.json", OUT / "logs", require_idle_gpu=True)
    plan = [(k, s) for k in args.backbones for s in args.seeds]
    if args.dry_run:
        return runner.run([st for k, s in plan for st in stages_for(k, s)], dry_run=True)
    failed = []
    for key, seed in plan:
        record_schedule(key, seed)
        if runner.run(stages_for(key, seed)) != 0:
            failed.append(f"{key}_s{seed}")
            runner.log(f"{key} seed {seed}: some stages failed; see local_state.json")
    runner.log(f"seeds3 local complete; failed groups: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
