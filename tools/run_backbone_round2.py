#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second-backbone replication, ROUND 2: training length chosen per backbone on a dev set.

Rules: results/courtdyn/backbone2_r2/PREREGISTRATION.json (written before this ran).
Per backbone (smol, then qwen25vl3b):
  1. For k = 1..4: train the metre-only (v3) adapter up to epoch k of a 4-epoch cosine
     schedule (train/train_qlora_epochs.py --stop_at_step 70k; resumes from epoch k-1),
     evaluate its step-70k checkpoint on the 280-item dev set, and stop at the first k with
     parse rate >= 0.90 and >= 5 distinct answers in each family.  None qualifies -> k* = 4,
     "degenerate at the 4-epoch cap".  The decision is written to stopping.json before any
     test cell is evaluated, and a recorded decision is never recomputed.
  2. Train the mixed-unit adapter on the same schedule to k* (not selected separately).
  3. Evaluate both at k* on the round-1 test cells (explicit, first-frame x4, in-template)
     and the text probe.  Base-model results are reused from round 1.
Everything is resumable: rerun after an interruption.

Usage:  python tools/run_backbone_round2.py --run [--backbones smol qwen25vl3b]
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

OUT = ROOT / "results" / "courtdyn" / "backbone2_r2"
DEV_INPUTS = OUT / "dev_inputs"
DEV_SEQ = "Q1_side_0-30"
STEPS_PER_EPOCH, MAX_EPOCHS, SEED = 70, 4, 42
MIN_PARSE, MIN_DISTINCT = 0.90, 5
WRAPPER = ROOT / "train" / "train_qlora_epochs.py"


def adapter_root(key, pool):
    return ROOT / "models" / f"courtdyn-{key}-{pool}-r2ep4-s{SEED}"


def checkpoint(key, pool, k):
    return adapter_root(key, pool) / f"checkpoint-{STEPS_PER_EPOCH * k}"


def train_stage(key, pool, k):
    a = C.TRAIN_ARGS
    ckpt = checkpoint(key, pool, k)
    cmd = python_cmd(
        WRAPPER, "--model_path", R1.BACKBONES[key], "--qa_json", R1.POOLS[pool],
        "--img_root", C.DATA, "--output_dir", adapter_root(key, pool),
        "--num_samples", a["num_samples"], "--allow_base_init",
        "--budget_slice", a["budget_slice"], "--num_train_epochs", MAX_EPOCHS,
        "--max_steps", -1, "--seed", SEED, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", STEPS_PER_EPOCH, "--stop_at_step", STEPS_PER_EPOCH * k)
    return Stage(name=f"train_{key}_{pool}_epoch{k}", cmd=cmd,
                 verify=lambda: (ckpt / "adapter_model.safetensors").is_file(),
                 log_name=f"train_{key}_{pool}.log",
                 needs=[R1.POOLS[pool], Path(R1.BACKBONES[key]) / "config.json"],
                 stall_minutes=90, max_restarts=2,
                 note=f"4-epoch schedule, stop at step {STEPS_PER_EPOCH * k}")


def eval_stage(key, label, adapter, seq, arms, mode, output, inputs=None):
    cmd = python_cmd(ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                     "--model", R1.BACKBONES[key], "--label", label, "--adapter", adapter,
                     "--seq", seq, "--arms", *arms, "--frame-mode", mode, "--output", output)
    if inputs:
        cmd += ["--inputs", str(inputs)]
    return Stage(name=f"eval_{key}_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{key}_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"])


def dev_passes(summary_path: Path):
    s = json.loads(summary_path.read_text(encoding="utf-8"))["legacy_m"]
    fams = {f: {"parse_rate": v["parse_rate"], "n_distinct": v["n_distinct"]} for f, v in s.items()}
    ok = all(v["parse_rate"] >= MIN_PARSE and v["n_distinct"] >= MIN_DISTINCT for v in fams.values())
    return ok, fams


def choose_k(key, runner) -> int:
    decision_path = OUT / key / "stopping.json"
    if decision_path.is_file():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        runner.log(f"{key}: stopping decision already recorded: k* = {d['k_star']}")
        return d["k_star"]
    trail = []
    for k in range(1, MAX_EPOCHS + 1):
        if runner.run([train_stage(key, "v3", k)]) != 0:
            raise SystemExit(f"{key}: training to epoch {k} failed")
        out = OUT / key / "dev" / f"v3_epoch{k}"
        if runner.run([eval_stage(key, f"v3_epoch{k}", checkpoint(key, "v3", k), DEV_SEQ,
                                  ["legacy_m"], "full", out, DEV_INPUTS)]) != 0:
            raise SystemExit(f"{key}: dev evaluation at epoch {k} failed")
        ok, fams = dev_passes(out / "summary.json")
        trail.append({"epoch": k, "passes": ok, "families": fams})
        runner.log(f"{key}: dev epoch {k}: {'PASS' if ok else 'fail'} {fams}")
        if ok:
            break
    k_star = next((t["epoch"] for t in trail if t["passes"]), MAX_EPOCHS)
    decision = {"backbone": key, "k_star": k_star,
                "degenerate_at_cap": not any(t["passes"] for t in trail),
                "rule": f"parse >= {MIN_PARSE} and >= {MIN_DISTINCT} distinct per family on dev",
                "trail": trail, "decided_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_cells_evaluated_before_decision": False}
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    return k_star


def test_stages(key, k):
    stages = []
    for label, pool in (("v3", "v3"), ("mixunit", "mixunit")):
        ckpt = checkpoint(key, pool, k)
        ev = OUT / key / "eval"
        for seq in R1.SEQS:
            stages.append(eval_stage(key, label, ckpt, seq, ["m_height", "cm_height", "px_height"],
                                     "full", ev / f"{label}_{seq}_full"))
            stages.append(eval_stage(key, label, ckpt, seq, ["m_height"], "static4",
                                     ev / f"{label}_{seq}_static4"))
            stages.append(eval_stage(key, label, ckpt, seq, ["legacy_m", "legacy_cm"], "full",
                                     ev / f"{label}_{seq}_full_intemplate", R1.INTEMPLATE))
        runs = OUT / key / "unit_probe"
        stages.append(Stage(
            name=f"probe_{key}_{label}",
            cmd=python_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", label,
                           "--adapter", str(ckpt), "--model-path", R1.BACKBONES[key],
                           "--out", runs),
            done_file=runs / label / "summary.json", log_name=f"probe_{key}_{label}.log",
            needs=[ckpt / "adapter_model.safetensors"]))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backbones", nargs="+", default=["smol", "qwen25vl3b"])
    args = ap.parse_args()
    if not args.run:
        ap.error("pass --run")
    if not (OUT / "PREREGISTRATION.json").is_file():
        raise SystemExit("no PREREGISTRATION.json: record the rules before running")
    if not (DEV_INPUTS / "manifest.json").is_file():
        raise SystemExit("no dev inputs: run tools/build_backbone_dev.py first")
    runner = Runner("backbone2-r2", OUT / "state.json", OUT / "logs", require_idle_gpu=True)
    for key in args.backbones:
        k = choose_k(key, runner)
        if runner.run([train_stage(key, "mixunit", k)]) != 0:
            raise SystemExit(f"{key}: mixed-unit training failed")
        if runner.run(test_stages(key, k)) != 0:
            runner.log(f"{key}: some test stages failed; see state.json")
    runner.log("round 2 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
