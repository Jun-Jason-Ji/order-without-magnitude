#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RTC ("read-then-convert") supervision: GPU queue.  Rules: results/courtdyn/rtc/PREREGISTRATION.json.

Inputs from tools/build_rtc_inputs.py.  Stages run through a100.runner (its own gpuq batch "rtc",
so it waits behind any running batch) and are resumable: rerun the same command.

Stages (8 GB card, estimated from M1 timings; RTC answers are ~15 tokens, so decoding is slower)
  train  RTC s42 (1,120 items, 70 steps, M1 recipe)                        ~65 min
  eval   RTC s42: rtc_{m,cm,U1,U2,pxK} full + rtc_m first-frame x4, Q1/Q2  ~60 min
  eval   v3 s42 and M1 s42: plain_{U1,U2,pxK} + rtc_{m,cm,U1,U2,pxK}       ~2 x 60 min
  eval   M1-fine s42 (only if its adapter exists): plain_{U1,U2,pxK}       ~20 min
  probe  392-item text probe, RTC adapter                                  ~10 min
Evaluation reuses the frozen cell runner (a100.eval_controls.run_cell through
tools/run_m1_mixunit.py eval-cell); RTC-format answers fail its strict parse by design and are
re-parsed from raw_answer by tools/analyze_rtc.py under the preregistered regex.

  python tools/run_rtc.py --dry-run | --run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import config as C                            # noqa: E402
from a100.runner import Runner, Stage, python_cmd       # noqa: E402
import run_m1_mixunit as M1                             # noqa: E402

OUT = ROOT / "results" / "courtdyn" / "rtc"
POOL = OUT / "rtc_v3_pool.json"
INPUTS = OUT / "inputs"
SEQS = ["Q2_top_480-510", "Q1_top_0-30"]
RTC_ARMS = ["rtc_m", "rtc_cm", "rtc_U1", "rtc_U2", "rtc_pxK"]
PLAIN_NEW = ["plain_U1", "plain_U2", "plain_pxK"]


def rtc_adapter(seed: int) -> Path:
    return ROOT / "models" / f"courtdyn-rtc-v3-s{seed}"


COMPARATORS = {"v3s42": ROOT / "models" / "courtdyn-revision-v3-eventholdout-s42",
               "m1s42": ROOT / "models" / "courtdyn-mixunit-v3-s42"}
M1_FINE = ROOT / "models" / "courtdyn-mixunit-fine-v3-s42"


def train_stage(model, seed):
    a = C.TRAIN_ARGS
    adapter = rtc_adapter(seed)
    cmd = python_cmd(
        ROOT / "train" / "train_qlora.py",
        "--model_path", model, "--qa_json", POOL, "--img_root", C.DATA,
        "--output_dir", adapter, "--num_samples", a["num_samples"],
        "--allow_base_init", "--budget_slice", a["budget_slice"],
        "--num_train_epochs", a["num_train_epochs"], "--max_steps", a["max_steps"],
        "--seed", seed, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", a["save_steps"])

    def verify():
        receipt = adapter / "training_completion.json"
        if not receipt.is_file():
            return False
        d = json.loads(receipt.read_text(encoding="utf-8"))
        return bool(d.get("completed") and d.get("seed") == seed
                    and d.get("selected_samples") == 1120 and d.get("global_step") == 70
                    and (adapter / "adapter_model.safetensors").is_file())

    return Stage(name=f"train_rtc_v3_s{seed}", cmd=cmd, verify=verify,
                 log_name=f"train_rtc_v3_s{seed}.log", needs=[POOL], stall_minutes=90,
                 max_restarts=2, note=f"1120 items, 70 steps, seed {seed}")


def eval_stage(label, adapter, seq, arms, frame_mode="full"):
    output = OUT / "eval" / f"{label}_{seq}_{frame_mode}_{'-'.join(a.split('_', 1)[1] if '_' in a else a for a in arms)}_{arms[0].split('_')[0]}"
    cmd = python_cmd(ROOT / "tools" / "run_m1_mixunit.py", "eval-cell",
                     "--label", label, "--adapter", adapter, "--seq", seq,
                     "--arms", *arms, "--frame-mode", frame_mode, "--output", output,
                     "--inputs", INPUTS)
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"],
                 note=f"{len(arms) * 280} items")


def probe_stage(seed):
    label = f"rtc_s{seed}"
    done = ROOT / "results" / "courtdyn" / "unit_probe_v2_runs" / label / "summary.json"
    return Stage(name=f"probe_{label}",
                 cmd=python_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", label,
                                "--adapter", rtc_adapter(seed).relative_to(ROOT)),
                 done_file=done, log_name=f"probe_{label}.log",
                 needs=[rtc_adapter(seed) / "adapter_model.safetensors"], note="392 text items")


def build(model, seeds=(42,)):
    stages = []
    for seed in seeds:
        stages.append(train_stage(model, seed))
        for seq in SEQS:
            stages.append(eval_stage(f"rtc_s{seed}", rtc_adapter(seed), seq, RTC_ARMS))
            stages.append(eval_stage(f"rtc_s{seed}", rtc_adapter(seed), seq, ["rtc_m"], "static4"))
    if 42 in seeds:
        for label, adapter in COMPARATORS.items():
            for seq in SEQS:
                stages.append(eval_stage(label, adapter, seq, PLAIN_NEW))
                stages.append(eval_stage(label, adapter, seq, RTC_ARMS))
        if (M1_FINE / "adapter_model.safetensors").is_file():
            for seq in SEQS:
                stages.append(eval_stage("m1fines42", M1_FINE, seq, PLAIN_NEW))
    for seed in seeds:
        stages.append(probe_stage(seed))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    if not (OUT / "PREREGISTRATION.json").is_file():
        raise SystemExit("no PREREGISTRATION.json")
    stages = build(M1.backbone(), args.seeds)
    runner = Runner("rtc", OUT / "state.json", OUT / "logs")
    runner.log(f"{len(stages)} stages")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
