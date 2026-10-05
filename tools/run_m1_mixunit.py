#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1: does metre-only supervision explain the unit failure?  GPU queue.

Inputs come from tools/build_mixunit_pool.py.  One adapter is trained on the
event-holdout v3 pool with half of its targets re-expressed in centimetres, at
the published v3 adapter's exact hyperparameters and seed (only the unit mix
differs), then evaluated on the same cells as the published v3 seed-42 adapter
plus an in-template centimetre test.  Stages run through a100.runner, so the
queue is resumable after any interruption: rerun the same command.

Stages (about 3 GPU-hours on the 8 GB card, measured from earlier cells)
  train   mixunit v3 s42                                         ~62 min
  eval    explicit Q1/Q2 full (m, cm, px) + static4 (m)          ~55 min
  eval    in-template Q1/Q2 (m, cm), mixunit adapter            ~28 min
  eval    in-template Q1/Q2 (m, cm), published v3 s42 adapter   ~28 min  (matched baseline)
  probe   392-item text probe, mixunit adapter                  ~ 8 min

Preregistered decision rule (recorded before training)
------------------------------------------------------
Primary statistic: the paired centimetre/metre prediction ratio R (median of
itemwise ratios, Eq. pairedratio of the manuscript) in the four cells
{Q1, Q2} x {speed, path}.  The published v3 seed-42 adapter gives R =
1.38 / 1.00 / 1.57 / 1.00 on the explicit arms; the exact factor is 100.

  S  "supervision explains it":      >= 3 of 4 cells have R >= 10
  N  "supervision does not explain": >= 3 of 4 cells have R <= 3
  I  otherwise: intermediate, reported as a distribution, no verdict forced

Applied separately to
  E  the explicit-coordinate arms (the published comparison), and
  T  the in-template arms (training wording), against the v3 s42 adapter
     evaluated on the same in-template arms.

Interpretation fixed in advance
  E=S           unit non-response in the paper is a consequence of single-unit
                supervision; the paper's claim must be reframed around it.
  E=N and T=S   mixed supervision teaches the unit only in its trained wording;
                the explicit-arm failure is a failure to transfer across
                question templates, and the claim is reframed accordingly.
  E=N and T=N   metre-only supervision does not explain the centimetre failure;
                the confound is ruled out for the metric-unit contrast.
  any I         reported as intermediate; no reframing is claimed from it.

Reported without a verdict: metre-arm rho and T-MRA against v3 s42 (mixing
halves the metre targets, so a metre cost is possible and is reported, not
tested); px/m ratio (an unseen unit); first-frame drop; text-probe accuracy.

Usage
  python tools/run_m1_mixunit.py --dry-run
  python tools/run_m1_mixunit.py --run
  python tools/run_m1_mixunit.py eval-cell ...      (internal, one cell)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100 import config as C                            # noqa: E402
from a100.runner import Runner, Stage, python_cmd       # noqa: E402

OUT = ROOT / "results" / "courtdyn" / "m1_mixunit"
POOL = OUT / "mixunit_v3_pool.json"
INTEMPLATE = OUT / "intemplate"
MIX_ADAPTER = ROOT / "models" / "courtdyn-mixunit-v3-s42"
V3_ADAPTER = ROOT / "models" / "courtdyn-revision-v3-eventholdout-s42"
SEQS = ["Q2_top_480-510", "Q1_top_0-30"]
SEED = 42
EXPECTED_ITEMS, EXPECTED_STEPS = 1120, 70


def backbone() -> str:
    path = os.environ.get("QWEN35_4B")
    if not path:
        raise SystemExit("set QWEN35_4B to the reproduction backbone")
    return path


def mix_adapter(seed: int) -> Path:
    return ROOT / "models" / f"courtdyn-mixunit-v3-s{seed}"


def v3_adapter(seed: int) -> Path:
    """The metre-only v3 adapter of the same seed (the matched baseline)."""
    return C.adapter_dir("v3", seed)


def tag(seed: int) -> str:
    """Seed 42 keeps its original stage and directory names so finished work is reused."""
    return "" if seed == SEED else f"_s{seed}"


def train_stage(model, seed=SEED):
    a = C.TRAIN_ARGS
    adapter = mix_adapter(seed)
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
                    and d.get("selected_samples") == EXPECTED_ITEMS
                    and d.get("global_step") == EXPECTED_STEPS
                    and (adapter / "adapter_model.safetensors").is_file())

    return Stage(name=f"train_mixunit_v3_s{seed}", cmd=cmd, verify=verify,
                 log_name=f"train_mixunit_v3_s{seed}.log", needs=[POOL],
                 stall_minutes=90, max_restarts=2,
                 note=f"{EXPECTED_ITEMS} items, {EXPECTED_STEPS} steps, seed {seed}")


def eval_stage(label, adapter, seq, arms, frame_mode, inputs=None):
    output = OUT / "eval" / (f"{label}_{seq}_{frame_mode}"
                             + ("_intemplate" if inputs else ""))
    cmd = python_cmd(Path(__file__).resolve(), "eval-cell",
                     "--label", label, "--adapter", adapter, "--seq", seq,
                     "--arms", *arms, "--frame-mode", frame_mode, "--output", output)
    if inputs:
        cmd += ["--inputs", str(inputs)]
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"],
                 note=f"{len(arms) * 280} items")


def probe_stage():
    done = ROOT / "results" / "courtdyn" / "unit_probe_v2_runs" / "mixunit" / "summary.json"
    return Stage(name="probe_mixunit",
                 cmd=python_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", "mixunit"),
                 done_file=done, log_name="probe_mixunit.log",
                 needs=[MIX_ADAPTER / "adapter_model.safetensors"], note="392 text items")


def build(model, seeds=(SEED,)):
    stages = []
    for seed in seeds:
        mix, t = mix_adapter(seed), tag(seed)
        stages.append(train_stage(model, seed))
        for seq in SEQS:
            stages.append(eval_stage(f"mixunit{t}", mix, seq, C.ARMS_THREE, "full"))
            stages.append(eval_stage(f"mixunit{t}", mix, seq, ["m_height"], "static4"))
        for label, adapter in ((f"mixunit{t}", mix), (f"v3s{seed}", v3_adapter(seed))):
            for seq in SEQS:
                stages.append(eval_stage(label, adapter, seq, ["legacy_m", "legacy_cm"],
                                         "full", inputs=INTEMPLATE))
        if seed == SEED:
            stages.append(probe_stage())
    return stages


def eval_cell(argv):
    ap = argparse.ArgumentParser(prog="run_m1_mixunit.py eval-cell")
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--frame-mode", default="full")
    ap.add_argument("--output", required=True)
    ap.add_argument("--inputs")
    a = ap.parse_args(argv)
    from a100.eval_controls import run_cell
    output = Path(a.output)
    # run_cell refuses an existing directory unless resuming; resume is only
    # allowed when run_config.json exists (a partial run of this same cell).
    resume = (output / "run_config.json").is_file()
    summary = run_cell(backbone(), "nf4", a.seq, a.arms, output, adapter=a.adapter,
                       frame_mode=a.frame_mode, resume=resume, inputs=a.inputs,
                       label=a.label)
    print(json.dumps({"status": "COMPLETE", "output": str(output),
                      "summary": summary}, indent=1, ensure_ascii=False)[:2000])
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "eval-cell":
        return eval_cell(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--require-idle-gpu", action="store_true",
                    help="wait before each stage until no other process holds the card")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    if not (OUT / "PREREGISTRATION.json").is_file():
        raise SystemExit("no PREREGISTRATION.json: record the decision rule before running")
    stages = build(backbone(), args.seeds)
    runner = Runner("m1-mixunit", OUT / "state.json", OUT / "logs",
                    require_idle_gpu=args.require_idle_gpu)
    runner.log(f"{len(stages)} stages")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
