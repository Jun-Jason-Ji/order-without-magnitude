#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RTC replication queue: seeds 43/44 on Qwen3.5-4B, then InternVL3-2B.

Rules: results/courtdyn/rtc/PREREGISTRATION.json (REPLICATION PLAN) with ERRATA items 2-3
(conversion band; pre-run specification of seeds, comparators, InternVL schedule, summary wording).
Gate (ERRATA 3a): run only if seed 42's H1_unseen_units_rtc = "holds"; --run refuses otherwise.

Outputs results/courtdyn/rtc_rep/{eval,logs,unit_probe,state.json}.  Stages go through
a100/runner.py as gpuq batch "rtc-rep" (waits behind any running batch) and are resumable:
rerun the same command.  tools/run_rtc.py and tools/analyze_rtc.py (seed 42) are imported
read-only and not modified.

  seed s in 43, 44 (Qwen3.5-4B, QWEN35_4B must be set):
    train RTC s   (run_rtc.train_stage: same pool and recipe, --seed s)        ~65 min
    eval  RTC s   rtc_{m,cm,U1,U2,pxK} full + rtc_m first-frame x4, Q1/Q2      ~60 min
    eval  v3 s, M1 s   plain_{U1,U2,pxK} + rtc_{m,cm,U1,U2,pxK}, Q1/Q2          ~2 x 60 min
    eval  v3 s   plain_m Q1 (no published Q1 metre cell)                        ~5 min
    probe 392 text items, RTC s                                                 ~10 min
  InternVL3-2B (through tools/internvl_launch.py):
    train RTC, 4-epoch cosine schedule stopped at step 140 (k* = 2)             ~50 min
    eval  RTC + InternVL v3 / mixed-unit (checkpoint-140), same arms            ~2.5 h
    probe 392 text items, RTC InternVL                                          ~10 min

  python tools/run_rtc_replication.py --dry-run | --run [--force-gate]
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
import run_rtc as RTC                                   # noqa: E402  (read-only reuse)
import run_m1_mixunit as M1                             # noqa: E402

CD = ROOT / "results" / "courtdyn"
OUT = CD / "rtc_rep"
SEEDS = [43, 44]
SEQS = RTC.SEQS
RTC_ARMS, PLAIN_NEW = RTC.RTC_ARMS, RTC.PLAIN_NEW

# InternVL3-2B (ERRATA 3c)
IVL_MODEL = (r"E:\models\hf\hub\models--OpenGVLab--InternVL3-2B-hf\snapshots"
             r"\cb57a075cb75a2e6d1b668b128d48bb00ae321d2")
IVL_LAUNCHER = ROOT / "tools" / "internvl_launch.py"
IVL_TRAINER = ROOT / "train" / "train_qlora_internvl.py"
IVL_STOPPING = CD / "backbone3_internvl" / "internvl3_2b" / "stopping.json"
IVL_STEPS_PER_EPOCH, IVL_EPOCHS = 70, 4


def ivl_k() -> int:
    return json.loads(IVL_STOPPING.read_text(encoding="utf-8"))["k_star"]


def ivl_rtc_root() -> Path:
    return ROOT / "models" / "courtdyn-internvl3_2b-rtc-r2ep4-s42"


def ivl_ckpt(root: Path) -> Path:
    return root / f"checkpoint-{IVL_STEPS_PER_EPOCH * ivl_k()}"


IVL_COMPARATORS = {"ivlv3": ROOT / "models" / "courtdyn-internvl3_2b-v3-r2ep4-s42",
                   "ivlmix": ROOT / "models" / "courtdyn-internvl3_2b-mixunit-r2ep4-s42"}


def cell_output(label, seq, frame_mode, arms) -> Path:
    """Same naming as tools/run_rtc.py / tools/analyze_rtc.py cell_dir."""
    suffix = "-".join(a.split("_", 1)[1] if "_" in a else a for a in arms)
    return OUT / "eval" / f"{label}_{seq}_{frame_mode}_{suffix}_{arms[0].split('_')[0]}"


# ---------------------------------------------------------------- Qwen3.5-4B seeds
def qwen_eval(label, adapter, seq, arms, frame_mode="full"):
    output = cell_output(label, seq, frame_mode, arms)
    cmd = python_cmd(ROOT / "tools" / "run_m1_mixunit.py", "eval-cell",
                     "--label", label, "--adapter", adapter, "--seq", seq,
                     "--arms", *arms, "--frame-mode", frame_mode, "--output", output,
                     "--inputs", RTC.INPUTS)
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"], note=f"{len(arms) * 280} items")


def probe(label, adapter, extra=(), launcher=None):
    runs = OUT / "unit_probe"
    args = ["--label", label, "--adapter", adapter, "--out", runs, *extra]
    cmd = (python_cmd(launcher, ROOT / "tools" / "run_unit_probe.py", *args) if launcher
           else python_cmd(ROOT / "tools" / "run_unit_probe.py", *args))
    return Stage(name=f"probe_{label}", cmd=cmd, done_file=runs / label / "summary.json",
                 log_name=f"probe_{label}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"], note="392 text items")


def seed_stages(model, seed):
    rtc = RTC.rtc_adapter(seed)
    st = [RTC.train_stage(model, seed)]
    for seq in SEQS:
        st.append(qwen_eval(f"rtc_s{seed}", rtc, seq, RTC_ARMS))
        st.append(qwen_eval(f"rtc_s{seed}", rtc, seq, ["rtc_m"], "static4"))
    comps = {f"v3s{seed}": ROOT / "models" / f"courtdyn-v3-s{seed}",
             f"m1s{seed}": ROOT / "models" / f"courtdyn-mixunit-v3-s{seed}"}
    for label, adapter in comps.items():
        for seq in SEQS:
            st.append(qwen_eval(label, adapter, seq, PLAIN_NEW))
            st.append(qwen_eval(label, adapter, seq, RTC_ARMS))
    # ERRATA 3b: v3 s43/s44 have no published Q1 metre cell
    st.append(qwen_eval(f"v3s{seed}", comps[f"v3s{seed}"], "Q1_top_0-30", ["plain_m"]))
    st.append(probe(f"rtc_s{seed}", rtc.relative_to(ROOT)))
    return st


# ---------------------------------------------------------------- InternVL3-2B
def ivl_train():
    a = C.TRAIN_ARGS
    root = ivl_rtc_root()
    stop = IVL_STEPS_PER_EPOCH * ivl_k()
    cmd = python_cmd(
        IVL_TRAINER, "--model_path", IVL_MODEL, "--qa_json", RTC.POOL, "--img_root", C.DATA,
        "--output_dir", root, "--num_samples", a["num_samples"], "--allow_base_init",
        "--budget_slice", a["budget_slice"], "--num_train_epochs", IVL_EPOCHS, "--max_steps", -1,
        "--seed", 42, "--lr", a["lr"], "--grad_accum", a["grad_accum"], "--lora_r", a["lora_r"],
        "--max_pixels", a["max_pixels"], "--save_steps", IVL_STEPS_PER_EPOCH, "--stop_at_step", stop)
    ckpt = ivl_ckpt(root)
    return Stage(name="train_rtc_internvl3_2b", cmd=cmd,
                 verify=lambda: (ckpt / "adapter_model.safetensors").is_file(),
                 log_name="train_rtc_internvl3_2b.log", needs=[RTC.POOL, Path(IVL_MODEL) / "config.json"],
                 stall_minutes=90, max_restarts=2, note=f"4-epoch schedule, stop at step {stop}")


def ivl_eval(label, adapter, seq, arms, frame_mode="full"):
    output = cell_output(label, seq, frame_mode, arms)
    cmd = python_cmd(IVL_LAUNCHER, ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                     "--model", IVL_MODEL, "--label", label, "--adapter", adapter, "--seq", seq,
                     "--arms", *arms, "--frame-mode", frame_mode, "--output", output,
                     "--inputs", RTC.INPUTS)
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors"], note=f"{len(arms) * 280} items")


def ivl_stages():
    rtc = ivl_ckpt(ivl_rtc_root())
    st = [ivl_train()]
    for seq in SEQS:
        st.append(ivl_eval("rtc_ivl", rtc, seq, RTC_ARMS))
        st.append(ivl_eval("rtc_ivl", rtc, seq, ["rtc_m"], "static4"))
    for label, root in IVL_COMPARATORS.items():
        for seq in SEQS:
            st.append(ivl_eval(label, ivl_ckpt(root), seq, PLAIN_NEW))
            st.append(ivl_eval(label, ivl_ckpt(root), seq, RTC_ARMS))
    st.append(probe("rtc_ivl", rtc, ("--model-path", IVL_MODEL), launcher=IVL_LAUNCHER))
    return st


def gate_ok() -> tuple[bool, str]:
    v = CD / "rtc" / "verdict.json"
    if not v.is_file():
        return False, "no results/courtdyn/rtc/verdict.json"
    h = json.loads(v.read_text(encoding="utf-8"))["hypotheses"]["H1_unseen_units_rtc"]
    status = h[0] if isinstance(h, (list, tuple)) else h
    return status == "holds", f"seed-42 H1_unseen_units_rtc = {status}"


def build(model):
    stages = []
    for seed in SEEDS:
        stages += seed_stages(model, seed)
    return stages + ivl_stages()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-gate", action="store_true", help="tests only: ignore the seed-42 gate")
    args = ap.parse_args()
    if not (args.run or args.dry_run):
        ap.error("pass --run or --dry-run")
    ok, why = gate_ok()
    if args.run and not ok and not args.force_gate:
        raise SystemExit(f"replication gate closed (ERRATA 3a): {why}")
    model = M1.backbone() if args.run else (__import__("os").environ.get("QWEN35_4B") or "QWEN35_4B")
    stages = build(model)
    runner = Runner("rtc-rep", OUT / "state.json", OUT / "logs")
    runner.log(f"{len(stages)} stages; gate: {why}")
    return runner.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
