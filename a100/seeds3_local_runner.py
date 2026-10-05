#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-seed record, the groups moved from the laptop to the A100 (ERRATA 3 of
results/courtdyn/backbone_seeds3/ERRATA.md): Qwen2.5-VL-3B and SmolVLM2-2.2B, seeds 43 and 44.

Parts, run in this order by a100/run_seeds3_local.sh (each part in its own process, because the
round-2 builders keep seed and output directory as module globals):
    qwen25vl3b_s43 -> qwen25vl3b_s44 -> smol_s43 -> smol_s44
Per part, exactly the local orchestrator's stage list (tools/run_backbone_seeds3_local.py):
    train v3 to step 280 (k FIXED = 4, the seed-42 selection; 4-epoch cosine schedule = full schedule)
    -> dev answers at step 280 (legacy_m on Q1_side_0-30; descriptive only)
    -> train mixed-unit to step 280
    -> {v3, mixunit} x {Q2_top_480-510, Q1_top_0-30} x {full m/cm/px, static4 m, in-template
       legacy_m/cm} + the 392-item text probe for both adapters.
Base cells are reused from the local seed-42 round (results/courtdyn/backbone2).  Training goes
through train/train_durable.py --family epochs (train_qlora_epochs.py -> frozen train_qlora.py) with a
resumable checkpoint every 10 steps; evaluation calls tools/run_backbone_replication.py directly, as
the local seed-42 runs did (no family launcher: both are frozen base families of eval/run_bench).
Outputs: results/courtdyn/backbone_seeds3/s<seed>/<key>/{eval,dev,unit_probe,logs,state.json,
stopping.json} (the A100 layout tools/analyze_backbone_replication.py --round 6 reads);
adapters models/courtdyn-<key>-{v3,mixunit}-r2ep4-s<seed>.
Base weights: the T36 sweep's copies (a100/config.py BACKBONE_SWEEP keys qwen25vl-3b, smolvlm2-2b),
resolved with C.backbone_path; the snapshot revision is recorded in stopping.json (ERRATA 3).

  python a100/seeds3_local_runner.py --part qwen25vl3b_s43 --dry-run | --run
  python a100/seeds3_local_runner.py --progress
Exit codes: 0 done, 1 some stages failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import config as C                             # noqa: E402
from a100 import durable as D                            # noqa: E402
from a100.runner import Runner, Stage, now, python_cmd   # noqa: E402

CD = ROOT / "results" / "courtdyn"
OUT = CD / "backbone_seeds3"
PREREG = OUT / "PREREGISTRATION.json"
PREREG_SHA = "a259c9e07ecf2b095fcf67ae66113dbb8addcc53e7e91041cecfb725042bf2cb"
ERRATA = OUT / "ERRATA.md"
DURABLE = ROOT / "train" / "train_durable.py"
SAVE_EVERY = 10
K = 4                                                    # seed-42 k* of both backbones (round 2)
SWEEP_KEY = {"qwen25vl3b": "qwen25vl-3b", "smol": "smolvlm2-2b"}
SEED42_STOPPING = {"qwen25vl3b": CD / "backbone2_r2" / "qwen25vl3b" / "stopping.json",
                   "smol": CD / "backbone2_r2" / "smol" / "stopping.json"}
# part -> (key, seed, rough A100-40GB hours: local 5060 timings / 3)
PARTS = {
    "qwen25vl3b_s43": ("qwen25vl3b", 43, 3.5),
    "qwen25vl3b_s44": ("qwen25vl3b", 44, 3.5),
    "smol_s43": ("smol", 43, 1.5),
    "smol_s44": ("smol", 44, 1.5),
}


def seed_root(seed: int) -> Path:
    return OUT / f"s{seed}"


def state_path(key: str, seed: int) -> Path:
    return seed_root(seed) / key / "state.json"


def model_dir(key: str) -> Path:
    path = C.backbone_path(SWEEP_KEY[key])
    if path is None or not (Path(path) / "config.json").is_file():
        raise SystemExit(f"{key}: base weights not found on this machine (sweep key {SWEEP_KEY[key]}; "
                         f"set ${C.BACKBONES[SWEEP_KEY[key]]['env']} or place them under {C.MODEL_ROOT})")
    return Path(path)


# ------------------------------------------------------------------ progress page
def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render() -> str:
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# A100 three-seed run progress (groups moved from the laptop, ERRATA 3)", "",
             f"Updated {t} UTC.  Order: Qwen2.5-VL-3B s43 -> s44 -> SmolVLM2-2.2B s43 -> s44.", "",
             "| part | done | running | failed | planned | est. hours |", "|---|---|---|---|---|---|"]
    running_all, left = [], 0.0
    for part, (key, seed, hours) in PARTS.items():
        st = _load(state_path(key, seed)) or {}
        stages = st.get("stages", {})
        plan = len(st.get("plan", [])) or 0
        done = sum(s.get("status") == "done" for s in stages.values())
        failed = sum(s.get("status") == "failed" for s in stages.values())
        running = [f"{n} (since {s.get('started', '?')}, attempt {s.get('restarts', 0) + 1})"
                   for n, s in stages.items() if s.get("status") in ("running", "waiting_gpu", "queued_gpuq")]
        running_all += running
        frac = 1.0 if st.get("finished") and not running else (done / plan if plan else 0.0)
        left += hours * (1 - min(frac, 1.0))
        lines.append(f"| {part} | {done} | {len(running)} | {failed} | {plan or '-'} | {hours} |")
    lines += ["", "Running now: " + ("; ".join(running_all) if running_all else "nothing (between stages, "
                                     "waiting for the seeds3 / UnitDyn runs, or not started)"),
              "", f"Rough time left: ~{left:.1f} GPU-hours.", "",
              "After a restart: `bash a100/resume_seeds3_local.sh`.  Results: `bash a100/pack_seeds3_local.sh`."]
    return "\n".join(lines) + "\n"


def write_progress():
    D.atomic_write_text(C.ROOT / "PROGRESS_SEEDS3_LOCAL.md", render())


class LocalGroupsRunner(Runner):
    def __init__(self, *args, **kwargs):
        if "require_idle_gpu" not in kwargs:
            kwargs["require_idle_gpu"] = os.environ.get("A100_REQUIRE_IDLE_GPU", "1") != "0"
        super().__init__(*args, **kwargs)

    def save(self):
        self.state["updated"] = now()
        D.atomic_write_text(self.state_path,
                            json.dumps(self.state, indent=1, ensure_ascii=False) + "\n")
        try:
            write_progress()
        except Exception as error:                     # noqa: BLE001
            print(f"[progress-seeds3-local] not written: {error!r}", flush=True)


# ------------------------------------------------------------------ stages
def durable_train_stage(key, pool, k):
    """tools/run_backbone_round2.train_stage, entered through the durable wrapper (family 'epochs',
    i.e. train_qlora_epochs.py -> frozen train_qlora.py, exactly the local seed-42/43/44 path)."""
    import run_backbone_a100 as A4                       # checkpoint_done only
    import run_backbone_replication as R1
    import run_backbone_round2 as R2
    a = C.TRAIN_ARGS
    ckpt = R2.checkpoint(key, pool, k)
    cmd = python_cmd(
        DURABLE, "--family", "epochs", "--keep-every", R2.STEPS_PER_EPOCH,
        "--model_path", R1.BACKBONES[key], "--qa_json", R1.POOLS[pool],
        "--img_root", C.DATA, "--output_dir", R2.adapter_root(key, pool),
        "--num_samples", a["num_samples"], "--allow_base_init",
        "--budget_slice", a["budget_slice"], "--num_train_epochs", R2.MAX_EPOCHS,
        "--max_steps", -1, "--seed", R2.SEED, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", SAVE_EVERY, "--stop_at_step", R2.STEPS_PER_EPOCH * k)
    return Stage(name=f"train_{key}_{pool}_s{R2.SEED}", cmd=cmd,
                 verify=lambda: A4.checkpoint_done(ckpt),
                 log_name=f"train_{key}_{pool}_s{R2.SEED}.log",
                 needs=[R1.POOLS[pool], Path(R1.BACKBONES[key]) / "config.json"],
                 stall_minutes=60, max_restarts=6,
                 note=f"seed {R2.SEED}; 4-epoch schedule, stop at step {R2.STEPS_PER_EPOCH * k} (k fixed = seed-42 k*)")


def configure(key: str, seed: int):
    """Point the round-2 builders at this backbone, seed and output root (A100 layout)."""
    import run_backbone_replication as R1
    import run_backbone_round2 as R2
    R1.BACKBONES[key] = str(model_dir(key))
    R2.OUT = seed_root(seed)
    R2.SEED = seed
    R2.train_stage = durable_train_stage
    return R1, R2


def seed_stages(key: str, seed: int):
    R1, R2 = configure(key, seed)
    st = [durable_train_stage(key, "v3", K)]
    st.append(R2.eval_stage(key, f"v3_epoch{K}", R2.checkpoint(key, "v3", K), R2.DEV_SEQ, ["legacy_m"],
                            "full", R2.OUT / key / "dev" / f"v3_epoch{K}", R2.DEV_INPUTS))
    st.append(durable_train_stage(key, "mixunit", K))
    st += R2.test_stages(key, K)
    return st


def write_schedule_record(key: str, seed: int):
    """<key>/stopping.json: the FIXED schedule (seed-42 k*), plus the base snapshot used here."""
    path = seed_root(seed) / key / "stopping.json"
    if path.is_file():
        return
    s42 = json.loads(SEED42_STOPPING[key].read_text(encoding="utf-8"))
    if s42["k_star"] != K:
        raise SystemExit(f"{key}: seed-42 k* {s42['k_star']} != fixed k {K}")
    D.atomic_write_json(path, {
        "backbone": key, "seed": seed, "k_star": K,
        "degenerate_at_cap": bool(s42.get("degenerate_at_cap")),
        "schedule": "fixed to seed-42 selection (dev stopping rule not re-run)",
        "seed42_stopping": str(SEED42_STOPPING[key].relative_to(ROOT)),
        "machine": "A100-40GB (ERRATA 3; seed-42 cells were run on the local RTX 5060)",
        "base_model_dir": str(model_dir(key)),
        "base_model_snapshot": model_dir(key).name,
        "prereg_rules_sha256": PREREG_SHA,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_cells_evaluated_before_this": False})


def check_records():
    if not PREREG.is_file():
        raise SystemExit(f"no {PREREG}: record the rules first")
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    if hashlib.sha256(p["rules"].encode("utf-8")).hexdigest() != PREREG_SHA or p["rules_sha256"] != PREREG_SHA:
        raise SystemExit(f"{PREREG} does not match the recorded rules (sha256 {PREREG_SHA[:12]}...)")
    if not ERRATA.is_file() or "## 3." not in ERRATA.read_text(encoding="utf-8"):
        raise SystemExit(f"{ERRATA} lacks item 3 (machine assignment); refusing to run")
    for key, s in SEED42_STOPPING.items():
        if not s.is_file():
            raise SystemExit(f"missing seed-42 stopping record {s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=list(PARTS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if args.progress:
        write_progress()
        print(render())
        return 0
    if not args.part:
        ap.error("--part is required with --run / --dry-run")
    key, seed, _ = PARTS[args.part]
    check_records()
    stages = seed_stages(key, seed)
    runner = LocalGroupsRunner(f"seeds3-{args.part}", state_path(key, seed), seed_root(seed) / key / "logs")
    if args.run:
        write_schedule_record(key, seed)
    rc = runner.run(stages, dry_run=args.dry_run)
    if not args.dry_run:
        runner.log(f"{args.part}: complete" + (" (with failures)" if rc else ""))
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
