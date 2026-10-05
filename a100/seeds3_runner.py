#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A100 part of the three-seed record (results/courtdyn/backbone_seeds3/PREREGISTRATION.json).

Parts, run in this order by a100/run_seeds3.sh (each in its own process: the round-2 builders keep
the seed and output directory as module globals):
    gemma3_12b_s44 -> pixtral_12b_s44 -> idefics3_8b_s43 -> idefics3_8b_s44
Gemma and Pixtral first because their seed-44 results complete a backbone whose other two seeds
already exist (Pixtral-12B's three-seed R1 label decides the paper title).

Per part (the extension's seed-43 group, tools/run_a100_ext.py, with the seed as data):
    train v3 to step 140 (FIXED schedule = seed-42 k* 2; the dev rule is not re-run)
    -> dev answers at step 140 (descriptive only) -> train mixed-unit to step 140
    -> explicit m/cm/px full, first-frame x4, in-template cells on both test clips + text probe,
       for both adapters.  Unadapted-backbone cells are reused from results/courtdyn/backbone4_a100.
Outputs: results/courtdyn/backbone_seeds3/s<seed>/<key>/{eval,dev,unit_probe,logs,state.json,
stopping.json}, i.e. the inner layout of results/courtdyn/backbone4_ext/<key>/, so
tools/analyze_backbone_replication.py reads a seed by pointing its root at backbone_seeds3/s<seed>.
Adapters: models/courtdyn-<key>-{v3,mixunit}-r2ep4-s<seed>.

Reuses the round-4 machinery unchanged (tools/run_backbone_a100.py configure/train_stage/launcher,
tools/run_backbone_round2.py builders, train/train_durable.py, tools/family_launch.py); the frozen
a100/eval_controls.py, model_families.py and config.py are not touched.

  python a100/seeds3_runner.py --part gemma3_12b_s44 --dry-run | --run
  python a100/seeds3_runner.py --progress          (rewrite PROGRESS_SEEDS3.md)
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
from a100.runner import Runner, now                      # noqa: E402

CD = ROOT / "results" / "courtdyn"
OUT = CD / "backbone_seeds3"
PREREG = OUT / "PREREGISTRATION.json"
PREREG_SHA = "a259c9e07ecf2b095fcf67ae66113dbb8addcc53e7e91041cecfb725042bf2cb"
ROUND4_PREREG = CD / "backbone4_a100" / "PREREGISTRATION.json"
K = 2                                                    # seed-42 k* of all three backbones
# part -> (key, seed, rough A100-40GB hours: Gemma/Pixtral from the extension's seed-43 estimate,
# Idefics3 from its round-4 stage timings without the base cells and the epoch-1 dev step)
PARTS = {
    "gemma3_12b_s44": ("gemma3_12b", 44, 5.5),
    "pixtral_12b_s44": ("pixtral_12b", 44, 4.5),
    "idefics3_8b_s43": ("idefics3_8b", 43, 3.0),
    "idefics3_8b_s44": ("idefics3_8b", 44, 3.0),
}


def seed_root(seed: int) -> Path:
    return OUT / f"s{seed}"


def state_path(key: str, seed: int) -> Path:
    return seed_root(seed) / key / "state.json"


# ------------------------------------------------------------------ progress page
def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render() -> str:
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# A100 three-seed run progress", "",
             f"Updated {t} UTC.  Order: Gemma3-12B s44 -> Pixtral-12B s44 -> Idefics3-8B s43 -> "
             "Idefics3-8B s44.", "",
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
                                     "waiting for the extension run, or not started)"),
              "", f"Rough time left: ~{left:.1f} GPU-hours.", "",
              "After a restart: `bash a100/resume_seeds3.sh` (finished stages are skipped; training "
              "resumes from its last durable checkpoint).  Results: `bash a100/pack_seeds3.sh`."]
    return "\n".join(lines) + "\n"


def write_progress():
    D.atomic_write_text(C.ROOT / "PROGRESS_SEEDS3.md", render())


class Seeds3Runner(Runner):
    """Durable state writes; progress page rewritten on every state change (as a100/ext_runner.py)."""

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
            print(f"[progress-seeds3] not written: {error!r}", flush=True)


# ------------------------------------------------------------------ stages
def seed_stages(key: str, seed: int):
    import run_backbone_a100 as A4                       # read-only reuse
    import run_backbone_round2 as R2
    A4.configure(key)                                    # model path, launcher, durable train stage
    R2.OUT = seed_root(seed)
    R2.SEED = seed
    st = [A4.train_stage(key, "v3", K)]
    st.append(R2.eval_stage(key, f"v3_s{seed}_step{R2.STEPS_PER_EPOCH * K}", R2.checkpoint(key, "v3", K),
                            R2.DEV_SEQ, ["legacy_m"], "full", R2.OUT / key / "dev" / f"v3_epoch{K}",
                            R2.DEV_INPUTS))
    st.append(A4.train_stage(key, "mixunit", K))
    st += R2.test_stages(key, K)
    return st


def write_schedule_record(key: str, seed: int):
    """The analyzer reads <key>/stopping.json; here it records the FIXED schedule, not a decision."""
    path = seed_root(seed) / key / "stopping.json"
    if not path.is_file():
        D.atomic_write_json(path, {
            "backbone": key, "k_star": K, "degenerate_at_cap": False, "seed": seed,
            "fixed_schedule": True, "dev_rule_rerun": False,
            "source": "seed-42 round-4 selection (results/courtdyn/backbone4_a100/<key>/stopping.json)",
            "record": "results/courtdyn/backbone_seeds3/PREREGISTRATION.json",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat()})


def check_records(key: str):
    if not PREREG.is_file():
        raise SystemExit(f"no {PREREG}: record the rules first")
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    if hashlib.sha256(p["rules"].encode("utf-8")).hexdigest() != PREREG_SHA or p["rules_sha256"] != PREREG_SHA:
        raise SystemExit(f"{PREREG} does not match the recorded rules (sha256 {PREREG_SHA[:12]}...)")
    import run_backbone_a100 as A4
    named = json.loads(ROUND4_PREREG.read_text(encoding="utf-8")).get("revisions", {})
    m = A4.MODELS[key]
    if list(named.get(key, []))[:2] != [m["repo"], m["revision"]]:
        raise SystemExit(f"{key} at {m['revision'][:8]} is not the revision named in {ROUND4_PREREG}")


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
    check_records(key)
    stages = seed_stages(key, seed)
    runner = Seeds3Runner(f"seeds3-{args.part}", state_path(key, seed), seed_root(seed) / key / "logs")
    if args.run:
        write_schedule_record(key, seed)
    rc = runner.run(stages, dry_run=args.dry_run)
    if not args.dry_run:
        runner.log(f"{args.part}: complete" + (" (with failures)" if rc else ""))
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
