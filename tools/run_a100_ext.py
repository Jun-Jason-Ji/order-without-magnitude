#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A100 extension run: seed-43 replication (Gemma3-12B, Pixtral-12B) and RTC + H6 on both.

Rules: results/courtdyn/backbone4_ext/PREREGISTRATION.json (written before any of this ran).
Reuses the round-4 machinery unchanged (tools/run_backbone_a100.py configure/train_stage/launcher,
the round-2 stage builders, train/train_durable.py, tools/family_launch.py); only this file and
a100/ext_runner.py are new.  Each --part runs in its own process (the round-2 builders keep the
seed and output directory as module globals).

  part seed43_<key>  (fixed schedule, step 140 = seed-42 k* 2; the dev rule is not re-run):
    train v3 s43 to step 140 -> dev answers at step 140 (descriptive) -> train mixed-unit s43 to 140
    -> explicit / first-frame x4 / in-template test cells + text probe for both adapters
    Outputs results/courtdyn/backbone4_ext/<key>/ (analysed by analyze_backbone_replication --round 5;
    unadapted-backbone cells are reused from backbone4_a100).
  part rtc_<key>  (seed 42, step 140):
    train the RTC pool -> rtc {m,cm,U1,U2,pxK} + rtc_m first-frame x4, both clips
    -> comparators (round-4 v3 / mixed-unit s42, checkpoint-140): plain {U1,U2,pxK} and
       rtc {m,cm,U1,U2,pxK}, both clips -> 392-item text probe for the RTC adapter.
    Outputs results/courtdyn/backbone4_ext/rtc/ (analysed by tools/analyze_rtc_a100.py).

  python tools/run_a100_ext.py --part seed43_gemma3_12b --dry-run | --run
Exit codes: 0 done, 1 some stages failed.
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

from a100 import durable as D                            # noqa: E402
from a100.ext_runner import ExtRunner                    # noqa: E402
from a100.runner import Stage                            # noqa: E402
import run_backbone_a100 as A4                           # noqa: E402  (read-only reuse)
import run_backbone_replication as R1                    # noqa: E402
import run_backbone_round2 as R2                         # noqa: E402

CD = ROOT / "results" / "courtdyn"
OUT = CD / "backbone4_ext"
PREREG = OUT / "PREREGISTRATION.json"
K = 2                                                    # seed-42 k* of both backbones
RTC_DIR = CD / "rtc"
RTC_POOL = RTC_DIR / "rtc_v3_pool.json"
RTC_INPUTS = RTC_DIR / "inputs"
SEQS = ["Q2_top_480-510", "Q1_top_0-30"]
RTC_ARMS = ["rtc_m", "rtc_cm", "rtc_U1", "rtc_U2", "rtc_pxK"]
PLAIN_NEW = ["plain_U1", "plain_U2", "plain_pxK"]
KEYS = ["gemma3_12b", "pixtral_12b"]
PARTS = [f"seed43_{k}" for k in KEYS] + [f"rtc_{k}" for k in ("pixtral_12b", "gemma3_12b")]


def configure(key, seed):
    A4.configure(key)                  # model path, launcher, durable train stage
    R2.OUT = OUT
    R2.SEED = seed
    R1.POOLS["rtc"] = RTC_POOL


# ------------------------------------------------------------------ group 1: seed 43
def seed_stages(key):
    configure(key, 43)
    st = [A4.train_stage(key, "v3", K)]
    st.append(R2.eval_stage(key, f"v3_s43_step{R2.STEPS_PER_EPOCH * K}", R2.checkpoint(key, "v3", K),
                            R2.DEV_SEQ, ["legacy_m"], "full", OUT / key / "dev" / f"v3_epoch{K}",
                            R2.DEV_INPUTS))
    st.append(A4.train_stage(key, "mixunit", K))
    st += R2.test_stages(key, K)
    return st


def write_schedule_record(key):
    """The analyzer reads <key>/stopping.json; here it records the FIXED schedule, not a decision."""
    path = OUT / key / "stopping.json"
    if not path.is_file():
        D.atomic_write_json(path, {
            "backbone": key, "k_star": K, "degenerate_at_cap": False, "seed": 43,
            "fixed_schedule": True, "dev_rule_rerun": False,
            "source": "seed-42 round-4 selection (results/courtdyn/backbone4_a100/<key>/stopping.json)",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat()})


# ------------------------------------------------------------------ group 2: RTC
def cell_output(label, seq, mode, arms) -> Path:
    """Naming of tools/analyze_rtc.py / analyze_rtc_replication.py cell directories."""
    suffix = "-".join(a.split("_", 1)[1] for a in arms)
    return OUT / "rtc" / "eval" / f"{label}_{seq}_{mode}_{suffix}_{arms[0].split('_')[0]}"


def rtc_eval(key, label, adapter, seq, arms, mode="full"):
    output = cell_output(label, seq, mode, arms)
    cmd = A4.launched_cmd(ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                          "--model", R1.BACKBONES[key], "--label", label, "--adapter", adapter,
                          "--seq", seq, "--arms", *arms, "--frame-mode", mode, "--output", output,
                          "--inputs", RTC_INPUTS)
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log",
                 needs=[Path(adapter) / "adapter_model.safetensors", RTC_INPUTS / "manifest.json"],
                 note=f"{len(arms) * 280} items")


def rtc_stages(key):
    configure(key, 42)
    st = [A4.train_stage(key, "rtc", K)]
    rtc = R2.checkpoint(key, "rtc", K)
    for seq in SEQS:
        st.append(rtc_eval(key, f"rtc_{key}", rtc, seq, RTC_ARMS))
        st.append(rtc_eval(key, f"rtc_{key}", rtc, seq, ["rtc_m"], "static4"))
    comps = {f"{key}_v3": R2.checkpoint(key, "v3", K), f"{key}_mix": R2.checkpoint(key, "mixunit", K)}
    for label, adapter in comps.items():
        for seq in SEQS:
            st.append(rtc_eval(key, label, adapter, seq, PLAIN_NEW))
            st.append(rtc_eval(key, label, adapter, seq, RTC_ARMS))
    runs = OUT / "rtc" / "unit_probe"
    st.append(Stage(
        name=f"probe_rtc_{key}",
        cmd=A4.launched_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", f"rtc_{key}",
                            "--adapter", str(rtc), "--model-path", R1.BACKBONES[key], "--out", runs),
        done_file=runs / f"rtc_{key}" / "summary.json", log_name=f"probe_rtc_{key}.log",
        needs=[rtc / "adapter_model.safetensors"], note="392 text items"))
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=PARTS)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not PREREG.is_file():
        raise SystemExit("no results/courtdyn/backbone4_ext/PREREGISTRATION.json: record the rules first")
    group, key = args.part.split("_", 1)
    named = json.loads(PREREG.read_text(encoding="utf-8")).get("revisions", {})
    m = A4.MODELS[key]
    if list(named.get(key, []))[:2] != [m["repo"], m["revision"]]:
        raise SystemExit(f"{key} at {m['revision'][:8]} is not named in {PREREG}")
    if group == "seed43":
        stages = seed_stages(key)
        runner = ExtRunner(f"ext-{args.part}", OUT / key / "state.json", OUT / key / "logs")
        if args.run:
            write_schedule_record(key)
    else:
        stages = rtc_stages(key)
        runner = ExtRunner(f"ext-{args.part}", OUT / "rtc" / f"{key}_state.json", OUT / "rtc" / "logs")
    rc = runner.run(stages, dry_run=args.dry_run)
    if not args.dry_run:
        runner.log(f"{args.part}: complete" + (" (with failures)" if rc else ""))
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
