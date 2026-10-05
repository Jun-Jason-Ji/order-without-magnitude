#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage unit answering: read metres from the image, convert in text (D5.4).

Why this experiment
-------------------
The 392-item text probe (tools/run_unit_probe.py) found that every checkpoint
multiplies a *stated* metre value by K = 27.498 correctly (median ratio 27.50),
while with the image in view the same weights return pixel/metre ratios of
1.45-1.83.  The conversion is intact; it is simply not applied to the number the
model reads from the image.

That predicts a remedy with no training at all: make the model state the metre
quantity first, then ask it -- in a second, text-only turn -- to convert its own
stated value.  The ceiling of that pipeline is computable from the frozen
predictions without a GPU (exact conversion of each checkpoint's own metre
answers), and it is about three times the direct-prompt score:

    pixel target   direct 21.6-23.4 T-MRA   ->   exact conversion 63.9-77.9
    cm target      direct 20.2-21.6 T-MRA   ->   exact conversion 64.1-77.9

This script measures how much of that ceiling the model's *own* conversion
recovers.  Stage 1 is not rerun: it reuses each checkpoint's published metre-arm
predictions, whose raw answers are passed on verbatim, so stage 2 converts
exactly the number the model wrote.

Preregistered predictions (written 2026-09-25, before any stage-2 inference)
---------------------------------------------------------------------------
From the text probe, not from any stage-2 output:

  P1  pixels, every checkpoint: median converted/metre ratio within 5% of
      27.498, and two-stage T-MRA within 5 points of the exact-conversion
      ceiling for each family.  (The probe shows K applied correctly.)
  P2  centimetres, the three adapters: median converted/metre ratio far from
      100 -- the probe puts it at 10 -- and two-stage T-MRA at least 20 points
      below the ceiling for each family.  (The probe shows a tenfold error.)
  P3  centimetres, the unadapted backbone: median ratio within 5% of 100 and
      T-MRA within 5 points of its own ceiling.  (The probe shows x100 correct.)

Each is scored as stated; none is adjusted after the fact.  A failed prediction
is reported as failed.

Scoring
-------
Stage-2 answers are scored against the image arm's own references (rendered
pixels for px, floor motion x 100 for cm) with the same scalar_score and strict
parse as every visual cell, so the three numbers compared -- direct prompt,
exact conversion, model's own conversion -- share one scale.

Usage
  python tools/run_two_stage_units.py --arm native
  python tools/run_two_stage_units.py --all
  python tools/run_two_stage_units.py --ceiling-only     # CPU: direct vs exact only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEQ = "Q2_top_480-510"
K = 27.498
PUBLISHED = ROOT / "results" / "courtdyn" / "revision_execution_20260912"
OUT = ROOT / "results" / "courtdyn" / "two_stage_units"

# arm -> (directory holding m_height/px_height/cm_height jsonl, adapter or None)
ARMS = {
    "base": (ROOT / "results" / "a100" / f"base_nf4_{SEQ}_full", None),
    "native": (PUBLISHED / f"native_original_{SEQ}_full", "models/courtdyn-native-sft"),
    "v1": (PUBLISHED / f"eventholdout_v1_s42_{SEQ}_full",
           "models/courtdyn-revision-v1-eventholdout-s42"),
    "v3": (PUBLISHED / f"eventholdout_v3_s42_{SEQ}_full",
           "models/courtdyn-revision-v3-eventholdout-s42"),
}

TARGETS = {
    "px": dict(arm="px_height", factor=K,
               sentence=f"One meter equals {K:g} pixels in this image.",
               path_unit="pixels", speed_unit="pixels per second"),
    "cm": dict(arm="cm_height", factor=100.0,
               sentence="One meter equals 100 centimeters.",
               path_unit="centimeters", speed_unit="centimeters per second"),
}
FAMILIES = ["dynamics_speed_player", "dynamics_path_player"]


def item_key(row):
    return (row["category"], row["track"], tuple(row["window"]))


def load_cell(directory: Path, arm: str):
    path = directory / f"{arm}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {item_key(r): r for r in rows}


def stage2_question(row_m, target):
    """The second turn: the checkpoint's own metre answer, verbatim, plus the factor."""
    value = (row_m["raw_answer"] or "").strip()
    t = TARGETS[target]
    if row_m["category"] == "dynamics_path_player":
        return (f"The player marked with the red box has a path length of {value} "
                f"meters over the clip. {t['sentence']} Express that path length "
                f"in {t['path_unit']}. Output only the number, one decimal place.")
    return (f"The player marked with the red box moves at an average speed of "
            f"{value} meters per second. {t['sentence']} Express that speed in "
            f"{t['speed_unit']}. Output only the number, one decimal place.")


def score_block(pairs):
    """pairs: [(prediction or None, reference row)] -> T-MRA over all items, rho."""
    from a100.eval_controls import scalar_score, spearman
    tmra = sum(scalar_score(p, r["reference_value"], r["score_tolerance"],
                            r["score_floor"]) if p is not None else 0.0
               for p, r in pairs) / len(pairs)
    parsed = [(p, r) for p, r in pairs if p is not None]
    rho = spearman([p for p, _ in parsed], [r["reference_value"] for _, r in parsed]) \
        if len(parsed) > 2 else None
    return {"tmra": tmra, "rho": rho, "n": len(pairs), "n_parsed": len(parsed)}


def ceiling(arm):
    directory, _ = ARMS[arm]
    m = load_cell(directory, "m_height")
    out = {}
    for target, t in TARGETS.items():
        ref = load_cell(directory, t["arm"])
        if set(ref) != set(m):
            raise SystemExit(f"{arm}: item keys differ between m_height and {t['arm']}")
        for family in FAMILIES:
            keys = sorted(k for k in m if k[0] == family)
            direct = [(ref[k]["prediction"], ref[k]) for k in keys]
            exact = [(m[k]["prediction"] * t["factor"]
                      if m[k]["prediction"] is not None else None, ref[k]) for k in keys]
            out[f"{target}/{family}"] = {"direct": score_block(direct),
                                         "exact_conversion": score_block(exact)}
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_complete(arm) -> bool:
    """A stage-1 cell is usable only once its runner has written summary.json.

    Checking for m_height.jsonl alone is not enough: the runner writes the metre
    arm first, so a cell still in progress has it while its px and cm arms --
    the references stage 2 is scored against -- do not exist yet.
    """
    directory = ARMS[arm][0]
    return (directory / "summary.json").is_file() and all(
        (directory / f"{a}.jsonl").is_file() for a in ("m_height", "px_height", "cm_height"))


def run_arm(arm, base_model):
    from a100.eval_controls import build_infer, parse
    directory, adapter = ARMS[arm]
    if not cell_complete(arm):
        raise SystemExit(f"{arm}: stage-1 cell in {directory} is not complete")
    m = load_cell(directory, "m_height")
    out = OUT / arm
    out.mkdir(parents=True, exist_ok=True)
    results = out / "stage2.jsonl"
    done = {}
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            done[(row["target"], tuple(row["key"][:2]) + (tuple(row["key"][2]),))] = row
    todo = [(target, k) for target in TARGETS for k in sorted(m)
            if (target, k) not in done]
    print(f"[{arm}] {len(done)} done, {len(todo)} to run", flush=True)
    if todo:
        adapter_path = str((ROOT / adapter).resolve()) if adapter else None
        infer, receipt = build_infer(base_model, "nf4", adapter=adapter_path)
        config = {"protocol": "courtdyn-two-stage-units-v1", "arm": arm, "seq": SEQ,
                  "stage1_dir": str(directory), "adapter": adapter_path,
                  "model_path": str(base_model), "precision": "nf4",
                  "decoding": "greedy", "text_only_stage2": True, "K": K,
                  "loader": receipt.get("loader"),
                  "stage1_sha256": sha256_file(directory / "m_height.jsonl"),
                  "code_sha256": sha256_file(Path(__file__)),
                  "started_at": datetime.now(timezone.utc).isoformat()}
        (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n",
                                             encoding="utf-8")
        started = time.time()
        with open(results, "a", encoding="utf-8") as handle:
            for index, (target, k) in enumerate(todo, 1):
                row_m = m[k]
                if row_m["prediction"] is None:
                    raw, pred = None, None          # nothing stated, nothing to convert
                else:
                    question = stage2_question(row_m, target)
                    raw = infer([], question)
                    pred = parse((raw or "").strip())
                row = {"target": target, "key": [k[0], k[1], list(k[2])],
                       "stage1_raw": row_m["raw_answer"], "stage1": row_m["prediction"],
                       "stage2_raw": raw, "stage2": pred}
                done[(target, k)] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 50 == 0 or index == len(todo):
                    print(f"[{arm}] {index}/{len(todo)} "
                          f"{(time.time() - started) / index:.2f}s/item", flush=True)
    return summarize(arm, m, done)


def summarize(arm, m, done):
    directory, _ = ARMS[arm]
    ceil = ceiling(arm)
    summary = {"arm": arm, "cells": {}}
    for target, t in TARGETS.items():
        ref = load_cell(directory, t["arm"])
        for family in FAMILIES:
            keys = sorted(k for k in m if k[0] == family)
            two = [(done[(target, k)]["stage2"], ref[k]) for k in keys]
            ratios = [done[(target, k)]["stage2"] / m[k]["prediction"] for k in keys
                      if done[(target, k)]["stage2"] is not None
                      and m[k]["prediction"] not in (None, 0)]
            cell = dict(ceil[f"{target}/{family}"])
            cell["two_stage"] = score_block(two)
            cell["two_stage_median_ratio"] = st.median(ratios) if ratios else None
            cell["required_ratio"] = t["factor"]
            summary["cells"][f"{target}/{family}"] = cell
    (OUT / arm / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                            encoding="utf-8")
    return summary


def print_summary(summary):
    print(f"== {summary['arm']}")
    for name, c in summary["cells"].items():
        ts = c.get("two_stage")
        ts_text = (f"two-stage T {ts['tmra']:5.1f}  ratio "
                   f"{c['two_stage_median_ratio']:.2f}/{c['required_ratio']:g}"
                   if ts else "")
        print(f"   {name:34s} direct T {c['direct']['tmra']:5.1f} | exact T "
              f"{c['exact_conversion']['tmra']:5.1f} | {ts_text}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ceiling-only", action="store_true")
    args = ap.parse_args()

    if args.ceiling_only:
        for arm in ARMS:
            if cell_complete(arm):
                print_summary({"arm": arm, "cells": ceiling(arm)})
            else:
                print(f"== {arm}: stage-1 cell not complete yet, skipped")
        return 0
    if args.all:
        # One process per checkpoint: a second 4-bit model in the same process
        # overflows 8 GB (see run_unit_probe.py).
        failed = []
        for arm in ARMS:
            if subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               "--arm", arm]).returncode != 0:
                failed.append(arm)
        print(f"FAILED: {failed}" if failed else "all arms complete")
        return 1 if failed else 0
    if not args.arm:
        ap.error("pass --arm, --all or --ceiling-only")
    base_model = os.environ.get("QWEN35_4B")
    if not base_model:
        raise SystemExit("set QWEN35_4B to the reproduction backbone")
    print_summary(run_arm(args.arm, base_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
