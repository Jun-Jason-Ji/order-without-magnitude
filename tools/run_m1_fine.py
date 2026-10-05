#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-fine: the M1 seed-42 mixed-unit pool with all 280 centimetre SPEED targets at 1 cm/s resolution.

Preregistration: results/courtdyn/m1_fine/PREREGISTRATION.json (written before this pool existed).
Everything else is the published M1 pool (results/courtdyn/m1_mixunit/mixunit_v3_pool.json): same
items, images, order, units, path targets, metre targets and training arguments; asserted below.

  python tools/run_m1_fine.py --build          # write the pool (refuses to overwrite a different one)
  python tools/run_m1_fine.py --dry-run | --run
  python tools/run_m1_fine.py --analyse        # decide the preregistered rule
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import config as C                            # noqa: E402
from a100.runner import Runner, Stage, python_cmd       # noqa: E402
import run_m1_mixunit as M1                             # noqa: E402

OUT = ROOT / "results" / "courtdyn" / "m1_fine"
SRC = ROOT / "results" / "courtdyn" / "m1_mixunit" / "mixunit_v3_pool.json"
POOL = OUT / "mixunit_fine_v3_pool.json"
ADAPTER = ROOT / "models" / "courtdyn-mixunit-fine-v3-s42"
SEQS = ["Q2_top_480-510", "Q1_top_0-30"]


def build_pool():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out, changed = [], 0
    for r in src:
        r = copy.deepcopy(r)
        if r["category"] == "dynamics_speed_player" and r["meta"].get("m1_unit") == "cm":
            new = f"{round(r['meta']['speed_mps'] * 100):.1f}"
            r["meta"]["m1_fine_previous_answer"] = r["answer"]
            r["answer"] = new
            changed += 1
        out.append(r)
    assert changed == 280, changed   # prereg text says 140: erratum, see ERRATA.md
    assert [(r["question"], r["image_ids"], r["category"]) for r in out] == \
           [(r["question"], r["image_ids"], r["category"]) for r in src]
    diff = sum(a["answer"] != b["answer"] for a, b in zip(out, src))
    assert all(a["answer"] == b["answer"] for a, b in zip(out, src)
               if not (a["category"] == "dynamics_speed_player" and a["meta"].get("m1_unit") == "cm"))
    return out, changed, diff


def train_stage(model):
    a = C.TRAIN_ARGS
    cmd = python_cmd(ROOT / "train" / "train_qlora.py",
                     "--model_path", model, "--qa_json", POOL, "--img_root", C.DATA,
                     "--output_dir", ADAPTER, "--num_samples", a["num_samples"],
                     "--allow_base_init", "--budget_slice", a["budget_slice"],
                     "--num_train_epochs", a["num_train_epochs"], "--max_steps", a["max_steps"],
                     "--seed", 42, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
                     "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
                     "--save_steps", a["save_steps"])

    def verify():
        receipt = ADAPTER / "training_completion.json"
        if not receipt.is_file():
            return False
        d = json.loads(receipt.read_text(encoding="utf-8"))
        return bool(d.get("completed") and d.get("seed") == 42 and d.get("selected_samples") == 1120
                    and d.get("global_step") == 70 and (ADAPTER / "adapter_model.safetensors").is_file())

    return Stage(name="train_mixunit_fine_v3_s42", cmd=cmd, verify=verify,
                 log_name="train_mixunit_fine_v3_s42.log", needs=[POOL], stall_minutes=90,
                 max_restarts=2, note="1120 items, 70 steps, seed 42")


def eval_stage(seq):
    output = OUT / "eval" / f"mixfine_{seq}_full"
    cmd = python_cmd(ROOT / "tools" / "run_m1_mixunit.py", "eval-cell", "--label", "mixfine",
                     "--adapter", ADAPTER, "--seq", seq, "--arms", "m_height", "cm_height",
                     "--frame-mode", "full", "--output", output)
    return Stage(name=f"eval_{output.name}", cmd=cmd, done_file=output / "summary.json",
                 log_name=f"eval_{output.name}.log", needs=[ADAPTER / "adapter_model.safetensors"],
                 note="560 items")


def analyse():
    import numpy as np
    from scipy.stats import spearmanr
    from a100.analyze import paired_ratio, read_cell
    pre = json.loads((OUT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pre["rules"].encode("utf-8")).hexdigest() == pre["rules_sha256"]
    rep = {"rules_sha256": pre["rules_sha256"], "cells": {}}
    for seq in SEQS:
        cell = read_cell(OUT / "eval" / f"mixfine_{seq}_full")
        old = read_cell(ROOT / "results" / "courtdyn" / "m1_mixunit" / "eval" / f"mixunit_{seq}_full")
        for fam, short in (("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")):
            res = {}
            for tag, c in (("fine", cell), ("published_m1_s42", old)):
                m = [r for r in c["arms"]["m_height"] if r["category"] == fam]
                cm = [r for r in c["arms"]["cm_height"] if r["category"] == fam]
                ratio = paired_ratio(cm, m)
                v = [r for r in cm if r["prediction"] is not None]
                preds = [r["prediction"] for r in v]
                rho = (spearmanr(preds, [r["reference_value"] for r in v])[0]
                       if len(set(preds)) > 1 else None)
                res[tag] = {"R": ratio["median"] if ratio else None,
                            "distinct_cm": len(set(preds)), "rho_cm": rho,
                            "cm_median": float(np.median(preds)) if preds else None,
                            "metre_median": float(np.median([r["prediction"] for r in m
                                                             if r["prediction"] is not None]))}
            f = res["fine"]
            res["converts"] = bool(f["distinct_cm"] > 2 and f["R"] is not None and f["R"] >= 10
                                   and f["rho_cm"] is not None and f["rho_cm"] >= 0.30)
            rep["cells"][f"{seq}/{short}"] = res
    speed = [rep["cells"][f"{s}/speed"]["converts"] for s in SEQS]
    path_ok = all((rep["cells"][f"{s}/path"]["fine"]["R"] or 0) >= 10 for s in SEQS)
    rep["verdict"] = ("confounded (path no longer converts)" if not path_ok else
                      "SUPPORTED" if all(speed) else "NOT SUPPORTED" if not any(speed) else "intermediate")
    (OUT / "verdict.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if not (OUT / "PREREGISTRATION.json").is_file():
        raise SystemExit("no PREREGISTRATION.json")
    if a.analyse:
        return analyse()
    if a.build:
        pool, changed, diff = build_pool()
        text = json.dumps(pool, ensure_ascii=False, indent=1) + "\n"
        if POOL.is_file() and POOL.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{POOL} exists with different content; refusing to overwrite")
        POOL.write_text(text, encoding="utf-8")
        print(f"pool written: {changed} cm-speed targets re-expressed at 1 cm/s; "
              f"{diff} answers differ from M1 (the rest equal to the 10-cm/s rounding)")
        return 0
    stages = [train_stage(M1.backbone())] + [eval_stage(s) for s in SEQS]
    runner = Runner("m1-fine", OUT / "state.json", OUT / "logs")
    return runner.run(stages, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
