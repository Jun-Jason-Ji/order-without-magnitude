#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run this first on the A100 box. It answers "will the sweeps work here?"

Checks, in order of how expensive they are to discover late:

  paths       every directory the sweeps read or write resolves
  inputs      the prepared control pools match their manifest hashes
  frames      every frame the evaluation and training pools reference exists
  adapters    the shipped seed-42 adapters are intact
  backbone    the replication backbone is present or pullable
  runtime     torch sees a CUDA device with enough memory, and the library
              versions are recorded
  smoke       (--smoke) loads the backbone and answers four real items, so a
              broken processor policy or chat template fails in two minutes
              rather than six hours into a sweep

Exit code 0 means the sweeps can start. Anything else names the blocker.

Usage
  python -m a100.preflight
  python -m a100.preflight --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C            # noqa: E402

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append(dict(status=status, check=name, detail=str(detail)))
        print(f"[{status:4s}] {name}" + (f"  {detail}" if detail else ""), flush=True)
        return status != FAIL

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == FAIL]


def check_paths(report):
    ok = True
    for name, path, must_exist in (
            ("repository root", C.ROOT, True),
            ("frame root", C.DATA, True),
            ("prepared inputs", C.INPUTS, True),
            ("output root", C.OUT, False),
            ("model root", C.MODEL_ROOT, False)):
        if path.exists():
            report.add(OK, name, path)
        elif must_exist:
            ok = report.add(FAIL, name, f"{path} does not exist") and ok
        else:
            report.add(WARN, name, f"{path} will be created")
    return ok


def check_inputs(report):
    from a100.eval_controls import sha256
    manifest_path = C.INPUTS / "manifest.json"
    if not manifest_path.is_file():
        return report.add(FAIL, "manifest", f"{manifest_path} missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    bad = []
    for cell in manifest["cells"]:
        path = C.INPUTS / cell["file"].replace("\\", "/")
        if not path.is_file():
            bad.append(f"missing {cell['file']}")
        elif sha256(path) != cell["sha256"]:
            bad.append(f"hash mismatch {cell['file']}")
    if bad:
        return report.add(FAIL, "prepared control pools", "; ".join(bad[:4]))
    arithmetic = C.INPUTS / "arithmetic.json"
    if not arithmetic.is_file():
        return report.add(FAIL, "arithmetic probes", f"{arithmetic} missing")
    return report.add(OK, "prepared control pools",
                      f"{len(manifest['cells'])} cells verified, "
                      f"{len(json.loads(arithmetic.read_text(encoding='utf8')))} probes")


def check_frames(report):
    from a100.eval_controls import load_cells
    ok = True
    for seq in C.EVAL_SEQS:
        try:
            _, cells = load_cells(seq, C.ARMS_THREE)
        except SystemExit as error:
            ok = report.add(FAIL, f"eval frames {seq}", error) and ok
            continue
        root = C.image_root(seq)
        wanted = {i for _, rows in cells for row in rows for i in row["image_ids"]}
        missing = [i for i in sorted(wanted) if not (root / i).is_file()]
        if missing:
            ok = report.add(FAIL, f"eval frames {seq}",
                            f"{len(missing)} missing under {root}, "
                            f"e.g. {missing[0]}") and ok
        else:
            report.add(OK, f"eval frames {seq}", f"{len(wanted)} frames under {root}")

    for pool, relative in C.TRAIN_POOLS.items():
        path = C.ROOT / relative
        if not path.is_file():
            report.add(WARN, f"training pool {pool}", f"{path} missing "
                                                      "(only needed to train)")
            continue
        rows = json.loads(path.read_text(encoding="utf8"))
        wanted = {i for row in rows for i in (row.get("image_ids") or [row["image_id"]])}
        missing = [i for i in sorted(wanted) if not (C.DATA / i).is_file()]
        if missing:
            ok = report.add(FAIL, f"training frames {pool}",
                            f"{len(missing)}/{len(wanted)} missing under "
                            f"{C.DATA}, e.g. {missing[0]}") and ok
        else:
            report.add(OK, f"training frames {pool}",
                       f"{len(rows)} items, {len(wanted)} frames")
    return ok


def check_adapters(report):
    ok = True
    for pool in C.TRAIN_POOLS:
        directory = C.adapter_dir(pool, 42)
        weights = directory / "adapter_model.safetensors"
        if not weights.is_file():
            if os.environ.get("A100_ADAPTERS_OPTIONAL") == "1":
                # The split upload bundle (2026-09-28) ships no seed-42 adapters: backbone 4
                # and T36 do not use them; only exp_seeds and `preflight --smoke` would.
                report.add(WARN, f"adapter {pool} seed 42",
                           "not shipped; not needed for this run (backbone 4, T36)")
                continue
            ok = report.add(FAIL, f"adapter {pool} seed 42",
                            f"{weights} missing; the bundle should contain it") and ok
            continue
        receipt = directory / "training_completion.json"
        detail = f"{weights.stat().st_size / 2**20:.0f} MiB"
        if receipt.is_file():
            data = json.loads(receipt.read_text(encoding="utf8"))
            detail += (f", seed {data.get('seed')}, "
                       f"{data.get('global_step')} steps, "
                       f"{data.get('selected_samples')} items")
            if data.get("global_step") != C.EXPECTED_STEPS[pool]:
                report.add(WARN, f"adapter {pool} seed 42",
                           f"step count {data.get('global_step')} != expected "
                           f"{C.EXPECTED_STEPS[pool]}")
        report.add(OK, f"adapter {pool} seed 42", detail)
    return ok


def check_backbone(report):
    resolved = C.backbone_path("qwen35-4b")
    if resolved and (Path(resolved) / "config.json").is_file():
        return report.add(OK, "replication backbone", resolved)
    return report.add(
        FAIL, "replication backbone",
        "not found locally. Set QWEN35_4B=/path/to/Qwen3.5-4B (or place it "
        "under $COURTDYN_MODELS). Multi-seed replication must use the same "
        "weights every published cell used.")


def check_runtime(report):
    versions = {}
    try:
        import importlib.metadata as md
        for package in ("torch", "transformers", "peft", "bitsandbytes",
                        "accelerate", "safetensors"):
            try:
                versions[package] = md.version(package)
            except Exception:                              # noqa: BLE001
                versions[package] = None
    except Exception as error:                             # noqa: BLE001
        return report.add(FAIL, "python packages", error)
    missing = [k for k, v in versions.items() if v is None]
    if missing:
        report.add(FAIL, "python packages", "not installed: " + ", ".join(missing))
    else:
        report.add(OK, "python packages",
                   ", ".join(f"{k} {v}" for k, v in versions.items()))
    try:
        import torch
    except Exception as error:                             # noqa: BLE001
        return report.add(FAIL, "torch import", error)
    if not torch.cuda.is_available():
        return report.add(FAIL, "cuda", "torch.cuda.is_available() is False")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2**30
    report.add(OK, "cuda device", f"{props.name}, {total:.0f} GiB, "
                                  f"capability {props.major}.{props.minor}")
    if total < 20:
        report.add(WARN, "cuda memory",
                   f"{total:.0f} GiB; the bf16 8B arms of the backbone sweep "
                   "expect roughly 40 GiB")
    return not missing


def smoke(report):
    """Four real items through the replication path."""
    from a100.eval_controls import build_infer, load_cells, parse
    seq = C.EVAL_SEQS[0]
    _, cells = load_cells(seq, ["m_height"])
    cell, rows = cells[0]
    backbone = C.backbone_path("qwen35-4b")
    adapter = C.adapter_dir("native", 42)
    try:
        infer, receipt = build_infer(backbone, "nf4", adapter=str(adapter))
    except Exception as error:                             # noqa: BLE001
        return report.add(FAIL, "smoke: load", repr(error))
    report.add(OK, "smoke: load", json.dumps(
        {k: receipt[k] for k in ("model_family", "precision", "loader")}))
    root = Path(cell["image_root"])
    answers = []
    for item in rows[:4]:
        paths = [str(root / i) for i in item["image_ids"]]
        try:
            raw = infer(paths, item["question"])
        except Exception as error:                         # noqa: BLE001
            return report.add(FAIL, "smoke: generate", repr(error))
        answers.append((raw.strip(), parse(raw.strip()), item["answer"]))
    parsed = sum(1 for _, p, _ in answers if p is not None)
    detail = "; ".join(f"got {a!r} want {g}" for a, _, g in answers)
    if parsed == 0:
        return report.add(FAIL, "smoke: parse",
                          "no answer parsed as a bare number. " + detail)
    return report.add(OK if parsed == len(answers) else WARN, "smoke: parse",
                      f"{parsed}/{len(answers)} parsed. " + detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="also load the backbone and answer four real items")
    ap.add_argument("--json", help="write the machine-readable report here")
    args = ap.parse_args()

    report = Report()
    print(json.dumps(C.describe(), indent=2))
    print()
    check_paths(report)
    check_inputs(report)
    check_frames(report)
    check_adapters(report)
    check_backbone(report)
    check_runtime(report)
    if args.smoke and not report.failed:
        smoke(report)
    elif args.smoke:
        report.add(WARN, "smoke", "skipped: fix the failures above first")

    print()
    if report.failed:
        print("BLOCKED: " + "; ".join(r["check"] for r in report.failed))
    else:
        print("ready: python -m a100.exp_seeds --dry-run")
    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(paths=C.describe(), checks=report.rows),
                       indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
