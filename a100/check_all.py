#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automatic pre-run checks for the A100 run.  Fails fast, prints a summary table.

  gpu         CUDA device visible to torch, bf16 matmul, a bitsandbytes NF4 layer forward
  disk        free space under $A100_WORK
  durability  atomic write + fsync + rename works on this filesystem
  manifest    every bundle file matches MANIFEST.json (sha256)
  inputs      every prepared cell (test, dev, in-template) matches its manifest hash, every
              frame they and the two training pools reference exists, the probe file exists
  selftest    python -m a100.selftest (CPU pipeline self-test, protocol-equality assertions)
  preflight   python -m a100.preflight (paths, pools, adapters, backbone, runtime)
  smoke:<m>   tools/run_backbone_a100.py --smoke per backbone (GPU; disclosed in the
              preregistration): 2 training steps with an interruption and resume, 4 bare
              dev answers, 1 probe answer with the scratch adapter.  Skipped when it already
              passed for the same code (stamp file).

Usage: python -m a100.check_all [--no-gpu] [--backbones gemma3_4b ...] [--skip-manifest]
Exit 0 = the run may start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                 # noqa: E402
from a100 import durable as D                # noqa: E402

ROWS = []
from a100 import backbone4 as B4             # noqa: E402


def work() -> Path:
    return Path(os.environ.get("A100_WORK") or C.ROOT)


def add(status, name, detail=""):
    ROWS.append((status, name, str(detail)))
    print(f"[{status:4s}] {name}  {detail}", flush=True)
    return status != "FAIL"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return add("FAIL", "gpu", "torch.cuda.is_available() is False "
                       f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})")
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        float((a @ a).float().sum())
        import bitsandbytes as bnb
        layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
        out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
        assert out.shape == (2, 64)
        ok = total > 30
        return add("ok" if ok else "warn", "gpu",
                   f"{name}, {total:.0f} GiB, torch {torch.__version__} cuda {torch.version.cuda}, "
                   f"bf16 + NF4 ok" + ("" if ok else "  (less than 30 GiB: gemma3_12b may not fit)"))
    except Exception as error:                               # noqa: BLE001
        return add("FAIL", "gpu", repr(error)[:300])


def check_disk():
    free = shutil.disk_usage(work()).free / 2**30
    have = C.MODEL_ROOT / "hf" / "hub"
    status = "ok" if free > 120 else ("warn" if free > 60 else "FAIL")
    return add(status, "disk", f"{free:.0f} GiB free under {work()} (models ~110 GiB, "
                               f"adapters+results ~15 GiB; already cached: {have.is_dir()})")


def check_durability():
    probe = work() / ".durability_probe.json"
    try:
        D.atomic_write_json(probe, {"t": time.time()})
        json.loads(probe.read_text(encoding="utf-8"))
        probe.unlink()
        return add("ok", "durability", "atomic write + fsync + rename ok")
    except Exception as error:                               # noqa: BLE001
        return add("FAIL", "durability", repr(error))


def check_manifest():
    path = C.ROOT / "MANIFEST.json"
    if not path.is_file():
        return add("warn", "manifest", "no MANIFEST.json (not running from a bundle)")
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    bad = [rel for rel, meta in files.items()
           if not (C.ROOT / rel).is_file() or (C.ROOT / rel).stat().st_size != meta["bytes"]
           or sha256(C.ROOT / rel) != meta["sha256"]]
    return add("FAIL" if bad else "ok", "manifest",
               f"{len(files) - len(bad)}/{len(files)} files match" + (f"; e.g. {bad[:3]}" if bad else ""))


def check_inputs():
    problems, n_cells, frames = [], 0, set()
    for inputs in (C.INPUTS, C.ROOT / "results/courtdyn/backbone2_r2/dev_inputs",
                   C.ROOT / "results/courtdyn/m1_mixunit/intemplate"):
        manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
        for cell in manifest["cells"]:
            if inputs == C.INPUTS and cell["seq"] not in C.EVAL_SEQS:
                continue
            path = inputs / cell["file"].replace("\\", "/")
            n_cells += 1
            if not path.is_file() or sha256(path) != cell["sha256"]:
                problems.append(f"hash {path}")
                continue
            root = C.image_root(cell["seq"])
            for row in json.loads(path.read_text(encoding="utf-8")):
                for image_id in row["image_ids"]:
                    frames.add(root / image_id)
    for pool in ("results/courtdyn/revision_controls_20260912/training/event_holdout_v3.json",
                 "results/courtdyn/m1_mixunit/mixunit_v3_pool.json"):
        for row in json.loads((C.ROOT / pool).read_text(encoding="utf-8")):
            for image_id in row.get("image_ids") or [row["image_id"]]:
                frames.add(C.DATA / image_id)
    missing = [f for f in frames if not f.is_file()]
    if missing:
        problems.append(f"{len(missing)} frames missing, e.g. {missing[0]}")
    if not (C.ROOT / "results/courtdyn/unit_probe_v2.json").is_file():
        problems.append("unit_probe_v2.json missing")
    return add("FAIL" if problems else "ok", "inputs",
               f"{n_cells} cells, {len(frames)} frames" + (f"; {problems[:3]}" if problems else ""))


def run_module(name, args, label, timeout=1800):
    log = work() / "logs" / f"check_{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as handle:
        rc = subprocess.call([sys.executable, "-u", *args], stdout=handle, stderr=subprocess.STDOUT,
                             cwd=str(C.ROOT), timeout=timeout)
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] or [""]
    return add("ok" if rc == 0 else "FAIL", label, f"rc={rc}; {tail[0][:160]}; log {log}")


def code_stamp():
    h = hashlib.sha256()
    for rel in ("tools/run_backbone_a100.py", "tools/family_launch.py", "tools/internvl_launch.py",
                "train/train_durable.py", "train/train_qlora_family.py", "a100/family_policy.py",
                "train/train_qlora_internvl.py", "train/train_qlora_epochs.py",
                "train/train_qlora.py", "a100/eval_controls.py", "a100/model_families.py"):
        h.update((C.ROOT / rel).read_bytes())
    return h.hexdigest()


def check_smoke(key):
    m = B4.MODELS[key]
    snap = (C.MODEL_ROOT / "hf" / "hub" / ("models--" + m["repo"].replace("/", "--"))
            / "snapshots" / m["revision"])
    if m["conditional"] and not (snap / "config.json").is_file():
        return add("warn", f"smoke:{key}", "conditional arm without weights (no gated access yet): "
                                           "not run, skipped")
    stamp = work() / ".stamps" / f"smoke_{key}.ok"
    code = code_stamp()
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == code:
        return add("ok", f"smoke:{key}", "passed earlier for this code (stamp)")
    ok = run_module(None, ["tools/run_backbone_a100.py", "--backbone", key, "--smoke"],
                    f"smoke_{key}", timeout=5400)
    if ok:
        D.atomic_write_text(stamp, code + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gpu", action="store_true", help="CPU-only checks (no gpu, no smoke)")
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--backbones", nargs="*", default=B4.active())
    args = ap.parse_args()
    fatal = False
    if not args.no_gpu:
        fatal |= not check_gpu()
    check_disk()
    fatal |= not check_durability()
    if not args.skip_manifest:
        fatal |= not check_manifest()
    fatal |= not check_inputs()
    fatal |= not run_module(None, ["-m", "a100.selftest"], "selftest", timeout=1800)
    if not args.no_gpu:
        fatal |= not run_module(None, ["-m", "a100.preflight"], "preflight", timeout=1800)
        if not fatal:
            for key in args.backbones:
                fatal |= not check_smoke(key)
    width = max(len(r[1]) for r in ROWS)
    print("\n==== check summary ====")
    for status, name, detail in ROWS:
        print(f"  {status:4s}  {name:<{width}}  {detail[:150]}")
    D.atomic_write_json(work() / "logs" / "check_all.json",
                        [dict(status=s, check=n, detail=d) for s, n, d in ROWS])
    print("RESULT:", "FAIL" if fatal else "PASS")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
