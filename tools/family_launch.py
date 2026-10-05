#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run an evaluation entry point with the per-family image policy pinned and crash-safe I/O.

Used for every evaluation cell and text probe of the A100 runs (backbone 4 and the T36
zero-shot sweep).  It changes nothing about what is computed; it fixes two things from outside,
because a100/eval_controls.py and a100/model_families.py are hashed into published receipts:

  image policy  InternVL: one 448x448 tile per frame (tools/internvl_launch.py, identical to the
                third-backbone run).  Gemma3: pan-and-scan explicitly off (the processor's fixed
                896x896 input per frame).  Other families are untouched.  Receipt objects that
                json cannot encode (transformers' SizeDict) are converted.
  durability    the machine may be restarted at any moment, so inside this process
                  * every *.json written with Path.write_text is written atomically (tmp, fsync,
                    rename, fsync dir): run_config.json and summary.json can never be torn;
                  * every *.jsonl opened for writing or appending fsyncs on each flush(), and
                    eval_controls flushes after every prediction;
                  * before starting, torn tails of the cell's *.jsonl logs are cut off (kept as
                    *.torn-<time>), and an empty output directory left by a crash before its
                    run_config.json existed is removed (otherwise eval_controls refuses to start:
                    the 2026-09-27 mkdir loop).

Usage: python tools/family_launch.py <script.py | -m module> [args...]
"""
from __future__ import annotations

import builtins
import pathlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import durable as D                      # noqa: E402
from a100 import model_families as MF              # noqa: E402
import internvl_launch                             # noqa: E402,F401  InternVL policy + jsonable receipt

from a100 import family_policy as FP              # noqa: E402

_inspect = MF.inspect
_LAST_SPEC = [None]          # set by inspect(); build_infer inspects before loading the processor
_kwargs_after_internvl = MF.processor_kwargs
_observed_after_internvl = MF.observed_policy


def inspect(model_path):
    """MF.inspect, plus the families registered in a100/family_policy.py."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    found = FP.detect(config)
    result = (found[1], config) if found else _inspect(model_path)
    _LAST_SPEC[0] = result[0]
    return result


def processor_kwargs(spec, max_pixels):
    pinned = FP.policy(spec)
    return pinned if pinned is not None else _kwargs_after_internvl(spec, max_pixels)


def observed_policy(processor):
    if _LAST_SPEC[0] is not None:
        FP.check(_LAST_SPEC[0], processor)      # fail closed if the pinned policy did not apply
    out = dict(_observed_after_internvl(processor))
    ip = getattr(processor, "image_processor", None)
    for key in ("do_pan_and_scan", "max_patches", "crop_to_patches", "patch_size"):
        if ip is not None and getattr(ip, key, None) is not None:
            out[key] = getattr(ip, key)
    return D.jsonable(out)


MF.inspect = inspect
MF.processor_kwargs = processor_kwargs
MF.observed_policy = observed_policy

# ---------------------------------------------------------------- durable I/O
_write_text = pathlib.Path.write_text
_path_open = pathlib.Path.open
_open = builtins.open


def write_text(self, data, encoding=None, errors=None, newline=None):
    if self.suffix == ".json" and errors is None and newline is None:
        D.atomic_write_text(self, data, encoding or "utf-8")
        return len(data)
    return _write_text(self, data, encoding=encoding, errors=errors, newline=newline)


def _wants_fsync(file, mode):
    return (isinstance(file, (str, Path)) and str(file).endswith(".jsonl")
            and any(m in mode for m in ("a", "w")) and "+" not in mode)


def path_open(self, mode="r", *args, **kwargs):
    handle = _path_open(self, mode, *args, **kwargs)
    return D.FsyncOnFlush(handle) if _wants_fsync(self, mode) else handle


def open_(file, mode="r", *args, **kwargs):
    handle = _open(file, mode, *args, **kwargs)
    return D.FsyncOnFlush(handle) if _wants_fsync(file, mode) else handle


pathlib.Path.write_text = write_text
pathlib.Path.open = path_open
builtins.open = open_


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def prepare_outputs(argv):
    """Repair what an interrupted attempt of this cell left behind."""
    dirs = []
    output = _arg(argv, "--output")
    if output:
        dirs.append(Path(output))
    out, label = _arg(argv, "--out"), _arg(argv, "--label")
    if out and label:
        dirs.append(Path(out) / label)
    for directory in dirs:
        if not directory.is_dir():
            continue
        for log in sorted(directory.glob("*.jsonl")):
            removed = D.repair_jsonl(log)
            if removed:
                print(f"[family_launch] cut a torn tail of {removed} bytes from {log}", flush=True)
        if not any(directory.iterdir()):
            directory.rmdir()
            print(f"[family_launch] removed empty output directory {directory}", flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "-m":
        module, rest = sys.argv[2], sys.argv[3:]
        prepare_outputs(rest)
        sys.argv = [module] + rest
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        target, rest = Path(sys.argv[1]).resolve(), sys.argv[2:]
        prepare_outputs(rest)
        sys.argv = [str(target)] + rest
        if str(target.parent) not in sys.path:
            sys.path.insert(0, str(target.parent))
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
