#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T36 zero-shot backbone sweep (a100/exp_backbones.py) as run on the A100 in 2026-09.

Identical stage list and cells to `python -m a100.exp_backbones`; two machine-level changes:
  * every cell enters a100/eval_controls.py through tools/family_launch.py, which pins the
    image policy the registry cannot express (InternVL: max_patches = 1, one 448x448 tile per
    frame -- the registry's max_num_tiles is a no-op on the -hf processor -- identical to the
    local InternVL3-2B runs; Gemma3: pan-and-scan off) and makes result files crash-safe;
  * the runner writes its state atomically and refreshes PROGRESS.md.
Descriptive sweep: no decision rule is recorded for it (PREREGISTRATION (f) of backbone 4).

Usage: python tools/a100_t36.py --dry-run | --run [--models ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import exp_backbones as T36                    # noqa: E402
from a100.durable_runner import DurableRunner            # noqa: E402
from a100.runner import python_cmd                       # noqa: E402

LAUNCHER = ROOT / "tools" / "family_launch.py"


def launched(*args):
    args = list(args)
    if args[:2] == ["-m", "a100.eval_controls"]:
        return python_cmd(LAUNCHER, *args)
    return python_cmd(*args)


T36.python_cmd = launched
T36.Runner = DurableRunner


def default_models():
    """The sweep's default model list, minus models whose weights are not on disk (gated ones
    without access); those are logged as not run rather than pulled at evaluation time."""
    from a100 import config as C
    present = [m for m in C.BACKBONE_SWEEP if C.backbone_path(m) is not None]
    missing = [m for m in C.BACKBONE_SWEEP if m not in present]
    if missing:
        print(f"[t36] not run (weights not downloaded, e.g. gated access not granted): {missing}",
              flush=True)
    return present


if __name__ == "__main__":
    if "--models" not in sys.argv and "--run" in sys.argv:
        sys.argv += ["--models", *default_models()]
    sys.exit(T36.main())
