#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a repository script with the InternVL image policy pinned to one tile per frame.

a100/model_families.py asks InternVL for {"max_num_tiles": 1}, but the -hf InternVLProcessor
has no such knob: its processor-level default crop_to_patches=True tiles a 960x540 frame into
2 tiles + a thumbnail (12 tiles, ~3.2k tokens per 4-frame item, which also overflows 8 GB in
training). The effective knob is the image processor's max_patches. That file's sha256 is
recorded in ~100 published cell receipts, so it is not edited; the policy is replaced here, at
runtime, for the InternVL family only (every other family is untouched), and the replacement
is what each receipt records under "processor_policy".

Usage: python tools/internvl_launch.py <script.py> [script args...]
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100 import model_families as MF              # noqa: E402

INTERNVL_POLICY = {"max_patches": 1}
_processor_kwargs = MF.processor_kwargs


def processor_kwargs(spec, max_pixels):
    if spec.name == MF.INTERNVL.name:
        return dict(INTERNVL_POLICY)
    return _processor_kwargs(spec, max_pixels)


MF.processor_kwargs = processor_kwargs


def _jsonable(o):
    # The -hf InternVL image processor reports sizes as transformers' SizeDict, which
    # json.dumps rejects when the receipt is written to run_config.json (2026-09-27, first
    # base cell).  Converted here, for this launcher only; the receipt's values are unchanged.
    import dataclasses
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if dataclasses.is_dataclass(o):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if hasattr(o, "keys"):
        return {k: _jsonable(o[k]) for k in o.keys()}
    return repr(o)


_observed_policy = MF.observed_policy
MF.observed_policy = lambda processor: _jsonable(_observed_policy(processor))

if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target)] + sys.argv[2:]
    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
