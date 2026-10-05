#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run train/train_qlora_epochs.py (and through it train/train_qlora.py) for InternVL3.

train/train_qlora.py and train/model_family.py are byte-frozen (their hashes sit in execution
plans and audits) and know two families, Qwen-VL and SmolVLM2. This wrapper adds the InternVL
family from a100/model_families.py by patching three functions of the imported
train.model_family module before the trainer imports them; for any other model type the
original functions are called, so this file is inert outside InternVL.

  inspect_local_model      -> INTERNVL spec for model_type "internvl"
  processor_policy         -> the same kwargs the InternVL evaluation uses
                              (tools/internvl_launch.py: max_patches=1, i.e. one 448x448
                              tile per frame, no dynamic tiling)
  assert_processor_policy  -> fail closed unless the loaded processor really does that

Usage: the train_qlora_epochs.py arguments, unchanged.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.model_family as TMF                   # noqa: E402
from a100 import model_families as MF              # noqa: E402
sys.path.insert(0, str(ROOT / "tools"))
import internvl_launch                             # noqa: E402,F401  (pins MF.processor_kwargs)

_inspect, _policy, _assert = (TMF.inspect_local_model, TMF.processor_policy,
                              TMF.assert_processor_policy)


def inspect_local_model(model_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True,
                                        trust_remote_code=False)
    if MF.INTERNVL.matches(config.model_type):
        return MF.INTERNVL, config
    return _inspect(model_path)


def processor_policy(spec, requested_max_pixels=200704):
    if spec.name == MF.INTERNVL.name:
        return MF.processor_kwargs(spec, requested_max_pixels)
    return _policy(spec, requested_max_pixels)


def assert_processor_policy(spec, processor):
    if spec.name != MF.INTERNVL.name:
        return _assert(spec, processor)
    # The processor-level default crop_to_patches=True overrides the image processor's own
    # flag, so the cap that actually bounds tiling is max_patches.
    ip = processor.image_processor
    if int(getattr(ip, "max_patches", -1)) != 1:
        raise AssertionError(f"InternVL tiling not capped: max_patches={ip.max_patches}")


TMF.inspect_local_model = inspect_local_model
TMF.processor_policy = processor_policy
TMF.assert_processor_policy = assert_processor_policy

TARGET = Path(__file__).resolve().parent / "train_qlora_epochs.py"
sys.argv = [str(TARGET)] + sys.argv[1:]
runpy.run_path(str(TARGET), run_name="__main__")
