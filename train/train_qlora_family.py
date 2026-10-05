#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run train/train_qlora_epochs.py (and through it train/train_qlora.py) for an added family.

The generic counterpart of train/train_qlora_internvl.py for every family registered in
a100/family_policy.py (Gemma3, DeepSeek-VL hybrid, Pixtral, ...).  train/train_qlora.py and
train/model_family.py are byte-frozen and know Qwen-VL and SmolVLM2 only, so three functions of
train.model_family are patched before the trainer imports them; the family is recognised from
the model's own config, and for any model that is not an added family the original functions
are called, so this file is inert for Qwen and SmolVLM2.

  inspect_local_model      -> the added family's ModelFamilySpec
  processor_policy         -> the family's pinned kwargs (the same ones tools/family_launch.py
                              pins for evaluation)
  assert_processor_policy  -> the family's fail-closed check on the loaded processor

Usage: the train_qlora_epochs.py arguments, unchanged.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.model_family as TMF                   # noqa: E402
from a100 import family_policy as FP               # noqa: E402

_inspect, _policy, _assert = (TMF.inspect_local_model, TMF.processor_policy,
                              TMF.assert_processor_policy)


def inspect_local_model(model_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True,
                                        trust_remote_code=False)
    found = FP.detect(config)
    if found:
        return found[1], config
    return _inspect(model_path)


def processor_policy(spec, requested_max_pixels=200704):
    pinned = FP.policy(spec)
    return pinned if pinned is not None else _policy(spec, requested_max_pixels)


def assert_processor_policy(spec, processor):
    if not FP.check(spec, processor):
        return _assert(spec, processor)


TMF.inspect_local_model = inspect_local_model
TMF.processor_policy = processor_policy
TMF.assert_processor_policy = assert_processor_policy

TARGET = Path(__file__).resolve().parent / "train_qlora_epochs.py"
sys.argv = [str(TARGET)] + sys.argv[1:]
runpy.run_path(str(TARGET), run_name="__main__")
