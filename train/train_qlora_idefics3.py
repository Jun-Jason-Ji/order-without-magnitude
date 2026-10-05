#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run train/train_qlora_epochs.py (and through it train/train_qlora.py) for Idefics3-8B.

model_type "idefics3" already belongs to the frozen SmolVLM2 family of train/model_family.py
(same image policy: no splitting, longest edge 384), so no family injection is needed.  One
thing is: train/train_qlora.py fixes a bf16-autocast dtype mismatch of that family by hooking
exactly one module whose class is named "SmolVLMConnector" and raises otherwise
(T23 smoke, 2026-09-03).  Idefics3 -- the architecture SmolVLM is derived from -- has the same
connector under the name "Idefics3Connector", so the class is given the SmolVLM name before the
model is built: the frozen trainer then applies its unchanged hook (connector output cast to the
embedding dtype) to it.  Nothing else changes; for other models this file is inert.

Usage: the train_qlora_epochs.py arguments, unchanged.
"""
import runpy
import sys
from pathlib import Path

from transformers.models.idefics3 import modeling_idefics3 as _M

_M.Idefics3Connector.__name__ = "SmolVLMConnector"

TARGET = Path(__file__).resolve().parent / "train_qlora_epochs.py"
sys.argv = [str(TARGET)] + sys.argv[1:]
runpy.run_path(str(TARGET), run_name="__main__")
