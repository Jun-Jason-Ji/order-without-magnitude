#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Families added for the A100 runs: spec, pinned image policy and a fail-closed check.

a100/model_families.py and train/model_family.py are hashed into published receipts, so they
are not edited.  A family added here is injected at runtime -- into evaluation by
tools/family_launch.py and into training by train/train_qlora_family.py -- from this single
table, so training and evaluation cannot drift apart.  InternVL is not here: its policy lives in
tools/internvl_launch.py, which predates this file and already ran locally.

Adding a family = one FAMILIES entry: how to recognise its config, its ModelFamilySpec, the
processor kwargs that pin its image budget, and a check that the loaded processor obeys them.

  gemma3              SigLIP + Gemma.  Fixed 896x896 per frame (256 tokens); pan-and-scan OFF.
  deepseek_vl_hybrid  SigLIP (384) + SAM (1024) + LLaMA (DeepSeek-VL-7B).  Fixed dual-resolution
                      input (576 tokens per frame); no knob, the processor defaults are recorded.
  pixtral             Pixtral-ViT + Mistral (Pixtral-12B, llava layout).  Variable resolution;
                      longest edge capped at 512 px, i.e. 512x288 for a 960x540 frame: 576
                      patches of 16 px + 18 row breaks per frame, ~2.4k visual tokens per
                      4-frame item (147k px per frame, 73% of Qwen's 200,704-px budget).  The
                      pixel-equal cap (592 px, ~800 tokens per frame) would triple the visual
                      tokens of every other family; 512 is the largest 16-px-aligned cap that
                      keeps the item within ~2.5k visual tokens.
  idefics3            (Idefics3-8B-Llama3) is NOT here: model_type idefics3 is the frozen SmolVLM2
                      family (train/model_family.py): no image splitting, longest edge 384,
                      which the 364-px vision encoder turns into one 364x364 view per frame.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.model_family import ModelFamilySpec           # noqa: E402
from a100 import model_families as MF                    # noqa: E402

LANG = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

DEEPSEEK_VL_HYBRID = ModelFamilySpec(
    name="deepseek_vl_hybrid", model_types=("deepseek_vl_hybrid",), trust_remote_code=False,
    language_lora_targets=LANG,
    vision_exclude_pattern=r".*(vision_model|high_res_vision|aligner).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values", "high_res_pixel_values"}),
    image_budget_kind="processor_default", smoke_longest_edge=1024, supports_video=False)

PIXTRAL = ModelFamilySpec(
    name="pixtral", model_types=("llava",), trust_remote_code=False,
    language_lora_targets=LANG,
    vision_exclude_pattern=r".*(vision_tower|multi_modal_projector).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values", "image_sizes"}),
    image_budget_kind="longest_edge", smoke_longest_edge=512, supports_video=False)

GEMMA3_POLICY = {"do_pan_and_scan": False}
PIXTRAL_POLICY = {"size": {"longest_edge": 512}}


def _check_gemma3(processor):
    if bool(getattr(processor.image_processor, "do_pan_and_scan", False)):
        raise AssertionError("Gemma3 pan-and-scan is on; the pinned policy is off")


def _check_pixtral(processor):
    size = getattr(processor.image_processor, "size", None)
    edge = size.get("longest_edge") if hasattr(size, "get") else getattr(size, "longest_edge", None)
    if int(edge or 0) != 512:
        raise AssertionError(f"Pixtral longest_edge is {edge}, pinned 512")


def _is_pixtral(config) -> bool:
    vision = getattr(config, "vision_config", None)
    return config.model_type == "llava" and getattr(vision, "model_type", None) == "pixtral"


# name -> (recognise(config), spec, processor kwargs, check(processor))
FAMILIES: dict[str, tuple[Callable[[Any], bool], ModelFamilySpec, dict, Callable]] = {
    "gemma3": (lambda c: MF.GEMMA3.matches(c.model_type), MF.GEMMA3, GEMMA3_POLICY, _check_gemma3),
    "deepseek_vl_hybrid": (lambda c: c.model_type == "deepseek_vl_hybrid", DEEPSEEK_VL_HYBRID,
                           {}, lambda p: None),
    "pixtral": (_is_pixtral, PIXTRAL, PIXTRAL_POLICY, _check_pixtral),
}


def detect(config):
    """(name, spec) of an added family, or None for anything else."""
    for name, (recognise, spec, _, _) in FAMILIES.items():
        if recognise(config):
            return name, spec
    return None


def policy(spec) -> dict | None:
    for _, s, kwargs, _ in FAMILIES.values():
        if s.name == spec.name:
            return dict(kwargs)
    return None


def check(spec, processor) -> bool:
    """Run the family's processor check; False if the spec is not an added family."""
    for _, s, _, fn in FAMILIES.values():
        if s.name == spec.name:
            fn(processor)
            return True
    return False
