#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocessing policy for the backbones added by the A100 sweep.

train/model_family.py knows two families (Qwen-VL and SmolVLM2) and its file
hash is recorded inside frozen execution plans, so extending it in place would
invalidate those receipts. This module imports those two specs unchanged and
adds the families the multi-backbone arm needs.

Honest boundary, to be carried into the paper: these families do not share an
image budget. Qwen takes a pixel cap, Idefics3/SmolVLM2 a longest edge,
InternVL a tile count, LLaVA-OneVision and Gemma3 a fixed processor grid.
We therefore do not equalise resolution across families; we record what each
one actually did and compare rank statistics, which are invariant to a
per-family monotone change of scale, rather than claiming matched inputs.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.model_family import (QWEN_VL, SMOLVLM2, ModelFamilySpec,  # noqa: E402
                                processor_policy as base_processor_policy)

INTERNVL = ModelFamilySpec(
    name="internvl",
    model_types=("internvl", "internvl_chat", "internvl3"),
    trust_remote_code=False,          # the -hf conversions are native
    language_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    vision_exclude_pattern=r".*(vision_tower|vision_model|mlp1|multi_modal_projector).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values", "image_flags"}),
    image_budget_kind="tiles",
    smoke_longest_edge=448,
    supports_video=True,
)

LLAVA_ONEVISION = ModelFamilySpec(
    name="llava_onevision",
    model_types=("llava_onevision", "llava_next", "llava"),
    trust_remote_code=False,
    language_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    vision_exclude_pattern=r".*(vision_tower|multi_modal_projector).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values", "image_sizes"}),
    image_budget_kind="processor_default",
    smoke_longest_edge=384,
    supports_video=True,
)

GEMMA3 = ModelFamilySpec(
    name="gemma3",
    model_types=("gemma3", "gemma3_text"),
    trust_remote_code=False,
    language_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    vision_exclude_pattern=r".*(vision_tower|multi_modal_projector).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values"}),
    image_budget_kind="processor_default",
    smoke_longest_edge=896,
    supports_video=False,
)

EXTRA_SPECS = (INTERNVL, LLAVA_ONEVISION, GEMMA3)
ALL_SPECS = (QWEN_VL, SMOLVLM2) + EXTRA_SPECS
BASE_FAMILY_NAMES = {QWEN_VL.name, SMOLVLM2.name}


def from_model_type(model_type: str) -> ModelFamilySpec:
    normalized = str(model_type).lower()
    for spec in ALL_SPECS:
        if spec.matches(normalized):
            return spec
    raise ValueError(
        "unsupported multimodal model_type=%r; known=%s. Add a spec here "
        "rather than letting a model run with another family's image budget."
        % (model_type, sorted(t for s in ALL_SPECS for t in s.model_types)))


def inspect(model_path: str | Path) -> tuple[ModelFamilySpec, Any]:
    """AutoConfig only: no tensors, no CUDA context."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    return from_model_type(config.model_type), config


def processor_kwargs(spec: ModelFamilySpec, max_pixels: int) -> dict:
    """Explicit, family-appropriate image budget.

    Qwen and SmolVLM2 delegate to the frozen policy so their cells stay
    byte-comparable with the published runs. The added families have no
    equivalent single knob, so we pass nothing and record the processor's own
    defaults in the receipt instead of pretending to match.
    """
    if spec.name in BASE_FAMILY_NAMES:
        return base_processor_policy(spec, max_pixels)
    if spec.image_budget_kind == "tiles":
        # InternVL splits a frame into up to N 448-px tiles; four frames per
        # item make the default explode, so cap it and say so.
        return {"max_num_tiles": 1}
    return {}


def observed_policy(processor: Any) -> dict:
    """What the loaded processor will actually do, for the run receipt."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return {"image_processor": None}
    out = {}
    for key in ("size", "max_pixels", "min_pixels", "do_image_splitting",
                "do_resize", "patch_size", "max_num_tiles", "crop_size"):
        value = getattr(ip, key, None)
        if value is None:
            continue
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        try:
            out[key] = dict(value) if isinstance(value, dict) else value
        except (TypeError, ValueError):
            out[key] = repr(value)
    return out
