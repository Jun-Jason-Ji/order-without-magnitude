#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU-only model-family registry shared by future SFT/GRPO adapters.

Keeping these differences declarative prevents the second-backbone experiment from
silently using Qwen-only image budgets, LoRA exclusions or tensor-padding rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    model_types: tuple[str, ...]
    trust_remote_code: bool
    language_lora_targets: tuple[str, ...]
    vision_exclude_pattern: str
    non_sequence_multimodal_keys: frozenset[str]
    image_budget_kind: str
    smoke_longest_edge: int
    supports_video: bool

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in self.model_types


QWEN_VL = ModelFamilySpec(
    name="qwen_vl",
    model_types=("qwen3_5", "qwen3_5_moe", "qwen2_vl", "qwen2_5_vl", "qwen3_vl"),
    trust_remote_code=True,
    language_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    vision_exclude_pattern=r".*(visual|vision_tower|vision_model|merger|projector).*",
    non_sequence_multimodal_keys=frozenset({"pixel_values", "image_grid_thw"}),
    image_budget_kind="max_pixels",
    smoke_longest_edge=448,
    supports_video=True,
)

SMOLVLM2 = ModelFamilySpec(
    name="smolvlm2",
    model_types=("smolvlm", "idefics3"),
    trust_remote_code=False,
    language_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    vision_exclude_pattern=r".*(vision_model|connector|modality_projection).*",
    non_sequence_multimodal_keys=frozenset(
        {"pixel_values", "pixel_attention_mask", "image_sizes"}),
    image_budget_kind="longest_edge_no_split",
    smoke_longest_edge=384,
    supports_video=True,
)

SPECS = (QWEN_VL, SMOLVLM2)


def from_model_type(model_type: str) -> ModelFamilySpec:
    normalized = str(model_type).lower()
    for spec in SPECS:
        if spec.matches(normalized):
            return spec
    raise ValueError(f"unsupported multimodal model_type={model_type!r}; "
                     f"known={sorted(t for s in SPECS for t in s.model_types)}")


def inspect_local_model(model_path: str | Path) -> tuple[ModelFamilySpec, Any]:
    """Load only AutoConfig; no tensors and no CUDA context are created."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    return from_model_type(config.model_type), config


def processor_policy(spec: ModelFamilySpec, requested_max_pixels: int = 200704) -> dict:
    """Return explicit, non-interchangeable image preprocessing policy."""
    if spec.image_budget_kind == "max_pixels":
        return {"max_pixels": int(requested_max_pixels)}
    return {"do_image_splitting": False,
            # Transformers forwards ``size`` to Idefics3ImageProcessor.  A
            # top-level ``longest_edge`` kwarg is silently ignored and leaves
            # SmolVLM2 at 1536 px, which can break an 8GB smoke budget.
            "size": {"longest_edge": int(spec.smoke_longest_edge)}}


def assert_processor_policy(spec: ModelFamilySpec, processor: Any) -> None:
    """Fail closed when Transformers silently ignores an image-budget kwarg."""

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise AssertionError(f"{spec.name} processor has no image_processor")
    if spec.image_budget_kind == "longest_edge_no_split":
        if getattr(image_processor, "do_image_splitting", None) is not False:
            raise AssertionError("SmolVLM2 image splitting was not disabled")
        size = getattr(image_processor, "size", None)
        if hasattr(size, "to_dict"):
            size = size.to_dict()
        if not isinstance(size, dict):
            try:
                size = dict(size)
            except (TypeError, ValueError):
                size = {}
        actual = size.get("longest_edge")
        if int(actual or -1) != int(spec.smoke_longest_edge):
            raise AssertionError(
                f"SmolVLM2 longest_edge ignored: expected "
                f"{spec.smoke_longest_edge}, got {actual}"
            )


def is_token_sequence_tensor(spec: ModelFamilySpec, key: str, tensor: Any,
                             prompt_length: int) -> bool:
    """Whether completion padding must extend the tensor's last dimension."""
    if key in spec.non_sequence_multimodal_keys:
        return False
    shape = getattr(tensor, "shape", ())
    return bool(shape) and shape[-1] == prompt_length
