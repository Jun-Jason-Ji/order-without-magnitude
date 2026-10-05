#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paths, model registry and protocol constants for the single-A100 runs.

Everything the other machine needs to know is resolved here from environment
variables, so no script in this package contains a machine-specific path.

    COURTDYN_ROOT    repository root (default: the parent of this package)
    COURTDYN_DATA    rendered frame root (default: $ROOT/data/courtdyn)
    COURTDYN_INPUTS  prepared control pools (default: the frozen 20260912 set)
    COURTDYN_OUT     where new results are written (default: $ROOT/results/a100)
    COURTDYN_MODELS  local base-weight root; each entry below may also be
                     overridden individually by its own variable
    HF_HOME          standard Hugging Face cache, used when a model is pulled

Nothing here imports torch, so it can be read and validated on a CPU box.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COURTDYN_ROOT") or PKG.parent).resolve()
DATA = Path(os.environ.get("COURTDYN_DATA") or ROOT / "data" / "courtdyn").resolve()
INPUTS = Path(os.environ.get("COURTDYN_INPUTS")
              or ROOT / "results/courtdyn/revision_controls_20260912").resolve()
OUT = Path(os.environ.get("COURTDYN_OUT") or ROOT / "results" / "a100").resolve()
MODEL_ROOT = Path(os.environ.get("COURTDYN_MODELS") or ROOT / "models").resolve()

# --------------------------------------------------------------------------
# Protocol constants.  These reproduce the frozen 2026-09-12 control protocol
# exactly; changing one makes a run incomparable with the published cells, so
# each is recorded in every receipt.
# --------------------------------------------------------------------------
MAX_PIXELS = 200704          # 448^2, the Qwen image budget used for every
                             # existing CourtDyn cell
MAX_NEW_TOKENS = 128
CANVAS = (960, 540)
ITEMS_PER_ARM = 280
ARMS_THREE = ["m_height", "cm_height", "px_height"]
ARMS_SIX = ARMS_THREE + ["m_noheight", "cm_noheight", "px_noheight"]
EVAL_SEQS = ["Q2_top_480-510", "Q1_top_0-30"]
TRAIN_SEEDS = [42, 43, 44]

# --------------------------------------------------------------------------
# Base weights.  `local` is tried first (so an air-gapped A100 works from the
# shipped bundle), then `hf` through the normal Hugging Face cache.
# `family` selects the preprocessing policy in a100/model_families.py.
# `precision` is the default for that model; nf4 reproduces the frozen
# protocol, bf16 is the A100-native alternative and is a *different arm*.
# --------------------------------------------------------------------------
BACKBONES = {
    # The backbone every existing CourtDyn adapter was trained on.  Keep nf4:
    # multi-seed replication is only meaningful against the frozen protocol.
    "qwen35-4b": dict(env="QWEN35_4B", hf="Qwen/Qwen3.5-4B",
                      family="qwen_vl", precision="nf4",
                      role="replication backbone"),
    # Same weights, bf16: the precision control that tells us how much of any
    # public-model gap is quantisation rather than the model.
    "qwen35-4b-bf16": dict(env="QWEN35_4B", hf="Qwen/Qwen3.5-4B",
                           family="qwen_vl", precision="bf16",
                           role="precision control"),
    "qwen25vl-3b": dict(env="QWEN25_VL_3B", hf="Qwen/Qwen2.5-VL-3B-Instruct",
                        family="qwen_vl", precision="bf16",
                        role="public arm"),
    "qwen25vl-7b": dict(env="QWEN25_VL_7B", hf="Qwen/Qwen2.5-VL-7B-Instruct",
                        family="qwen_vl", precision="bf16",
                        role="public arm (has frozen nf4 cells to compare)"),
    "internvl3-2b": dict(env="INTERNVL3_2B", hf="OpenGVLab/InternVL3-2B-hf",
                         family="internvl", precision="bf16",
                         role="public arm, second family"),
    "internvl3-8b": dict(env="INTERNVL3_8B", hf="OpenGVLab/InternVL3-8B-hf",
                         family="internvl", precision="bf16",
                         role="public arm, second family"),
    "llava-ov-7b": dict(env="LLAVA_OV_7B",
                        hf="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                        family="llava_onevision", precision="bf16",
                        role="public arm, third family"),
    "gemma3-4b": dict(env="GEMMA3_4B", hf="google/gemma-3-4b-it",
                      family="gemma3", precision="bf16",
                      role="public arm, fourth family"),
    "smolvlm2-2b": dict(env="SMOLVLM2_2B", hf="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                        family="smolvlm2", precision="bf16",
                        role="public arm, fifth family (known degenerate on "
                             "CourtDyn: answers letters)"),
}

# Public models that go into the zero-shot backbone sweep, in run order:
# cheapest first so a short session still produces usable rows.
BACKBONE_SWEEP = ["qwen35-4b", "qwen35-4b-bf16", "qwen25vl-3b", "internvl3-2b",
                  "gemma3-4b", "smolvlm2-2b", "qwen25vl-7b", "internvl3-8b",
                  "llava-ov-7b"]

# --------------------------------------------------------------------------
# Adapters.  `pool` is relative to $ROOT; `seed` and `label` name the cell.
# The three seed-42 adapters already exist and ship in the bundle; the rest
# are trained by a100/exp_seeds.py.
# --------------------------------------------------------------------------
TRAIN_POOLS = {
    "v1": "results/courtdyn/revision_controls_20260912/training/event_holdout_v1.json",
    "v3": "results/courtdyn/revision_controls_20260912/training/event_holdout_v3.json",
    # The original 1,400-item pool.  Its Q1_top cell is event-overlapped; see
    # tools/audit_event_splits.py.  Kept so the seed sweep can also cover the
    # checkpoint the manuscript calls "original".
    "native": "results/courtdyn/courtdyn_native_sft_train.json",
}

# Hyperparameters of the frozen matched-training protocol.  train_qlora.py is
# called with exactly these; only --seed varies across the sweep.
TRAIN_ARGS = dict(num_samples=0, allow_base_init=True, budget_slice="diagnostic",
                  num_train_epochs=1, max_steps=-1, lr=1e-4, grad_accum=16,
                  lora_r=16, max_pixels=MAX_PIXELS, save_steps=35)

# Expected optimizer steps, used as a completion assertion per pool.
EXPECTED_STEPS = {"v1": 70, "v3": 70, "native": 88}
EXPECTED_ITEMS = {"v1": 1120, "v3": 1120, "native": 1400}


def adapter_dir(pool: str, seed: int) -> Path:
    """Where a (pool, seed) adapter lives.

    Seed 42 maps onto the existing frozen directories so the sweep reuses them
    instead of retraining and silently producing a second seed-42 checkpoint.
    """
    if seed == 42:
        frozen = {"v1": "courtdyn-revision-v1-eventholdout-s42",
                  "v3": "courtdyn-revision-v3-eventholdout-s42",
                  "native": "courtdyn-native-sft"}
        return ROOT / "models" / frozen[pool]
    return ROOT / "models" / f"courtdyn-{pool}-s{seed}"


def adapter_label(pool: str, seed: int) -> str:
    return f"{pool}_s{seed}"


def backbone_path(name: str) -> Path | None:
    """Resolve a backbone to a local directory, or None to pull from the hub."""
    spec = BACKBONES[name]
    explicit = os.environ.get(spec["env"])
    if explicit:
        return Path(explicit)
    guess = MODEL_ROOT / spec["hf"].split("/")[-1]
    if (guess / "config.json").is_file():
        return guess
    hub = MODEL_ROOT / "hf" / "hub" / ("models--" + spec["hf"].replace("/", "--"))
    if hub.is_dir():
        snapshots = sorted((hub / "snapshots").glob("*"))
        if snapshots:
            return snapshots[-1]
    return None


def image_root(seq: str) -> Path:
    """Frames for one clip, resolved on this machine rather than the manifest.

    The prepared manifest stores the absolute Windows path of the machine that
    built it. Honouring that path would make every run non-portable, so the
    runners resolve frames here and record both paths in the receipt.
    """
    # The first clip built was rendered into the unsuffixed default directory.
    if seq == "Q4_side_480-510" and (DATA / "frames").is_dir():
        return DATA / "frames"
    return DATA / f"frames_{seq}"


def describe() -> dict:
    return {
        "root": str(ROOT), "data": str(DATA), "inputs": str(INPUTS),
        "out": str(OUT), "model_root": str(MODEL_ROOT),
        "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "canvas": list(CANVAS), "eval_seqs": list(EVAL_SEQS),
        "train_seeds": list(TRAIN_SEEDS),
    }
