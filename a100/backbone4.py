#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backbone-4 registry: the A100 trained-replication backbones, pinned, in run order.

Single source for tools/run_backbone_a100.py, a100/download_models.py, a100/check_all.py,
a100/progress.py and a100/run_all.sh.  ACTIVE is the preregistered list, in run order
(results/courtdyn/backbone4_a100/PREREGISTRATION.json names each key with its revision; the
orchestrator refuses a key it does not name).  A100_BACKBONES="a b" narrows it for one call.

  repo, revision  pinned Hugging Face snapshot
  wrapper         training entry used by train/train_durable.py:
                    family    train/train_qlora_family.py   (a100/family_policy.py families)
                    idefics3  train/train_qlora_idefics3.py (frozen SmolVLM2 family)
                    internvl  train/train_qlora_internvl.py
  gated           needs HF_TOKEN with accepted licence
  conditional     run iff access is available when the run reaches it; otherwise recorded as
                  not run (no substitution)
  gb, hours       download size; rough A100-40GB hours for the whole protocol
"""
from __future__ import annotations

import os

MODELS = {
    "pixtral_12b": dict(repo="mistral-experimental/pixtral-12b",   # = mistral-community/pixtral-12b (redirect)
                        revision="c2756cbbb9422eba9f6c5c439a214b0392dfc998", wrapper="family",
                        gated=False, conditional=False, gb=25.4, hours=12),
    "idefics3_8b": dict(repo="HuggingFaceM4/Idefics3-8B-Llama3",
                        revision="fddb4ff79181e55a994674777e06cd5456ce3dc3", wrapper="idefics3",
                        gated=False, conditional=False, gb=17.0, hours=7),
    "gemma3_12b": dict(repo="google/gemma-3-12b-it",
                       revision="96b6f1eccf38110c56df3a15bffe176da04bfd80", wrapper="family",
                       gated=True, conditional=True, gb=24.4, hours=11),
    # registered, not preregistered for this run (kept so a later record can add them)
    "gemma3_4b": dict(repo="google/gemma-3-4b-it",
                      revision="093f9f388b31de276ce2de164bdc2081324b9767", wrapper="family",
                      gated=True, conditional=True, gb=8.6, hours=5),
    # Llama 3.2 Vision (mllama, cross-attention).  Weights are the byte-identical ModelScope mirror
    # LLM-Research/Llama-3.2-11B-Vision-Instruct (a100/fetch_modelscope_model.py); "revision" is the
    # snapshot pin = sha256 over the sorted "path sha256" lines of its 18 files (original/ excluded),
    # checked against MODELSCOPE_MANIFEST.json before any stage.  Record: results/courtdyn/backbone_llama32.
    "llama32_11b_vision": dict(repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
                               revision="a99bf12aac301325dae596ebc1b094eb42c81502a96aad47106b02bb099c779b",
                               wrapper="llama32", gated=True, conditional=True, gb=21.4, hours=15,
                               source="modelscope:LLM-Research/Llama-3.2-11B-Vision-Instruct@master"),
    "internvl3_8b": dict(repo="OpenGVLab/InternVL3-8B-hf",
                         revision="259a3b64a14623c0ec91a045cb43f7c5af5fa6af", wrapper="internvl",
                         gated=False, conditional=False, gb=15.9, hours=7),
    "deepseek_vl_7b": dict(repo="deepseek-community/deepseek-vl-7b-chat",
                           revision="4d6f7f4aad464b5426a5321d176a11da6191ba60", wrapper="family",
                           gated=False, conditional=False, gb=14.7, hours=7),
}
ACTIVE = ["pixtral_12b", "idefics3_8b", "gemma3_12b"]


def active() -> list[str]:
    chosen = os.environ.get("A100_BACKBONES", "").split() or ACTIVE
    unknown = [k for k in chosen if k not in MODELS]
    if unknown:
        raise SystemExit(f"unknown backbone(s) {unknown}; known: {sorted(MODELS)}")
    return chosen
